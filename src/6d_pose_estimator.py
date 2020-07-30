#!/usr/bin/env python

import json
import rospy
import cv2, cv_bridge
import numpy as np
import PIL
import message_filters
import yaml
import sys

from std_msgs.msg import String
from sensor_msgs.msg import RegionOfInterest, Image, CameraInfo, PointCloud2
from assembly_part_recognition.msg import InstanceSegmentation2D
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped

import cv2
import argparse
import tensorflow as tf
import numpy as np
import os
import configparser

import open3d as o3d # import both o3d and pcl occurs segmentation faults
import zivid_helper
import copy

import tf.transformations as tf_trans


color_dict = [(255,255,0),(0,0,255),(255,0,0),(255,255,0)] * 10


class PoseEstimator:

    def __init__(self):

        # initalize node
        rospy.init_node('6d_pose_estimator')
        rospy.loginfo("Starting 6d_pose_estimator.py")

        self.params = rospy.get_param("6d_pose_estimation")
        self.class_name = ["side_right", "long_short", "middle", "bottom"]

        # augmented autoencoder and renderer      
        sys.path.append(self.params["augauto_module_path"])
        sys.path.append(self.params["augauto_module_path"] + "/auto_pose/test")
        from aae_maskrcnn_pose_estimator import AePoseEstimator
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params["gpu_id"]
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        self.workspace_path = os.environ.get('AE_WORKSPACE_PATH')
        if self.workspace_path == None:
            print 'Please define a workspace path:\n'
            print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
            exit(-1)
        test_configpath = os.path.join(self.workspace_path,'cfg_eval', self.params["test_config"])
        test_args = configparser.ConfigParser()
        test_args.read(test_configpath)
        self.ae_pose_est = AePoseEstimator(test_configpath)
        self.ply_model_paths = [str(train_args.get('Paths','MODEL_PATH')) for train_args in self.ae_pose_est.all_train_args]
        self.bridge = cv_bridge.CvBridge()

        # subscribers
        rgb_sub = message_filters.Subscriber("/zivid_camera/color/image_color", Image)
        depth_sub = message_filters.Subscriber("/zivid_camera/depth/image_raw", Image)
        point_sub = message_filters.Subscriber("/zivid_camera/points", PointCloud2)
        is_sub = message_filters.Subscriber("/assembly/zivid/furniture_part/is_results", InstanceSegmentation2D)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, point_sub, is_sub], queue_size=1, slop=0.5)
        rospy.loginfo("Starting zivid rgb-d subscriber with time synchronizer")
        self.ts.registerCallback(self.estimate_6d_pose)

        # publishers
        self.vis_pub = rospy.Publisher('/assembly/zivid/furniture_part/6d_pose_vis_results', Image, queue_size=1)
        self.pose_pubs = [] # "side_right", "long_short", "middle", "bottom"
        self.idx2color = [[13, 255, 128], [0, 104, 255], [217, 12, 232], [232, 222, 12]]
        self.dims = []
        self.cloud_GTs = []
        for ply_model in self.ply_model_paths:
            cloud = o3d.io.read_point_cloud(ply_model)
            self.dims.append(cloud.get_max_bound())
            self.cloud_GTs.append(cloud)
            model_name = ply_model.split('/')[-1][5:-4]
            self.pose_pubs.append(rospy.Publisher(
                '/assembly/zivid/furniture_part/pose_{}'.format(model_name), PoseStamped, queue_size=1))
        self.pub_detections = rospy.Publisher('/assembly/zivid/furniture_part/detected_objects', Detection3DArray, queue_size=1)
        self.pub_markers = rospy.Publisher('/assembly/zivid/furniture_part/markers', MarkerArray, queue_size=1)
        
    def estimate_6d_pose(self, rgb, depth, cloud, is_results):

        # get rgb, depth, point cloud, is_results, K
        rospy.loginfo_once("Estimating 6D pose of objects")
        rgb_original = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb_resized = cv2.resize(rgb_original.copy(), (self.params["width"], self.params["height"]))

        depth_original = self.bridge.imgmsg_to_cv2(depth)
        depth_resized = cv2.resize(depth_original.copy(), dsize=(self.params["width"], self.params["height"]), interpolation=cv2.INTER_AREA)
        depth_resized = np.where(np.isnan(depth_resized), 0, depth_resized)  

        # gather all detected crops
        boxes, scores, labels = [], [], []
        boxes, scores, labels, is_masks = self.gather_is_results(rgb_resized, depth_resized, boxes, scores, labels, is_results)

        # 6D pose estimation for all detected crops using AAE
        all_pose_estimates, all_class_idcs = self.ae_pose_est.process_pose(boxes, labels, rgb_resized, depth_resized)

        # visualize pose estimation results
        self.visualize_pose_estimation_results(all_pose_estimates, all_class_idcs, labels, boxes, scores, rgb_resized)

        detection_array = Detection3DArray()
        detection_array.header = rgb.header
        npcloud_zivid = zivid_helper.convertZividCloudFromRosToNumpy(cloud)
        
        # 6D object pose refinement using ICP on pointcloud
        for i, (pose_estimate, class_id) in enumerate(zip(all_pose_estimates, all_class_idcs)):

            # crop zivid cloud with instance mask        
            is_mask_original = cv2.resize(is_masks[i], (rgb_original.shape[1], rgb_original.shape[0]), interpolation=cv2.INTER_AREA)
            is_mask_original = is_mask_original[np.isfinite(is_mask_original)]

            # remove outliers and scale it from m to mm
            cloud_zivid = zivid_helper.convertZividCloudFromNumpyToOpen3d(npcloud_zivid.copy(), mask=is_mask_original)
            cloud_zivid, _ = cloud_zivid.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            cloud_zivid.scale(1000)

            cloud_GT = copy.deepcopy(self.cloud_GTs[class_id])
            # rotate cloud_GT according to the estimated rotation then translate cloud_GT from origin to the center of cloud_zivid
            T_rot = np.eye(4)
            T_rot[:3, :3] = pose_estimate[:3, :3] # test needed: trans from 2D bbox or center
            T_trans = np.eye(4)
            T_trans[:3, 3] = cloud_zivid.get_center()
            H_init_cam2obj = np.matmul(T_trans, T_rot)
            cloud_GT.transform(H_init_cam2obj)
            icp_result = self.icp_refinement(source_cloud=copy.deepcopy(cloud_GT), target_cloud=copy.deepcopy(cloud_zivid))
            rospy.loginfo_once("icp result- fitness: {}, RMSE: {}, T: {}".format(
                icp_result.fitness, icp_result.inlier_rmse, icp_result.transformation))
            cloud_GT.transform(icp_result.transformation)
            # o3d.visualization.draw_geometries([cloud_zivid, cloud_GT])

            # publish the estimated pose and cube 
            H_refined_cam2obj = np.eye(4)
            H_refined_cam2obj[:3, :3] = np.matmul(icp_result.transformation[:3, :3], H_init_cam2obj[:3, :3])
            H_refined_cam2obj[:3, 3] = icp_result.transformation[:3, 3]/1000 + H_init_cam2obj[:3, 3]
            trans = H_refined_cam2obj[:3, 3]
            rot = tf_trans.quaternion_from_matrix(H_refined_cam2obj)

            pose_msg = PoseStamped()
            pose_msg.header = rgb.header
            pose_msg.pose.position.x = trans[0] 
            pose_msg.pose.position.y = trans[1] 
            pose_msg.pose.position.z = trans[2] 
            pose_msg.pose.orientation.x = rot[0]
            pose_msg.pose.orientation.y = rot[1]
            pose_msg.pose.orientation.z = rot[2]
            pose_msg.pose.orientation.w = rot[3]
            self.pose_pubs[class_id].publish(pose_msg)

            # Add to Detection3DArray
            detection = Detection3D()
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = class_id
            hypothesis.score = icp_result.fitness
            hypothesis.pose.pose = pose_msg.pose
            detection.results.append(hypothesis)
            detection.bbox.center = pose_msg.pose
            detection.bbox.size.x = self.dims[class_id][0] / 1000 * 2
            detection.bbox.size.y = self.dims[class_id][1] / 1000 * 2
            detection.bbox.size.z = self.dims[class_id][2] / 1000 * 2
            detection_array.detections.append(detection)
        
        self.pub_detections.publish(detection_array)
        self.publish_markers(detection_array)


    def gather_is_results(self, rgb, depth, boxes, scores, labels, is_results):
        
        is_masks = []
        for i, class_id in enumerate(is_results.class_ids):
            if is_results.scores[i] >= self.params["is_thresh"]:
                x = is_results.boxes[i].x_offset
                y = is_results.boxes[i].y_offset
                h = is_results.boxes[i].height
                w = is_results.boxes[i].width
                h_offset = int(h*self.params["crop_offset"])
                w_offset = int(w*self.params["crop_offset"])
                x1 = max(0, x-w_offset)
                y1 = max(0, y-h_offset)
                x2 = min(self.params["width"]-1, x+w+w_offset)
                y2 = min(self.params["height"]-1, y+h+h_offset)
                rgb_crop = rgb[y1:y2, x1:x2].copy()
                if self.params["padding"]:
                    pad = abs(((y2-y1)-(x2-x1))//2)
                    if h >= w:
                        rgb_crop = cv2.copyMakeBorder(rgb_crop, 0, 0, pad, pad, borderType=cv2.BORDER_CONSTANT, value=0)
                    else:
                        rgb_crop = cv2.copyMakeBorder(rgb_crop, pad, pad, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)
                rgb_crop = cv2.resize(rgb_crop, (128, 128))
                depth_crop = depth[y1:y2, x1:x2].copy()
                depth_crop = cv2.resize(depth_crop, (128, 128))
                is_masks.append(self.bridge.imgmsg_to_cv2(is_results.masks[i]))
                boxes.append(np.array([x, y, w, h]))
                scores.append(is_results.scores[i])
                labels.append(class_id)
        return boxes, scores, labels, is_masks


    def visualize_pose_estimation_results(self, all_pose_estimates, all_class_idcs, labels, boxes, scores, rgb):

        sys.path.append(self.params["augauto_module_path"] + "/auto_pose")
        from auto_pose.ae.utils import get_dataset_path
        from meshrenderer import meshrenderer_phong

        renderer = meshrenderer_phong.Renderer(self.ply_model_paths, 
            samples=1, 
            vertex_tmp_store_folder=get_dataset_path(self.workspace_path),
            vertex_scale=float(1)) # float(1) for some models

        bgr, depth, _ = renderer.render_many(obj_ids = [clas_idx for clas_idx in all_class_idcs],
                            W = self.ae_pose_est._width,
                            H = self.ae_pose_est._height,
                            K = self.ae_pose_est._camK, 
                            Rs = [pose_est[:3,:3] for pose_est in all_pose_estimates],
                            ts = [pose_est[:3,3] for pose_est in all_pose_estimates],
                            near = 10,
                            far = 10000,
                            random_light=False,
                            phong={'ambient':0.4,'diffuse':0.8, 'specular':0.3})
        bgr = cv2.resize(bgr, (self.ae_pose_est._width, self.ae_pose_est._height))
        
        g_y = np.zeros_like(bgr)
        g_y[:,:,1]= bgr[:,:,1]    
        im_bg = cv2.bitwise_and(rgb, rgb, mask=(g_y[:,:,1]==0).astype(np.uint8))                 
        image_show = cv2.addWeighted(im_bg, 1, g_y, 1, 0)

        for label, box, score in zip(labels, boxes, scores):
            box = box.astype(np.int32)
            xmin, ymin, xmax, ymax = box[0], box[1], box[0] + box[2], box[1] + box[3]
            cv2.putText(image_show, '%s : %1.3f' % (label,score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, color_dict[int(label)], 2)
            cv2.rectangle(image_show, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        self.vis_pub.publish(self.bridge.cv2_to_imgmsg(image_show, "bgr8"))    

    def icp_refinement(self, source_cloud, target_cloud, N=10000):

        n_source_points = len(source_cloud.points)
        n_target_points = len(target_cloud.points)
        n_sample = np.min([n_source_points, n_target_points, N])
        source_idxes = np.random.choice(n_source_points, n_sample)
        target_idxes = np.random.choice(n_target_points, n_sample)

        source_cloud = source_cloud.select_down_sample(source_idxes)
        target_cloud = target_cloud.select_down_sample(target_idxes)

        threshold = 0.02
        trans_init = np.eye(4)
        evaluation = o3d.registration.evaluate_registration(source_cloud, target_cloud, threshold, trans_init)
        icp_result = o3d.registration.registration_icp(
            source = source_cloud, target = target_cloud, max_correspondence_distance=500, # unit in millimeter
            init = np.eye(4),  
            estimation_method = o3d.registration.TransformationEstimationPointToPoint(), 
            criteria = o3d.registration.ICPConvergenceCriteria(
                                                relative_fitness=1e-10,
                                                relative_rmse=1e-8,
                                                max_iteration=500))                                               
        return icp_result

    def publish_markers(self, detection_array):
            # Delete all existing markers
            markers = MarkerArray()
            marker = Marker()
            marker.action = Marker.DELETEALL
            markers.markers.append(marker)
            self.pub_markers.publish(markers)

            # Object markers
            markers = MarkerArray()
            for i, det in enumerate(detection_array.detections):
                name = self.ply_model_paths[det.results[0].id].split('/')[-1][5:-4]
                color = self.idx2color[det.results[0].id]

                # cube marker
                marker = Marker()
                marker.header = detection_array.header
                marker.action = Marker.ADD
                marker.pose = det.bbox.center
                marker.color.r = color[0] / 255.0
                marker.color.g = color[1] / 255.0
                marker.color.b = color[2] / 255.0
                marker.color.a = 0.3
                marker.ns = "bboxes"
                marker.id = i
                marker.type = Marker.CUBE
                marker.scale = det.bbox.size
                markers.markers.append(marker)

                # text marker
                marker = Marker()
                marker.header = detection_array.header
                marker.action = Marker.ADD
                marker.pose = det.bbox.center
                marker.color.r = color[0] / 255.0
                marker.color.g = color[1] / 255.0
                marker.color.b = color[2] / 255.0
                marker.color.a = 1.0
                marker.id = i
                marker.ns = "texts"
                marker.type = Marker.TEXT_VIEW_FACING
                marker.scale.z = 0.1
                marker.text = '{} ({:.2f})'.format(name, det.results[0].score)
                markers.markers.append(marker)

                # mesh marker
                marker = Marker()
                marker.header = detection_array.header
                marker.action = Marker.ADD
                marker.pose = det.bbox.center
                marker.color.r = color[0] / 255.0
                marker.color.g = color[1] / 255.0
                marker.color.b = color[2] / 255.0
                marker.color.a = 0.8
                marker.ns = "meshes"
                marker.id = i
                marker.type = Marker.MESH_RESOURCE
                marker.scale.x = 0.001
                marker.scale.y = 0.001
                marker.scale.z = 0.001
                marker.mesh_resource = "file://" + self.ply_model_paths[det.results[0].id]
                marker.mesh_use_embedded_materials = True
                markers.markers.append(marker)


            self.pub_markers.publish(markers)
    
    
if __name__ == '__main__':

    pose_estimator = PoseEstimator()
    rospy.spin()




