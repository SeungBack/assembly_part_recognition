#!/usr/bin/env python

import json
import rospy
import cv2, cv_bridge
import numpy as np
import PIL
import message_filters
import yaml
import sys

from std_msgs.msg import String, Header
from sensor_msgs.msg import RegionOfInterest, Image, CameraInfo, PointCloud2
from assembly_part_recognition.msg import InstanceSegmentation2D
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped

import cv2
import argparse
import numpy as np
import os
import configparser

import open3d as o3d # import both o3d and pcl occurs segmentation faults
import cam_helper
import copy
import time
import tf.transformations as tf_trans

class PoseEstimator:

    def __init__(self):

        # initalize node
        rospy.init_node('furniture_pose_estimator')
        rospy.loginfo("Starting furniture_pose_estimator.py")

        self.params = rospy.get_param("furniture_pose_estimator")
        self.class_names = ["background", "side", "longshort", "middle", "bottom"]
        self.classidx2color = [[13, 128, 255], [255, 12, 12], [217, 12, 232], [232, 222, 12]]
        self.color_dict = [(255,255,0), (0,0,255), (255,0,0), (255,255,0)] * 10
        self.roi = self.params["roi"]

        self.bridge = cv_bridge.CvBridge()

        # subscribers
        self.initialize_is_model()
        self.initialize_pose_est_model()
        rgb_sub = message_filters.Subscriber(self.params["rgb"], Image, buff_size=1536*2048)
        depth_sub = message_filters.Subscriber(self.params["depth"], Image, buff_size=1536*2048)
        point_sub = message_filters.Subscriber(self.params["point"], PointCloud2, buff_size=1536*2048*3)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, point_sub], queue_size=1, slop=1)
        self.ts.registerCallback(self.inference)

        # publishers
        self.pose_pubs = [] # "side_right", "long_short", "middle", "bottom"
        self.idx2color = [[13, 128, 255], [255, 12, 12], [217, 12, 232], [232, 222, 12]]
        self.dims = []
        self.cloud_GTs = []
        for ply_model in self.ply_model_paths:
            cloud = o3d.io.read_point_cloud(ply_model)
            self.dims.append(cloud.get_max_bound())
            self.cloud_GTs.append(cloud)
            model_name = ply_model.split('/')[-1][5:-4]
            self.pose_pubs.append(rospy.Publisher(
                '/assembly/furniture/pose_{}'.format(model_name), PoseStamped, queue_size=1))
        self.is_pub = rospy.Publisher('/assembly/furniture/is_results', InstanceSegmentation2D, queue_size=1)
        self.detections_pub = rospy.Publisher('/assembly/furniture/pose', Detection3DArray, queue_size=1)
        self.markers_pub = rospy.Publisher('/assembly/furniture/markers', MarkerArray, queue_size=1)
        if self.params["debug"]:
            self.is_vis_pub = rospy.Publisher('/assembly/furniture/is_vis_results', Image, queue_size=1)
            self.pose_vis_pub = rospy.Publisher('/assembly/furniture/pose_vis_results', Image, queue_size=1)
        
    def initialize_is_model(self):
        import torch
        import torchvision.transforms as transforms
        # import maskrcnn from pytorch module        
        sys.path.append(self.params["pytorch_module_path"])
        from models import maskrcnn
        # load config files
        with open(os.path.join(self.params["pytorch_module_path"], 'config', self.params["config_file"])) as config_file:
            self.config = json.load(config_file)
        # build model
        self.model = maskrcnn.get_model_instance_segmentation(num_classes=5, config=self.config)
        self.model.load_state_dict(torch.load(self.params["weight_path"]))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.rgb_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                    ])

    def initialize_pose_est_model(self):
        import tensorflow as tf
        # augmented autoencoder and renderer      
        rospy.loginfo("Loading AAE model")
        sys.path.append(self.params["augauto_module_path"])
        sys.path.append(self.params["augauto_module_path"] + "/auto_pose/test")
        from aae_maskrcnn_pose_estimator import AePoseEstimator
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params["gpu_id"]
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.workspace_path = os.environ.get('AE_WORKSPACE_PATH')
        if self.workspace_path == None:
            print 'Please define a workspace path:\n'
            print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
            exit(-1)

        test_configpath = os.path.join(self.workspace_path, 'cfg_eval', self.params["test_config"])
        test_args = configparser.ConfigParser()
        test_args.read(test_configpath)
        self.ae_pose_est = AePoseEstimator(test_configpath)
        self.ply_model_paths = [str(train_args.get('Paths','MODEL_PATH')) for train_args in self.ae_pose_est.all_train_args]
        

    def inference(self, rgb, depth, cloud):

        # get rgb, depth, point cloud
        img_header = rgb.header
        rgb_original = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb_resized = cv2.resize(rgb_original.copy(), (self.params["width"], self.params["height"]))

        depth_original = self.bridge.imgmsg_to_cv2(depth, desired_encoding='32FC1')
        depth_resized = cv2.resize(depth_original.copy(), dsize=(self.params["width"], self.params["height"]), interpolation=cv2.INTER_AREA)
        if "zivid" in self.params["depth"]:
            depth_resized = np.where(np.isnan(depth_resized), 0, depth_resized)  
            npcloud_cam = cam_helper.convertZividCloudFromRosToNumpy(cloud)
        else:
            npcloud_cam = cam_helper.convertAzureCloudFromRosToNumpy(cloud)
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb = PIL.Image.fromarray(np.uint8(rgb), mode="RGB")
        rgb_img = rgb.resize((self.params["width"], self.params["height"]), PIL.Image.BICUBIC)
        rgb = self.rgb_transform(rgb_img).unsqueeze(0)
    
        # 2D instance segmentation
        rospy.loginfo_once("Segmenting furniture part area")
        pred_results = self.model(rgb.to(self.device))[0]
        pred_masks = pred_results["masks"].cpu().detach().numpy()
        pred_boxes = pred_results['boxes'].cpu().detach().numpy()
        pred_labels = pred_results['labels'].cpu().detach().numpy()
        pred_scores = pred_results['scores'].cpu().detach().numpy()

        # inference result -> ros message (Detection2D)
        is_msg = InstanceSegmentation2D()
        is_msg.header = Header()
        is_msg.header.stamp = rospy.get_rostime()
        scores = []
        for i, (x1, y1, x2, y2) in enumerate(pred_boxes):
            if x1 < self.roi[0] or x2 > self.roi[1] or y1 < self.roi[2] or y2 > self.roi[3]:
                continue
            box = RegionOfInterest()
            box.x_offset = np.asscalar(x1)
            box.y_offset = np.asscalar(y1)
            box.height = np.asscalar(y2 - y1)
            box.width = np.asscalar(x2 - x1)
            is_msg.boxes.append(box)
            
            class_id = pred_labels[i] - 1 # remove background label
            is_msg.class_ids.append(class_id) 
            class_name = self.class_names[class_id]
            is_msg.class_names.append(class_name)
            score = pred_scores[i]
            is_msg.scores.append(score)

            mask = Image()
            mask.header = is_msg.header
            mask.height = pred_masks[i].shape[1]
            mask.width = pred_masks[i].shape[2]
            mask.encoding = "mono8"
            mask.is_bigendian = False
            mask.step = mask.width
            mask.data = (np.uint8(pred_masks[i][0] * 255)).tobytes()
            is_msg.masks.append(mask)        

        self.is_pub.publish(is_msg)
        if self.params["debug"]:
            vis_results = self.visualize_instance_segmentation_results(rgb_img, pred_masks, pred_boxes, pred_labels, pred_scores, thresh=self.params["is_thresh"])
            self.is_vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_results, "bgr8"))

        # gather all detected crops
        boxes, scores, labels = [], [], []
        boxes, scores, labels, is_masks = self.gather_is_results(rgb_resized, depth_resized, boxes, scores, labels, is_msg)
        # 6D pose estimation for all detected crops using AAE
        rospy.loginfo_once("Estimating 6D object pose")
        all_pose_estimates, all_class_idcs = self.ae_pose_est.process_pose(boxes, labels, rgb_resized, depth_resized)
        # visualize pose estimation results using only AAE
        if self.params["debug"]:
            self.visualize_pose_estimation_results(all_pose_estimates, all_class_idcs, labels, boxes, scores, rgb_resized)

        detection_array = Detection3DArray()
        detection_array.header = img_header
        # 6D object pose refinement using ICP on pointcloud
        for i, (pose_estimate, class_id) in enumerate(zip(all_pose_estimates, all_class_idcs)):

            # crop zivid cloud with instance mask        
            is_mask_original = cv2.resize(is_masks[i], (rgb_original.shape[1], rgb_original.shape[0]), interpolation=cv2.INTER_AREA)
            is_mask_original = is_mask_original[np.isfinite(is_mask_original)]

            # remove outliers and scale it from m to mm
            cloud_cam = cam_helper.convertCloudFromNumpyToOpen3d(npcloud_cam.copy(), mask=is_mask_original)
            cloud_cam, _ = cloud_cam.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            cloud_cam.scale(1000)

            cloud_GT = copy.deepcopy(self.cloud_GTs[class_id])
            # rotate cloud_GT according to the estimated rotation then translate cloud_GT from origin to the center of cloud_cam
            T_rot = np.eye(4)
            T_rot[:3, :3] = pose_estimate[:3, :3] # test needed: trans from 2D bbox or center
            T_trans = np.eye(4)
            T_trans[:3, 3] = cloud_cam.get_center()
            H_init_cam2obj = np.matmul(T_trans, T_rot)
            cloud_GT.transform(H_init_cam2obj)
            icp_result = self.icp_refinement(source_cloud=copy.deepcopy(cloud_GT), target_cloud=copy.deepcopy(cloud_cam), N=self.params["icp_iter"])
            rospy.loginfo_once("icp result- fitness: {}, RMSE: {}, T: {}".format(
                icp_result.fitness, icp_result.inlier_rmse, icp_result.transformation))
            cloud_GT.transform(icp_result.transformation)
            # o3d.visualization.draw_geometries([cloud_cam, cloud_GT])

            # publish the estimated pose and cube 
            H_refined_cam2obj = np.eye(4)
            H_refined_cam2obj[:3, :3] = np.matmul(icp_result.transformation[:3, :3], H_init_cam2obj[:3, :3])
            H_refined_cam2obj[:3, 3] = icp_result.transformation[:3, 3]/1000 + H_init_cam2obj[:3, 3]
            trans = H_refined_cam2obj[:3, 3]
            rot = tf_trans.quaternion_from_matrix(H_refined_cam2obj)

            pose_msg = PoseStamped()
            pose_msg.header = img_header
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

        self.detections_pub.publish(detection_array)
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
                x1 = int(max(0, x-w_offset))
                y1 = int(max(0, y-h_offset))
                x2 = int(min(self.params["width"]-1, x+w+w_offset))
                y2 = int(min(self.params["height"]-1, y+h+h_offset))
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

    def visualize_instance_segmentation_results(self, rgb_img, masks, boxes, labels, score, thresh=0.5):
        
        rgb_img = np.uint8(rgb_img)
        if len(labels) == 0:
            return rgb_img

        cv2.rectangle(rgb_img, (self.roi[0], self.roi[2]), (self.roi[1], self.roi[3]), (0, 0, 255), 3)
        cv2.putText(rgb_img, "ROI", (self.roi[0], self.roi[2] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


        for i in range(len(labels)):
            if score[i] > thresh:
                x1, y1, x2, y2 = boxes[i]
                if x1 < self.roi[0] or x2 > self.roi[1] or y1 < self.roi[2] or y2 > self.roi[3]:
                    color = (50, 150, 0)
                    lw = 1
                else:
                    color = (0, 255, 0)
                    lw = 3
                mask = masks[i][0]
                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0

                r = mask * self.classidx2color[labels[i]-1][0]
                g = mask * self.classidx2color[labels[i]-1][1]
                b = mask * self.classidx2color[labels[i]-1][2]
                stacked_img = np.stack((r, g, b), axis=0)
                stacked_img = stacked_img.transpose(1, 2, 0)

                rgb_img = cv2.addWeighted(rgb_img, 1, stacked_img.astype(np.uint8), 1, 0.5)
                cv2.rectangle(rgb_img, (x1, y1), (x2, y2), color, lw)
                cv2.putText(rgb_img, self.class_names[labels[i]] + str(score[i].item())[:4], \
                    (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, lw)

        return np.uint8(rgb_img)

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
            cv2.putText(image_show, '%s : %1.3f' % (label,score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, self.color_dict[int(label)], 2)
            cv2.rectangle(image_show, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        self.pose_vis_pub.publish(self.bridge.cv2_to_imgmsg(image_show, "bgr8"))    

    def icp_refinement(self, source_cloud, target_cloud, N=1000):

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
            self.markers_pub.publish(markers)

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
                marker.scale.z = 0.07
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
                marker.color.a = 0.9
                marker.ns = "meshes"
                marker.id = i
                marker.type = Marker.MESH_RESOURCE
                marker.scale.x = 0.001
                marker.scale.y = 0.001
                marker.scale.z = 0.001
                marker.mesh_resource = "file://" + self.ply_model_paths[det.results[0].id]
                marker.mesh_use_embedded_materials = True
                markers.markers.append(marker)


            self.markers_pub.publish(markers)
    
    
if __name__ == '__main__':

    pose_estimator = PoseEstimator()
    rospy.spin()




