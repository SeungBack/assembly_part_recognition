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
import tf2_ros
sys.path.append('/home/demo/Workspace/seung/open3d-ros-helper')
from open3d_ros_helper.utils import *
################
import scipy as sp
import numpy.matlib as npm
from scipy.spatial.transform import Rotation as R

import torch


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
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1.0))
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        is_sucess = False
        while not is_sucess:
            try:
                self.transform_map_to_cam = self.tf_buffer.lookup_transform(self.params["camera_frame"], "map", rospy.Time(), rospy.Duration(1.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                print(e)
                rospy.sleep(0.5)
                continue
            is_sucess = True
        self.initialize_is_model()
        self.initialize_pose_est_model()
        self.camera_info = rospy.wait_for_message(self.params["camera_info"], CameraInfo)
        rgb_sub = message_filters.Subscriber(self.params["rgb"], Image, buff_size=1536*2048)
        depth_sub = message_filters.Subscriber(self.params["depth"], Image, buff_size=1536*2048)
        point_sub = message_filters.Subscriber(self.params["point"], PointCloud2, buff_size=1536*2048*3)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, point_sub], queue_size=1, slop=1)
        self.ts.registerCallback(self.inference)
    
        # publishers
        self.init_pose_pubs = [] # "side_right", "long_short", "middle", "bottom"
        self.icp_pose_pubs = []
        self.idx2color = [[13, 128, 255], [255, 12, 12], [217, 12, 232], [232, 222, 12]]
        self.dims = []
        self.cloud_objs = []
        for ply_model in self.ply_model_paths:
            cloud = o3d.io.read_point_cloud(ply_model)
            cloud = cloud.scale(0.001)
            self.dims.append(cloud.get_max_bound())
            self.cloud_objs.append(cloud)
            model_name = ply_model.split('/')[-1][5:-4]
            self.init_pose_pubs.append(rospy.Publisher(
                '/assembly/furniture/pose/init/{}'.format(model_name), PoseStamped, queue_size=1))
            self.icp_pose_pubs.append(rospy.Publisher(
                '/assembly/furniture/pose/icp/{}'.format(model_name), PoseStamped, queue_size=1))
        self.is_pub = rospy.Publisher('/assembly/furniture/is_results', InstanceSegmentation2D, queue_size=1)
        self.init_detection_pub = rospy.Publisher('/assembly/furniture/detection/init', Detection3DArray, queue_size=1)
        self.icp_detection_pub = rospy.Publisher('/assembly/furniture/detection/icp', Detection3DArray, queue_size=1)
        self.init_markers_pub = rospy.Publisher('/assembly/furniture/markers/init', MarkerArray, queue_size=1)
        self.icp_markers_pub = rospy.Publisher('/assembly/furniture/markers/icp', MarkerArray, queue_size=1)


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
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params["gpu_id"]
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
        

    def inference(self, rgb, depth, pcl_msg):
        # get rgb, depth, point cloud
        start_time = time.time()

        rgb_original = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb_resized = cv2.resize(rgb_original.copy(), (self.params["width"], self.params["height"]))
        camera_header = rgb.header

        depth_original = self.bridge.imgmsg_to_cv2(depth, desired_encoding='32FC1')
        depth_resized = cv2.resize(depth_original.copy(), dsize=(self.params["width"], self.params["height"]), interpolation=cv2.INTER_AREA)

        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb = PIL.Image.fromarray(np.uint8(rgb), mode="RGB")
        rgb_img = rgb.resize((self.params["width"], self.params["height"]), PIL.Image.BICUBIC)
        rgb = self.rgb_transform(rgb_img).unsqueeze(0)
        rospy.loginfo("input {}".format(time.time()-start_time))

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
        rospy.loginfo("instance segmentation {}".format(time.time()-start_time))

        # gather all detected crops
        boxes, scores, labels = [], [], []
        rgb_crops = []
        # 6D pose estimation for all detected crops using AAE
        boxes, scores, labels, rgb_crops, is_masks = self.gather_is_results(rgb_resized, depth_resized, boxes, scores, labels, rgb_crops, is_msg)
        if len(is_masks) == 0:
            return
        all_pose_estimates, all_class_idcs = self.ae_pose_est.process_pose(boxes, labels, rgb_resized, depth_resized)
        # visualize pose estimation results using only AAE
        if self.params["debug"]:
            self.visualize_pose_estimation_results(all_pose_estimates, all_class_idcs, labels, boxes, scores, rgb_resized)
        rospy.loginfo("pose estimation {}".format(time.time()-start_time))

        # 6D object pose refinement using ICP on pointcloud
        init_detection_array = Detection3DArray()
        init_detection_array.header = camera_header
        icp_detection_array = Detection3DArray()
        icp_detection_array.header = camera_header

        cloud_map = convert_ros_to_o3d(pcl_msg) 
        cloud_cam = do_transform_o3d_cloud(copy.deepcopy(cloud_map), self.transform_map_to_cam)
        is_mask_original = cv2.resize(is_masks[0], (rgb_original.shape[1], rgb_original.shape[0]), interpolation=cv2.INTER_AREA)
        cloud_cam_obj = crop_o3d_cloud_with_mask(cloud_cam, is_mask_original, camera_info=self.camera_info)
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        for i, (pose_estimate, class_id) in enumerate(zip(all_pose_estimates, all_class_idcs)):
            # crop cloud with instance mask   
            is_mask = is_masks[i].copy()
            is_mask[is_mask < 128] = 0
            is_mask_original = cv2.resize(is_mask, (rgb_original.shape[1], rgb_original.shape[0]), interpolation=cv2.INTER_AREA)
            cloud_cam_obj = crop_o3d_cloud_with_mask(cloud_cam, is_mask_original, camera_info=self.camera_info)
            # transform cloud_obj to the origin of camera frame
            cloud_obj = copy.deepcopy(self.cloud_objs[class_id])
            H_obj2cam = np.eye(4)
            H_obj2cam[:3, 3] = - cloud_obj.get_center()
            cloud_obj = cloud_obj.transform(H_obj2cam)
            # transform cloud_obj to the estimated 6d pose
            H_init_cam2obj = np.eye(4)
            H_init_cam2obj[:3, :3] = pose_estimate[:3, :3]
            H_init_cam2obj[:3, 3] = 0.001 * pose_estimate[:3, 3] # align scale
            cloud_obj = cloud_obj.transform(H_init_cam2obj)

            # translate cloud_obj to the centroid of cloud cam
            H_obj_to_cam_centroid = cloud_cam_obj.get_center()-cloud_obj.get_center()
            cloud_obj = cloud_obj.translate(H_obj_to_cam_centroid)

            # icp refinement
            icp_result, evaluation = icp_refinement(source_cloud=cloud_obj, target_cloud=cloud_cam_obj, max_iteration=self.params["icp_iter"], n_points=1000, max_correspondence_distance=5)
            rospy.loginfo("{}".format(evaluation))            
            H_init_cam2obj[:3, 3] = H_init_cam2obj[:3, 3] + H_obj_to_cam_centroid
            H_refined_cam2obj = np.matmul(icp_result.transformation, H_init_cam2obj)

            # gather 6d pose estimation results and publish it
            translation = H_init_cam2obj[:3, 3]
            rotation = tf_trans.quaternion_from_matrix(H_init_cam2obj)
            init_pose_msg, init_detection = self.gather_pose_results(camera_header, class_id, translation, rotation)
            translation = H_refined_cam2obj[:3, 3]
            rotation = tf_trans.quaternion_from_matrix(H_refined_cam2obj)
            icp_pose_msg, icp_detection = self.gather_pose_results(camera_header, class_id, translation, rotation)

            self.init_pose_pubs[class_id].publish(init_pose_msg)
            self.icp_pose_pubs[class_id].publish(icp_pose_msg)
            init_detection_array.detections.append(init_detection)
            icp_detection_array.detections.append(icp_detection)
            rospy.loginfo("icp {}".format(time.time()-start_time))

        self.init_detection_pub.publish(init_detection_array)
        self.icp_detection_pub.publish(icp_detection_array)
        self.publish_markers(init_detection_array, self.init_markers_pub, [13, 128, 255])
        self.publish_markers(icp_detection_array, self.icp_markers_pub, [255, 13, 13])

        rospy.loginfo("callback {}".format(time.time()-start_time))

    def gather_is_results(self, rgb, depth, boxes, scores, labels, rgb_crops, is_results):
        
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
                rgb_crops.append(rgb_crop)
        return boxes, scores, labels, rgb_crops, is_masks


    def gather_pose_results(self, header, class_id, translation, rotation):
        
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.pose.position.x = translation[0] 
        pose_msg.pose.position.y = translation[1] 
        pose_msg.pose.position.z = translation[2] 
        pose_msg.pose.orientation.x = rotation[0]
        pose_msg.pose.orientation.y = rotation[1]
        pose_msg.pose.orientation.z = rotation[2]
        pose_msg.pose.orientation.w = rotation[3]

        # Add to Detection3DArray
        detection = Detection3D()
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.id = class_id
        hypothesis.pose.pose = pose_msg.pose
        detection.results.append(hypothesis)
        detection.bbox.center = pose_msg.pose
        detection.bbox.size.x = self.dims[class_id][0] / 1000 * 2
        detection.bbox.size.y = self.dims[class_id][1] / 1000 * 2
        detection.bbox.size.z = self.dims[class_id][2] / 1000 * 2
        return pose_msg, detection


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
        g_y[:,:,1] = bgr[:,:,1]    
        im_bg = cv2.bitwise_and(rgb, rgb, mask=(g_y[:,:,1]==0).astype(np.uint8))                 
        image_show = cv2.addWeighted(im_bg, 1, g_y, 1, 0)

        for label, box, score in zip(labels, boxes, scores):
            box = box.astype(np.int32)
            xmin, ymin, xmax, ymax = box[0], box[1], box[0] + box[2], box[1] + box[3]
            cv2.putText(image_show, '%s : %1.3f' % (label,score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, self.color_dict[int(label)], 2)
            cv2.rectangle(image_show, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        self.pose_vis_pub.publish(self.bridge.cv2_to_imgmsg(image_show, "bgr8"))    

    def publish_markers(self, detection_array, publisher, color):
        # Delete all existing markers
        markers = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        markers.markers.append(marker)
        publisher.publish(markers)

        # Object markers
        markers = MarkerArray()
        for i, det in enumerate(detection_array.detections):
            name = self.ply_model_paths[det.results[0].id].split('/')[-1][5:-4]

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


        publisher.publish(markers)
    
    
if __name__ == '__main__':

    pose_estimator = PoseEstimator()
    rospy.spin()




