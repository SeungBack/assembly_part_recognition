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
import numpy as np
import os
import configparser

import open3d as o3d # import both o3d and pcl occurs segmentation faults
import zivid_helper
import copy

from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped

import tensorflow as tf
import scipy as sp


color_dict = [(255,255,0),(0,0,255),(255,0,0),(255,255,0)] * 10


class PoseTracker:

    def __init__(self):

        # initalize node
        rospy.init_node('6d_pose_tracker')
        rospy.loginfo("Starting 6d_pose_tracker.py")

        self.params = rospy.get_param("6d_pose_estimation")
        self.class_name = ["side_right", "long_short", "middle", "bottom"]

        # augmented autoencoder and renderer      
        sys.path.append(self.params["augauto_module_path"])
        sys.path.append(self.params["augauto_module_path"] + "/auto_pose/test")
        from ae.ae_factory import build_encoder

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
        self.codebooks = self.ae_pose_est.all_codebooks

        self.ply_model_paths = [str(train_args.get('Paths','MODEL_PATH')) for train_args in self.ae_pose_est.all_train_args]
        self.bridge = cv_bridge.CvBridge()


        # subscribers
        rgb_sub = message_filters.Subscriber("/zivid_camera/color/image_color", Image)
        pose_sub = message_filters.Subscriber("/assembly/zivid/furniture_part/6d_pose_est_results", Detection3DArray)
        is_sub = message_filters.Subscriber("/assembly/zivid/furniture_part/is_results", InstanceSegmentation2D)

        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, pose_sub, is_sub], queue_size=1, slop=0.5)
        rospy.loginfo("Starting zivid rgb-d subscriber with time synchronizer")
        self.ts.registerCallback(self.estimate_6d_pose)
        

    def estimate_6d_pose(self, rgb, detection3darray, is_results):
        
        rgb_original = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb_resized = cv2.resize(rgb_original.copy(), (self.params["width"], self.params["height"]))

        boxes, scores, labels = [], [], []
        crop_imgs, boxes, scores, labels = self.gather_is_results(rgb_resized, boxes, scores, labels, is_results)

        poes_dict = self.pose_read(detection3darray)

        s
        for crop_img, label in zip(crop_imgs, labels):
            class_idx = self.ae_pose_est.class_names.index(label)
            codebook = self.codebooks[class_idx]
            session = self.ae_pose_est.sess
            encoder = codebook.encoder(is_training=False)

            x = crop_img
            if x.dtype == 'uint8':
                x = x/255.
            if x.ndim == 3:
                x = np.expand_dims(x, 0)
            cosine_distance = session(codebook.cos_similarity, {encoder.x: x})

            roation_distribution = sp.stats.norm.pdf(cosine_distance, loc=np.max(cosine_distance))

            translation_distirbution = np.sum(roation_distribution)

            observation_likelihood = roation_distribution * translation_distirbution


            Roation_arr = np.random.rand(10, 3) # 임시로 배치
            cov = np.cov(Roation_arr.T)
            roation_prior = np.random.multivariate_normal(mean=Roation_arr[-1], cov=cov)

            updated_roation_dist = roation_distribution * roation_prior
           
        


        

    def particle_filter(self, img_code, codebook):
        self.rotation_dists = []
        

    def pose_read(self, detection3darray):
        """
        detection3darray.detections[index].results[0] (index is detect object num on the workspace)
        results[0].id => class id, results[0].score, results[0].pose.pose => pos and  value
        results[0].pose.pose.position => x, y, z
        results[0].pose.pose.orientation => x, y, z, w

        format example
        detections:[
            results:[
                id:1
                score:1.0
                pose:
                    pose: 
                        position: 
                            x: 0.250910286658
                            y: -0.288555985471
                            z: 2.0146174813
                        orientation: 
                            x: -0.153185811236
                            y: 0.159266977727
                            z: 0.85797919622
                            w: 0.463723878936
                    covariance:[...]
            ],
            results:[...]
        ]

        """
        pose_dict = {}

        for detect in detection3darray.detections:
   
            pose_info = detect.results[0].pose.pose
            obj_pose = pose_info.position
            obj_ori = pose_info.orientation
            pose_dict[str(detect.results[0].id)] = {"pose": [obj_pose.x, obj_pose.y, obj_pose.z], 
                                                    "ori":[obj_ori.x, obj_ori.y, obj_ori.z, obj_ori.w]}
        return pose_dict


    def gather_is_results(self, rgb, boxes, scores, labels, is_results):
        
        is_masks = []
        crop_imgs = []

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

                rgb_crop = cv2.resize(rgb_crop, (128, 128), interpolation=cv2.INTER_NEAREST)

                crop_imgs.append(rgb_crop)
                is_masks.append(self.bridge.imgmsg_to_cv2(is_results.masks[i]))
                boxes.append(np.array([x, y, w, h]))
                scores.append(is_results.scores[i])
                labels.append(class_id)

        return crop_imgs, boxes, scores, labels
    
    
if __name__ == '__main__':

    pose_tracker = PoseTracker()
    rospy.spin()




