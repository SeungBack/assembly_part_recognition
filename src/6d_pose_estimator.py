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
from sensor_msgs.msg import RegionOfInterest, Image
from assembly_part_recognition.msg import InstanceSegmentation2D
 
import cv2
import argparse
import tensorflow as tf
import numpy as np
import os
import configparser




class PoseEstimator:

    def __init__(self):

        """
        !TODO: support for azure
        """

        # initalize node
        rospy.init_node('6d_pose_estimator')
        rospy.loginfo("Starting 6d_pose_estimator.py")

        self.params = rospy.get_param("6d_pose_estimation")
        self.vis_pub = rospy.Publisher('/assembly/zivid/furniture_part/6d_pose_vis_results', Image, queue_size=1)

        self.initialize_model()
        self.bridge = cv_bridge.CvBridge()

        rgb_sub = message_filters.Subscriber("/zivid_camera/color/image_color", Image)
        depth_sub = message_filters.Subscriber("/zivid_camera/depth/image_raw", Image)
        is_sub = message_filters.Subscriber("/assembly/zivid/furniture_part/is_results", InstanceSegmentation2D)

        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, is_sub], queue_size=5, slop=3)
        rospy.loginfo("Starting zivid rgb-d subscriber with time synchronizer")
        # from rgb-depth images, inference the results and publish it
        self.ts.registerCallback(self.inference)

    def initialize_model(self):

        # import augmented autoencoder module        
        sys.path.append(self.params["augauto_module_path"])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params["gpu_id"]
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        from auto_pose.ae import factory
        from auto_pose.ae import utils as u

        full_name = self.params["exp_group"].split('/')
        experiment_name = full_name.pop()
        experiment_group = full_name.pop() if len(full_name) > 0 else ''
        self.codebook, self.dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)

        workspace_path = os.environ.get('AE_WORKSPACE_PATH')
        log_dir = u.get_log_dir(workspace_path,experiment_name, experiment_group)
        ckpt_dir = u.get_checkpoint_dir(log_dir)

        train_cfg_file_path = u.get_train_config_exp_file_path(log_dir, experiment_name)
        train_args = configparser.ConfigParser()
        train_args.read(train_cfg_file_path)  

        gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.9)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True

        self.session = tf.Session(config=config) 
        factory.restore_checkpoint(self.session, tf.train.Saver(), ckpt_dir)


    def inference(self, rgb, depth, is_results):

        rospy.loginfo_once("Estimating 6D pose of objects")
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb = cv2.resize(rgb, (self.params["width"], self.params["height"]))

        for i, class_id in enumerate(is_results.class_ids):
            if class_id == self.params["class_id"] and is_results.scores[i] >= self.params["is_thresh"]:
                # crop the object area with offset
                x = is_results.boxes[i].x_offset
                y = is_results.boxes[i].y_offset
                h = is_results.boxes[i].height
                w = is_results.boxes[i].width
                h_offset = int(h*self.params["crop_offset"])
                w_offset = int(w*self.params["crop_offset"])
                new_x1 = max(0, x-w_offset)
                new_y1 = max(0, y-h_offset)
                new_x2 = min(self.params["width"]-1, x+w+w_offset)
                new_y2 = min(self.params["height"]-1, y+h+h_offset)
                crop_rgb = rgb[new_y1:new_y2, new_x1:new_x2].copy()
                crop_rgb = cv2.resize(crop_rgb, (128, 128))
                break # currently, detect only single objects per class

        R = self.codebook.nearest_rotation(self.session, crop_rgb)
        pred_view = self.dataset.render_rot(R, downSample = 1)
        print(R)
        vis_results = np.hstack([crop_rgb, pred_view])
        self.vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_results, "bgr8"))


if __name__ == '__main__':

    pose_estimator = PoseEstimator()
    rospy.spin()




