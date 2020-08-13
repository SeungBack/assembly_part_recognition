#!/usr/bin/env python

import torch
import torchvision
import argparse
import json
import rospy
from zivid_camera.srv import *
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, Image
import cv2, cv_bridge
import torchvision.transforms as transforms
import numpy as np
import PIL
import message_filters
import torch
import yaml
import sys
import os
import torch.nn.functional as F
import torch.nn as nn

class RobotArmSegmenter:

    def __init__(self):

        # initalize node
        rospy.init_node('zivid_robotarm_segmenter')
        rospy.loginfo("Starting zivid_robotarm_segmenter.py")

        self.params = rospy.get_param("azure_robotarm_seg")
        if self.params["debug"]:
            self.hs_pub = rospy.Publisher('/assembly/robotarm/vis_results', Image, queue_size=10)

        self.initialize_model()
        self.rgb_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                            ])
        self.bridge = cv_bridge.CvBridge()
        rgb_sub = rospy.Subscriber(self.params["rgb"], Image, self.inference)

    def initialize_model(self):

        # import u-net from pytorch module        
        sys.path.append(self.params["pytorch_module_path"])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params["gpu_id"]
        from unet import UNet
        self.model = nn.DataParallel(UNet(n_channels=3, n_classes=2, bilinear=True))
        self.model.cuda()
        self.model.load_state_dict(torch.load(self.params['weight_path']))
        self.model.eval()

    def inference(self, rgb):

        rospy.loginfo_once("Segmenting robotarm")
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb_cv = cv2.resize(rgb, (self.params["width"], self.params["height"]), interpolation=cv2.INTER_AREA)
        rgb = PIL.Image.fromarray(rgb_cv)
        rgb = self.rgb_transform(rgb).unsqueeze(0)

        pred = self.model(rgb).squeeze(dim=0)
        pred = pred.cpu().detach().numpy()
        pred = np.argmax(pred, axis=0) # 400, 640

        if self.params["debug"]:
            pred = np.uint8(pred*255)
            color = np.array([10, 255, 10])
            r = pred * color[0]
            g = pred * color[1] 
            b = pred * color[2]
            stacked_img = np.stack((r, g, b), axis=0) # 3, 1200, 1920
            stacked_img = stacked_img.transpose(1, 2, 0)
            hs_vis = cv2.addWeighted(rgb_cv, 0.5, stacked_img.astype(np.uint8), 1, 0)
            self.hs_pub.publish(self.bridge.cv2_to_imgmsg(hs_vis, "bgr8"))
        


if __name__ == '__main__':

    robotarm_segmenter = RobotArmSegmenter()
    rospy.spin()




