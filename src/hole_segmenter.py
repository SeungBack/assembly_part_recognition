#!/usr/bin/env python

import torch
import torchvision
import argparse
import json
import rospy
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


#pred_mask is numpy.argmax of model output (with 0,1)
def hole_catcher(pred_mask, real_img, threshold):
    id_size = pred_mask.shape[0]*pred_mask.shape[1]
    id_array = np.arange(id_size)+1
    id_array = id_array.reshape(pred_mask.shape[0],pred_mask.shape[1])
    cnd = pred_mask[:,:]>0
    holes = id_array*cnd
    for i in range(pred_mask.shape[0]-1):
        for j in range(pred_mask.shape[1]-1):
            if holes[i][j] > 0:
                if holes[i+1][j]>0:
                    holes[i+1][j]=holes[i][j]
                if holes[i][j+1]>0:
                    holes[i][j+1]=holes[i][j]
                if holes[i+1][j+1]>0:
                    holes[i+1][j+1] = holes[i][j]
                if i>0:
                    if holes[i-1][j]>0:
                        holes[i-1][j]=holes[i][j]
                    if holes[i-1][j+1]>0:
                        holes[i-1][j+1]=holes[i][j]
                    if j>0:
                        if holes[i-1][j-1]>0:
                            holes[i-1][j-1]=holes[i][j]
                if j>0:
                    if holes[i][j-1]>0:
                        holes[i][j-1]=holes[i][j]
                    if holes[i+1][j-1]>0:
                        holes[i+1][j-1]=holes[i][j]
    label = np.expand_dims(pred_mask*255, axis=2)
    label = cv2.cvtColor(np.float32(label), cv2.COLOR_GRAY2BGR)
    holes_id, pixels = np.unique(holes, return_counts=True)
    holes_id = np.delete(holes_id, 0)
    pixels = np.delete(pixels, [0])
    real_img = real_img
    center_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1]), dtype=np.int8)
    for k in range(len(holes_id)):
        if pixels[k]>threshold:
            dots = np.argwhere(holes==holes_id[k])
            avg = np.average(dots, axis=0)
            x_avg = int(avg[0])
            y_avg = int(avg[1])
            cv2.circle(real_img, (y_avg,x_avg), 15, [0,0,255], 3)
            center_mask[x_avg][y_avg]=1
            cv2.putText(real_img, '{}'.format(k),(y_avg+20,x_avg), fontFace=2, fontScale=0.5, color=[0,150,0], thickness = 2)
    checked_img = real_img+label

    return center_mask, checked_img

class HoleSegmenter:

    def __init__(self):

        # initalize node
        rospy.init_node('hole_segmenter')
        rospy.loginfo("Starting hole_segmenter.py")

        self.params = rospy.get_param("hole_seg")
        self.bridge = cv_bridge.CvBridge()
        self.initialize_model()
        rgb_sub = rospy.Subscriber(self.params["rgb"], Image, self.inference)
        if self.params["debug"]:
            self.hs_pub = rospy.Publisher('/assembly/hole/vis_results', Image, queue_size=10)

    def initialize_model(self):

        # import u-net from pytorch module        
        sys.path.append(self.params["pytorch_module_path"])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params["gpu_id"]
        from unet import UNet
        self.model = nn.DataParallel(UNet(n_channels=3, n_classes=2, bilinear=True))
        self.model.cuda()
        self.model.load_state_dict(torch.load(self.params['weight_path']))
        self.model.eval()
        self.rgb_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                    ])
        self.roi = self.params["roi"]

    def inference(self, rgb):

        rospy.loginfo_once("Segmenting hole")
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb_cv = cv2.resize(rgb, (self.params["width"], self.params["height"]), interpolation=cv2.INTER_AREA)
        rgb_cv = rgb_cv[self.roi[2]:self.roi[3], self.roi[0]:self.roi[1]]
        rgb = self.rgb_transform(rgb_cv).unsqueeze(0)

        input_data = rgb
        pred = self.model(input_data).squeeze(dim=0)
        pred = pred.cpu().detach().numpy()
        pred = np.argmax(pred, axis=0) 
        hole_results, hole_vis_results = hole_catcher(pred, rgb_cv, self.params["cluster_thresh"])
        if self.params["debug"]:
            self.hs_pub.publish(self.bridge.cv2_to_imgmsg(np.uint8(hole_vis_results), "bgr8"))
        


if __name__ == '__main__':

    hole_segmenter = HoleSegmenter()
    rospy.spin()




