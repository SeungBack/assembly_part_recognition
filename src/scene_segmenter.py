#!/usr/bin/env python

import torch
import torchvision
import argparse
import json
import rospy
from sensor_msgs.msg import Image
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


class SceneSegmenter:

    def __init__(self):

        # initalize node
        rospy.init_node('scene_recognizier')
        rospy.loginfo("Starting scene_segmenter.py")

        self.seg_params = rospy.get_param("scene_segmentation")
        self.mask_pub = rospy.Publisher('seg_mask/scene', Image, queue_size=10)


        self.initialize_model()
        self.bridge = cv_bridge.CvBridge()

        # get rgb-depth images of same time step
        rgb_sub = message_filters.Subscriber('azure1/rgb/image_raw', Image)
        depth_sub = message_filters.Subscriber('azure1/depth_to_rgb/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=5, slop=0.1)
        rospy.loginfo("Starting rgb-d subscriber with time synchronizer")
        # from rgb-depth images, inference the results and publish it
        self.ts.registerCallback(self.inference)

    def initialize_model(self):

        # import u-net from pytorch module        
        sys.path.append(self.seg_params["pytorch_module_path"])
        from unet import UNet
        # build model
        self.model = UNet(n_channels=3, n_classes=3)
        self.model.load_state_dict(torch.load(self.seg_params["weight_path"]))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def inference(self, rgb, depth):

        rospy.loginfo_once("Segmenting scene area")
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb = PIL.Image.fromarray(np.uint8(rgb), mode="RGB")
        rgb_img = rgb.resize((self.seg_params["width"], self.seg_params["height"]), PIL.Image.BICUBIC)
        rgb = np.array(rgb_img)
        # transform to C, H, W add batch dimension
        rgb = rgb.transpose((2, 0, 1))
        rgb = rgb.reshape([1, 3, self.seg_params["height"], self.seg_params["width"]])
        rgb = torch.from_numpy(rgb).float()
        rgb = rgb / 255 # [1, 3, 768, 1024]
        pred_results = self.model(rgb.to(self.device))
        probs = F.softmax(pred_results, dim=1)
        probs = probs.squeeze(0)

        pred_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.seg_params["width"], self.seg_params["height"])),
                transforms.ToTensor()
            ]
        )

        probs = pred_transform(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
        full_mask = full_mask > self.seg_params["out_threshold"]

        vis_results = self.visualize_prediction(rgb_img, full_mask)

        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(vis_results, "bgr8"))

    def visualize_prediction(self, rgb_img, full_mask):
        
        rgb_img = np.uint8(rgb_img)

        full_mask = full_mask.astype(np.uint8)
        full_mask = 255*full_mask.transpose((2, 1, 0))
        rgb_img = cv2.addWeighted(rgb_img, 1, full_mask, 0.5, 0)

        return np.uint8(rgb_img)




if __name__ == '__main__':

    scene_semantic_segmenter = SceneSegmenter()
    rospy.spin()




