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


class SceneSegmenter:

    def __init__(self):

        # initalize node
        rospy.init_node('zivid_scene_segmenter')
        rospy.loginfo("Starting zivid_scene_segmenter.py")

        self.seg_params = rospy.get_param("zivid_scene")
        self.mask_pub = rospy.Publisher('seg_mask/zivid/scene', Image, queue_size=2)

        self.initialize_model()
        self.rgb_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                            ])
        self.bridge = cv_bridge.CvBridge()

        self.depth_val_min = 0.4
        self.depth_val_max = 2.25

        rgb_sub = message_filters.Subscriber("/zivid_camera/color/image_color", Image)
        depth_sub = message_filters.Subscriber("/zivid_camera/depth/image_raw", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=5, slop=0.1)
        rospy.loginfo("Starting zivid rgb-d subscriber with time synchronizer")
        # from rgb-depth images, inference the results and publish it
        self.ts.registerCallback(self.inference)

    def initialize_model(self):

        # import u-net from pytorch module        
        sys.path.append(self.seg_params["pytorch_module_path"])
        from unet import UNet
        # build model
        self.model = UNet(n_channels=4, n_classes=3)
        self.model.load_state_dict(torch.load(self.seg_params["weight_path"]))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def inference(self, rgb, depth):

        rospy.loginfo_once("Segmenting scene area")
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb = PIL.Image.fromarray(np.uint8(rgb), mode="RGB")
        # [768, 1024, 3]
        rgb_img = rgb.resize((self.seg_params["width"], self.seg_params["height"]), PIL.Image.BICUBIC)
        rgb = np.array(rgb_img)
        # transform to C, H, W add batch dimension
        rgb = rgb.transpose((2, 0, 1))
        rgb = rgb.reshape([1, 3, self.seg_params["height"], self.seg_params["width"]])
        rgb = torch.from_numpy(rgb).float()
        rgb = rgb / 255 # [1, 3, 768, 1024]
        rgb.unsqueeze(0)


        depth = self.bridge.imgmsg_to_cv2(depth)
        depth = cv2.resize(depth, dsize=(self.seg_params["width"], self.seg_params["height"]), interpolation=cv2.INTER_AREA)
        mask = np.isnan(depth.copy()).astype('uint8')
        depth = np.where(np.isnan(depth), 0, depth)           

        values = np.unique(depth)
        max_val = max(values)
        min_val = min(values)
        depth = np.uint8((depth - min_val) / (max_val - min_val) * 255)
        inpainted = cv2.inpaint(depth, mask, 3, cv2.INPAINT_NS)

        depth = np.expand_dims(np.asarray(depth), -1)
        depth = np.repeat(depth, 3, -1)
        depth = torch.from_numpy(np.float32(depth))
        depth = depth.transpose(0, 2).transpose(1, 2)

        depth = torch.clamp(depth, min=self.depth_val_min, max=self.depth_val_max)
        depth = (depth-self.depth_val_min) / (self.depth_val_max-self.depth_val_min) # 3500-8000 to 0-1
        depth = depth.unsqueeze(0)

       
        rgbd = torch.cat((rgb, depth), axis=1)[:,:4] # rgb 3channel + depth 1channel condat rgbd 4channel input
      


        pred_results = self.model(rgbd.to(self.device))
        probs = F.softmax(pred_results, dim=1)
        probs = probs.squeeze(0)

        pred_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.seg_params["height"], self.seg_params["width"])),
                transforms.ToTensor()
            ]
        )

        probs = pred_transform(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
        full_mask = full_mask > self.seg_params["out_threshold"]

        vis_results = self.visualize_prediction(rgb_img, full_mask)

        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(vis_results, "bgr8"))

    def visualize_prediction(self, rgb_img, full_mask):
        
        rgb_img = np.uint8(rgb_img) # [768, 1024, 3]

        full_mask = full_mask.astype(np.uint8)
        full_mask = 255*full_mask.transpose((1, 2, 0))
        rgb_img = cv2.addWeighted(rgb_img, 1, full_mask, 0.5, 0)

        return np.uint8(rgb_img)


if __name__ == '__main__':

    scene_semantic_segmenter = SceneSegmenter()
    rospy.spin()




