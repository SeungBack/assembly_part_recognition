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

class_idx2name = {
    1: "bracket",
    2: "wood_pin",
    3: "flathead_screw",
    4: "panhead_screw",
}

class_idx2color = {
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 0, 255),
    4: (0, 255, 255)}

class PartSegmenter:

    def __init__(self):

        # initalize node
        rospy.init_node('zivid_connector_segmenter')
        rospy.loginfo("Starting zivid_connector_segmenter.py")

        self.seg_params = rospy.get_param("zivid_connector")
        self.mask_pub = rospy.Publisher('seg_mask/zivid/connector', Image, queue_size=2)

        self.initialize_model()
        self.rgb_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                            ])
        self.bridge = cv_bridge.CvBridge()

        rgb_sub = message_filters.Subscriber("/zivid_camera/color/image_color", Image)
        depth_sub = message_filters.Subscriber("/zivid_camera/depth/image_raw", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=5, slop=0.1)
        rospy.loginfo("Starting zivid rgb-d subscriber with time synchronizer")
        # from rgb-depth images, inference the results and publish it
        self.ts.registerCallback(self.inference)

    def initialize_model(self):

        # import maskrcnn from pytorch module        
        sys.path.append(self.seg_params["pytorch_module_path"])
        from models import maskrcnn
        # load config files
        with open(os.path.join(self.seg_params["pytorch_module_path"], 'config', self.seg_params["config_file"])) as config_file:
            config = json.load(config_file)
        # build model
        self.model = maskrcnn.get_model_instance_segmentation(num_classes=5, config=config)
        self.model.load_state_dict(torch.load(self.seg_params["weight_path"]))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def inference(self, rgb, depth):
        rospy.loginfo_once("Segmenting connector area")
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb = PIL.Image.fromarray(np.uint8(rgb), mode="RGB")
        rgb_img = rgb.resize((self.seg_params["width"], self.seg_params["height"]), PIL.Image.BICUBIC)
        rgb = self.rgb_transform(rgb_img)

        pred_results = self.model([rgb.to(self.device)])[0]

        pred_masks = pred_results["masks"].cpu().detach().numpy()
        pred_boxes = pred_results['boxes'].cpu().detach().numpy()
        pred_labels = pred_results['labels'].cpu().detach().numpy()
        pred_scores = pred_results['scores'].cpu().detach().numpy()
        
        vis_results = self.visualize_prediction(rgb_img, pred_masks, pred_boxes, pred_labels, pred_scores)

        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(vis_results, "bgr8"))

    def visualize_prediction(self, rgb_img, masks, boxes, labels, score, thresh=0.5):
        
        rgb_img = np.uint8(rgb_img)
        if len(labels) == 0:
            return rgb_img

        for i in range(len(labels)):
            if score[i] > thresh:
                mask = masks[i][0]
                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0

                r = mask * class_idx2color[labels[i]][0]
                g = mask * class_idx2color[labels[i]][1]
                b = mask * class_idx2color[labels[i]][2]
                stacked_img = np.stack((r, g, b), axis=0)
                stacked_img = stacked_img.transpose(1, 2, 0)

                rgb_img = cv2.addWeighted(rgb_img, 1, stacked_img.astype(np.uint8), 0.5, 0)
                cv2.rectangle(rgb_img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 255, 0), 1)
                cv2.putText(rgb_img, class_idx2name[labels[i]] + str(score[i].item())[:4], \
                    (boxes[i][0], boxes[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255 ,0), 1)

                # draw object axis
                _, cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rect = cv2.minAreaRect(cnt[0])
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                center_p = tuple(np.int0(rect[0]))
                size = tuple(np.int0(rect[1]))
                angle = np.int0(rect[2])
                cv2.circle(rgb_img, center_p, 2, (255, 0, 0), -1)
                
                cv2.drawContours(rgb_img, [box], 0, (0, 0, 255), 1)

                left_l = np.sqrt((box[1][0]-box[0][0])**2 + (box[1][1]-box[0][1])**2).astype(int)
                right_l = np.sqrt((box[3][0]-box[0][0])**2 + (box[3][1]-box[0][1])**2).astype(int)

                if left_l > right_l:
                    angle += 90

                pt1 = (center_p[0] - int(max(size)/2 * np.cos(np.radians(angle))), center_p[1]  - int(max(size)/2 * np.sin(np.radians(angle))))
                pt2 = (center_p[0] + int(max(size)/2 * np.cos(np.radians(angle))), center_p[1]  + int(max(size)/2 * np.sin(np.radians(angle))))
                cv2.line(rgb_img, pt1, pt2,(255,0,0),2)
           

        return np.uint8(rgb_img)


if __name__ == '__main__':

    connector_segmenter = PartSegmenter()
    rospy.spin()




