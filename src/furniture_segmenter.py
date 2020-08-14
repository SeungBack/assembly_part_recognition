#!/usr/bin/env python

import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import json
import rospy

import cv2, cv_bridge
import numpy as np
import PIL
import message_filters
import yaml
import sys
import os
import time
import math

from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2, Image, RegionOfInterest
from assembly_part_recognition.msg import InstanceSegmentation2D


class PartSegmenter:

    def __init__(self):

        # initalize node
        rospy.init_node('furniture_segmenter')
        rospy.loginfo("Starting furniture_segmenter.py")

        self.params = rospy.get_param("furniture_seg")
        self.initialize_model()
        self.bridge = cv_bridge.CvBridge()
        self.class_names = ["background", "side", "longshort", "middle", "bottom"]
        self.classidx2color = [[13, 128, 255], [255, 12, 12], [217, 12, 232], [232, 222, 12]]
        self.roi = self.params["roi"]
        rgb_sub = rospy.Subscriber(self.params["rgb"], Image, self.inference)
        self.is_pub = rospy.Publisher('/assembly/furniture/is_results', InstanceSegmentation2D, queue_size=1)
        if self.params["debug"]:
            self.vis_pub = rospy.Publisher('/assembly/furniture/is_vis_results', Image, queue_size=1)

    def initialize_model(self):

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

    def save_inference_data(self, rgb, depth, save_dir=None):
        rgb_save = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb_save = cv2.cvtColor(rgb_save, cv2.COLOR_BGR2RGB)
        rgb_save = PIL.Image.fromarray(np.uint8(rgb_save), mode="RGB")
        depth_save = self.bridge.imgmsg_to_cv2(depth)

        save_root = "/home/demo/catkin_ws/src/assembly_part_recognition/{}".format(save_dir)
        save_dir_rgb = os.path.join(save_root, "rgb")
        save_dir_depth = os.path.join(save_root, "depth_value")
        if not os.path.isdir(save_dir_rgb): os.makedirs(save_dir_rgb)
        if not os.path.isdir(save_dir_depth): os.makedirs(save_dir_depth)

        save_name = "{}".format(time.time())
        rgb_save.save(os.path.join(save_dir_rgb, "{}.png".format(save_name)))
        np.save(os.path.join(save_dir_depth, "{}.npy".format(save_name)), depth_save)


    def inference(self, rgb):

        rospy.loginfo_once("Segmenting furniture part area")
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb = PIL.Image.fromarray(np.uint8(rgb), mode="RGB")
        rgb_img = rgb.resize((self.params["width"], self.params["height"]), PIL.Image.BICUBIC)
        rgb = self.rgb_transform(rgb_img).unsqueeze(0)
            
        # inference
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
            vis_results = self.visualize_prediction(rgb_img, pred_masks, pred_boxes, pred_labels, pred_scores, thresh=self.params["is_thresh"])
            self.vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_results, "bgr8"))

    def visualize_prediction(self, rgb_img, masks, boxes, labels, score, thresh=0.5):
        
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




if __name__ == '__main__':

    furniture_segmenter = PartSegmenter()
    rospy.spin()




