#!/usr/bin/env python

import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import json
import rospy

from zivid_camera.srv import *

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


class_idx2color = {
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 0, 255),
    4: (0, 255, 255),
    5: (255, 255, 0),
    6: (0, 255, 128)}


class PartSegmenter:

    def __init__(self):

        # initalize node
        rospy.init_node('zivid_furniture_part_segmenter')
        rospy.loginfo("Starting zivid_furniture_part_segmenter.py")

        self.params = rospy.get_param("zivid_furniture_part")
        self.vis_pub = rospy.Publisher('/assembly/zivid/furniture_part/is_vis_results', Image, queue_size=1)
        self.is_pub = rospy.Publisher('/assembly/zivid/furniture_part/is_results', InstanceSegmentation2D, queue_size=1)
        self.imdepth_pub = rospy.Publisher('/zivid_camera/depth/image_impainted', Image, queue_size=1)


        self.initialize_model()
        self.rgb_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                            ])
        self.bridge = cv_bridge.CvBridge()
        self.class_names = ["background", "side_right", "long_short", "middle", "bottom"]

        rgb_sub = message_filters.Subscriber("/zivid_camera/color/image_color", Image)
        depth_sub = message_filters.Subscriber("/zivid_camera/depth/image_raw", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=5, slop=1)
        rospy.loginfo("Starting zivid rgb-d subscriber with time synchronizer")
        # from rgb-depth images, inference the results and publish it
        self.ts.registerCallback(self.inference)

    def initialize_model(self):

        # import maskrcnn from pytorch module        
        sys.path.append(self.params["pytorch_module_path"])
        from models import maskrcnn
        # load config files
        with open(os.path.join(self.params["pytorch_module_path"], 'config', self.params["config_file"])) as config_file:
            config = json.load(config_file)
        # build model
        self.model = maskrcnn.get_model_instance_segmentation(num_classes=self.params["num_classes"], config=config)
        self.model.load_state_dict(torch.load(self.params["weight_path"]))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def save_inference_data(self, rgb, depth):
        rgb_save = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb_save = cv2.cvtColor(rgb_save, cv2.COLOR_BGR2RGB)
        rgb_save = PIL.Image.fromarray(np.uint8(rgb_save), mode="RGB")
        depth_save = self.bridge.imgmsg_to_cv2(depth)
        save_dir = "/home/demo/catkin_ws/src/assembly_part_recognition/inference_data_zivid_white_2"
        if not os.path.isdir(save_dir): os.makedirs(save_dir)
        save_name = "{}".format(time.time())

        rgb_save.save(os.path.join(save_dir, "{}.png".format(save_name)))
        np.save(os.path.join(save_dir, "{}.npy".format(save_name)), depth_save)

        # np.save(os.path.join(save_dir, save_name + "_rgb.npy"), rgb_save)
        # np.save(os.path.join(save_dir, save_name + "_depth.npy"), depth_save)


    def inference(self, rgb, depth):
        self.save_inference_data(rgb, depth)

        rospy.loginfo_once("Segmenting furniture_part area")
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb = PIL.Image.fromarray(np.uint8(rgb), mode="RGB")
        rgb_img = rgb.resize((self.params["width"], self.params["height"]), PIL.Image.BICUBIC)
        rgb = self.rgb_transform(rgb_img)

        # load config files
        with open(os.path.join(self.params["pytorch_module_path"], 'config', self.params["config_file"])) as config_file:
            config = json.load(config_file)
        
        if config["depth_type"] != "None":

            depth = self.bridge.imgmsg_to_cv2(depth)
            depth = cv2.resize(depth, dsize=(self.params["width"], self.params["height"]), interpolation=cv2.INTER_AREA)
            mask = np.isnan(depth.copy()).astype('uint8')
            depth = np.where(np.isnan(depth), 0, depth)           

            values = np.unique(depth)
            min_depth = self.params["min_depth"]
            max_depth = self.params["max_depth"]
            depth = np.uint8((depth - min_depth) / (max_depth - min_depth) * 255) # normalize 
            depth = cv2.inpaint(depth, mask, 3, cv2.INPAINT_NS)
            depth_vis = cv2.resize(depth.copy(), dsize=(self.params["width"], self.params["height"]), interpolation=cv2.INTER_AREA )
            self.imdepth_pub.publish(self.bridge.cv2_to_imgmsg(depth_vis))

            depth = np.expand_dims(np.asarray(depth), -1)
            depth = np.repeat(depth, 3, -1)
            depth = torch.from_numpy(np.float32(depth))
            depth = depth.transpose(0, 2).transpose(1, 2)
            depth = depth/255 # 0~255 to 0~1

            # create corresponding mask
            val_mask = torch.ones([depth.shape[1], depth.shape[2]])
            val_mask[np.where(depth[0] == 0.0)] = 0
            val_mask = val_mask.unsqueeze(0)

            input_tensor = torch.cat([rgb, depth, val_mask], dim=0)
            input_tensor = input_tensor.unsqueeze(0)
        else:
            input_tensor = rgb.unsqueeze(0)

        # pred_results = self.model([rgb.to(self.device)])[0]
        pred_results = self.model(input_tensor.to(self.device))[0]

        pred_masks = pred_results["masks"].cpu().detach().numpy()
        pred_boxes = pred_results['boxes'].cpu().detach().numpy()
        pred_labels = pred_results['labels'].cpu().detach().numpy()
        pred_scores = pred_results['scores'].cpu().detach().numpy()
        vis_results = self.visualize_prediction(rgb_img, pred_masks, pred_boxes, pred_labels, pred_scores, thresh=0.3)

        # inference result -> ros message (Detection2D)
        is_msg = InstanceSegmentation2D()
        is_msg.header = Header()
        is_msg.header.stamp = rospy.get_rostime()
        
        scores = []
        for i, (x1, y1, x2, y2) in enumerate(pred_boxes):
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
        self.vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_results, "bgr8"))

    def visualize_prediction(self, rgb_img, masks, boxes, labels, score, thresh=0.5):
        
        rgb_img = np.uint8(rgb_img)
        if len(labels) == 0:
            return rgb_img

        for i in range(len(labels)):
            if score[i] > thresh:
                mask = masks[i][0]
                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0

                color = 255*np.random.random(3)
                r = mask * class_idx2color[labels[i]][0]
                g = mask * class_idx2color[labels[i]][1]
                b = mask * class_idx2color[labels[i]][2]
                stacked_img = np.stack((r, g, b), axis=0)
                stacked_img = stacked_img.transpose(1, 2, 0)

                rgb_img = cv2.addWeighted(rgb_img, 1, stacked_img.astype(np.uint8), 1, 0)
                cv2.rectangle(rgb_img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 255, 0), 1)
                cv2.putText(rgb_img, self.class_names[labels[i]] + str(score[i].item())[:4], \
                    (boxes[i][0], boxes[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255 ,0), 1)

        return np.uint8(rgb_img)

if __name__ == '__main__':

    furniture_part_segmenter = PartSegmenter()
    rospy.spin()




