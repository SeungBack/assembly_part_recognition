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

class_idx2name = {
    1: "side",
    2: "long_short",
    3: "middle",
    4: "bottom",
}

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
        rospy.init_node('part_recognizier')
        rospy.loginfo("Starting part_segmenter.py")

        self.seg_params = rospy.get_param("furniture_part_segmentation")
        self.mask_pub = rospy.Publisher('seg_mask/furniture_part', Image, queue_size=10)


        self.initialize_model()
        self.rgb_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                            ])

        self.bridge = cv_bridge.CvBridge()

        # get rgb-depth images of same time step
        rgb_sub = message_filters.Subscriber('azure1/rgb/image_raw', Image)
        depth_sub = message_filters.Subscriber('azure1/depth_to_rgb/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=5, slop=0.1)
        rospy.loginfo("Starting rgb-d subscriber with time synchronizer")
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

        rospy.loginfo_once("Segmenting furniture part area")
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

                color = 255*np.random.random(3)
                r = mask * class_idx2color[labels[i]][0]
                g = mask * class_idx2color[labels[i]][1]
                b = mask * class_idx2color[labels[i]][2]
                stacked_img = np.stack((r, g, b), axis=0)
                stacked_img = stacked_img.transpose(1, 2, 0)

                rgb_img = cv2.addWeighted(rgb_img, 1, stacked_img.astype(np.uint8), 0.5, 0)
                cv2.rectangle(rgb_img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 255, 0), 1)
                cv2.putText(rgb_img, class_idx2name[labels[i]] + str(score[i].item())[:4], \
                    (boxes[i][0], boxes[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255 ,0), 1)

        return np.uint8(rgb_img)




if __name__ == '__main__':

    furniture_part_segmenter = PartSegmenter()
    rospy.spin()




