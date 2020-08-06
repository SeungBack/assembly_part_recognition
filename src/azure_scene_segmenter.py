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
import time


def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    
    for name, param in state_dict["state_dict"].items():
        if name not in own_state:
            print('{} not in model_state'.format(name))
            continue
        else:
            own_state[name].copy_(param)

    return model

class SceneSegmenter:

    def __init__(self):

        # initalize node
        rospy.init_node('azure_scene_segmenter')
        rospy.loginfo("Starting azure_scene_segmenter.py")

        self.seg_params = rospy.get_param("azure_scene")
        self.mask_pub = rospy.Publisher('seg_mask/azure/scene', Image, queue_size=10)

        self.initialize_model()

        self.bridge = cv_bridge.CvBridge()

        self.depth_val_min = 0.1
        self.depth_val_max = 3.25

        # get rgb-depth images of same time step
        rgb_sub = message_filters.Subscriber('azure1/rgb/image_raw', Image)
        depth_sub = message_filters.Subscriber('azure1/depth_to_rgb/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=5, slop=0.1)
        rospy.loginfo("Starting rgb-d subscriber with time synchronizer")
        # from rgb-depth images, inference the results and publish it
        self.ts.registerCallback(self.inference)

    def initialize_model(self):

        # import RFNet from pytorch module        
        sys.path.append(self.seg_params["pytorch_module_path"])

        from models.rfnet import RFNet
        from models.resnet.resnet_single_scale_single_attention import *
        from dataloaders import custom_transforms as tr
        

        self.transform = composed_transforms = transforms.Compose([
            
            tr.FixedResize(size=768),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        # build model
        self.resnet = resnet18(pretrained=True, efficient=False, use_bn= True)
        self.model = RFNet(self.resnet, num_classes=4, use_bn=True)
    
        #self.model.load_state_dict(torch.load(self.seg_params["weight_path"]))
        
        self.model = load_my_state_dict(self.model, torch.load(self.seg_params["weight_path"]))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

   
    def save_inference_data(self, rgb, depth):
        rgb_save = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb_save = cv2.cvtColor(rgb_save, cv2.COLOR_BGR2RGB)
        rgb_save = PIL.Image.fromarray(np.uint8(rgb_save), mode="RGB")
        depth_save = self.bridge.imgmsg_to_cv2(depth)
        save_dir = "/home/demo/catkin_ws/src/assembly_part_recognition/inference_data_azure"

        save_dir_rgb = os.path.join(save_dir, 'rgb')
        save_dir_depth = os.path.join(save_dir, 'depth_value')
        if not os.path.isdir(save_dir_rgb):
            os.makedirs(save_dir_rgb)
        if not os.path.isdir(save_dir_depth):
            os.makedirs(save_dir_depth)

        save_name = "{}".format(time.time())

        rgb_save.save(os.path.join(save_dir_rgb, "{}.png".format(save_name)))
        np.save(os.path.join(save_dir_depth, "{}.npy".format(save_name)), depth_save)

        # np.save(os.path.join(save_dir, save_name + "_rgb.npy"), rgb_save)
        # np.save(os.path.join(save_dir, save_name + "_depth.npy"), depth_save)

    def inference(self, rgb, depth):
        #self.save_inference_data(rgb, depth)

        sys.path.append(self.seg_params["pytorch_module_path"])
        from dataloaders.utils import Colorize
        from torchvision.transforms import ToPILImage

        rospy.loginfo_once("Segmenting scene area")
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb = PIL.Image.fromarray(np.uint8(rgb), mode="RGB") # PIL image
        rgb_img = rgb
        
        depth = self.bridge.imgmsg_to_cv2(depth) * 1000
        mask = np.uint8(np.where(depth==0, 1, 0))

        #max_val = 3.0
        #min_val = 0.2
        vals = np.unique(depth)
        max_val = max(vals)
        min_val = min(vals)

        depth = np.uint8((depth - min_val) / (max_val - min_val) * 255)
        depth = cv2.inpaint(depth, mask, 3, cv2.INPAINT_NS)

        depth = np.expand_dims(np.asarray(depth), -1)
        depth = np.repeat(depth, 3, -1)

        depth[depth>100] = 255

        # depth_max = 160
        # depth_min = 30
        # depth_arr = np.clip(depth, depth_min, depth_max)
        # depth = (depth_arr - depth_min) / (depth_max - depth_min)
        # depth = depth * 255

        # depth_min = 0.25
        # depth_max = 3    
        # depth_arr = np.clip(depth, depth_min, depth_max)
        # depth = (depth_arr - depth_min) / (depth_max - depth_min)
        # depth = depth * 255

        depth = PIL.Image.fromarray(depth).convert("L")
        depth.save("/home/demo/Workspace/Assembly-scene-segmentation/depth_infer/img.png")
        sample = {'image': rgb,'depth': depth}

        sample = self.transform(sample)
        image, depth = sample['image'], sample['depth']
        image = image.unsqueeze(0)
        depth = depth.unsqueeze(0)
      
        output = self.model(image.to(self.device), depth.to(self.device))
        pre_colors = Colorize()(torch.max(output, 1)[1].detach().cpu().byte())

        pre_color_image = ToPILImage()(pre_colors[0])
        full_mask = pre_color_image

        vis_results = self.visualize_prediction(rgb_img, full_mask)


        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(vis_results, "bgr8"))

    def visualize_prediction(self, rgb_img, label):
        image = rgb_img
       
        width, height = image.size
        left = 140
        top = 30
        right = 2030
        bottom = 900
        # crop

        # resize
        label = label.resize(image.size, PIL.Image.BILINEAR)

        image = image.convert('RGBA')
        label = label.convert('RGBA')
        image = PIL.Image.blend(image, label, 0.3)
       
        return np.asarray(image.convert("RGB"))


if __name__ == '__main__':

    scene_semantic_segmenter = SceneSegmenter()
    rospy.spin()




