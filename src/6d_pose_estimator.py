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
from sensor_msgs.msg import RegionOfInterest, Image, CameraInfo, PointCloud2
from assembly_part_recognition.msg import InstanceSegmentation2D
 
import cv2
import argparse
import tensorflow as tf
import numpy as np
import os
import configparser

import pcl
import pcl_ros

import open3d as o3d
import copy

import open3d_helper

color_dict = [(255,255,0),(0,0,255),(255,0,0),(255,255,0)] * 10


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
        caminfo_sub = message_filters.Subscriber("/zivid_camera/color/camera_info", CameraInfo)

        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, is_sub, caminfo_sub], queue_size=5, slop=5)
        rospy.loginfo("Starting zivid rgb-d subscriber with time synchronizer")
        # from rgb-depth images, inference the results and publish it
        self.ts.registerCallback(self.inference_aae_pose)


        # cloud_sub = message_filters.Subscriber("/zivid_camera/points", PointCloud2)
        # self.ts = message_filters.ApproximateTimeSynchronizer([cloud_sub], queue_size=5, slop=5)
        # self.ts.registerCallback(self.test_cloud)

    def initialize_model(self):

        # initialize augmented auto encoder and renderer      
        sys.path.append(self.params["augauto_module_path"])
        sys.path.append(self.params["augauto_module_path"] + "/auto_pose/test")

        from aae_maskrcnn_pose_estimator import AePoseEstimator
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params["gpu_id"]
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

        self.workspace_path = os.environ.get('AE_WORKSPACE_PATH')
        if self.workspace_path == None:
            print 'Please define a workspace path:\n'
            print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
            exit(-1)

        test_configpath = os.path.join(self.workspace_path,'cfg_eval', self.params["test_config"])
        test_args = configparser.ConfigParser()
        test_args.read(test_configpath)
        self.ae_pose_est = AePoseEstimator(test_configpath)
        self.ply_model_paths = [str(train_args.get('Paths','MODEL_PATH')) for train_args in self.ae_pose_est.all_train_args]

    def inference_aae_pose(self, rgb, depth, is_results, caminfo):

        # TODO: undistort rgb_original, depth_original

        # get rgb, depth, is_results K, pointcloud
        rospy.loginfo_once("Estimating 6D pose of objects")
        rgb_original = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb_resized = cv2.resize(rgb_original.copy(), (self.params["width"], self.params["height"]))

        depth_original = self.bridge.imgmsg_to_cv2(depth)
        depth_resized = cv2.resize(depth_original.copy(), dsize=(self.params["width"], self.params["height"]), interpolation=cv2.INTER_AREA)
        depth_resized = np.where(np.isnan(depth_resized), 0, depth_resized)  

        K = np.array(caminfo.K).reshape([3, 3])


        # gather all detected crops
        boxes, scores, labels = [], [], []
        boxes, scores, labels, is_mask = self.gather_is_results(rgb_resized, depth_resized, boxes, scores, labels, is_results)
        is_mask_original = cv2.resize(is_mask, (rgb_original.shape[1], rgb_original.shape[0]), interpolation=cv2.INTER_AREA)

        # 6D pose estimation for all detected crops
        all_pose_estimates, all_class_idcs = self.ae_pose_est.process_pose(boxes, labels, rgb_resized, depth_resized)

        # visualize pose estimation results
        self.visualize_pose_estimation_results(all_pose_estimates, all_class_idcs, labels, boxes, scores, rgb_resized)

        # cloud_zivid
        list_points_local = []
        list_colors_local = []
        for idx_pixel in np.ndindex(is_mask_original.shape):
            if is_mask_original[idx_pixel] < 128 or np.isnan(depth_original[idx_pixel]):
                continue
            # gather depth (x, y, z)
            p_screen = np.array([idx_pixel[0], idx_pixel[1], depth_original[idx_pixel]], dtype=float)
            p_local = np.matmul(np.linalg.inv(K), p_screen) 
            list_points_local.append(tuple(p_local))

            c_local = np.array([rgb_original[idx_pixel][0], rgb_original[idx_pixel][1], rgb_original[idx_pixel][2]]) / 255.0
            list_colors_local.append(tuple(c_local))

        cloud_zivid = o3d.geometry.PointCloud()
        cloud_zivid.points = o3d.utility.Vector3dVector(list_points_local)
        cloud_zivid.colors = o3d.utility.Vector3dVector(list_colors_local)
        T_seg_to_center = np.eye(4)
        T_seg_to_center[:3, 3] = -cloud_zivid.get_center()
        cloud_zivid.transform(T_seg_to_center)
        cloud_zivid.transform([[1000, 0, 0, 0], [0, 1000, 0, 0], [0, 0, 1000, 0], [0, 0, 0, 1]]) # rescale ply from mm to m

        # 6D object pose refinement using ICP on pointcloud
        for i, (pose_estimate, class_id) in enumerate(zip(all_pose_estimates, all_class_idcs)):

            cloud_GT = o3d.io.read_point_cloud(self.ply_model_paths[class_id])     
            # estimated rotation
            T_rot = np.eye(4)
            T_rot[:3, :3] = pose_estimate[:3, :3]
            cloud_GT_rotated = copy.deepcopy(cloud_GT).transform(T_rot)

            # OpenGL->Real
            T_gl_to_zivid = np.eye(4)
            flip_xy_plane = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
            rotate_z_90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) # -90 along Z
            T_gl_to_zivid[:3, :3] = np.matmul(flip_xy_plane, T_gl_to_zivid[:3, :3])
            T_gl_to_zivid[:3, :3] = np.matmul(rotate_z_90, T_gl_to_zivid[:3, :3])

            flip = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float) # ok
            rot_y_90 = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float)
            flip = np.matmul(rot_y_90, flip)

            cloud_GT_to_zivid = copy.deepcopy(cloud_GT_rotated).transform(flip)
            o3d.visualization.draw_geometries([
                cloud_zivid, 
                cloud_GT_to_zivid, 
            ])

    def gather_is_results(self, rgb, depth, boxes, scores, labels, is_results):

        for i, class_id in enumerate(is_results.class_ids):
            if is_results.scores[i] >= self.params["is_thresh"]:
                x = is_results.boxes[i].x_offset
                y = is_results.boxes[i].y_offset
                h = is_results.boxes[i].height
                w = is_results.boxes[i].width
                h_offset = int(h*self.params["crop_offset"])
                w_offset = int(w*self.params["crop_offset"])
                x1 = max(0, x-w_offset)
                y1 = max(0, y-h_offset)
                x2 = min(self.params["width"]-1, x+w+w_offset)
                y2 = min(self.params["height"]-1, y+h+h_offset)
                rgb_crop = rgb[y1:y2, x1:x2].copy()
                if self.params["padding"]:
                    pad = abs(((y2-y1)-(x2-x1))//2)
                    if h >= w:
                        rgb_crop = cv2.copyMakeBorder(rgb_crop, 0, 0, pad, pad, borderType=cv2.BORDER_CONSTANT, value=0)
                    else:
                        rgb_crop = cv2.copyMakeBorder(rgb_crop, pad, pad, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)
                    cv2.imwrite("/home/demo/rgb_crop.png", rgb_crop)
                rgb_crop = cv2.resize(rgb_crop, (128, 128))
                depth_crop = depth[y1:y2, x1:x2].copy()
                depth_crop = cv2.resize(depth_crop, (128, 128))
                is_mask = self.bridge.imgmsg_to_cv2(is_results.masks[i])

                boxes.append(np.array([x, y, w, h]))
                scores.append(is_results.scores[i])
                labels.append(class_id-1)
        return boxes, scores, labels, is_mask


    def visualize_pose_estimation_results(self, all_pose_estimates, all_class_idcs, labels, boxes, scores, rgb):

        sys.path.append(self.params["augauto_module_path"] + "/auto_pose")
        from auto_pose.ae.utils import get_dataset_path
        from meshrenderer import meshrenderer_phong

        renderer = meshrenderer_phong.Renderer(self.ply_model_paths, 
            samples=1, 
            vertex_tmp_store_folder=get_dataset_path(self.workspace_path),
            vertex_scale=float(1)) # float(1) for some models

        bgr, depth, _ = renderer.render_many(obj_ids = [clas_idx for clas_idx in all_class_idcs],
                            W = self.ae_pose_est._width,
                            H = self.ae_pose_est._height,
                            K = self.ae_pose_est._camK, 
                            Rs = [pose_est[:3,:3] for pose_est in all_pose_estimates],
                            ts = [pose_est[:3,3] for pose_est in all_pose_estimates],
                            near = 10,
                            far = 10000,
                            random_light=False,
                            phong={'ambient':0.4,'diffuse':0.8, 'specular':0.3})
        bgr = cv2.resize(bgr, (self.ae_pose_est._width, self.ae_pose_est._height))
        
        g_y = np.zeros_like(bgr)
        g_y[:,:,1]= bgr[:,:,1]    
        im_bg = cv2.bitwise_and(rgb, rgb, mask=(g_y[:,:,1]==0).astype(np.uint8))                 
        image_show = cv2.addWeighted(im_bg, 1, g_y, 1, 0)

        for label, box, score in zip(labels, boxes, scores):
            box = box.astype(np.int32)
            xmin, ymin, xmax, ymax = box[0], box[1], box[0] + box[2], box[1] + box[3]
            cv2.putText(image_show, '%s : %1.3f' % (label,score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, color_dict[int(label)], 2)
            cv2.rectangle(image_show, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        self.vis_pub.publish(self.bridge.cv2_to_imgmsg(image_show, "bgr8"))          
    
if __name__ == '__main__':

    pose_estimator = PoseEstimator()
    rospy.spin()




