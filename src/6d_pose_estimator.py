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
from sensor_msgs.msg import RegionOfInterest, Image, CameraInfo
from assembly_part_recognition.msg import InstanceSegmentation2D
 
import cv2
import argparse
import tensorflow as tf
import numpy as np
import os
import configparser

import pcl

import open3d as o3d
import copy

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

        # self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, is_sub], queue_size=5, slop=3)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, is_sub, caminfo_sub], queue_size=5, slop=3)
        rospy.loginfo("Starting zivid rgb-d subscriber with time synchronizer")
        # from rgb-depth images, inference the results and publish it
        # self.ts.registerCallback(self.inference)
        self.ts.registerCallback(self.inference_aae_pose)

    # def initialize_model(self):

    #     # import augmented autoencoder module        
    #     sys.path.append(self.params["augauto_module_path"])
    #     os.environ["CUDA_VISIBLE_DEVICES"] = self.params["gpu_id"]
    #     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    #     from auto_pose.ae import factory
    #     from auto_pose.ae import utils as u

    #     full_name = self.params["exp_group"].split('/')
    #     experiment_name = full_name.pop()
    #     experiment_group = full_name.pop() if len(full_name) > 0 else ''
    #     self.codebook, self.dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=True)

    #     workspace_path = os.environ.get('AE_WORKSPACE_PATH')
    #     log_dir = u.get_log_dir(workspace_path,experiment_name, experiment_group)
    #     ckpt_dir = u.get_checkpoint_dir(log_dir)

    #     train_cfg_file_path = u.get_train_config_exp_file_path(log_dir, experiment_name)
    #     train_args = configparser.ConfigParser()
    #     train_args.read(train_cfg_file_path)  

    #     gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.9)
    #     config = tf.ConfigProto(gpu_options=gpu_options)
    #     config.gpu_options.allow_growth = True

    #     self.session = tf.Session(config=config) 
    #     factory.restore_checkpoint(self.session, tf.train.Saver(), ckpt_dir)

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

    def inference_aae_pose(self, rgb, depth, is_results, caminfo):

        
        rospy.loginfo_once("Estimating 6D pose of objects")
        rgb_original = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb = cv2.resize(rgb_original.copy(), (self.params["width"], self.params["height"]))

        depth_original = self.bridge.imgmsg_to_cv2(depth)
        depth = cv2.resize(depth_original.copy(), dsize=(self.params["width"], self.params["height"]), interpolation=cv2.INTER_AREA)
        depth = np.where(np.isnan(depth), 0, depth)  

        K = np.array(caminfo.K).reshape([3, 3])

        # mask = np.where(depth==0, 1, 0).astype('uint8')
        # values = np.unique(depth)
        # max_val = max(values) 
        # min_val = min(values)
        # depth = np.uint8((depth - min_val) / (max_val - min_val) * 255)
        # depth = cv2.inpaint(depth, mask, 3, cv2.INPAINT_NS)
    
        # cv2.imwrite("/home/demo/depth.png", depth)
        # depth = self.bridge.imgmsg_to_cv2(depth)
        # depth = cv2.resize(depth, dsize=(self.params["width"], self.params["height"]), interpolation=cv2.INTER_AREA)

        for i, class_id in enumerate(is_results.class_ids):
            boxes, scores, labels = [], [], []
            if class_id == self.params["class_id"] and is_results.scores[i] >= self.params["is_thresh"]:
                # crop the object area with offset
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
                rgb_crop = cv2.resize(rgb_crop, (128, 128))
                depth_crop = depth[y1:y2, x1:x2].copy()
                depth_crop = cv2.resize(depth_crop, (128, 128))
                is_mask = self.bridge.imgmsg_to_cv2(is_results.masks[i])

                boxes.append(np.array([x, y, w, h]))
                scores.append(is_results.scores[i])
                # labels.append(is_results.class_ids[i]-1)
                labels.append(0)

                all_pose_estimates, all_class_idcs = self.ae_pose_est.process_pose(boxes, labels, rgb, depth)
                ply_model_paths = [str(train_args.get('Paths','MODEL_PATH')) for train_args in self.ae_pose_est.all_train_args]
                
                sys.path.append(self.params["augauto_module_path"] + "/auto_pose")
                from auto_pose.ae.utils import get_dataset_path
                from meshrenderer import meshrenderer_phong
                renderer = meshrenderer_phong.Renderer(ply_model_paths, 
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
                bgr = cv2.resize(bgr, (self.ae_pose_est._width,self.ae_pose_est._height))
                
                g_y = np.zeros_like(bgr)
                g_y[:,:,1]= bgr[:,:,1]    
                im_bg = cv2.bitwise_and(rgb, rgb, mask=(g_y[:,:,1]==0).astype(np.uint8))                 
                image_show = cv2.addWeighted(im_bg, 1, g_y, 1, 0)

                for label,box,score in zip(labels,boxes,scores):
                    box = box.astype(np.int32)
                    xmin, ymin, xmax, ymax = box[0], box[1], box[0] + box[2], box[1] + box[3]
                    cv2.putText(image_show, '%s : %1.3f' % (label,score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, color_dict[int(label)], 2)
                    cv2.rectangle(image_show, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                self.vis_pub.publish(self.bridge.cv2_to_imgmsg(image_show, "bgr8"))          


                # target_width = rgb_original.shape[1]
                # target_height = rgb_original.shape[0]
                # target_dim = (target_width, target_height)
                # is_mask_resized = cv2.resize(is_mask, target_dim, interpolation=cv2.INTER_AREA)
                
                # list_points_local = []
                # for idx_pixel in np.ndindex(is_mask_resized.shape):
                #     if is_mask_resized[idx_pixel] < 128 or np.isnan(depth_original[idx_pixel]):
                #         continue

                #     # gather depth (x, y, z)
                #     p_screen = np.array([idx_pixel[0], idx_pixel[1], depth_original[idx_pixel]], dtype=float)
                #     # print(p_screen)
                #     p_local = np.matmul(np.linalg.inv(K), p_screen)
                #     # print(p_local) # OK
                    
                #     list_points_local.append(tuple(p_local))

                break # currently, detect only single object with the highest confidence per class 

        # cloud_zivid = o3d.geometry.PointCloud()
        # cloud_zivid.points = o3d.utility.Vector3dVector(list_points_local)
        # cloud_zivid.paint_uniform_color([1, 0.706, 0]) # yellow

        # o3d.io.write_point_cloud("/home/demo/Downloads/cloud_zivid.ply", cloud_zivid)

        # cloud_gt = o3d.io.read_point_cloud("/home/demo/Workspace/seung/AugmentedAutoencoder/3d_models/ply/Ikea_stefan_middle_cloud.ply")        
        

        # T = np.eye(4)
        # T[:3, :3] = all_pose_estimates[0][:3, :3]
        # T[:3, 3] = cloud_zivid.get_center()
        # print("[DEBUG] cloud_zivid.get_center() = ", cloud_zivid.get_center())
        # print("[DEBUG] T[:3, 3] = ", T[:3, 3])

        # flip_xy_plane = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        # rotate_z_90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) # -90 along Z
        # T[:3, :3] = np.matmul(flip_xy_plane, T[:3, :3])
        # T[:3, :3] = np.matmul(rotate_z_90, T[:3, :3])

        # rospy.loginfo_once("T \n{}".format(T))
        # cloud_zivid_transformed = copy.deepcopy(cloud_zivid).transform(T) # to origin
        # cloud_zivid_transformed.paint_uniform_color([0.5, 0, 0]) # dark red 
        # cloud_zivid_transformed2 = copy.deepcopy(cloud_zivid).transform(np.linalg.inv(T))
        # cloud_zivid_transformed2.paint_uniform_color([0, 0.5, 0]) # dark green

        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame()

        # o3d.visualization.draw_geometries([coordinate, cloud_gt, cloud_zivid_transformed, cloud_zivid_transformed2, cloud_zivid])



    def inference_caminfo(self, rgb, depth, is_results, caminfo):

        rospy.loginfo_once("Estimating 6D pose of objects")
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb_original = rgb.copy()
        rgb = cv2.resize(rgb, (self.params["width"], self.params["height"]))
        

        for i, class_id in enumerate(is_results.class_ids):
            if class_id == self.params["class_id"] and is_results.scores[i] >= self.params["is_thresh"]:
                # crop the object area with offset
                x = is_results.boxes[i].x_offset
                y = is_results.boxes[i].y_offset
                h = is_results.boxes[i].height
                w = is_results.boxes[i].width
                h_offset = int(h*self.params["crop_offset"])
                w_offset = int(w*self.params["crop_offset"])
                new_x1 = max(0, x-w_offset)
                new_y1 = max(0, y-h_offset)
                new_x2 = min(self.params["width"]-1, x+w+w_offset)
                new_y2 = min(self.params["height"]-1, y+h+h_offset)
                rgb_crop = rgb[new_y1:new_y2, new_x1:new_x2].copy()
                rgb_crop = cv2.resize(rgb_crop, (128, 128))
                # get translation
                # median filter on segmented region

                depth = self.bridge.imgmsg_to_cv2(depth)
                depth_original = depth.copy()
                depth = cv2.resize(depth, dsize=(self.params["width"], self.params["height"]), interpolation=cv2.INTER_AREA)
                mask = np.isnan(depth.copy()).astype('uint8')
                depth = np.where(np.isnan(depth), 0, depth)
                is_mask = self.bridge.imgmsg_to_cv2(is_results.masks[i])

                # DEBUG: plot depth
                # print(depth.shape)
                # for idx_pixel in np.ndindex(depth.shape):
                #     print("[DEBUG] depth #", idx_pixel, " = ", depth[idx_pixel]) # OK

                # DEBUG: visualize mask
                # print(is_mask.shape)
                # cv2.imshow("mask", is_mask)
                # cv2.waitKey(0) # OK
                
                # Resize mask image & color & depth to original resolution
                target_width = rgb_original.shape[1]
                target_height = rgb_original.shape[0]
                target_dim = (target_width, target_height)
                is_mask_resized = cv2.resize(is_mask, target_dim, interpolation=cv2.INTER_AREA)
                # print(is_mask_resized.shape)
                # cv2.imshow("mask_resized", is_mask_resized)
                # cv2.imshow("rgb_original", rgb_original)
                # cv2.waitKey(0) # OK

                # Calibration matrix
                K = np.array(caminfo.K).reshape([3, 3])
                # print(K)

                # Iterate color image over mask index
                list_points_local = []
                for idx_pixel in np.ndindex(is_mask_resized.shape):
                    if is_mask_resized[idx_pixel] < 128 or np.isnan(depth_original[idx_pixel]):
                        continue

                    # gather depth (x, y, z)
                    p_screen = np.array([idx_pixel[0], idx_pixel[1], depth_original[idx_pixel]], dtype=float)
                    # print(p_screen)
                    p_local = np.matmul(np.linalg.inv(K), p_screen)
                    # print(p_local) # OK
                    
                    list_points_local.append(tuple(p_local))
                    
 
                # Median values
                t = np.mean(np.array(list_points_local), axis = 0)

                # bbox center
                bbox_min = np.matmul(np.linalg.inv(K), np.array([x, y, depth_original[x, y]]))
                bbox_max = np.matmul(np.linalg.inv(K), np.array([x+w, y+h ,depth_original[x+w, y+h]]))
                print(bbox_min, bbox_max)
                t = (bbox_min + bbox_max) / 2.0 
                
                
                print("centroid = ", t)

                break # currently, detect only single object with the highest confidence per class 
        # get rotation
        R = self.codebook.nearest_rotation(self.session, rgb_crop)
        pred_view = self.dataset.render_rot(R, downSample = 1)
        print(R)

        # cloud_gt = o3d.io.read_point_cloud("/home/demo/catkin_ws/src/assembly_point_cloud_manager/sample/pcd/stefan_middle.pcd")
        cloud_gt = o3d.io.read_point_cloud("/home/demo/Workspace/seung/AugmentedAutoencoder/3d_models/ply/Ikea_stefan_middle_cloud.ply")
        # cloud_gt.scale(1e-3, center=cloud_gt.get_center())
        cloud_zivid = o3d.geometry.PointCloud()
        # print(np.array(list_points_local).shape) # OK
        cloud_zivid.points = o3d.utility.Vector3dVector(list_points_local)
        cloud_zivid.paint_uniform_color([1, 0.706, 0])

        # Transformation

        T = np.eye(4)
        T[:3, :3] = R # T->R->S
        T[:3, 3] = cloud_zivid.get_center()
        cloud_zivid_transformed = copy.deepcopy(cloud_gt).transform(T) # to origin
        cloud_zivid_transformed.paint_uniform_color([0.5, 0, 0])


        T = np.eye(4)
        T[2, 2] =  -1
        cloud_zivid_transformed2 = copy.deepcopy(cloud_zivid_transformed).transform(T)
        cloud_zivid_transformed2.paint_uniform_color([0, 0.5, 0])


        # cloud_zivid_transformed3 = copy.deepcopy(cloud_zivid_transformed).transform(T)
        # cloud_zivid_transformed3.paint_uniform_color([0, 0, 0.5])

        
        vis_results = np.hstack([rgb_crop, pred_view])
        self.vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_results, "bgr8"))

        o3d.visualization.draw_geometries([cloud_gt, cloud_zivid, cloud_zivid_transformed, cloud_zivid_transformed2])




if __name__ == '__main__':

    pose_estimator = PoseEstimator()
    rospy.spin()




