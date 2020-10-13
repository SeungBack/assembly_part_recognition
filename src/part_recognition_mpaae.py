#!/usr/bin/env python

import json
import rospy
import cv2, cv_bridge
import numpy as np
import PIL
import message_filters
import yaml
import sys
import glob

from std_msgs.msg import String, Header
from sensor_msgs.msg import RegionOfInterest, Image, CameraInfo, PointCloud2
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped
import tf.transformations as tf_trans

import cv2
import argparse
import numpy as np
import os
import configparser

import open3d as o3d # import both o3d and pcl occurs segmentation faults
from open3d_ros_helper.utils import *
import copy
import time
import tf2_ros
import scipy as sp
import numpy.matlib as npm

from auto_pose.ae import utils as u
from auto_pose.ae import ae_factory as factory
from auto_pose.ae.utils import get_dataset_path
from auto_pose.meshrenderer import meshrenderer_phong
from auto_pose.meshrenderer.pysixd import misc, visibility
import matplotlib.pyplot as plt


class PoseEstimator:

    def __init__(self):

        # initalize node
        rospy.init_node('part_recognition')
        rospy.loginfo("Starting part_recognition_mpaae.py")

        self.params = rospy.get_param("part_recognition")
        self.is_class_names = self.params["is_class_names"]
        self.pad_factors = [1.2, 1.2, 1.2, 1.2]
        self.use_masked_crop = self.params["use_mask_crop"]
        self.use_sameWH_crop =self.params["use_sameWHcrop"]
        self.pe_class_names = self.params["pe_class_names"]
        self.classidx2color = [[13, 128, 255], [255, 12, 12], [217, 12, 232], [232, 222, 12]]
        self.color_dict = [(255,255,0), (0,0,255), (255,0,0), (255,255,0)] * 10
        self.roi = self.params["roi"]
        self.bridge = cv_bridge.CvBridge()

        # subscribers
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1.0))
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        is_sucess = False
        while not is_sucess:
            try:
                self.transform_map_to_cam = self.tf_buffer.lookup_transform(self.params["camera_frame"], "map", rospy.Time(), rospy.Duration(1.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                print(e)
                rospy.sleep(0.5)
            is_sucess = True
        self.initialize_is_model()
        self.initialize_pose_est_model()
        rgb_sub = message_filters.Subscriber(self.params["rgb"], Image, buff_size=720*1280*3)
        # rgb_rect_sub = message_filters.Subscriber(self.params["rgb_rect"], Image, buff_size=720*1280*3)
        depth_sub = message_filters.Subscriber(self.params["depth"], Image, buff_size=720*1280)
        point_sub = message_filters.Subscriber(self.params["point"], PointCloud2, buff_size=720*1280*3)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, point_sub], queue_size=1, slop=1)
        self.ts.registerCallback(self.inference)
        self.camera_info = rospy.wait_for_message(self.params["camera_info"], CameraInfo)
        self.intrinsic_matrix = np.array(self.camera_info.K).reshape(3, 3)
        # publishers
        self.aae_pose_pubs = [] # "side_right", "long_short", "middle", "bottom"
        self.icp_pose_pubs = []
        self.idx2color = [[13, 128, 255], [255, 12, 12], [217, 12, 232], [232, 222, 12]]
        self.aae_detections_pub = rospy.Publisher('/assembly/detections/aae', Detection3DArray, queue_size=1)
        self.icp_detections_pub = rospy.Publisher('/assembly/detections/icp', Detection3DArray, queue_size=1)
        self.aae_markers_pub = rospy.Publisher('/assembly/markers/aae', MarkerArray, queue_size=1)
        self.icp_markers_pub = rospy.Publisher('/assembly/markers/icp', MarkerArray, queue_size=1)
        self.vis_is_pub = rospy.Publisher('/assembly/vis_is', Image, queue_size=1)
        self.pose_aae_vis_pub = rospy.Publisher('/assembly/vis_pe_aae', Image, queue_size=1)
        self.vis_vsd_costmap = rospy.Publisher('/assembly/vis_vsd_costmap', Image, queue_size=1)
        self.vis_cou_costmap = rospy.Publisher('/assembly/vis_cou_costmap', Image, queue_size=1)


    def initialize_is_model(self):
        import torch
        import torchvision.transforms as transforms
        # import maskrcnn from pytorch module        
        from MaskRCNN import maskrcnn
        # load config files
        with open(self.params["is_config_path"]) as config_file:
            self.config = json.load(config_file)
        # build model
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params["is_gpu_id"]
        self.model = maskrcnn.get_model_instance_segmentation(num_classes=5, config=self.config)
        self.model.load_state_dict(torch.load(self.params["is_weight_path"]))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.rgb_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                    ])

    def initialize_pose_est_model(self):
        import tensorflow as tf

        rospy.loginfo("Loading AAE model")
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params["pe_gpu_id"]
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.workspace_path = os.environ.get('AE_WORKSPACE_PATH')
        if self.workspace_path == None:
            print 'Please define a workspace path:\n'
            print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
            exit(-1)
        # load all code books
        full_name = self.params["pe_experiment_name"].split('/')    
        experiment_name = full_name.pop()
        experiment_group = full_name.pop() if len(full_name) > 0 else ''
        self.ply_paths = glob.glob(self.params["model_dir"] + '/ply/*.ply')
        self.ply_paths.sort()
        self.ply_centered_paths = glob.glob(self.params["model_dir"] + '/ply_centered/*.ply')
        self.ply_centered_paths.sort()
        self.codebook, self.dataset = factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset = True, joint=True)

        self.dims = []
        self.centroids = []
        self.cloud_objs = []
        for ply_centered in self.ply_centered_paths:
            cloud = o3d.io.read_point_cloud(ply_centered)
            self.dims.append(cloud.get_max_bound())
        for ply in self.ply_paths:
            cloud = o3d.io.read_point_cloud(ply)
            centroid = cloud.get_center()
            self.centroids.append(centroid)
            self.cloud_objs.append(cloud)

        gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.5)
        config = tf.ConfigProto(gpu_options=gpu_options)

        log_dir = u.get_log_dir(self.workspace_path, experiment_name, experiment_group)
        train_cfg_file_path = u.get_train_config_exp_file_path(log_dir, experiment_name)
        self.train_args = configparser.ConfigParser(inline_comment_prefixes="#")
        self.train_args.read(train_cfg_file_path)
        test_configpath = os.path.join(self.workspace_path, 'cfg_eval', self.params["pe_test_config"])
        test_args = configparser.ConfigParser()
        test_args.read(test_configpath)

        self.sess = tf.Session(config=config)
        saver = tf.train.Saver()
        checkpoint_file = u.get_checkpoint_basefilename(log_dir, False, latest=self.train_args.getint('Training', 'NUM_ITER'), joint=True)
        saver.restore(self.sess, checkpoint_file)
       


    def inference(self, rgb, depth, pcl_msg):
        ## 1. Get rgb, depth, point cloud
        start_time = time.time()
        camera_header = rgb.header
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb = PIL.Image.fromarray(np.uint8(rgb), mode="RGB")
        rgb_img = np.uint8(rgb.resize((self.params["width"], self.params["height"]), PIL.Image.BICUBIC))

        # rgb_rect = self.bridge.imgmsg_to_cv2(rgb_rect, desired_encoding='bgr8')
        # rgb_rect = PIL.Image.fromarray(np.uint8(rgb_rect), mode="RGB")
        # rgb_rect_img = np.uint8(rgb_rect.resize((self.params["width"], self.params["height"]), PIL.Image.BICUBIC))

        depth = self.bridge.imgmsg_to_cv2(depth, desired_encoding='32FC1')
        depth = cv2.resize(depth, dsize=(self.params["width"], self.params["height"]), interpolation=cv2.INTER_AREA)

        rgb = self.rgb_transform(rgb_img).unsqueeze(0)
        print("-------------------------------------")
        # print("=> Get Input \t {}".format(time.time()-start_time))

        ## 2. Instance Segmentation using Mask R-CNN
        is_results = self.model(rgb.to(self.device))[0]
        pred_masks = is_results["masks"].cpu().detach().numpy()
        pred_boxes = is_results['boxes'].cpu().detach().numpy()
        pred_labels = is_results['labels'].cpu().detach().numpy()
        pred_scores = is_results['scores'].cpu().detach().numpy()
        boxes, scores, is_obj_ids, rgb_crops, is_masks = [], [], [], [], []
        for i, (label, (x1, y1, x2, y2), score, mask) in enumerate(zip(pred_labels, pred_boxes, pred_scores, pred_masks)):
            if score < self.params["is_thresh"]:
                continue
            if x1 < self.roi[0] or x2 > self.roi[1] or y1 < self.roi[2] or y2 > self.roi[3]:
                continue
            boxes.append(np.array([x1, y1, x2, y2]))
            scores.append(score)
            label = label-1
            is_obj_ids.append(label)
            is_masks.append(mask[0])
            if self.use_masked_crop[label]:
                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0.3
                mask = np.repeat(mask, 3, 0)
                mask = np.transpose(mask, (1, 2, 0))
                masked_img = rgb_img * mask 
            else:
                masked_img = rgb_img 
            x = int(x1)
            y = int(y1)
            h = int(y2 - y1)
            w = int(x2 - x1)
            if self.use_sameWH_crop[label]:
                w_offset = int(np.maximum(h, w) * self.pad_factors[label])
                h_offset = int(np.maximum(h, w) * self.pad_factors[label])
            else:
                w_offset = int(w * self.pad_factors[label])
                h_offset = int(h * self.pad_factors[label])
            left = np.maximum(x+w//2-w_offset//2, 0)
            right = x+w//2+w_offset/2
            top = np.maximum(y+h//2-h_offset//2, 0)
            bottom = y+h//2+h_offset//2
            rgb_crop = masked_img[top:bottom, left:right].copy()
            if self.params["black_borders"]:
                rgb_crop[:(y-top),:] = 0
                rgb_crop[(y+h-top):,:] = 0
                rgb_crop[:,:(x-left)] = 0
                rgb_crop[:,(x+w-left):] = 0
            rgb_crop = cv2.resize(rgb_crop, (128, 128), interpolation=cv2.INTER_NEAREST)
            rgb_crops.append(rgb_crop)
        # print("=> Inst Seg \t {}".format(time.time()-start_time))

        ## 3. 6D Object Pose Estimation using MPAAE
        if len(is_masks) == 0:
            return
        all_pose_estimates = []
        for i, (box, score, label, rgb_crop) in enumerate(zip(boxes, scores, is_obj_ids, rgb_crops)):
            box_xywh = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            H_aae = np.eye(4)
            class_idx = self.pe_class_names.index(self.is_class_names[label])
            R_est, t_est, _ = self.codebook.auto_pose6d(self.sess, 
                                                    rgb_crop, 
                                                    box_xywh, 
                                                    self.intrinsic_matrix, 
                                                    1, 
                                                    self.train_args,
                                                    self.codebook._get_codebook_name(self.ply_centered_paths[class_idx]),
                                                    refine=False)
            H_aae[:3,:3] = R_est
            H_aae[:3, 3] = t_est 
            all_pose_estimates.append(H_aae)     
        # print("=> Pose Est \t {}".format(time.time()-start_time))

        ## 4. 6D Object Pose Refinement using ICP 
        if "merged" in self.params["point"]:
            cloud_map = convert_ros_to_o3d(pcl_msg) 
            cloud_cam = do_transform_o3d_cloud(copy.deepcopy(cloud_map), self.transform_map_to_cam)
        else:
            cloud_cam = convert_ros_to_o3d(pcl_msg, remove_nans=True) 

        aae_trans = []
        aae_rot = []
        icp_trans = []
        icp_rot = []
        for i, (pose_estimate, label) in enumerate(zip(all_pose_estimates, is_obj_ids)):
            # crop cloud with instance mask   
            is_mask = is_masks[i].copy()
            is_mask[is_mask < 0.5] = 0
            cloud_cam_obj = crop_o3d_cloud_with_mask(cloud_cam, is_mask, camera_info=self.camera_info)
            pe_obj_id = self.pe_class_names.index(self.is_class_names[label])
            # cloud_cam_obj, index = cloud_cam_obj.remove_statistical_outlier(nb_neighbors = 50, std_ratio=0.2)
            
            # transform cloud_obj to the origin of camera frame
            cloud_obj = copy.deepcopy(self.cloud_objs[pe_obj_id])
            H_obj2cam = np.eye(4)
            # add centeroids for misaligned CAD
            H_obj2cam[:3, 3] = - cloud_obj.get_center() + self.centroids[pe_obj_id]
            cloud_obj = cloud_obj.transform(H_obj2cam)
            # transform cloud_obj to the estimated 6d pose
            H_aae_cam2obj = np.eye(4)
            H_aae_cam2obj[:3, :3] = pose_estimate[:3, :3]
            H_aae_cam2obj[:3, 3] = pose_estimate[:3, 3]   # align scale
            cloud_obj = cloud_obj.transform(H_aae_cam2obj)
            # translate cloud_obj to the centroid of cloud cam
            H_obj_to_cam_centroid = cloud_cam_obj.get_center() - cloud_obj.get_center()
            # print(H_aae_cam2obj[:3, 3][2], H_obj_to_cam_centroid[2])
            H_aae_cam2obj[:3, 3] = H_aae_cam2obj[:3, 3] + H_obj_to_cam_centroid
            all_pose_estimates[i][:3, 3] = H_aae_cam2obj[:3, 3] * 1000
            cloud_obj = cloud_obj.translate(H_obj_to_cam_centroid)
            # o3d.visualization.draw_geometries([cloud_obj, cloud_cam_obj])
            
            # icp refinement
            icp_result, residual = icp_refinement_with_ppf_match(cloud_obj, cloud_cam_obj, 
                                n_points=self.params["n_points"], n_iter=self.params["n_iter"], tolerance=self.params["tolerance"], num_levels=self.params["num_levels"])
            # print("{}: \t res: {}".format(self.is_class_names[label], residual))
            if residual < 1:
                H_refined_cam2obj = np.matmul(icp_result, H_aae_cam2obj)
            else:
                H_refined_cam2obj = H_aae_cam2obj

            all_pose_estimates[i][:3, :3] = H_refined_cam2obj[:3, :3]
            T_centroid = np.eye(4)
            T_centroid[:3, 3] = self.centroids[pe_obj_id]

            translation = H_aae_cam2obj[:3, 3] 
            rotation = tf_trans.quaternion_from_matrix(H_aae_cam2obj)
            aae_trans.append(translation)
            aae_rot.append(rotation)

            translation = H_refined_cam2obj[:3, 3] 
            rotation = tf_trans.quaternion_from_matrix(H_refined_cam2obj)
            icp_trans.append(translation)
            icp_rot.append(rotation)
        # print("=> ICP \t {}".format(time.time()-start_time))

        # 5. calculate reprojection error (VSD) and publish the detection3D
        pe_obj_ids, pe_masks, pe_depth = self.reproject_pe(self.pose_aae_vis_pub, all_pose_estimates, is_obj_ids, boxes, scores, rgb_img)
        vsd_errors = self.get_vsd_error(self.vis_vsd_costmap, is_masks, is_obj_ids, depth, pe_masks, pe_obj_ids, pe_depth)
        cou_errors = self.get_cou_error(self.vis_cou_costmap, is_masks, is_obj_ids, depth, pe_masks, pe_obj_ids, pe_depth)
        reproj_errors = np.mean(np.vstack([vsd_errors, cou_errors]), axis=0)
        aae_detections_array = Detection3DArray()
        aae_detections_array.header = camera_header
        icp_detections_array = Detection3DArray()
        icp_detections_array.header = camera_header
        # print("=> Get reprojErr \t {}".format(time.time()-start_time))
        for i, is_obj_id in enumerate(is_obj_ids):
            class_id = self.pe_class_names.index(self.is_class_names[is_obj_id])
            aae_pose_msg, aae_detection = self.gather_pose_results(camera_header, class_id, aae_trans[i], aae_rot[i], reproj_errors[i])
            icp_pose_msg, icp_detection = self.gather_pose_results(camera_header, class_id, icp_trans[i], icp_rot[i], reproj_errors[i])
            aae_detections_array.detections.append(aae_detection)
            icp_detections_array.detections.append(icp_detection)

        self.publish_vis_is(rgb_img, is_masks, boxes, is_obj_ids, scores)
        self.aae_detections_pub.publish(aae_detections_array)
        self.icp_detections_pub.publish(icp_detections_array)
        self.publish_markers(self.aae_markers_pub, aae_detections_array, [13, 255, 128])
        self.publish_markers(self.icp_markers_pub, icp_detections_array, [128, 128, 255])


    def gather_pose_results(self, header, class_id, translation, rotation, score):
        
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.pose.position.x = translation[0] 
        pose_msg.pose.position.y = translation[1] 
        pose_msg.pose.position.z = translation[2] 
        pose_msg.pose.orientation.x = rotation[0]
        pose_msg.pose.orientation.y = rotation[1]
        pose_msg.pose.orientation.z = rotation[2]
        pose_msg.pose.orientation.w = rotation[3]

        # Add to detection3Darray
        detections = Detection3D()
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.id = class_id
        hypothesis.pose.pose = pose_msg.pose
        hypothesis.score = score
        detections.results.append(hypothesis)
        detections.bbox.center = pose_msg.pose
        detections.bbox.size.x = self.dims[class_id][0] * 2
        detections.bbox.size.y = self.dims[class_id][1] * 2
        detections.bbox.size.z = self.dims[class_id][2] * 2
        return pose_msg, detections


    def publish_vis_is(self, rgb_img, masks, boxes, is_obj_ids, score):
        
        if len(is_obj_ids) == 0:
            return rgb_img
        cv2.rectangle(rgb_img, (self.roi[0], self.roi[2]), (self.roi[1], self.roi[3]), (0, 0, 255), 3)
        cv2.putText(rgb_img, "ROI", (self.roi[0], self.roi[2] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        mask_all = np.zeros_like(rgb_img)

        for i in range(len(is_obj_ids)):
            x1, y1, x2, y2 = boxes[i]
            if x1 < self.roi[0] or x2 > self.roi[1] or y1 < self.roi[2] or y2 > self.roi[3]:
                color = (128, 128, 0)
                lw = 1
            else:
                color = (0, 255, 0)
                lw = 3
            mask = masks[i]
            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
            mask_all[mask.astype(np.bool)] = self.classidx2color[is_obj_ids[i]-1]

            cv2.rectangle(rgb_img, (x1, y1), (x2, y2), color, lw)
            cv2.putText(rgb_img, self.is_class_names[is_obj_ids[i]] + str(score[i].item())[:4], \
                (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, lw)


        rgb_img = np.uint8(cv2.addWeighted(rgb_img, 0.6, mask_all, 0.4, 0.5))
        self.vis_is_pub.publish(self.bridge.cv2_to_imgmsg(rgb_img))


    def reproject_pe(self, publisher, all_pose_estimates, is_obj_ids, boxes, scores, rgb):

        os.environ["CUDA_VISIBLE_DEVICES"] = self.params["pe_gpu_id"]

        self.renderer = meshrenderer_phong.Renderer(self.ply_paths, 
            samples=1, 
            vertex_tmp_store_folder=get_dataset_path(self.workspace_path),
            vertex_scale=float(1000)) # float(1) for some models
        pe_obj_ids = [self.pe_class_names.index(self.is_class_names[label]) for label in is_obj_ids]
        bgr, depth, _, masks = self.renderer.render_many(obj_ids = pe_obj_ids,
                W = self.params["width"],
                H = self.params["height"],
                K = self.intrinsic_matrix, 
                Rs = [pose_est[:3,:3] for pose_est in all_pose_estimates],
                ts = [pose_est[:3, 3] for pose_est in all_pose_estimates],
                near = 10,
                far = 10000,
                random_light=False,
                phong={'ambient':0.4,'diffuse':0.8, 'specular':0.3},
                return_masks=True
                )
        self.renderer.close()

        bgr = cv2.resize(bgr, (self.params["width"], self.params["height"]))
        g_y = np.zeros_like(bgr)
        g_y[:,:,0] = bgr[:,:,0]    
        im_bg = cv2.bitwise_and(rgb, rgb, mask=(g_y[:,:,0]==0).astype(np.uint8))                 
        image_show = cv2.addWeighted(im_bg, 1, g_y, 1, 0)
        for label, box, score in zip(is_obj_ids, boxes, scores):
            box = box.astype(np.int32)
            x1, y1, x2, y2 = box
            x = int(x1)
            y = int(y1)
            h = int(y2 - y1)
            w = int(x2 - x1)
            if self.use_sameWH_crop[label]:
                w_offset = int(np.maximum(h, w) * self.pad_factors[label])
                h_offset = int(np.maximum(h, w) * self.pad_factors[label])
            else:
                w_offset = int(w * self.pad_factors[label])
                h_offset = int(h * self.pad_factors[label])
            left = np.maximum(x+w//2-w_offset//2, 0)
            right = x+w//2+w_offset/2
            top = np.maximum(y+h//2-h_offset//2, 0)
            bottom = y+h//2+h_offset//2
            cv2.putText(image_show, self.is_class_names[label], (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)            
            cv2.rectangle(image_show, (left, top), (right, bottom), (0, 255, 0), 3)
        imgmsg = self.bridge.cv2_to_imgmsg(image_show, "bgr8")
        publisher.publish(imgmsg)
        return pe_obj_ids, masks, depth


    def get_vsd_error(self, publisher, is_masks, is_obj_ids, cam_depth, pe_masks, pe_obj_ids, pe_depth):
        print(" vsd\t| score\t|")
        print("----------------")
        vsd_errors = []
        cost_map = np.zeros([self.params["height"], self.params["width"]])
        for i, is_obj_id in enumerate(is_obj_ids):
            for j, pe_obj_id in enumerate(pe_obj_ids):
                if pe_obj_id == self.pe_class_names.index(self.is_class_names[is_obj_id]):
                    is_mask = is_masks[i]
                    is_mask[is_mask > 0.5] = 1
                    is_mask[is_mask < 0.5] = 0
                    pe_mask = pe_masks[j]
                    # from https://github.com/thodan/sixd_toolkit
                    # Convert depth images to distance images
                    dist_cam = misc.depth_im_to_dist_im(cam_depth, self.intrinsic_matrix) * 1000
                    dist_est = misc.depth_im_to_dist_im(pe_depth, self.intrinsic_matrix)
                    # Visibility mask of the model in the ground truth pose
                    visb_cam = visibility.estimate_visib_mask_gt(dist_cam, dist_cam, self.params["delta"])
                    # Visibility mask of the model in the estimated pose
                    visib_est = visibility.estimate_visib_mask_est(dist_cam, dist_est, visb_cam, self.params["delta"])
                    # Intersection and union of the visibility masks
                    visib_inter = np.logical_and(is_mask, pe_mask)
                    visib_union = np.logical_or(is_mask, pe_mask)
                    # Pixel-wise matching cost
                    costs = np.abs(dist_cam[visib_inter] - dist_est[visib_inter]) 
                    if self.params["cost_type"] == 'step':
                        costs = costs >= self.params["tau"]
                    elif self.params["cost_type"] == 'tlinear': # Truncated linear
                        costs *= (1.0 / self.params["tau"])
                        costs[costs > 1.0] = 1.0
                    else:
                        print('Error: Unknown pixel matching cost.')
                        exit(-1)
                    costs_vis = np.ones(dist_cam.shape)
                    costs_vis[visib_inter] = costs
                    cost_map += costs_vis

                    # Visible Surface Discrepancy
                    visib_union_count = visib_union.sum()
                    visib_comp_count = visib_union_count - visib_inter.sum()
                    if visib_union_count > 0:
                        e = (costs.sum() + visib_comp_count) / float(visib_union_count)
                    else:
                        e = 1.0
                    print("{}\t{}".format(self.is_class_names[is_obj_id].split('_')[-1], e))
                    vsd_errors.append(e)
        cost_map = cv2.normalize(cost_map, cost_map, 0, 255, cv2.NORM_MINMAX)
        cost_map = cv2.applyColorMap(np.uint8(cost_map), cv2.COLORMAP_JET)
        publisher.publish(self.bridge.cv2_to_imgmsg(cost_map, "bgr8"))
        return vsd_errors

    def get_cou_error(self, publisher, is_masks, is_obj_ids, cam_depth, pe_masks, pe_obj_ids, pe_depth):
        print("\n cou\t| score\t|")
        print("----------------")
        cou_errors = []
        cost_map = np.zeros([self.params["height"], self.params["width"]])
        for i, is_obj_id in enumerate(is_obj_ids):
            for j, pe_obj_id in enumerate(pe_obj_ids):
                if pe_obj_id == self.pe_class_names.index(self.is_class_names[is_obj_id]):
                    is_mask = is_masks[i]
                    is_mask[is_mask > 0.5] = 1
                    is_mask[is_mask < 0.5] = 0
                    pe_mask = pe_masks[j]
                    # from https://github.com/thodan/sixd_toolkit    
                    inter = np.logical_and(is_mask, pe_mask)
                    union = np.logical_or(is_mask, pe_mask)
                    union_count = float(union.sum())
                    if union_count > 0:
                        e = 1.0 - inter.sum() / union_count
                    else:
                        e = 1.0
                    print("{}\t{}".format(self.is_class_names[is_obj_id].split('_')[-1], e))
                    cou_errors.append(e)
                    cost_map = np.logical_or(inter, cost_map)
        cost_map = np.uint8(cost_map) * 255
        cost_map = cv2.applyColorMap(np.uint8(cost_map), cv2.COLORMAP_JET)
        publisher.publish(self.bridge.cv2_to_imgmsg(cost_map, "rgb8"))
        return cou_errors


    def publish_markers(self, publisher, detections_array, color):
        # Delete all existing markers
        markers = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        markers.markers.append(marker)
        publisher.publish(markers)

        # Object markers
        markers = MarkerArray()
        for i, det in enumerate(detections_array.detections):
            name = self.ply_paths[det.results[0].id].split('/')[-1][5:-4]
            # cube marker
            marker = Marker()
            marker.header = detections_array.header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 0.5
            marker.ns = "bboxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.scale = det.bbox.size
            markers.markers.append(marker)

            # text marker
            marker = Marker()
            marker.header = detections_array.header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 1.0
            marker.id = i
            marker.ns = "texts"
            marker.type = Marker.TEXT_VIEW_FACING
            marker.scale.z = 0.07
            marker.text = '{} ({:.2f})'.format(name, det.results[0].score)
            markers.markers.append(marker)

            # mesh marker
            marker = Marker()
            marker.header = detections_array.header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 0.9
            marker.ns = "meshes"
            marker.id = i
            marker.type = Marker.MESH_RESOURCE
            marker.scale.x = 1
            marker.scale.y = 1
            marker.scale.z = 1
            marker.mesh_resource = "file://" + self.ply_paths[det.results[0].id]
            marker.mesh_use_embedded_materials = True
            markers.markers.append(marker)
        publisher.publish(markers)
    
    
if __name__ == '__main__':

    pose_estimator = PoseEstimator()
    rospy.spin()




