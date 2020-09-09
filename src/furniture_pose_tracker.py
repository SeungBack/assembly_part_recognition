#!/usr/bin/env python

import json
import rospy
import cv2, cv_bridge
import numpy as np
import PIL
import message_filters
import yaml
import sys

from std_msgs.msg import String, Header
from sensor_msgs.msg import RegionOfInterest, Image, CameraInfo, PointCloud2
from assembly_part_recognition.msg import InstanceSegmentation2D
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped

import cv2
import argparse
import numpy as np
import os
import configparser

import open3d as o3d # import both o3d and pcl occurs segmentation faults
import cam_helper
import copy
import time
import tf.transformations as tf_trans

################
import scipy as sp
import numpy.matlib as npm
from scipy.spatial.transform import Rotation as R

import torch




focal_lenghts = [2782.7666015625, 2782.5361328125]
principal_point = [957.6043701171875, 590.627583007812]
###############

class PoseEstimator:

    def __init__(self):

        # initalize node
        rospy.init_node('furniture_pose_estimator')
        rospy.loginfo("Starting furniture_pose_estimator.py")

        self.params = rospy.get_param("furniture_pose_estimator")
        self.class_names = ["background", "side", "longshort", "middle", "bottom"]
        self.classidx2color = [[13, 128, 255], [255, 12, 12], [217, 12, 232], [232, 222, 12]]
        self.color_dict = [(255,255,0), (0,0,255), (255,0,0), (255,255,0)] * 10
        self.roi = self.params["roi"]

        ########################
        sys.path.append("/home/demo/Workspace/PoseRBPF")
        from pose_rbpf import particle_filter



        #######################

        self.bridge = cv_bridge.CvBridge()

        # subscribers
        self.initialize_is_model()
        self.initialize_pose_est_model()
        rgb_sub = message_filters.Subscriber(self.params["rgb"], Image, buff_size=1536*2048)
        depth_sub = message_filters.Subscriber(self.params["depth"], Image, buff_size=1536*2048)
        point_sub = message_filters.Subscriber(self.params["point"], PointCloud2, buff_size=1536*2048*3)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, point_sub], queue_size=1, slop=1)
        self.ts.registerCallback(self.inference)

        # publishers
        self.pose_pubs = [] # "side_right", "long_short", "middle", "bottom"
        self.idx2color = [[13, 128, 255], [255, 12, 12], [217, 12, 232], [232, 222, 12]]
        self.dims = []
        self.cloud_GTs = []
        for ply_model in self.ply_model_paths:
            cloud = o3d.io.read_point_cloud(ply_model)
            self.dims.append(cloud.get_max_bound())
            self.cloud_GTs.append(cloud)
            model_name = ply_model.split('/')[-1][5:-4]
            self.pose_pubs.append(rospy.Publisher(
                '/assembly/furniture/pose_{}'.format(model_name), PoseStamped, queue_size=1))
        self.is_pub = rospy.Publisher('/assembly/furniture/is_results', InstanceSegmentation2D, queue_size=1)
        self.icp_pub = rospy.Publisher('/assembly/furniture/icp_pose', Detection3DArray, queue_size=1)
        self.markers_pub = rospy.Publisher('/assembly/furniture/markers', MarkerArray, queue_size=1)

        ################################




        self.tracking_pub = rospy.Publisher('/assembly/furniture/track_pose', Detection3DArray, queue_size=1)
        self.t_markers_pub = rospy.Publisher('/assembly/furniture/tracking_markers', MarkerArray, queue_size=1)
        ################################
        if self.params["debug"]:
            self.is_vis_pub = rospy.Publisher('/assembly/furniture/is_vis_results', Image, queue_size=1)
            self.pose_vis_pub = rospy.Publisher('/assembly/furniture/pose_vis_results', Image, queue_size=1)
        
    def initialize_is_model(self):
        import torch
        import torchvision.transforms as transforms
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

    def initialize_pose_est_model(self):
        import tensorflow as tf
        # augmented autoencoder and renderer      
        rospy.loginfo("Loading AAE model")
        sys.path.append(self.params["augauto_module_path"])
        sys.path.append(self.params["augauto_module_path"] + "/auto_pose/test")
        from aae_maskrcnn_pose_estimator import AePoseEstimator
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params["gpu_id"]
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.workspace_path = os.environ.get('AE_WORKSPACE_PATH')
        if self.workspace_path == None:
            print 'Please define a workspace path:\n'
            print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
            exit(-1)

        test_configpath = os.path.join(self.workspace_path, 'cfg_eval', self.params["test_config"])
        test_args = configparser.ConfigParser()
        test_args.read(test_configpath)
        self.ae_pose_est = AePoseEstimator(test_configpath)
        self.ply_model_paths = [str(train_args.get('Paths','MODEL_PATH')) for train_args in self.ae_pose_est.all_train_args]
        

    def inference(self, rgb, depth, cloud):
        inference_start_time = time.time()
        # get rgb, depth, point cloud
        img_header = rgb.header
        rgb_original = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb_resized = cv2.resize(rgb_original.copy(), (self.params["width"], self.params["height"]))

        depth_original = self.bridge.imgmsg_to_cv2(depth, desired_encoding='32FC1')
        depth_resized = cv2.resize(depth_original.copy(), dsize=(self.params["width"], self.params["height"]), interpolation=cv2.INTER_AREA)
        if "zivid" in self.params["depth"]:
            depth_resized = np.where(np.isnan(depth_resized), 0, depth_resized)  
            npcloud_cam = cam_helper.convertZividCloudFromRosToNumpy(cloud)
        else:
            npcloud_cam = cam_helper.convertAzureCloudFromRosToNumpy(cloud)
        rgb = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb = PIL.Image.fromarray(np.uint8(rgb), mode="RGB")
        rgb_img = rgb.resize((self.params["width"], self.params["height"]), PIL.Image.BICUBIC)
        rgb = self.rgb_transform(rgb_img).unsqueeze(0)

        get_input_data = time.time()
        print("get input data : ", get_input_data - inference_start_time)

        # 2D instance segmentation
        rospy.loginfo_once("Segmenting furniture part area")
        pred_results = self.model(rgb.to(self.device))[0]
        pred_masks = pred_results["masks"].cpu().detach().numpy()
        pred_boxes = pred_results['boxes'].cpu().detach().numpy()
        pred_labels = pred_results['labels'].cpu().detach().numpy()
        pred_scores = pred_results['scores'].cpu().detach().numpy()

        get_pred_result = time.time()
        print("get pred result from maskrcnn : ", get_pred_result - get_input_data)

        
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
        infer_result_to_message = time.time()
        print("inference result -> ros message (Detection2D) : ", infer_result_to_message - get_pred_result)

        self.is_pub.publish(is_msg)
        if self.params["debug"]:
            vis_results = self.visualize_instance_segmentation_results(rgb_img, pred_masks, pred_boxes, pred_labels, pred_scores, thresh=self.params["is_thresh"])
            self.is_vis_pub.publish(self.bridge.cv2_to_imgmsg(vis_results, "bgr8"))

        # gather all detected crops
        boxes, scores, labels = [], [], []
        ###############
        rgb_crops = []
        ##############
        boxes, scores, labels, rgb_crops, is_masks = self.gather_is_results(rgb_resized, depth_resized, boxes, scores, labels, rgb_crops, is_msg)
        # 6D pose estimation for all detected crops using AAE
        rospy.loginfo_once("Estimating 6D object pose")
        all_pose_estimates, all_class_idcs = self.ae_pose_est.process_pose(boxes, labels, rgb_resized, depth_resized)
        # visualize pose estimation results using only AAE
        if self.params["debug"]:
            self.visualize_pose_estimation_results(all_pose_estimates, all_class_idcs, labels, boxes, scores, rgb_resized)

        detection_array = Detection3DArray()
        detection_array.header = img_header
        # 6D object pose refinement using ICP on pointcloud
        
        print("get 6D object pose", time.time() - infer_result_to_message)
        #####################
        refined_pose_array = [] # For pose tracker function
        #####################

        for i, (pose_estimate, class_id) in enumerate(zip(all_pose_estimates, all_class_idcs)):
            icp_inner_start = time.time()

            # crop zivid cloud with instance mask        
            is_mask_original = cv2.resize(is_masks[i], (rgb_original.shape[1], rgb_original.shape[0]), interpolation=cv2.INTER_AREA)
            is_mask_original = is_mask_original[np.isfinite(is_mask_original)]

            # remove outliers and scale it from m to mm
            cloud_cam = cam_helper.convertCloudFromNumpyToOpen3d(npcloud_cam.copy(), mask=is_mask_original)
            cloud_cam, _ = cloud_cam.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            cloud_cam.scale(1000)

            cloud_GT = copy.deepcopy(self.cloud_GTs[class_id])
            # rotate cloud_GT according to the estimated rotation then translate cloud_GT from origin to the center of cloud_cam
            T_rot = np.eye(4)
            T_rot[:3, :3] = pose_estimate[:3, :3] # test needed: trans from 2D bbox or center
            T_trans = np.eye(4)
            T_trans[:3, 3] = cloud_cam.get_center()
            H_init_cam2obj = np.matmul(T_trans, T_rot)
            cloud_GT.transform(H_init_cam2obj)
            icp_result = self.icp_refinement(source_cloud=copy.deepcopy(cloud_GT), target_cloud=copy.deepcopy(cloud_cam), N=self.params["icp_iter"])
            rospy.loginfo_once("icp result- fitness: {}, RMSE: {}, T: {}".format(
                icp_result.fitness, icp_result.inlier_rmse, icp_result.transformation))
            cloud_GT.transform(icp_result.transformation)
            # o3d.visualization.draw_geometries([cloud_cam, cloud_GT])
            get_icp_result = time.time()
            print("get icp result : ", get_icp_result - icp_inner_start)
            # publish the estimated pose and cube 
            H_refined_cam2obj = np.eye(4)
            H_refined_cam2obj[:3, :3] = np.matmul(icp_result.transformation[:3, :3], H_init_cam2obj[:3, :3])
            H_refined_cam2obj[:3, 3] = icp_result.transformation[:3, 3]/1000 + H_init_cam2obj[:3, 3]
            trans = H_refined_cam2obj[:3, 3]
            rot = tf_trans.quaternion_from_matrix(H_refined_cam2obj)

            #####################
            refined_pose_array.append([class_id ,trans, rot]) # For pose tracker function
            #####################

            pose_msg = PoseStamped()
            pose_msg.header = img_header
            pose_msg.pose.position.x = trans[0] 
            pose_msg.pose.position.y = trans[1] 
            pose_msg.pose.position.z = trans[2] 
            pose_msg.pose.orientation.x = rot[0]
            pose_msg.pose.orientation.y = rot[1]
            pose_msg.pose.orientation.z = rot[2]
            pose_msg.pose.orientation.w = rot[3]
            self.pose_pubs[class_id].publish(pose_msg)

            # Add to Detection3DArray
            detection = Detection3D()
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = class_id
            hypothesis.score = icp_result.fitness
            hypothesis.pose.pose = pose_msg.pose
            detection.results.append(hypothesis)
            detection.bbox.center = pose_msg.pose
            detection.bbox.size.x = self.dims[class_id][0] / 1000 * 2
            detection.bbox.size.y = self.dims[class_id][1] / 1000 * 2
            detection.bbox.size.z = self.dims[class_id][2] / 1000 * 2
            detection_array.detections.append(detection)
            make_pose_msg = time.time()
            print("make_pose_msg : ", make_pose_msg - get_icp_result)

        self.icp_pub.publish(detection_array)
        self.publish_markers(detection_array)
        '''
        st_tracking = time.time()

        all_tracked_pose = self.pose_tracker(rgb_resized, refined_pose_array, boxes, labels)

        tracking_array = Detection3DArray()
        tracking_array.header = img_header
        print("############### get tracking result : ", time.time() - st_tracking)
        for i, class_id in enumerate(all_class_idcs):
            make_tracking_msg_start = time.time()
            trans = self.queue_dict[self.class_names[class_id]][0].items[-1]
            rot = self.queue_dict[self.class_names[class_id]][1].items[-1]

            pose_msg = PoseStamped()
            pose_msg.header = img_header
            pose_msg.pose.position.x = trans[0] 
            pose_msg.pose.position.y = trans[1] 
            pose_msg.pose.position.z = trans[2] 
            pose_msg.pose.orientation.x = rot[0]
            pose_msg.pose.orientation.y = rot[1]
            pose_msg.pose.orientation.z = rot[2]
            pose_msg.pose.orientation.w = rot[3]
            self.pose_pubs[class_id].publish(pose_msg)

            # Add to Detection3DArray
            detection = Detection3D()
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = class_id
            hypothesis.score = icp_result.fitness
            hypothesis.pose.pose = pose_msg.pose
            detection.results.append(hypothesis)
            detection.bbox.center = pose_msg.pose
            detection.bbox.size.x = self.dims[class_id][0] / 1000 * 2
            detection.bbox.size.y = self.dims[class_id][1] / 1000 * 2
            detection.bbox.size.z = self.dims[class_id][2] / 1000 * 2
            tracking_array.detections.append(detection)

            make_tracking_msg = time.time()
            print("make tracking_msg", make_tracking_msg - make_tracking_msg_start)

        self.tracking_pub.publish(tracking_array)
        self.publish_t_markers(tracking_array)
        '''


    def gather_is_results(self, rgb, depth, boxes, scores, labels, rgb_crops, is_results):
        
        is_masks = []
        for i, class_id in enumerate(is_results.class_ids):
            if is_results.scores[i] >= self.params["is_thresh"]:
                x = is_results.boxes[i].x_offset
                y = is_results.boxes[i].y_offset
                h = is_results.boxes[i].height
                w = is_results.boxes[i].width
                h_offset = int(h*self.params["crop_offset"])
                w_offset = int(w*self.params["crop_offset"])
                x1 = int(max(0, x-w_offset))
                y1 = int(max(0, y-h_offset))
                x2 = int(min(self.params["width"]-1, x+w+w_offset))
                y2 = int(min(self.params["height"]-1, y+h+h_offset))
                rgb_crop = rgb[y1:y2, x1:x2].copy()
                if self.params["padding"]:
                    pad = abs(((y2-y1)-(x2-x1))//2)
                    if h >= w:
                        rgb_crop = cv2.copyMakeBorder(rgb_crop, 0, 0, pad, pad, borderType=cv2.BORDER_CONSTANT, value=0)
                    else:
                        rgb_crop = cv2.copyMakeBorder(rgb_crop, pad, pad, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)
                rgb_crop = cv2.resize(rgb_crop, (128, 128))
                depth_crop = depth[y1:y2, x1:x2].copy()
                depth_crop = cv2.resize(depth_crop, (128, 128))
                is_masks.append(self.bridge.imgmsg_to_cv2(is_results.masks[i]))
                boxes.append(np.array([x, y, w, h]))
                scores.append(is_results.scores[i])
                labels.append(class_id)
                rgb_crops.append(rgb_crop)
        return boxes, scores, labels, rgb_crops, is_masks


    def visualize_instance_segmentation_results(self, rgb_img, masks, boxes, labels, score, thresh=0.5):
        
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
            cv2.putText(image_show, '%s : %1.3f' % (label,score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, self.color_dict[int(label)], 2)
            cv2.rectangle(image_show, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        self.pose_vis_pub.publish(self.bridge.cv2_to_imgmsg(image_show, "bgr8"))    

    def icp_refinement(self, source_cloud, target_cloud, N=1000):

        n_source_points = len(source_cloud.points)
        n_target_points = len(target_cloud.points)
        n_sample = np.min([n_source_points, n_target_points, N])
        source_idxes = np.random.choice(n_source_points, n_sample)
        target_idxes = np.random.choice(n_target_points, n_sample)

        source_cloud = source_cloud.select_down_sample(source_idxes)
        target_cloud = target_cloud.select_down_sample(target_idxes)

        threshold = 0.02
        trans_init = np.eye(4)
        evaluation = o3d.registration.evaluate_registration(source_cloud, target_cloud, threshold, trans_init)
        icp_result = o3d.registration.registration_icp(
            source = source_cloud, target = target_cloud, max_correspondence_distance=500, # unit in millimeter
            init = np.eye(4),  
            estimation_method = o3d.registration.TransformationEstimationPointToPoint(), 
            criteria = o3d.registration.ICPConvergenceCriteria(
                                                relative_fitness=1e-10,
                                                relative_rmse=1e-8,
                                                max_iteration=500))                                               
        return icp_result


    def pose_tracker(self, rgb, refined_pose_array, boxes, labels):
        all_pose_estimates = []
        tracking_start_time = time.time()

        session = self.ae_pose_est.sess
        codebooks = self.ae_pose_est.all_codebooks
        
        print("tracking : start")
        for i, (class_id ,trans, rot) in enumerate(refined_pose_array):
            loop_start_time = time.time()
            self.queue_dict[self.class_names[class_id]][0].enqueue(trans)
            self.queue_dict[self.class_names[class_id]][1].enqueue(rot)

            codebook = codebooks[class_id]  # rotation particle
            encoder = codebook._encoder  # 2d image encoder

            # init weighted sample
            samples = []

            # T and P(R) { 1 to N particle } load from last time k -1
            translation_arr = copy.deepcopy(self.queue_dict[self.class_names[class_id]][0].items)
            translation_arr = torch.tensor(translation_arr).cuda()

            rotation_arr = copy.deepcopy(self.queue_dict[self.class_names[class_id]][1].items)
        

            if len(self.queue_dict[self.class_names[class_id]][0].items) >= 2:
                # motion prior
                alpha = 0.5 # constant parameter
                alpha = torch.tensor(alpha).cuda()

                #translation_prior_mean = translation_arr[-1] + np.multiply(alpha, (translation_arr[-1] - translation_arr[-2]))
                #translation_prior_mean = torch.add(translation_arr[-1], alpha*(translation_arr[-1] - translation_arr[-2]))

                #translation_est = np.random.multivariate_normal(mean=translation_prior_mean, cov=np.cov(np.array(translation_arr).T))
                #translation_arr.append(translation_est)

                euler_rotation_arr = R.from_quat(rotation_arr).as_euler('xyz', degrees=False)

                roation_prior = np.random.multivariate_normal(mean=euler_rotation_arr[-1], cov=np.cov(np.array(euler_rotation_arr).T))
                roation_prior = torch.from_numpy(np.flip(roation_prior,axis=0).copy()).cuda()

            else:
                roation_prior = np.squeeze(R.from_quat(rotation_arr).as_euler('xyz', degrees=False))
                roation_prior = torch.from_numpy(np.flip(roation_prior,axis=0).copy()).cuda()
                


            time1 = time.time()
            print("inner for loop start : ", time1 - loop_start_time) 

            cosine_similarity = codebook.cos_similarity

            

            for pose_3d in translation_arr:

                inner_loop_time = time.time()
               
                #x = rgb_crops[i]
                x = self.roi_img(pose_3d, rgb)
                crop_time = time.time()
                print("crop image : ", crop_time - inner_loop_time)
                if x.dtype == 'uint8':
                    x = x/255.
     
                x = np.expand_dims(x, 0)
                
                cosine_distance = session.run(cosine_similarity, {encoder.x: x})
              
                time1 = time.time()
                print("get cos distance : ", time1 - crop_time)
                
                rotation_distribution = np.squeeze(sp.stats.norm.pdf(cosine_distance, loc=np.max(cosine_distance)))
                rotation_distribution = torch.tensor(rotation_distribution).cuda()
                rotation_distribution = rotation_distribution / torch.sum(rotation_distribution)
                #rotation_distribution = np.divide(rotation_distribution, np.sum(rotation_distribution))
                time2 = time.time()
                print("get rot dist : ", time2 - time1)

                #translation_distirbution = np.sum(rotation_distribution)
                translation_distirbution = torch.sum(rotation_distribution)

                #observation_likelihood = np.multiply(rotation_distribution, translation_distirbution)
                observation_likelihood = rotation_distribution * translation_distirbution

                #observation_likelihood = np.divide(np.squeeze(observation_likelihood), np.sum(observation_likelihood))
                observation_likelihood = torch.squeeze(observation_likelihood) / torch.sum(observation_likelihood)

                time3 = time.time()
                print("observation likelihood : ", time3 - time2)


                # rotation distribution update by rotation prior

                #rotation_distribution = np.expand_dims(rotation_distribution, axis=1)
                rotation_distribution = rotation_distribution.unsqueeze(1)
                
                #updated_roation_dist = np.sum(np.dot(rotation_distribution, np.expand_dims(roation_prior, axis=0)), axis=1)
                updated_roation_dist = torch.sum(torch.matmul(rotation_distribution, roation_prior.unsqueeze(0)), axis=1)
                
                #updated_roation_dist = np.divide(updated_roation_dist, np.sum(updated_roation_dist))
                updated_roation_dist = updated_roation_dist / torch.sum(updated_roation_dist)
                
                #posterior_translation = np.sum(np.dot(observation_likelihood, updated_roation_dist))
                posterior_translation = torch.sum(torch.dot(observation_likelihood, updated_roation_dist))

                # weight of particle
                samples.append([pose_3d.cpu().detach().numpy(), updated_roation_dist.cpu().detach().numpy(), posterior_translation.cpu().detach().numpy()])
                
                print("inner for loop : ", time.time() - inner_loop_time)
                
            

            samples = np.array(samples)
            #weight_arr = np.array([sample[-1] for sample in samples])
            
            weight_arr = samples[:,2]
            weight_arr = np.divide(weight_arr, np.sum(weight_arr))
            
            # resampling step
            time2 = time.time()
            print("resampling start : ", time2-time1)
            
            if len(samples) < 2:
                resampled_sample = samples
                T_est = np.array(trans)
                R_est = np.array(rot)
                R_est = R_est[...,[3,0,1,2]]
            
            else:
                indexes = self.systematic_resample(weight_arr)
                resampled_sample = samples[indexes]
   
                T_est = np.mean(resampled_sample[:, 0])

                idcs = np.array([np.argmax(sample[1]) for sample in resampled_sample])
                max_rotation_dist = codebook._dataset.viewsphere_for_embedding[idcs]
             
                rot_matrix = []
                for i, mat in enumerate(max_rotation_dist):
                    Rs_est, ts_est = self.convert_rotate_mat(mat, T_est, class_id, codebook, idcs[i])
                    T_rot = np.eye(4)
                    T_rot[:3, :3] = Rs_est

                    rot_matrix.append(tf_trans.quaternion_from_matrix(T_rot))

                rot_matrix = np.array(rot_matrix)

                R_est = weightedAverageQuaternions(Q=rot_matrix[...,[3,0,1,2]].copy(), w=weight_arr)
                R_est = R_est[...,[3,0,1,2]]

            time3 = time.time()
            # send pose info
            all_pose_estimates.append([T_est.copy(), R_est.copy()])
            print("outter for loop : ", time3- time2)

        print("finish tracking : ", time.time() - tracking_start_time)
        return all_pose_estimates

    def filter(self, pose_3d, rgb, session, cosine_similarity, encoder, samples):
        inner_loop_time = time.time()
        #x = rgb_crops[i]
        x = self.roi_img(pose_3d, rgb)
        crop_time = time.time()
        print("crop image : ", crop_time - inner_loop_time)
        if x.dtype == 'uint8':
            x = x/255.

        x = np.expand_dims(x, 0)
        
        cosine_distance = session.run(cosine_similarity, {encoder.x: x})
        
        time1 = time.time()
        print("get cos distance : ", time1 - crop_time)
        
        rotation_distribution = np.squeeze(sp.stats.norm.pdf(cosine_distance, loc=np.max(cosine_distance)))
        rotation_distribution = torch.tensor(rotation_distribution).cuda()
        rotation_distribution = rotation_distribution / torch.sum(rotation_distribution)
        #rotation_distribution = np.divide(rotation_distribution, np.sum(rotation_distribution))
        time2 = time.time()
        print("get rot dist : ", time2 - time1)

        #translation_distirbution = np.sum(rotation_distribution)
        translation_distirbution = torch.sum(rotation_distribution)

        #observation_likelihood = np.multiply(rotation_distribution, translation_distirbution)
        observation_likelihood = rotation_distribution * translation_distirbution

        #observation_likelihood = np.divide(np.squeeze(observation_likelihood), np.sum(observation_likelihood))
        observation_likelihood = torch.squeeze(observation_likelihood) / torch.sum(observation_likelihood)

        time3 = time.time()
        print("observation likelihood : ", time3 - time2)


        # rotation distribution update by rotation prior

        #rotation_distribution = np.expand_dims(rotation_distribution, axis=1)
        rotation_distribution = rotation_distribution.unsqueeze(1)
        
        #updated_roation_dist = np.sum(np.dot(rotation_distribution, np.expand_dims(roation_prior, axis=0)), axis=1)
        updated_roation_dist = torch.sum(torch.matmul(rotation_distribution, roation_prior.unsqueeze(0)), axis=1)
        
        #updated_roation_dist = np.divide(updated_roation_dist, np.sum(updated_roation_dist))
        updated_roation_dist = updated_roation_dist / torch.sum(updated_roation_dist)
        
        #posterior_translation = np.sum(np.dot(observation_likelihood, updated_roation_dist))
        posterior_translation = torch.sum(torch.dot(observation_likelihood, updated_roation_dist))

        # weight of particle
        return pose_3d.cpu().detach().numpy(), updated_roation_dist.cpu().detach().numpy(), posterior_translation.cpu().detach().numpy()


    def roi_img(self, pose_3d, rgb):
        canonical_z = 0.01
        x, y, z = pose_3d
        u = np.multiply(focal_lenghts[0], np.divide(x,z)) + principal_point[0]
        v = np.multiply(focal_lenghts[1], np.divide(y,z)) + principal_point[1]

        roi_size = np.multiply(np.divide(z,canonical_z), 128)
        x1 = int(u - (roi_size/2))
        y1 = int(v - (roi_size/2))
        x2 = int(u + (roi_size/2))
        y2 = int(v + (roi_size/2))

        rgb_crop = rgb[y1:y2, x1:x2].copy()
 
        rgb_crop = cv2.resize(rgb_crop, (128, 128))
    
        return rgb_crop


    def convert_rotate_mat(self, Rs_est, pose_3d, clas_idx, codebook, idx, depth_pred=None):

        canonical_z = 0.1
        x, y, z = pose_3d
        u = np.multiply(focal_lenghts[0], (x/z)) + principal_point[0]
        v = np.multiply(focal_lenghts[1], (y/z)) + principal_point[1]
        roi_size = (z/canonical_z) * 128
        x1 = int(u - (roi_size/2))
        y1 = int(v - (roi_size/2))
        x2 = int(u + (roi_size/2))
        y2 = int(v + (roi_size/2))

        predicted_bb = [x1, y1, x2-x1, y2-y1]

        K_test = self.ae_pose_est._camK
        top_n = 1
        train_args = self.ae_pose_est.all_train_args[clas_idx]

        # test_depth = f_test / f_train * render_radius * diag_bb_ratio
        K_train = np.array(eval(train_args.get('Dataset','K'))).reshape(3,3)
        render_radius = train_args.getfloat('Dataset','RADIUS')

        K00_ratio = K_test[0,0] / K_train[0,0]  
        K11_ratio = K_test[1,1] / K_train[1,1]  
        
        mean_K_ratio = np.mean([K00_ratio,K11_ratio])


        embed_obj_bbs_values = codebook.embed_obj_bbs_values

        ts_est = np.empty((top_n,3))
      
        rendered_bb = embed_obj_bbs_values[idx].squeeze()
        if depth_pred is None:
            diag_bb_ratio = np.linalg.norm(np.float32(rendered_bb[2:])) / np.linalg.norm(np.float32(predicted_bb[2:]))
            z = diag_bb_ratio * mean_K_ratio * render_radius
        else:
            z = depth_pred


        # object center in image plane (bb center =/= object center)
        center_obj_x_train = rendered_bb[0] + rendered_bb[2]/2. - K_train[0,2]
        center_obj_y_train = rendered_bb[1] + rendered_bb[3]/2. - K_train[1,2]

        center_obj_x_test = predicted_bb[0] + predicted_bb[2]/2 - K_test[0,2]
        center_obj_y_test = predicted_bb[1] + predicted_bb[3]/2 - K_test[1,2]
        
        center_obj_mm_x = center_obj_x_test * z / K_test[0,0] - center_obj_x_train * render_radius / K_train[0,0]  
        center_obj_mm_y = center_obj_y_test * z / K_test[1,1] - center_obj_y_train * render_radius / K_train[1,1]  


        t_est = np.array([center_obj_mm_x, center_obj_mm_y, z])
        ts_est = t_est

        # correcting the rotation matrix 
        # the codebook consists of centered object views, but the test image crop is not centered
        # we determine the rotation that preserves appearance when translating the object
        d_alpha_x = - np.arctan(t_est[0]/t_est[2])
        d_alpha_y = - np.arctan(t_est[1]/t_est[2])
        R_corr_x = np.array([[1,0,0],
                            [0,np.cos(d_alpha_y),-np.sin(d_alpha_y)],
                            [0,np.sin(d_alpha_y),np.cos(d_alpha_y)]]) 
        R_corr_y = np.array([[np.cos(d_alpha_x),0,-np.sin(d_alpha_x)],
                            [0,1,0],
                            [np.sin(d_alpha_x),0,np.cos(d_alpha_x)]]) 
        R_corrected = np.dot(R_corr_y,np.dot(R_corr_x,Rs_est))
        Rs_est = R_corrected

        return (Rs_est, ts_est)

    
    def systematic_resample(self, weights):
        N = len(weights)

        # make N subdivisions, and choose positions with a consistent random offset
        positions = (np.random.random() + np.arange(N)) / N
        indexes = np.zeros(N, 'i')

        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes


    def publish_markers(self, detection_array):
        # Delete all existing markers
        markers = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        markers.markers.append(marker)
        self.markers_pub.publish(markers)

        # Object markers
        markers = MarkerArray()
        for i, det in enumerate(detection_array.detections):
            name = self.ply_model_paths[det.results[0].id].split('/')[-1][5:-4]
            color = self.idx2color[det.results[0].id]

            # cube marker
            marker = Marker()
            marker.header = detection_array.header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 0.3
            marker.ns = "bboxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.scale = det.bbox.size
            markers.markers.append(marker)

            # text marker
            marker = Marker()
            marker.header = detection_array.header
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
            marker.header = detection_array.header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 0.9
            marker.ns = "meshes"
            marker.id = i
            marker.type = Marker.MESH_RESOURCE
            marker.scale.x = 0.001
            marker.scale.y = 0.001
            marker.scale.z = 0.001
            marker.mesh_resource = "file://" + self.ply_model_paths[det.results[0].id]
            marker.mesh_use_embedded_materials = True
            markers.markers.append(marker)


        self.markers_pub.publish(markers)
    

    def publish_t_markers(self, detection_array):
        # Delete all existing markers
        markers = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        markers.markers.append(marker)
        self.t_markers_pub.publish(markers)

        # "side_right", "long_short", "middle", "bottom"
        idx2color = [[24, 28, 128], [128, 128, 232], [129, 9, 100], [100, 222, 12]]
        # Object markers
        markers = MarkerArray()
        for i, det in enumerate(detection_array.detections):
            name = self.ply_model_paths[det.results[0].id].split('/')[-1][5:-4]
            color = idx2color[det.results[0].id]

            # cube marker
            marker = Marker()
            marker.header = detection_array.header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 0.3
            marker.ns = "bboxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.scale = det.bbox.size
            markers.markers.append(marker)

            # text marker
            marker = Marker()
            marker.header = detection_array.header
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
            marker.header = detection_array.header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 0.9
            marker.ns = "meshes"
            marker.id = i
            marker.type = Marker.MESH_RESOURCE
            marker.scale.x = 0.001
            marker.scale.y = 0.001
            marker.scale.z = 0.001
            marker.mesh_resource = "file://" + self.ply_model_paths[det.results[0].id]
            marker.mesh_use_embedded_materials = True
            markers.markers.append(marker)


        self.t_markers_pub.publish(markers)


class memory_queue:
    def __init__(self, max_size):
        self.items = list()
        self.max_size = max_size
    
    def enqueue(self, data):
        self.items.append(data)
        if len(self.items) > self.max_size:
            self.dequeue()
    
    def dequeue(self):
        data = self.items[0]
        del self.items[0]


def weightedAverageQuaternions(Q, w):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4,4))
    weightSum = 0
   
    for i in range(0,M):
      
        q = Q[i,:]
        A = w[i] * np.outer(q,q) + A
        weightSum += w[i]

    # scale
    A = (1.0/weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0].A1)

    
if __name__ == '__main__':

    pose_estimator = PoseEstimator()
    rospy.spin()




