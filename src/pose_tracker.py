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
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped

import cv2
import argparse
import numpy as np
import os
import configparser

import open3d as o3d # import both o3d and pcl occurs segmentation faults
import zivid_helper
import copy

from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped

import tensorflow as tf
import scipy as sp
import numpy.matlib as npm
from scipy.spatial.transform import Rotation as R


color_dict = [(255,255,0),(0,0,255),(255,0,0),(255,255,0)] * 10


class PoseTracker:

    def __init__(self):

        # initalize node
        rospy.init_node('6d_pose_tracker')
        rospy.loginfo("Starting 6d_pose_tracker.py")

        self.params = rospy.get_param("6d_pose_estimation")
        self.class_name = ["side_right", "long_short", "middle", "bottom"]

        # augmented autoencoder and renderer      
        sys.path.append(self.params["augauto_module_path"])
        sys.path.append(self.params["augauto_module_path"] + "/auto_pose/test")
        from ae.ae_factory import build_encoder

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
        self.codebooks = self.ae_pose_est.all_codebooks

        self.ply_model_paths = [str(train_args.get('Paths','MODEL_PATH')) for train_args in self.ae_pose_est.all_train_args]
        self.bridge = cv_bridge.CvBridge()

        self.translation_dict = {cls:[] for cls in self.class_name}
        self.rotation_dict = {cls:[] for cls in self.class_name}
        self.crop_img_dict = {cls:[] for cls in self.class_name}

        self.max_similarity_score = {cls:0 for cls in self.class_name}

        # subscribers
        rgb_sub = message_filters.Subscriber("/zivid_camera/color/image_color", Image)
        pose_sub = message_filters.Subscriber("/assembly/zivid/furniture_part/6d_pose_est_results", Detection3DArray)
        is_sub = message_filters.Subscriber("/assembly/zivid/furniture_part/is_results", InstanceSegmentation2D)

        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, pose_sub, is_sub], queue_size=1, slop=0.5)
        rospy.loginfo("Starting zivid rgb-d subscriber with time synchronizer")
        self.ts.registerCallback(self.estimate_6d_pose)
        

    def estimate_6d_pose(self, rgb, detection3darray, is_results):
        
        rgb_original = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        rgb_resized = cv2.resize(rgb_original.copy(), (self.params["width"], self.params["height"]))

        detection_array = Detection3DArray()
        detection_array.header = rgb.header

        boxes, scores, labels = [], [], []
        crop_imgs, boxes, scores, labels = self.gather_is_results(rgb_resized, boxes, scores, labels, is_results)

        poes_dict = self.pose_read(detection3darray)

        init_cls_list = [idx for idx in range(len(self.class_name)) if idx not in labels]
        
        for init_cls in init_cls_list:
            self.translation_dict[self.class_name[init_cls]] = []
            self.rotation_dict[self.class_name[init_cls]] = []
            self.crop_img_dict[self.class_name[init_cls]] = []

        for crop_img, label in zip(crop_imgs, labels):
            class_idx = label
            codebook = self.codebooks[class_idx]  # rotation particle
            session = self.ae_pose_est.sess  # tensorflow session
            encoder = codebook.encoder(is_training=False)  # 2d image encoder

            # init weighted sample
            samples = []

            # get init pose info from icp
            if len(self.rotation_dict[self.class_name[class_idx]]) == 0:
                self.translation_dict[self.class_name[class_idx]].append(poes_dict[class_idx]["pose"])
                self.rotation_dict[self.class_name[class_idx]].append(poes_dict[class_idx]["ori"])

            # T and P(R) { 1 to N particle } load from last time k -1
            translation_arr = copy.deepcopy(self.translation_dict[self.class_name[class_idx]])
            rotation_arr = copy.deepcopy(self.rotation_dict[self.class_name[class_idx]])

            # motion prior
            alpha = 0.5 # constant parameter
            translation_prior_mean = translation_arr[-1] + alpha * (translation_arr[-1] - translation_arr[-2])
            translation_est = np.random.multivariate_normal(mean=translation_prior_mean, cov=np.cov(translation_arr.T))

            euler_rotation_arr = R.from_quat(rotation_arr).as_euler('xyz', degrees=False)
            roation_prior = np.random.multivariate_normal(mean=euler_rotation_arr[-1], cov=np.cov(euler_rotation_arr.T))

            self.crop_img_dict[class_idx].append(crop_img)

            for crop_img in self.crop_img_dict[class_idx]:

                x = crop_img

                if x.dtype == 'uint8':
                    x = x/255.
     
                x = np.expand_dims(x, 0)

                cosine_distance = session(codebook.cos_similarity, {encoder.x: x})
                roation_distribution = sp.stats.norm.pdf(cosine_distance, loc=np.max(cosine_distance))
                translation_distirbution = np.sum(roation_distribution)
                observation_likelihood = roation_distribution * translation_distirbution

                # rotation distribution update by rotation prior
                updated_roation_dist = np.array([np.sum(i * roation_prior) for i in roation_distribution])
                
                posterior_translation = np.sum(np.dot(observation_likelihood, updated_roation_dist))
                # weight of particle
                samples.append([translation_est, updated_roation_dist, posterior_translation])
            
            samples = np.array(samples)
            weight_arr = [samples[-1] for sample in samples]
            
            # resampling step
            indexes = self.systematic_resample(weight_arr)
            resampled_sample = samples[indexes]

            T_est = np.mean(resampled_sample[:, 0])

            max_rotation_dist = [codebook._dataset.viewsphere_for_embedding[np.argmax(sample[1])] for sample in samples]
            R_est = weightedAverageQuaternions(Q=max_rotation_dist[...,[3,0,1,2]].copy(), w=weight_arr)

            # send pose info
            self.translation_dict[self.class_name[class_idx]].append(R_est[[1,2,3,0].copy()])
            self.rotation_dict[self.class_name[class_idx]].append(T_est.copy())
        
    
    def systematic_resample(self, weights):
        """ Performs the systemic resampling algorithm used by particle filters.

        This algorithm separates the sample space into N divisions. A single random
        offset is used to to choose where to sample from for all divisions. This
        guarantees that every sample is exactly 1/N apart.

        Parameters
        ----------
        weights : list-like of float
            list of weights as floats

        Returns
        -------

        indexes : ndarray of ints
            array of indexes into the weights defining the resample. i.e. the
            index of the zeroth resample is indexes[0], etc.
        """
        N = len(weights)

        # make N subdivisions, and choose positions with a consistent random offset
        positions = (random() + np.arange(N)) / N

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
        

    def pose_read(self, detection3darray):
        """
        detection3darray.detections[index].results[0] (index is detect object num on the workspace)
        results[0].id => class id, results[0].score, results[0].pose.pose => pos and  value
        results[0].pose.pose.position => x, y, z
        results[0].pose.pose.orientation => x, y, z, w

        format example
        detections:[
            results:[
                id:1
                score:1.0
                pose:
                    pose: 
                        position: 
                            x: 0.250910286658
                            y: -0.288555985471
                            z: 2.0146174813
                        orientation: 
                            x: -0.153185811236
                            y: 0.159266977727
                            z: 0.85797919622
                            w: 0.463723878936
                    covariance:[...]
            ],
            results:[...]
        ]

        """
        pose_dict = {}

        for detect in detection3darray.detections:
   
            pose_info = detect.results[0].pose.pose
            obj_pose = pose_info.position
            obj_ori = pose_info.orientation
            pose_dict[str(detect.results[0].id)] = {"pose": [obj_pose.x, obj_pose.y, obj_pose.z], 
                                                    "ori":[obj_ori.x, obj_ori.y, obj_ori.z, obj_ori.w]}
        return pose_dict


    def gather_is_results(self, rgb, boxes, scores, labels, is_results):
        
        is_masks = []
        crop_imgs = []

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

                rgb_crop = cv2.resize(rgb_crop, (128, 128), interpolation=cv2.INTER_NEAREST)

                crop_imgs.append(rgb_crop)
                is_masks.append(self.bridge.imgmsg_to_cv2(is_results.masks[i]))
                boxes.append(np.array([x, y, w, h]))
                scores.append(is_results.scores[i])
                labels.append(class_id)

        return crop_imgs, boxes, scores, labels
    
    def publish_markers(self, detection_array):
            # Delete all existing markers
            markers = MarkerArray()
            marker = Marker()
            marker.action = Marker.DELETEALL
            markers.markers.append(marker)
            self.pub_markers.publish(markers)

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


            self.pub_markers.publish(markers)
    

# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation

# Average multiple quaternions with specific weights
# The weight vector w must be of the same length as the number of rows in the
# quaternion maxtrix Q
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

    pose_tracker = PoseTracker()
    rospy.spin()




