import sys
import os
import numpy as np
import torch

sys.path.append("/home/demo/Workspace/PoseRBPF")

import copy
from config.config import cfg
from transforms3d.axangles import *
from .particle_filter import *
import os


class PoseRBPF:
    def __init__(self, obj_list, codebook_list, session, modality, visualize=False, refine=False):

        self.obj_list = obj_list

        # load the object information

        # load encoders and poses
        self.aae_list = []
        #self.codebook_list = []
        self.codebook_list = codebooks
        self.session = session

        self.instance_list = []
        self.rbpf_list = []
        self.rbpf_ok_list = []

        self.modality = modality

        for _codebook, obj in zip(codebook_list, obj_list):
       
            self.aae_full = AAE([obj], modality)
    
            for param in self.aae_full.encoder.parameters():
                param.requires_grad = False

            self.aae_list.append(copy.deepcopy(_codebook._encoder))

            #self.codebook_list.append(torch.load(codebook_file)[0]) original code
            self.codebook_list.append(torch.tensor(_codebook.embedding_normalized).cuda())

            #self.rbpf_codepose = torch.load(codebook_file)[1].cpu().numpy()  # all are identical / original code

        self.rbpf_codepose = _codebook._dataset.viewsphere_for_embedding

        self.intrinsics = np.array([[cfg.PF.FU, 0, cfg.PF.U0],
                                    [0, cfg.PF.FV, cfg.PF.V0],
                                    [0, 0, 1.]], dtype=np.float32)
            
        # target object property
        self.target_obj = None
        self.target_obj_idx = None
        self.target_obj_encoder = None
        self.target_obj_codebook = None
        self.target_obj_cfg = None
        self.target_box_sz = None

        self.max_sim_rgb = 0
        self.max_sim_depth = 0
        self.max_vis_ratio = 0

        # initialize the particle filters
        self.rbpf = particle_filter(cfg.PF, n_particles=100)
        self.rbpf_ok = False

        # pose rbpf for initialization
        self.rbpf_init_max_sim = 0

        # data properties
        self.data_with_gt = False
        self.data_with_est_bbox = False
        self.data_with_est_center = False
        self.data_intrinsics = np.ones((3, 3), dtype=np.float32)

        # initialize the PoseRBPF variables
        '''
        # ground truth information
        self.gt_available = False
        self.gt_bbox_center = np.zeros((3,))
        self.gt_bbox_size = 0
        self.gt_z = 0
        '''

        self.gt_t = [0, 0, 0]
        self.gt_rotm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.gt_uv = np.array([0, 0, 1], dtype=np.float32)

        # estimated states
        self.est_bbox_center = np.zeros((2, cfg.PF.N_PROCESS))
        self.est_bbox_size = np.zeros((cfg.PF.N_PROCESS,))
        self.est_bbox_weights = np.zeros((cfg.PF.N_PROCESS,))


        # posecnn prior
        self.prior_uv = [0, 0, 1]
        self.prior_z = 0
        self.prior_t = np.array([0, 0, 0], dtype=np.float32)
        self.prior_R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

        # flags for experiments
        self.exp_with_mask = True
        self.step = 0
        self.iskf = False
        self.init_step = False
        self.save_uncertainty = False
        self.show_prior = False

        # motion model
        self.T_c1c0 = np.eye(4, dtype=np.float32)
        self.T_o0o1 = np.eye(4, dtype=np.float32)
        self.T_c0o = np.eye(4, dtype=np.float32)
        self.T_c1o = np.eye(4, dtype=np.float32)
        self.Tbr1 = np.eye(4, dtype=np.float32)
        self.Tbr0 = np.eye(4, dtype=np.float32)
        self.Trc = np.eye(4, dtype=np.float32)

        # multiple object pose estimation (mope)
        self.mope_Tbo_list = []
        self.mope_pc_b_list = []
        # for i in range(len(self.rbpf_list)):
        #     self.mope_Tbo_list.append(np.eye(4, dtype=np.float32))
        #     self.mope_pc_b_list.append(None)

        # evaluation module
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

        # relative poses
        self.T_ob_o = []
        self.rel_pose_flag = []
        for i in range(len(self.obj_list)):
            self.T_ob_o.append(np.eye(4, dtype=np.float32))
            self.rel_pose_flag.append(False)

        # for visualization
        self.uv_init = None
        self.z_init = None

        self.image_recon = None


    # add object instance
    def add_object_instance(self, object_name):
        assert object_name in self.obj_list, "object {} is not in the list of test objects".format(object_name)

        idx_obj = self.obj_list.index(object_name)
        self.rbpf_list.append(particle_filter(self.cfg.PF, n_particles=self.cfg.PF.N_PROCESS))
        self.rbpf_ok_list.append(False)
        self.instance_list.append(object_name)
        self.mope_Tbo_list.append(np.eye(4, dtype=np.float32))
        self.mope_pc_b_list.append(None)


    # specify the target object for tracking
    def set_target_obj(self, target_instance_idx):
        target_object = self.instance_list[target_instance_idx]
        assert target_object in self.obj_list, "target object {} is not in the list of test objects".format(target_object)

        # set target object property
        self.target_obj = target_object
        self.target_obj_idx = self.obj_list.index(target_object)
        self.target_obj_encoder = self.aae_list[self.target_obj_idx]

        self.target_obj_codebook = self.codebook_list[self.target_obj_idx]
        self.target_obj_cfg = cfg

        self.target_box_sz = 2 * self.target_obj_cfg.TRAIN.U0 / self.target_obj_cfg.TRAIN.FU * \
                             self.target_obj_cfg.TRAIN.RENDER_DIST[0]

        self.target_obj_cfg.PF.FU = self.intrinsics[0, 0]
        self.target_obj_cfg.PF.FV = self.intrinsics[1, 1]
        self.target_obj_cfg.PF.U0 = self.intrinsics[0, 2]
        self.target_obj_cfg.PF.V0 = self.intrinsics[1, 2]

        # reset particle filter
        self.rbpf = self.rbpf_list[target_instance_idx]
        self.rbpf_ok = self.rbpf_ok_list[target_instance_idx]
        self.rbpf_init_max_sim = 0

        # estimated states
        self.est_bbox_center = np.zeros((2, self.target_obj_cfg.PF.N_PROCESS))
        self.est_bbox_size = np.zeros((self.target_obj_cfg.PF.N_PROCESS,))
        self.est_bbox_weights = np.zeros((self.target_obj_cfg.PF.N_PROCESS,))

        # prior
        self.prior_uv = [0, 0, 1]
        self.prior_z = 0
        self.prior_t = np.array([0, 0, 0], dtype=np.float32)
        self.prior_R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

        # motion model
        self.T_c1c0 = np.eye(4, dtype=np.float32)
        self.T_o0o1 = np.eye(4, dtype=np.float32)
        self.T_c0o = np.eye(4, dtype=np.float32)
        self.T_c1o = np.eye(4, dtype=np.float32)
        self.Tbr1 = np.eye(4, dtype=np.float32)
        self.Tbr0 = np.eye(4, dtype=np.float32)
        self.Trc = np.eye(4, dtype=np.float32)

        # print
        print('target object is set to {}'.format(self.target_obj_cfg.PF.TRACK_OBJ))


    def switch_target_obj(self, target_instance_idx):
        target_object = self.instance_list[target_instance_idx]

        # set target object property
        self.target_obj = target_object
        self.target_obj_idx = self.obj_list.index(target_object)
        self.target_obj_encoder = self.aae_list[self.target_obj_idx]
        self.target_obj_codebook = self.codebook_list[self.target_obj_idx]
        self.target_obj_cfg = cfg


        self.target_box_sz = 2 * self.target_obj_cfg.TRAIN.U0 / self.target_obj_cfg.TRAIN.FU * \
                             self.target_obj_cfg.TRAIN.RENDER_DIST[0]

        self.target_obj_cfg.PF.FU = self.intrinsics[0, 0]
        self.target_obj_cfg.PF.FV = self.intrinsics[1, 1]
        self.target_obj_cfg.PF.U0 = self.intrinsics[0, 2]
        self.target_obj_cfg.PF.V0 = self.intrinsics[1, 2]

        # reset particle filter
        self.rbpf = self.rbpf_list[target_instance_idx]
        self.rbpf_ok = self.rbpf_ok_list[target_instance_idx]
        self.rbpf_init_max_sim = 0
        

    def set_intrinsics(self, intrinsics, w, h):
        self.intrinsics = intrinsics
        self.data_intrinsics = intrinsics
        self.target_obj_cfg.PF.FU = self.intrinsics[0, 0]
        self.target_obj_cfg.PF.FV = self.intrinsics[1, 1]
        self.target_obj_cfg.PF.U0 = self.intrinsics[0, 2]
        self.target_obj_cfg.PF.V0 = self.intrinsics[1, 2]

    

    def use_detection_priors(self, n_particles):
        self.rbpf.uv[-n_particles:] = np.repeat([self.prior_uv], n_particles, axis=0)
        self.rbpf.uv[-n_particles:, :2] += np.random.uniform(-self.target_obj_cfg.PF.UV_NOISE_PRIOR,
                                                             self.target_obj_cfg.PF.UV_NOISE_PRIOR,
                                                             (n_particles, 2))

    # initialize PoseRBPF
    def initialize_poserbpf(self, image, intrinsics, uv_init, n_init_samples, z_init=None, depth=None):
        # sample around the center of bounding box
        uv_h = np.array([uv_init[0], uv_init[1], 1])
        uv_h = np.repeat(np.expand_dims(uv_h, axis=0), n_init_samples, axis=0)
        uv_h[:, :2] += np.random.uniform(-self.target_obj_cfg.PF.INIT_UV_NOISE, self.target_obj_cfg.PF.INIT_UV_NOISE,
                                         (n_init_samples, 2))
        uv_h[:, 0] = np.clip(uv_h[:, 0], 0, image.shape[1])
        uv_h[:, 1] = np.clip(uv_h[:, 1], 0, image.shape[0])

        self.uv_init = uv_h.copy()

        
        # sample around z
        if z_init == None:
            z = np.random.uniform(0.5, 1.5, (n_init_samples, 1))
        else:
            z = np.random.uniform(z_init - 0.2, z_init + 0.2, (n_init_samples, 1))

        self.z_init = z.copy()
        # evaluate translation
        distribution = self.evaluate_particles_rgb(image, uv_h, z,
                                                    self.target_obj_cfg.TRAIN.RENDER_DIST[0], 0.1, depth=depth,
                                                    initialization=True)

        # find the max pdf from the distribution matrix
        index_star = my_arg_max(distribution)
        #uv_star = uv_h[index_star[0], :]  # .copy()
        #z_star = z[index_star[0], :]  # .copy()
        #self.rbpf.update_trans_star_uvz(uv_star, z_star, intrinsics)
        distribution[index_star[0], :] /= torch.sum(distribution[index_star[0], :])
        self.rbpf.rot = distribution[index_star[0], :].view(1, 1, 37, 72, 72).repeat(self.rbpf.n_particles, 1, 1, 1, 1)


    # compute bounding box by projection
    def compute_box(self, pose, points):
        x3d = np.transpose(points)
        RT = np.zeros((3, 4), dtype=np.float32)
        RT[:3, :3] = quat2mat(pose[:4])
        RT[:, 3] = pose[4:]
        x2d = np.matmul(self.intrinsics, np.matmul(RT, x3d))
        x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
        x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

        x1 = np.min(x2d[0, :])
        y1 = np.min(x2d[1, :])
        x2 = np.max(x2d[0, :])
        y2 = np.max(x2d[1, :])
        box = np.array([x1, y1, x2, y2])
        return box


    def evaluate_particles_rgb(self, image, uv, z, render_dist, gaussian_std, depth=None, initialization=False):

        images_roi_cuda, scale_roi = get_rois_cuda(image.detach(), uv, z,
                                                   self.target_obj_cfg.PF.FU,
                                                   self.target_obj_cfg.PF.FV,
                                                   render_dist, out_size=128)

        # forward passing
        n_particles = z.shape[0]
        class_info = torch.ones((1, 1, 128, 128), dtype=torch.float32)
        class_info_cuda = class_info.cuda().repeat(n_particles, 1, 1, 1)
        images_input_cuda = torch.cat((images_roi_cuda.detach(), class_info_cuda.detach()), dim=1)
        

        ################
        # 인코더를 통해서 입력 이미지를 code로 변환하는 파트 => tensorflow로 수정.
        #codes = self.target_obj_encoder.forward(images_input_cuda).view(images_input_cuda.size(0), -1).detach() original code

        x = images_input_cuda.cpu().numpy()
        codes = self.session.run(self.target_obj_encoder, {self.target_obj_encoder.x, x})
        codes = torch.tensor(codes).cuda().view(images_input_cuda.size(0), -1).detach()
        ################

        # compute the similarity between particles' codes and the codebook
        # cosine distance 연산
        cosine_distance_matrix_rgb = self.aae_full.compute_distance_matrix(codes, self.target_obj_codebook)
        max_rgb = torch.max(cosine_distance_matrix_rgb)
        self.max_sim_rgb = max_rgb.cpu().numpy()
        cosine_distance_matrix = cosine_distance_matrix_rgb

        # get the maximum similarity for each particle
        v_sims, i_sims = torch.max(cosine_distance_matrix, dim=1)
        self.cos_dist_mat = v_sims

        max_sim_all = torch.max(v_sims)
        pdf_matrix = mat2pdf(cosine_distance_matrix / max_sim_all, 1, gaussian_std)

        return pdf_matrix

    # filtering
    def process_poserbpf(self, image, intrinsics, depth=None, mask=None, apply_motion_prior=False, use_detection_prior=False):
        # propagation
        if apply_motion_prior:
            self.rbpf.propagate_particles(self.T_c1c0, self.T_o0o1, 0, 0, intrinsics)
            uv_noise = self.target_obj_cfg.PF.UV_NOISE
            z_noise = self.target_obj_cfg.PF.Z_NOISE
            self.rbpf.add_noise_r3(uv_noise, z_noise)
            self.rbpf.add_noise_rot()
        else:
            uv_noise = self.target_obj_cfg.PF.UV_NOISE
            z_noise = self.target_obj_cfg.PF.Z_NOISE
            self.rbpf.add_noise_r3(uv_noise, z_noise)
            self.rbpf.add_noise_rot()

        if use_detection_prior:
            self.use_detection_priors(int(self.rbpf.n_particles/2))

        # compute pdf matrix for each particle
        est_pdf_matrix = self.evaluate_particles_rgb(image, self.rbpf.uv, self.rbpf.z,
                                                        self.target_obj_cfg.TRAIN.RENDER_DIST[0],
                                                        self.target_obj_cfg.PF.WT_RESHAPE_VAR,
                                                        depth=depth,
                                                        initialization=False)


        # most likely particle
        index_star = my_arg_max(est_pdf_matrix)

        self.rbpf.update_trans_star(self.rbpf.uv[index_star[0], :], self.rbpf.z[index_star[0], :], intrinsics)
        self.rbpf.update_rot_star_R(quat2mat(self.rbpf_codepose[index_star[1]][3:]))

        # match rotation distribution
        self.rbpf.rot = torch.clamp(self.rbpf.rot, 1e-5, 1)
        rot_dist = torch.exp(torch.add(torch.log(est_pdf_matrix), torch.log(self.rbpf.rot.view(self.rbpf.n_particles, -1))))
        normalizers = torch.sum(rot_dist, dim=1)

        normalizers_cpu = normalizers.cpu().numpy()
        if np.sum(normalizers_cpu) == 0:
            return 0
        self.rbpf.weights = normalizers_cpu / np.sum(normalizers_cpu)

        rot_dist /= normalizers.unsqueeze(1).repeat(1, self.target_obj_codebook.size(0))

        # matched distributions
        self.rbpf.rot = rot_dist.view(self.rbpf.n_particles, 1, 37, 72, 72)

        # resample
        self.rbpf.resample_ddpf(self.rbpf_codepose, intrinsics, self.target_obj_cfg.PF)


        return 0

    def propagate_with_forward_kinematics(self, target_instance_idx):
        self.switch_target_obj(target_instance_idx)
        self.rbpf.propagate_particles(self.T_c1c0, self.T_o0o1, 0, 0, torch.from_numpy(self.intrinsics).unsqueeze(0))

    # function used in ros node
    def pose_estimation_single(self, target_instance_idx, roi, image, depth, visualize=False, dry_run=False):

        # set target object
        self.switch_target_obj(target_instance_idx)

        if not roi is None:
            center = np.array([0.5 * (roi[2] + roi[4]), 0.5 * (roi[3] + roi[5]), 1], dtype=np.float32)

            # todo: use segmentation masks
            # idx_obj = self.obj_list_all.index(self.target_obj) + 1
            # # there is no large clamp
            # if self.target_obj == '061_foam_brick':
            #     idx_obj -= 1
            # mask_obj = (mask==idx_obj).float().repeat(1,1,3)
            # depth_input = depth * mask_obj[:, :, [0]]

            self.prior_uv = center

            if self.rbpf_ok_list[target_instance_idx] == False:
                # sample around the center of bounding box
                self.initialize_poserbpf(image, self.intrinsics,
                                       self.prior_uv[:2], 100, depth=depth)

                if self.max_sim_rgb > self.target_obj_cfg.PF.SIM_RGB_THRES and not dry_run:
                    print('===================is initialized!======================')
                    self.rbpf_ok_list[target_instance_idx] = True
                    self.process_poserbpf(image,
                                      torch.from_numpy(self.intrinsics).unsqueeze(0),
                                      depth=depth, use_detection_prior=True)

            else:
                self.process_poserbpf(image,
                                      torch.from_numpy(self.intrinsics).unsqueeze(0),
                                      depth=depth, use_detection_prior=True)
                if self.log_max_sim[-1] < 0.6:
                    self.rbpf_ok_list[target_instance_idx] = False

            if not dry_run:
                print('Estimating {}, rgb sim = {}, depth sim = {}'.format(self.instance_list[target_instance_idx], self.max_sim_rgb, self.max_sim_depth))

        Tco = np.eye(4, dtype=np.float32)
        Tco[:3, :3] = self.rbpf.rot_bar
        Tco[:3, 3] = self.rbpf.trans_bar
        max_sim = self.log_max_sim[-1]
        return Tco, max_sim


    def run_dataset(self, images, t_posecnn, q_posecnn):

        self.prior_t = t_posecnn[0].cpu().numpy()
        self.prior_R = quat2mat(q_posecnn[0].cpu().numpy())

        self.target_obj_cfg.PF.FU = self.intrinsics[0, 0]
        self.target_obj_cfg.PF.FV = self.intrinsics[1, 1]
        self.target_obj_cfg.PF.U0 = self.intrinsics[0, 2]
        self.target_obj_cfg.PF.V0 = self.intrinsics[1, 2]


        # motion prior
        self.T_c1o[:3, :3] = self.gt_rotm
        self.T_c1o[:3, 3] = self.gt_t

        if np.linalg.norm(self.T_c0o[:3, 3]) == 0:
            self.T_c1c0 = np.eye(4, dtype=np.float32)
        else:
            self.T_c1c0 = np.matmul(self.T_c1o, np.linalg.inv(self.T_c0o))
            
        self.T_c0o = self.T_c1o.copy()


        # initialization
        if step == 0 or self.rbpf_ok is False:
            print('[Initialization] Initialize PoseRBPF with detected center ... ')
            if np.linalg.norm(self.prior_uv[:2] - self.gt_uv[:2]) > 40:
                self.prior_uv[:2] = self.gt_uv[:2]

            self.initialize_poserbpf(images[0].detach(), self.intrinsics,
                                        self.prior_uv[:2], self.target_obj_cfg.PF.N_INIT,
                                        depth=None)
            self.rbpf_ok = True

        # filtering
        if self.rbpf_ok:
            torch.cuda.synchronize()
            time_start = time.time()
            self.process_poserbpf(images[0], self.intrinsics, depth=None)


            torch.cuda.synchronize()
            time_elapse = time.time() - time_start
            print('[Filtering] fps = ', 1 / time_elapse)

