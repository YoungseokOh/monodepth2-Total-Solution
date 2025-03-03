# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import warnings
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import json
import random
from PIL import Image

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed
warnings.filterwarnings("ignore")

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        # torch cudnn disabled
        torch.backends.cudnn.enabled = False
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.masks = {}
        self.parameters_to_train = []
        self.device = torch.device("cpu" if self.opt.no_cuda else f"cuda:{self.opt.gpu_number}")
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        
        # Encoder selection block
        if self.opt.depth_network == "DepthResNet":
            print(f'----- ResNet-{self.opt.num_layers} -----')
            # Network - DepthResNet(Monodepth2)
            # Encoder
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())
        elif self.opt.depth_network == "DepthResNet_latest":    
            print(f'-----Latest ResNet-{self.opt.num_layers} -----')
            # Network - Latest Depth ResNet-18
            # Encoder
            self.models["encoder"] = networks.LatestResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())
        elif self.opt.depth_network == "LwDepthResNet":    
            print(f'-----Lightweight ResNet-{self.opt.num_layers} -----')
            # Network - Lightweight Depth ResNet-18
            # Encoder
            self.models["encoder"] = networks.LwResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())
        elif self.opt.depth_network == "DepthResNet_CBAM":
            print(f'----- DepthResNet_CBAM-{self.opt.num_layers} -----')
            # Network - DepthResNet-CBAM
            # Encoder
            self.models["encoder"] = networks.ResnetCbamEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())
        elif self.opt.depth_network == "HRLiteNet":
            print('----- HRLiteNet(MobileNetv3) -----')
            # Network - HRLiteNet
            # Encoder
            self.models["encoder"] = networks.MobileEncoder(self.opt.weights_init == "pretrained")
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())
        elif self.opt.depth_network == "RepVGGNet":
            print('----- RepVGGNet -----')
            # Network - RepVGGNet
            # Encoder
            self.models["encoder"] = networks.RepVGGencoder(self.opt.weights_init == "pretrained")
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())
        
        # Decoder selection block
        if self.opt.decoder == 'Dnet':
            print('----- Dnet_Decoder is loaded -----')
            self.models["depth"] = networks.Dnet_DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())
        elif self.opt.decoder == 'NCDL_Decoder':
            self.models["depth"] = networks.NCDLDepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            # Head module
            self.models["head"] = networks.depth_head_module()
            self.models["head"].to(self.device) # GPU
            self.parameters_to_train += list(self.models["depth"].parameters())
        elif self.opt.decoder == 'CAD_Decoder':
            print('----- CAD_Decoder is loaded -----')
            self.models["depth"] = networks.CAD_DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())
        elif self.opt.decoder == 'ECA_Dnet':
            print('----- ECA_Dnet_Decoder is loaded -----')
            self.models["depth"] = networks.ECADnet_DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())
        elif self.opt.decoder == 'original':
            print('----- Original_Depth_Decoder is loaded -----')
            self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())
        elif self.opt.decoder == 'PS_Decoder':
            print('----- Depth_PS_Decoder is loaded -----')
            self.models["depth"] = networks.DepthPSDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())
        elif self.opt.decoder == 'Lite_Decoder':
            print('----- Original_Depth_Decoder is loaded -----')
            self.models["depth"] = networks.Lite_DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())
        elif self.opt.decoder == 'HR_decoder':
            print('----- HR_Depth_Decoder is loaded -----')
            self.models["depth"] = networks.HRDepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())
        if self.opt.depth_network == "HRLiteNet":
            print('----- HRDepth_decoder is loaded -----')
            self.models["depth"] = networks.HRDepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales, mobile_encoder=True)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())
        if self.opt.depth_network == "RepVGGNet" and self.opt.decoder == "PS_RepVGG_Decoder":
            print('----- PS_RepVGG_Decoder is loaded -----')
            self.models["depth"] = networks.DepthPSRepVGGDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())
        if self.opt.depth_network == "RepVGGNet" and self.opt.decoder == "PS_Dnet_RepVGG_Decoder":
            print('----- PS_Dnet_RepVGG_Decoder is loaded -----')
            self.models["depth"] = networks.DepthPSDnetRepVGGDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())
# PoseNet
        if self.use_pose_net:
            # Resnet-18
            if self.opt.pose_model_type == "separate_resnet":
                print(f"----- Using PoseNet type : {self.opt.pose_model_type} -----")
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)
            # resNet-CBAM
            if self.opt.pose_model_type == "separate_resnet_cbam":
                print(f"----- Using PoseNet type : {self.opt.pose_model_type} -----")
                self.models["pose_encoder"] = networks.ResnetCbamEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)
            # Separated_repVGG for PoseNet.
            elif self.opt.pose_model_type == "separate_repVGG":
                print(f"----- Using PoseNet type : {self.opt.pose_model_type} -----")
                self.models["pose_encoder"] = networks.RepVGGencoder(
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())
        
        # Vignetting mask
        if self.opt.vignetting_mask:
            self.vignetting_mask = Image.open('A5_vignetting_mask.jpg')
            for scale in self.opt.scales:
                self.masks["vignetting_mask", scale] = np.array(self.vignetting_mask.resize((self.opt.width // (2 ** scale), self.opt.height // (2 ** scale)), 0))


        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        if self.opt.lr_scheduler == "StepLR":
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        elif self.opt.lr_scheduler == "CosAnnLR":
            self.model_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.model_optimizer, T_max=50, eta_min=1e-6, last_epoch=-1)
        
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                        "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "nextchip":datasets.MonoDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        # Monodataset -> KITTIRAW or nextchip or Cityscapes
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=worker_seed_fn)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = next(iter(self.val_loader))

        # Tensorboard
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
        # AutoBlur
        if not self.opt.disable_auto_blur:
            assert self.opt.receptive_field_of_auto_blur % 2 == 1, \
                'receptive_field_of_auto_blur should be an odd number'
            print('--- AutoBlur is running ---')
            self.auto_blur = networks.AutoBlurModule(
                self.opt.receptive_field_of_auto_blur,
                hf_pixel_thresh=self.opt.hf_pixel_thresh,
                hf_area_percent_thresh=self.opt.hf_area_percent_thresh,
            )
            self.auto_blur.to(self.device)
        else:
            print('--- disable AutoBlur ---')


        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time
            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        # AutoBlur
        if not self.opt.disable_auto_blur:
            for scale in self.opt.scales:
                for f_i in self.opt.frame_ids:
                    inputs[('raw_color', f_i, scale)] = inputs[('color', f_i, scale)]
                    inputs[('color', f_i, scale)] = self.auto_blur(
                        inputs[('color', f_i, scale)])
        # Disable AutoBlur
        else:
            for scale in self.opt.scales:
                for f_i in self.opt.frame_ids:
                    inputs[('raw_color', f_i, scale)] = inputs[('color', f_i, scale)]

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            # Depth and Pose results in 'outputs'
            if self.opt.decoder == "NCDL_Decoder":
                # Divided Decoder - Decoder - Head
                outputs = self.models["depth"](features)    
                outputs = self.models["head"](outputs)
            else:
                outputs = self.models["depth"](features)
        # No need for now
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))
        # Warping.
        self.generate_images_pred(inputs, outputs)
        # Calculate loss
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet" or \
                    self.opt.pose_model_type == "separate_resnet_cbam":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "separate_repVGG":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            #inputs = self.val_iter.next()
            inputs = self.val_iter
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            # inputs = self.val_iter.next()
            inputs = self.val_iter

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0
            # Disparity to Depth
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        total_smooth_loss = 0
        for scale in self.opt.scales:
            loss = 0
            smooth_log = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            if self.opt.vignetting_mask:
                mask = torch.from_numpy(self.masks["vignetting_mask", scale]).to(torch.device('cuda'))
                source_mask = torch.from_numpy(self.masks["vignetting_mask", source_scale]).to(torch.device('cuda'))
                img_mask = mask.repeat(self.opt.batch_size, 3, 1, 1)
                source_img_mask = source_mask.repeat(self.opt.batch_size, 3, 1, 1)
                disp_mask = mask.repeat(self.opt.batch_size, 1, 1, 1)
                loss_mask = mask.repeat(self.opt.batch_size, 1, 1)

                disp = outputs[("disp", scale)]
                color = inputs[("color", 0, scale)] if self.opt.disable_ambiguity_mask \
                    else inputs[('raw_color', 0, scale)]
                target = inputs[("color", 0, source_scale)]
                target *= source_img_mask

                for frame_id in self.opt.frame_ids[1:]:
                    pred = outputs[("color", frame_id, scale)]
                    pred *= source_img_mask
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                
                reprojection_losses = torch.cat(reprojection_losses, 1)
                
                if not self.opt.disable_automasking:
                    identity_reprojection_losses = []
                    for frame_id in self.opt.frame_ids[1:]:
                        pred = inputs[("color", frame_id, source_scale)]
                        identity_reprojection_losses.append(
                            self.compute_reprojection_loss(pred, target))

                    identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                    if self.opt.avg_reprojection:
                        identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                    else:
                        # save both images, and do min all at once below
                        identity_reprojection_loss = identity_reprojection_losses

                elif self.opt.predictive_mask:
                    # use the predicted mask
                    mask = outputs["predictive_mask"]["disp", scale]
                    if not self.opt.v1_multiscale:
                        mask = F.interpolate(
                            mask, [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=False)

                    reprojection_losses *= mask
                    # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                    weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                    loss += weighting_loss.mean()

                if self.opt.avg_reprojection:
                    reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                else:
                    reprojection_loss = reprojection_losses

            else:
                disp = outputs[("disp", scale)]
                color = inputs[("color", 0, scale)] if self.opt.disable_ambiguity_mask \
                    else inputs[('raw_color', 0, scale)]
                target = inputs[("color", 0, source_scale)]

                for frame_id in self.opt.frame_ids[1:]:
                    pred = outputs[("color", frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                
                reprojection_losses = torch.cat(reprojection_losses, 1)
                
                if not self.opt.disable_automasking:
                    identity_reprojection_losses = []
                    for frame_id in self.opt.frame_ids[1:]:
                        pred = inputs[("color", frame_id, source_scale)]
                        identity_reprojection_losses.append(
                            self.compute_reprojection_loss(pred, target))

                    identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                    if self.opt.avg_reprojection:
                        identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                    else:
                        # save both images, and do min all at once below
                        identity_reprojection_loss = identity_reprojection_losses

                elif self.opt.predictive_mask:
                    # use the predicted mask
                    mask = outputs["predictive_mask"]["disp", scale]
                    if not self.opt.v1_multiscale:
                        mask = F.interpolate(
                            mask, [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=False)

                    reprojection_losses *= mask

                    # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                    weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                    loss += weighting_loss.mean()

                if self.opt.avg_reprojection:
                    reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                else:
                    reprojection_loss = reprojection_losses
            
            # AutoBlur
            if not self.opt.disable_ambiguity_mask:
                ambiguity_mask = self.compute_ambiguity_mask(
                    inputs, outputs, reprojection_loss, scale)
            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)


            if not self.opt.disable_ambiguity_mask:
                to_optimise = to_optimise * ambiguity_mask
            

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if self.opt.vignetting_mask:
                mask = torch.from_numpy(self.masks["vignetting_mask", 0]).to(torch.device('cuda'))
                mask = mask.repeat(self.opt.batch_size, 1, 1)
                to_optimise = to_optimise * mask

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

        
            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            smooth_log += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_smooth_loss += smooth_log
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        total_smooth_loss /= self.num_scales
        losses["loss"] = total_loss
        losses["smooth_loss"] = total_smooth_loss
        return losses
    
    @staticmethod
    def extract_ambiguity(ipt):
        grad_r = ipt[:, :, :, :-1] - ipt[:, :, :, 1:]
        grad_b = ipt[:, :, :-1, :] - ipt[:, :, 1:, :]

        grad_l = F.pad(grad_r, (1, 0))
        grad_r = F.pad(grad_r, (0, 1))

        grad_t = F.pad(grad_b, (0, 0, 1, 0))
        grad_b = F.pad(grad_b, (0, 0, 0, 1))

        is_u_same_sign = ((grad_l * grad_r) > 0).any(dim=1, keepdim=True)
        is_v_same_sign = ((grad_t * grad_b) > 0).any(dim=1, keepdim=True)
        is_same_sign = torch.logical_or(is_u_same_sign, is_v_same_sign)

        grad_u = (grad_l.abs() + grad_r.abs()).sum(1, keepdim=True) / 2
        grad_v = (grad_t.abs() + grad_b.abs()).sum(1, keepdim=True) / 2
        grad = torch.sqrt(grad_u ** 2 + grad_v ** 2)

        ambiguity = grad * is_same_sign
        return ambiguity


    def compute_ambiguity_mask(self, inputs, outputs,
                               reprojection_loss, scale):
        src_scale = scale if self.opt.v1_multiscale else 0
        min_reproj, min_idx = torch.min(reprojection_loss, dim=1)

        target_ambiguity = self.extract_ambiguity(inputs[("color", 0, src_scale)])

        reproj_ambiguities = []
        for f_i in self.opt.frame_ids[1:]:
            src_ambiguity = self.extract_ambiguity(inputs[("color", f_i, src_scale)])

            reproj_ambiguity = F.grid_sample(
                src_ambiguity, outputs[("sample", f_i, scale)],
                padding_mode="border", align_corners=True)
            reproj_ambiguities.append(reproj_ambiguity)

        reproj_ambiguities = torch.cat(reproj_ambiguities, dim=1)
        reproj_ambiguity = torch.gather(reproj_ambiguities, 1, min_idx.unsqueeze(1))

        synthetic_ambiguity, _ = torch.cat(
            [target_ambiguity, reproj_ambiguity], dim=1).max(dim=1)

        if self.opt.ambiguity_by_negative_exponential:
            ambiguity_mask = torch.exp(-self.opt.negative_exponential_coefficient
                                       * synthetic_ambiguity)
        else:
            ambiguity_mask = synthetic_ambiguity < self.opt.ambiguity_thresh
        return ambiguity_mask


    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    viz_inv_depth(normalize_image(outputs[("disp", s)][j])).transpose(2, 0, 1), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


def worker_seed_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

