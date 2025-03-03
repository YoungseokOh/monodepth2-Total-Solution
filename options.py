# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse
from select import select

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")
        # Deep Network
        self.parser.add_argument("--depth_network",
                                 type=str,
                                 help="choose the deep network",
                                 default="DepthResNet",
                                 choices=["DepthResNet", "DepthResNet_latest", "DepthPSResNet", 
                                          "DepthResNet_CBAM", "HRLiteNet", "LwDepthResNet", 
                                          "RepVGGNet"])
        self.parser.add_argument("--decoder",
                                 type=str,
                                 help="choose the depth decoder : [Lite_Decoder, original, ECA_Dnet, Dnet, HR_decoder, PS(Pixel Shuffle)_Decoder, PS_RepVGG_Decoder, PS_Dnet_RepVGG_Decoder]",
                                 default="original",
                                 choices=["Lite_Decoder", "original", "ECA_Dnet", "Dnet", 
                                          "HR_decoder", "PS_Decoder", "PS_RepVGG_Decoder", "PS_Dnet_RepVGG_Decoder", 
                                          "CAD_Decoder", "NCDL_Decoder"])
        
        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))
        
        # TRAINING options
        self.parser.add_argument("--disable_auto_blur",
                                 help="if set, disable Auto-Blur",
                                 default=False, action="store_true")
        self.parser.add_argument("--receptive_field_of_auto_blur",
                                 type=int, default=9)
        self.parser.add_argument("--disable_ambiguity_mask",
                                 help="if set, disable Ambiguity-Masking",
                                 default=False,
                                 action="store_true")
        self.parser.add_argument("--ambiguity_thresh",
                                 type=float,
                                 help="threshold for ambiguous pixels",
                                 default=0.3)
        self.parser.add_argument("--hf_pixel_thresh",
                                 type=float,
                                 help="hf pixel thresh in Auto-Blur",
                                 default=0.2)
        self.parser.add_argument("--hf_area_percent_thresh",
                                 type=int, default=60)
        self.parser.add_argument("--ambiguity_by_negative_exponential",
                                 help='if set, use negative exponential '
                                      'to replace threshold',
                                 default=False, action="store_true")
        self.parser.add_argument("--negative_exponential_coefficient",
                                 help='coefficient of negative '
                                      'exponential function',
                                 type=int, default=3)
        self.parser.add_argument("--random_seed",
                                 default=None, type=int)

        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", 
                                        "benchmark", "eigen_test", "cityscapes_preprocessed",
                                        "A5_v4_frontview", "A5_v4_frontview_denoise", "A5_v4_frontview_carhood",
                                        "A5_adj_3_rearview", "A5_verify_rearview_default", "A5_verify_rearview_selected", 
                                        "A5_fisheye_cropped_images_for_md2", "A5_fisheye_original_images_for_md2", "A6_ch2_md2_dataset"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test",
                                          "cityscapes_preprocessed", "nextchip"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])


        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=12)
        self.parser.add_argument("--lr_scheduler",
                                 type=str,
                                 help="Choose lr_sheduler",
                                 default="StepLR",
                                 choices=["StepLR", "CosAnnLR"])

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "separate_resnet_cbam", "separate_repVGG", "shared"])
        self.parser.add_argument("--vignetting_mask",
                                 help="if set, able vignetting mask in the loss. Use only fisheye lens",
                                 default=False,
                                 action="store_true")
        
        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)
        self.parser.add_argument("--gpu_number",
                                 help="When you want to choose GPU number",
                                 default=0)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--onepass",
                                 help="if set onepass model for evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

        # A5 Dataset Split Options
     #    self.parser.add_argument("--image_dataset_path",
     #                             type=str,
     #                             help="path to the custom training data for mono_dataset.py",
     #                             default=os.path.join(file_dir, "kitti_data"))
     #    self.parser.add_argument("--split_db_name",
     #                             type=str,
     #                             help="write dataset name in image dataset path",
     #                             default=os.path.join(file_dir, "kitti_data"))

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
