from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
from collections import OrderedDict
import datasets
import networks
import tqdm

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        if opt.decoder == 'NCDL_Decoder':
            depthead_path = os.path.join(opt.load_weights_folder, "head.pth")
        encoder_dict = torch.load(encoder_path, map_location='cuda:0')

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)
        print(f"Evauluation model : {opt.depth_network}")
        if opt.depth_network == "DepthResNet":
            encoder = networks.ResnetEncoder(opt.num_layers, False)
            # Depth original
            if opt.decoder == "NCDL_Decoder":
                depth_decoder = networks.NCDLDepthDecoder(encoder.num_ch_enc)
                depth_head = networks.depth_head_module()
            else:
                depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
            # CAD Decoder
            # depth_decoder = networks.CAD_DepthDecoder(encoder.num_ch_enc)
            # Dnet Decoder
            # depth_decoder = networks.Dnet_DepthDecoder(encoder.num_ch_enc)
        elif opt.depth_network == "DepthPSResNet":
            encoder = networks.ResnetEncoder(opt.num_layers, False)
            depth_decoder = networks.DepthPSDecoder(encoder.num_ch_enc)
        elif opt.depth_network == "DepthResNet_CBAM":
            encoder = networks.ResnetCbamEncoder(opt.num_layers, False)
            depth_decoder = networks.Dnet_DepthDecoder(encoder.num_ch_enc)
        elif opt.depth_network == "HRLiteNet":
            encoder = networks.MobileEncoder(False)
            depth_decoder = networks.HRDepthDecoder(encoder.num_ch_enc, opt.scales, mobile_encoder=True)
        elif opt.depth_network == "DepthRexNet":
            encoder = networks.RexnetEncoder(opt.num_layers, False)
            depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
        elif opt.depth_network == "RepVGGNet":
            if opt.onepass:
                print('----- onepass RepVGG -----')
                import networks.repVGG as RepVGG
                from networks.depth.DepthRepVGGNet import DepthRepVGGNet
                model = DepthRepVGGNet("18np", True)
                encoder = torch.load(os.path.join(opt.load_weights_folder, 'encoder.pth'))
                deploy_encoder = OrderedDict()
                for k, v in encoder.items():
                    if k == 'height' or k == 'width' or k == 'use_stereo':
                        continue
                    else:
                        deploy_encoder[k[8:]] = v
                build_repVGG_encoder = RepVGG.create_RepVGG_A0(False)
                build_repVGG_encoder.load_state_dict(deploy_encoder)
                deploy_repVGG_encoder = RepVGG.repvgg_model_convert(build_repVGG_encoder, os.path.join(opt.load_weights_folder, 'encoder_deploy.pth'))
                decoder = torch.load(os.path.join(opt.load_weights_folder, 'depth.pth'))
                endecoder = OrderedDict()
                for k, v in deploy_repVGG_encoder.state_dict().items():
                    if k == 'height' or k == 'width' or k == 'use_stereo':
                        continue
                    endecoder['encoder.encoder.' + k] = v
                for k, v in decoder.items():
                    if k in endecoder:
                        endecoder['decoder.' + k] += v
                    else:
                        endecoder['decoder.' + k] = v
                model.load_state_dict(endecoder)
            else:
                encoder = networks.RepVGGencoder(False)
                # Dnet_DepthDecoder
                # depth_decoder = networks.Dnet_DepthDecoder(encoder.num_ch_enc)
                # Lite DepthDecoder
                # depth_decoder = networks.lite_DepthDecoder(encoder.num_ch_enc)
                # Original Decoder
                depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
            
        
        if opt.onepass:
            model.cuda()
            model.eval()
        else:
            model_dict = encoder.state_dict()
            encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
            depth_decoder.load_state_dict(torch.load(decoder_path, map_location='cuda:0'))
            if opt.decoder == "NCDL_Decoder":
                depth_head.load_state_dict(torch.load(depthead_path, map_location='cuda:0'))
            
            encoder.cuda()
            encoder.eval()
            depth_decoder.cuda()
            depth_decoder.eval()
            depth_head.cuda()
            depth_head.eval()
        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(dataloader)):
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                if opt.onepass:
                    output = model(input_color)
                else:
                    if not opt.decoder =="NCDL_Decoder":
                        output = depth_decoder(encoder(input_color))
                    else:
                        output = depth_head(depth_decoder(encoder(input_color)))

                if opt.onepass:
                    pred_disp, _ = disp_to_depth(output, opt.min_depth, opt.max_depth)
                else:
                    pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()


    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in tqdm.tqdm(range(pred_disps.shape[0])):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
