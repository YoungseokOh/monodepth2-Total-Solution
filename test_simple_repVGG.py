# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import torch
import networks
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from layers import disp_to_depth
from collections import OrderedDict
from torchvision import transforms, datasets
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_path', type=str,
                        help='path of trained model',
                        default="")
    parser.add_argument('--model_name', type=str,
                        help='name of trained model',
                        default="")
    parser.add_argument('--num_epoch', type=int,
                        help='the number of epoch for searching weights',
                        default=19)
    parser.add_argument("--depth_network",
                                 type=str,
                                 help="choose the encoder network",
                                 default="RepVGGNet",
                                 choices=["DepthResNet", "DepthResNet_CBAM", "HRLiteNet", "DepthRexNet", "RepVGGNet"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_path is not None, \
        "You must specify the --model_name parameter; see README.md for an example"
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_path:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")
    # Download model for monodepth2
    # download_model_if_doesnt_exist(args.model_name)

    model_path = os.path.join(args.model_path, "models/weights_{}".format(args.num_epoch - 1))
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("Loading pretrained encoder")
    if args.depth_network == "RepVGGNet":
        print('----- RepVGGNet -----')
        # Network - RepVGGNet
        import networks.repVGG as RepVGG
        from networks.depth.DepthRepVGGNet import DepthRepVGGNet
        model = DepthRepVGGNet("18np", True)
        encoder = torch.load(os.path.join(model_path, 'encoder.pth'))
        deploy_encoder = OrderedDict()
        for k, v in encoder.items():
            if k == 'height' or k == 'width' or k == 'use_stereo':
                continue
            else:
                deploy_encoder[k[8:]] = v
        build_repVGG_encoder = RepVGG.create_RepVGG_A0(False)
        build_repVGG_encoder.load_state_dict(deploy_encoder)
        deploy_repVGG_encoder = RepVGG.repvgg_model_convert(build_repVGG_encoder, os.path.join(model_path, 'encoder_deploy.pth'))
        decoder = torch.load(os.path.join(model_path, 'depth.pth'))
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
        model.to(device)
        model.eval()
    # extract the height and width of image that this model was trained with
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']

    if not os.path.exists(os.path.join(args.image_path, 'results')):
        os.makedirs(os.path.join(args.image_path, 'results'))
    output_directory = os.path.join(args.image_path, 'results')
    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            disp = model(input_image)

            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            if args.pred_metric_depth:
                name_dest_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
                np.save(name_dest_npy, metric_depth)
            else:
                name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
