import os
import copy
import torch
import torch.nn as nn
from PIL import Image
from functools import partial
import torch.nn.functional as F
import networks.repVGG as RepVGG
from collections import OrderedDict
import torchvision.transforms as transforms
from networks.depth.LwDepthResNet import LwDepthResNet
from networks.depth.DepthResNet import DepthResNet
from networks.depth.DepthRepVGGNet import DepthRepVGGNet


def main():
    main_folder = '/home/seok436/tmp'
    depth_network  = 'LwDepthResNet' # Choice = ["DepthResNet", "DepthResNet_CBAM", "HRLiteNet", "LwDepthResNet", "RepVGGNet"]
    model_name = 'LwDepthResNet_A5_v4_denoise_640x384_pretrained_b12_e20_separate_repVGG_AutoBlur_original'
    convert_onnx = True
    input_height = 384
    input_width = 640
    epoch = 20
    load_path = os.path.join(os.path.join(main_folder, model_name), 'models/weights_{}'.format(epoch-1))
    print(load_path)
    # ResNet-18
    if depth_network == 'LwDepthResNet':
        model = LwDepthResNet("18np")
        encoder = torch.load(os.path.join(load_path, 'encoder.pth'))
        deploy_encoder = OrderedDict()
        for k, v in encoder.items():
            if k == 'height' or k == 'width' or k == 'use_stereo':
                continue
            else:
                deploy_encoder[k[8:]] = v
        decoder = torch.load(os.path.join(load_path, 'depth.pth'))
        endecoder = OrderedDict()
        for k, v in encoder.items():
            if k == 'height' or k == 'width' or k == 'use_stereo':
                continue
            endecoder['encoder.' + k] = v
        for k, v in decoder.items():
            if k in endecoder:
                endecoder['decoder.' + k] += v
            else:
                endecoder['decoder.' + k] = v
        model.load_state_dict(endecoder)
        print('----- Deploy model is created!')
        print('The converted model is saved!')
    elif depth_network == 'RepVGGNet':
    # model = DepthResDNet("18np")
    # RepVGGNet
        model = DepthRepVGGNet("18np", True)
        encoder = torch.load(os.path.join(load_path, 'encoder.pth'))
        deploy_encoder = OrderedDict()
        for k, v in encoder.items():
            if k == 'height' or k == 'width' or k == 'use_stereo':
                continue
            else:
                deploy_encoder[k[8:]] = v
        build_repVGG_encoder = RepVGG.create_RepVGG_A0(False)
        build_repVGG_encoder.load_state_dict(deploy_encoder)
        deploy_repVGG_encoder = RepVGG.repvgg_model_convert(build_repVGG_encoder, os.path.join(load_path, 'encoder_deploy.pth'))
        decoder = torch.load(os.path.join(load_path, 'depth.pth'))
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
        print('----- Deploy model is created!')
        # torch.save({'state_dict': model.state_dict()}, os.path.join(main_folder, '{}_deploy.ckpt'.format(model_name)))

        # .ckpt save

    if convert_onnx:
        model.eval()
        img = Image.open("assets/input_kitti_{}x{}.png".format(input_width, input_height))
        # Test image -NCDB
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        img.unsqueeze_(0)
        img_input = img
        print("Creating dummy input...")
        torch.onnx.export(model, img_input,
                "onnx/{}_{}x{}.onnx".format(model_name, input_width, input_height),
                opset_version=10)
        print("Pytorch to onnx (End to End version) is done!")
    print('The deploy model (ONNX) is saved!')
    print('The work is done.')
if __name__ == '__main__':
    # args = parse_args()
    main()
