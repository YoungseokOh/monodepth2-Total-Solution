import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
from networks.depth.DepthResNet import DepthResDNet
from networks.depth.DepthRepVGGNet import DepthRepVGGNet


def main():
    # model = DepthResDNet("18np")
    main_folder = '/home/seok436/tmp'
    model_name = 'repVGG_model_pretrained_b12_e30_separate_resnet'
    epoch = 30
    load_path = os.path.join(os.path.join(main_folder, model_name), 'models/weights_{}'.format(epoch-1))
    print(load_path)
    model = DepthRepVGGNet("18np")
    encoder = torch.load(os.path.join(load_path, 'encoder.pth'))
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
    torch.save({'state_dict': model.state_dict()}, os.path.join(main_folder, '{}.ckpt'.format(model_name)))

if __name__ == '__main__':
    # args = parse_args()
    main()
