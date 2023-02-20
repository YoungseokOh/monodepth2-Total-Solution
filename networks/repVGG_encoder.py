# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import networks.repVGG as RepVGG
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.utils.checkpoint as checkpoint


class RepVGGNetMultiImageInput(RepVGG.RepVGG):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, num_blocks, num_classes, width_multiplier, block_type, num_input_images=1):
        super(RepVGGNetMultiImageInput, self).__init__(num_blocks, num_classes, width_multiplier)
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = block_type(in_channels=num_input_images * 3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)

        # ResNet?
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def repVGG_multiimage_input(pretrained=False, num_input_images=1):
    """Constructs a RepVGG model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    blocks = [2, 4, 14, 1]
    num_classes = 1000
    block_type = RepVGG.RepVGGBlock
    width_multiplier=[0.75, 0.75, 0.75, 2.5]
    model = RepVGGNetMultiImageInput(blocks, num_classes, width_multiplier, block_type, num_input_images=num_input_images)

    if pretrained:
        loaded = torch.load('pretrained_model/repVGG/RepVGG-A0-train.pth')
        loaded['stage0.rbr_dense.conv.weight'] = torch.cat(
            [loaded['stage0.rbr_dense.conv.weight']] * num_input_images, 1) / num_input_images
        loaded['stage0.rbr_1x1.conv.weight'] = torch.cat(
            [loaded['stage0.rbr_1x1.conv.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class RepVGGencoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, pretrained, num_input_images=1):
        super(RepVGGencoder, self).__init__()
        self.num_ch_enc = np.array([48, 48, 96, 192, 1280])       
        if num_input_images > 1:
            self.encoder = repVGG_multiimage_input(pretrained, num_input_images)
        else:
            self.encoder = RepVGG.create_RepVGG_A0(deploy=False)
            if pretrained:
                print('----- RepVGG pretrained model is loaded.-----')
                self.encoder.load_state_dict(
                    torch.load('pretrained_model/repVGG/RepVGG-A0-train.pth'))

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        out = self.encoder.stage0(x)
        self.features.append(out)
        for stage in (self.encoder.stage1, self.encoder.stage2, self.encoder.stage3, self.encoder.stage4):
            for block in stage:
                if self.encoder.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
            self.features.append(out)
        return self.features
