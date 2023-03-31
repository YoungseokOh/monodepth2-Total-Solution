# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class DepthPSRepVGGDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthPSRepVGGDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 48, 48, 96, 192])
        self.num_ch_ps_dec = np.array([64, 32])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("eca"), i, 0]  = eca_layer(num_ch_in)
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            self.convs[("s1_conv", i, 1)] = S1_ConvBlock(num_ch_in, self.num_ch_ps_dec[0])
            self.convs[("s2_conv", i, 1)] = S2_ConvBlock(self.num_ch_ps_dec[0], self.num_ch_ps_dec[1])
            self.convs[("upconv", i, 1)] = UpConvBlock(self.num_ch_ps_dec[1], num_ch_out * 4)
            if i == 1:
                self.convs[("iconv", i, 1)] = iConvBlock(num_ch_in * 2, num_ch_out)
            elif i == 0:
                self.convs[("iconv", i, 1)] = iConvBlock(self.num_ch_dec[i], num_ch_out)
            else:
                self.convs[("iconv", i, 1)] = iConvBlock(num_ch_in * 2, num_ch_out)
            self.PixelShuffle = nn.PixelShuffle(2)
            self.ELU = nn.ELU(inplace=True)
        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x_f = self.convs[("eca", i, 0)](x)
            x = self.convs[("upconv", i, 0)](x_f)
            # PS Original
            # x = self.convs[("upconv", i, 0)](x)
            x = self.convs[("s1_conv", i, 1)](x)
            x = self.convs[("s2_conv", i, 1)](x)
            x = [self.PixelShuffle(self.convs[("upconv", i, 1)](x))]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("iconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
        return self.outputs

