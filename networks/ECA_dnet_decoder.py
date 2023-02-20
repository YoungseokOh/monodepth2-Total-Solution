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


class Dnet_DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(Dnet_DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            # ECA (Efficient Channel Attention)
            self.convs[("eca"), i]  = eca_layer(num_ch_in)
            # SE Block
            # self.convs[("SEblock", i)] = SEBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dconv", s)]  = Conv3x3(self.num_ch_dec[s], 8)
            self.convs[("Hier_SEblock", s)] = SEBlock(8, 8)
            self.convs[("dispconv", s)] = Conv3x3((4-s)*8, self.num_output_channels)
            
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.dfeats  = []
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            # SE Block
            # x = self.convs["SEblock", i](x)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                x_f = self.convs[("dconv", i)](x)
                x_att = self.convs[("eca", i)](x_f)
                self.dfeats.append(x_att + x_f)
                # Original
                # self.dfeats.append(self.convs[("dconv", i)](x))
        
        self.dfeats.reverse()

        for s in self.scales:
            dcfeats = []
            up = 2 ** (3-s)
            for i in range(3, s-1, -1):
                if i == s:
                    dcfeats.append(self.dfeats[i])
                else:
                    # Original Dnet
                    dcfeats.append(upsample_DNet(self.dfeats[i], sf=up))
                    # Hierarchical SEblock
                    # dcfeats.append(upsample_DNet(self.convs["Hier_SEblock", i](self.dfeats[i]), sf=up))
                    up /= 2
            dcfeats = torch.cat((dcfeats), 1)
            # dcfeats = SEBlock(dcfeats)
            dcfeats_f = self.convs[("dispconv", s)](dcfeats)
            dcfeats_att = self.convs[("eca", i)](dcfeats_f)
            self.outputs[("disp", s)] = self.sigmoid(dcfeats_f + dcfeats_att)
            # Original
            # self.outputs[("disp", s)] = self.sigmoid(self.convs[("dispconv", s)](dcfeats))

        return self.outputs


class BaselineDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(BaselineDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs