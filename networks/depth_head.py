import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict

class depth_head_module(nn.Module):
    def __init__(
    self, 
    scales=range(4),
    num_output_channels=1):
        super(depth_head_module, self).__init__()
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.num_output_channels = num_output_channels
        self.sigmoid = nn.Sigmoid()
        self.moduleset = nn.ModuleList(
            [
                nn.Conv2d(self.num_ch_dec[i], self.num_output_channels, 1, 1, 0)
                for i in scales
            ]
        )

    def _disp_to_depth(disp, min_depth, max_depth):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        """
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth


    def forward(self, x: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.outputs = {}

        for i, m in reversed(list(enumerate(self.moduleset))) :
            self.outputs[("disp", i)] = self.sigmoid(m(x["disp", i]))

        return self.outputs