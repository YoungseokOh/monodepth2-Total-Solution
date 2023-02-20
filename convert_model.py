import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from networks.depth.DepthResNet import DepthResDNet


def main():
    model = DepthResDNet("18np")
    print('done')


if __name__ == '__main__':
    # args = parse_args()
    main()
