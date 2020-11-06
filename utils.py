# -*- coding: utf-8 -*-
import torch
import pdb


def gen_defaults(intervals, device, centering=False):
    x = torch.arange(0, 1, intervals[0]).to(device)
    y = torch.arange(0, 1, intervals[1]).to(device)
    z = torch.arange(0, 1, intervals[2]).to(device)

    coord = torch.stack(torch.meshgrid([x, y, z])).permute(3, 2, 1, 0).reshape(-1, 3)
    if centering is True:
        coord = coord + intervals / 2

    coord.requires_grad = False

    return coord
