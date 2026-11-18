"""CNN that predicts, for every voxel, a molecule offset (x, y, z) and a confidence.

Parameter names (``convs.*``, ``loc_conv``, ``conf_conv``) are kept identical to the
original implementation so that checkpoints trained with the old code still load.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.conv(x)))


class Network(nn.Module):
    """Voxel-wise regression network.

    Input:  ``(B, D, H, W)`` multi-focal-plane image (D planes are treated as channels).
    Output: ``(B, D * W * H, 4)``; the last axis is ``(dx, dy, dz, confidence)``, all in [0, 1].
    The flattened voxel index is ``h + w * H + d * H * W`` (see :func:`vwr_smlm.utils.grid_size`).
    """

    def __init__(
        self,
        input_depth: int = 4,
        channels: Sequence[int] = (16, 32, 64, 128, 64, 32, 16),
        kernel_sizes: Sequence[int] = (5, 5, 3, 3, 3, 3, 3),
    ) -> None:
        super().__init__()
        if len(channels) != len(kernel_sizes):
            raise ValueError("channels and kernel_sizes must have the same length")
        self.input_depth = input_depth

        in_channels = [input_depth, *channels[:-1]]
        self.convs = nn.Sequential(
            *(
                ConvBlock(c_in, c_out, k, padding=k // 2)
                for c_in, c_out, k in zip(in_channels, channels, kernel_sizes, strict=False)
            )
        )
        # 3 coordinates per depth plane / 1 confidence per depth plane
        self.loc_conv = nn.Conv2d(channels[-1], 3 * input_depth, 3, padding=1)
        self.conf_conv = nn.Conv2d(channels[-1], input_depth, 3, padding=1)

    def _to_voxels(self, y: torch.Tensor) -> torch.Tensor:
        # (B, D*C, H, W) -> (B, D, C, H, W) -> (B, D, W, H, C) -> (B, D*W*H, C)
        b, _, h, w = y.shape
        y = y.view(b, self.input_depth, -1, h, w).permute(0, 1, 4, 3, 2)
        return y.reshape(b, -1, y.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        locations = self._to_voxels(torch.sigmoid(self.loc_conv(x)))
        confidences = self._to_voxels(torch.sigmoid(self.conf_conv(x)))
        return torch.cat((locations, confidences), dim=2)
