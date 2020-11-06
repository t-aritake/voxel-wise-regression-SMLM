# -*- coding: utf-8 -*-
import torch
import pdb


# convolution + batch normalization + ReLU
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.nn.functional.relu(self.bn(self.conv(x)))
        # return torch.nn.functional.relu(self.conv(x))



class Network(torch.nn.Module):
    def __init__(self, input_depth=4):
        super().__init__()

        self._input_depth = input_depth

        # channel numbers at each layer
        channels = [input_depth, 16, 32, 64, 128, 64, 32, 16]
        # kernel size of convolution at each layer
        kernel_sizes = [5, 5, 3, 3, 3, 3, 3]

        layers = []
        for l in range(len(channels)-1):
            layers.append(ConvBlock(channels[l], channels[l+1], kernel_sizes[l], padding=kernel_sizes[l]//2))


        self.convs = torch.nn.Sequential(*layers)
        # final convolution for molecule coordinates for each voxel
        self.loc_conv = torch.nn.Conv2d(16, 12, 3, padding=1)
        # final convolution for confindence for each voxel
        self.conf_conv = torch.nn.Conv2d(16, 4, 3, padding=1)

    def forward(self, x):
        x = self.convs(x)

        # predict moleucle coordinates
        pred_locations = torch.sigmoid(self.loc_conv(x))
        pred_locations = pred_locations.view(x.shape[0], self._input_depth, pred_locations.shape[1]//self._input_depth, pred_locations.shape[2], pred_locations.shape[3]).permute(0, 1, 4, 3, 2)
        pred_locations = pred_locations.reshape(pred_locations.shape[0], -1, pred_locations.shape[-1])

        # predict confindence
        pred_confidences = torch.sigmoid(self.conf_conv(x))
        pred_confidences = pred_confidences.view(x.shape[0], self._input_depth, pred_confidences.shape[1]//self._input_depth, pred_confidences.shape[2], pred_confidences.shape[3]).permute(0, 1, 4, 3, 2)
        pred_confidences= pred_confidences.reshape(pred_confidences.shape[0], -1, pred_confidences.shape[-1])

        x = torch.cat((pred_locations, pred_confidences), 2)

        return x

