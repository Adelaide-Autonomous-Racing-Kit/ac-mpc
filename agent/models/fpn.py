#!/usr/bin/env python3

"""
Implementation of feature pyramid network (FPN) decoder for semantic segmentation that is  based on descriptions in:
"Feature Pyramid Networks for Object Detection", Lin et al 2017, and,
"Panoptic Feature Pyramid Networks", Kirillov et al, 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                'Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)
    

class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class SumBlock(nn.Module):
    def __init__(self, weighted: bool = False, num_skip_connections: int = 4):
        super().__init__()
        self.weighted = weighted
        if self.weighted:
            self.num_skip_connections = num_skip_connections
            self.weight_params = nn.Parameter(torch.ones(num_skip_connections, dtype=torch.float32), requires_grad=True)
            self.weight_activation = nn.ReLU()

    def forward(self, x_list):
        if self.weighted:
            weights = self.weight_activation(self.weight_params)
            x = weights[0] * x_list[0]
            for i in range(1, self.num_skip_connections):
                x = x + weights[i] * x_list[i]
            return x
        else:
            return sum(x_list)


class ScaleFeaturePyramid(nn.Module):
    def __init__(
            self,
            num_skip_connections: int = 4,
            pyramid_in_channels: int = 256,
            pyramid_out_channels: int = 128,

    ):
        super().__init__()
        self.num_skip_connections = num_skip_connections

        if self.num_skip_connections == 4:
            self.seg_blocks = nn.ModuleList([
                SegmentationBlock(pyramid_in_channels, pyramid_out_channels, n_upsamples=n_upsamples)
                for n_upsamples in [3, 2, 1, 0]
            ])

        elif self.num_skip_connections == 5:
            self.seg_blocks = nn.ModuleList([
                SegmentationBlock(pyramid_in_channels, pyramid_out_channels, n_upsamples=n_upsamples)
                for n_upsamples in [4, 3, 2, 1, 0]
            ])

    def forward(self, p_list):
        if self.num_skip_connections == 4:
            return [seg_block(p) for seg_block, p in zip(self.seg_blocks, p_list)]

        elif self.num_skip_connections == 5:
            return [seg_block(p) for seg_block, p in zip(self.seg_blocks, p_list)]


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels, weighted: bool = False):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
        self.weighted = weighted
        if self.weighted:
            self.weight_params = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.weight_activation = nn.ReLU()

    def forward(self, x, skip=None):
        if self.weighted:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            skip = self.skip_conv(skip)
            weights = self.weight_activation(self.weight_params)
            x = weights[0] * x + weights[1] * skip
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            skip = self.skip_conv(skip)
            x = x + skip
        return x


class FeaturePyramid(nn.Module):
    def __init__(
            self,
            fpn_filter_depth: list,
            pyramid_in_channels: int = 256,
            weighted: bool = False
    ):
        super().__init__()

        self.fpn_filter_depth = fpn_filter_depth
        self.num_skip_connections = len(fpn_filter_depth)
        self.p_out = [torch.empty(0)] * self.num_skip_connections

        if self.num_skip_connections == 4:
            self.p5 = nn.Conv2d(fpn_filter_depth[0], pyramid_in_channels, kernel_size=1)
            self.p4 = FPNBlock(pyramid_in_channels, fpn_filter_depth[1], weighted=weighted)
            self.p3 = FPNBlock(pyramid_in_channels, fpn_filter_depth[2], weighted=weighted)
            self.p2 = FPNBlock(pyramid_in_channels, fpn_filter_depth[3], weighted=weighted)

        elif self.num_skip_connections == 5:
            self.p5 = nn.Conv2d(fpn_filter_depth[0], pyramid_in_channels, kernel_size=1)
            self.p4 = FPNBlock(pyramid_in_channels, fpn_filter_depth[1], weighted=weighted)
            self.p3 = FPNBlock(pyramid_in_channels, fpn_filter_depth[2], weighted=weighted)
            self.p2 = FPNBlock(pyramid_in_channels, fpn_filter_depth[3], weighted=weighted)
            self.p1 = FPNBlock(pyramid_in_channels, fpn_filter_depth[4], weighted=weighted)

    def forward(self, p_list):
        if self.num_skip_connections == 4:

            self.p_out[0] = self.p5(p_list[0])
            self.p_out[1] = self.p4(self.p_out[0], p_list[1])
            self.p_out[2] = self.p3(self.p_out[1], p_list[2])
            self.p_out[3] = self.p2(self.p_out[2], p_list[3])

        elif self.num_skip_connections == 5:
            self.p_out[0] = self.p5(p_list[0])
            self.p_out[1] = self.p4(self.p_out[0], p_list[1])
            self.p_out[2] = self.p3(self.p_out[1], p_list[2])
            self.p_out[3] = self.p2(self.p_out[2], p_list[3])
            self.p_out[4] = self.p1(self.p_out[3], p_list[4])

        return self.p_out


class FPNSegmentation(nn.Module):
    def __init__(
            self,
            n_classes: int,
            fpn_filter_depth: list = [256, 512, 1024, 2048],
            activation: str = 'softmax2d',
            pyramid_in_channels: int = 256,
            pyramid_out_channels: int = 128,
            dropout=0.2,
            final_upscale_ratio: int = 4,
            weighted: bool = False
    ):
        super().__init__()
        num_skip_connections = len(fpn_filter_depth)
        self.feature_pyramid = FeaturePyramid(fpn_filter_depth=fpn_filter_depth, pyramid_in_channels=pyramid_in_channels, weighted=weighted)
        self.feature_pyramid_scale = ScaleFeaturePyramid(num_skip_connections=num_skip_connections, pyramid_in_channels=pyramid_in_channels, pyramid_out_channels=pyramid_out_channels)
        self.merge = SumBlock(weighted=weighted, num_skip_connections=num_skip_connections)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.segmentation_head = SegmentationHead(
            in_channels=pyramid_out_channels,
            out_channels=n_classes,
            activation=activation,
            kernel_size=1,
            upsampling=final_upscale_ratio,
        )

    def forward(self, x):
        x = self.feature_pyramid_scale(self.feature_pyramid(x))
        x = self.merge(x)
        x = self.dropout(x)
        x = self.segmentation_head(x)
        return x




