#!/usr/bin/env python3

import copy
import os
import re
import subprocess
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam

from .fpn import FPNSegmentation


class DiceLoss(nn.Module):
    __name__ = "dice_loss"

    def __init__(self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta

    def f_score(self, gt, pr, beta=1.0, eps=1e-7):
        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
        fn = torch.sum(gt) - tp
        score = ((1 + beta**2) * tp + eps) / ((1 + beta**2) * tp + beta**2 * fn + fp + eps)
        return score

    def forward(self, y_pr, y_gt):
        return 1 - self.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
        )


LOG_STD_MAX = 2
LOG_STD_MIN = -20
VALID_MODELS = (
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
    "efficientnet_v2_xl",
    "efficientnet_v2_s_in21k",
    "efficientnet_v2_m_in21k",
    "efficientnet_v2_l_in21k",
    "efficientnet_v2_xl_in21k",
)

model_urls = {
    "efficientnet_v2_s": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-s.npy",
    "efficientnet_v2_m": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-m.npy",
    "efficientnet_v2_l": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-l.npy",
    "efficientnet_v2_xl": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-xl-21k.npy",
    "efficientnet_v2_s_in21k": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-s-21k.npy",
    "efficientnet_v2_m_in21k": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-m-21k.npy",
    "efficientnet_v2_l_in21k": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-l-21k.npy",
    "efficientnet_v2_xl_in21k": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-xl-21k.npy",
}

fpn_filter_depth_v2 = {
    # pre-calculated list
    "efficientnet_v2_s": [48, 64, 160, 256],
    "efficientnet_v2_m": [48, 80, 176, 512],
    "efficientnet_v2_l": [64, 96, 224, 640],
    "efficientnet_v2_xl": [64, 96, 256, 640],
    "efficientnet_v2_s_in21k": [48, 64, 160, 256],
    "efficientnet_v2_m_in21k": [48, 80, 176, 512],
    "efficientnet_v2_l_in21k": [36, 60, 164, 468],
    "efficientnet_v2_xl_in21k": [64, 96, 256, 640],
}


def load_from_zoo(model, model_name, pretrained_path="pretrained/official"):
    Path(os.path.join(pretrained_path, model_name)).mkdir(parents=True, exist_ok=True)
    file_name = os.path.join(pretrained_path, model_name, os.path.basename(model_urls[model_name]))
    load_npy(model, load_npy_from_url(url=model_urls[model_name], file_name=file_name))


def load_npy_from_url(url, file_name):
    if not Path(file_name).exists():
        subprocess.run(["wget", "-r", "-nc", "-O", file_name, url])
    return np.load(file_name, allow_pickle=True).item()


def npz_dim_convertor(name, weight):
    weight = torch.from_numpy(weight)
    if "kernel" in name:
        if weight.dim() == 4:
            if weight.shape[3] == 1:
                # depth-wise convolution 'h w in_c out_c -> in_c out_c h w'
                weight = weight.permute(2, 3, 0, 1)
            else:
                # 'h w in_c out_c -> out_c in_c h w'
                weight = weight.permute(3, 2, 0, 1)
        elif weight.dim() == 2:
            weight = weight.transpose(1, 0)
    elif "scale" in name or "bias" in name:
        weight = weight.squeeze()
    return weight


def load_npy(model, weight):
    name_convertor = [
        # stem
        ("stem.0.weight", "stem/conv2d/kernel/ExponentialMovingAverage"),
        (
            "stem.1.weight",
            "stem/tpu_batch_normalization/gamma/ExponentialMovingAverage",
        ),
        ("stem.1.bias", "stem/tpu_batch_normalization/beta/ExponentialMovingAverage"),
        (
            "stem.1.running_mean",
            "stem/tpu_batch_normalization/moving_mean/ExponentialMovingAverage",
        ),
        (
            "stem.1.running_var",
            "stem/tpu_batch_normalization/moving_variance/ExponentialMovingAverage",
        ),
        # fused layer
        ("block.fused.0.weight", "conv2d/kernel/ExponentialMovingAverage"),
        (
            "block.fused.1.weight",
            "tpu_batch_normalization/gamma/ExponentialMovingAverage",
        ),
        ("block.fused.1.bias", "tpu_batch_normalization/beta/ExponentialMovingAverage"),
        (
            "block.fused.1.running_mean",
            "tpu_batch_normalization/moving_mean/ExponentialMovingAverage",
        ),
        (
            "block.fused.1.running_var",
            "tpu_batch_normalization/moving_variance/ExponentialMovingAverage",
        ),
        # linear bottleneck
        ("block.linear_bottleneck.0.weight", "conv2d/kernel/ExponentialMovingAverage"),
        (
            "block.linear_bottleneck.1.weight",
            "tpu_batch_normalization/gamma/ExponentialMovingAverage",
        ),
        (
            "block.linear_bottleneck.1.bias",
            "tpu_batch_normalization/beta/ExponentialMovingAverage",
        ),
        (
            "block.linear_bottleneck.1.running_mean",
            "tpu_batch_normalization/moving_mean/ExponentialMovingAverage",
        ),
        (
            "block.linear_bottleneck.1.running_var",
            "tpu_batch_normalization/moving_variance/ExponentialMovingAverage",
        ),
        # depth wise layer
        (
            "block.depth_wise.0.weight",
            "depthwise_conv2d/depthwise_kernel/ExponentialMovingAverage",
        ),
        (
            "block.depth_wise.1.weight",
            "tpu_batch_normalization_1/gamma/ExponentialMovingAverage",
        ),
        (
            "block.depth_wise.1.bias",
            "tpu_batch_normalization_1/beta/ExponentialMovingAverage",
        ),
        (
            "block.depth_wise.1.running_mean",
            "tpu_batch_normalization_1/moving_mean/ExponentialMovingAverage",
        ),
        (
            "block.depth_wise.1.running_var",
            "tpu_batch_normalization_1/moving_variance/ExponentialMovingAverage",
        ),
        # se layer
        ("block.se.fc1.weight", "se/conv2d/kernel/ExponentialMovingAverage"),
        ("block.se.fc1.bias", "se/conv2d/bias/ExponentialMovingAverage"),
        ("block.se.fc2.weight", "se/conv2d_1/kernel/ExponentialMovingAverage"),
        ("block.se.fc2.bias", "se/conv2d_1/bias/ExponentialMovingAverage"),
        # point wise layer
        ("block.fused_point_wise.0.weight", "conv2d_1/kernel/ExponentialMovingAverage"),
        (
            "block.fused_point_wise.1.weight",
            "tpu_batch_normalization_1/gamma/ExponentialMovingAverage",
        ),
        (
            "block.fused_point_wise.1.bias",
            "tpu_batch_normalization_1/beta/ExponentialMovingAverage",
        ),
        (
            "block.fused_point_wise.1.running_mean",
            "tpu_batch_normalization_1/moving_mean/ExponentialMovingAverage",
        ),
        (
            "block.fused_point_wise.1.running_var",
            "tpu_batch_normalization_1/moving_variance/ExponentialMovingAverage",
        ),
        ("block.point_wise.0.weight", "conv2d_1/kernel/ExponentialMovingAverage"),
        (
            "block.point_wise.1.weight",
            "tpu_batch_normalization_2/gamma/ExponentialMovingAverage",
        ),
        (
            "block.point_wise.1.bias",
            "tpu_batch_normalization_2/beta/ExponentialMovingAverage",
        ),
        (
            "block.point_wise.1.running_mean",
            "tpu_batch_normalization_2/moving_mean/ExponentialMovingAverage",
        ),
        (
            "block.point_wise.1.running_var",
            "tpu_batch_normalization_2/moving_variance/ExponentialMovingAverage",
        ),
        # head
        ("head.bottleneck.0.weight", "head/conv2d/kernel/ExponentialMovingAverage"),
        (
            "head.bottleneck.1.weight",
            "head/tpu_batch_normalization/gamma/ExponentialMovingAverage",
        ),
        (
            "head.bottleneck.1.bias",
            "head/tpu_batch_normalization/beta/ExponentialMovingAverage",
        ),
        (
            "head.bottleneck.1.running_mean",
            "head/tpu_batch_normalization/moving_mean/ExponentialMovingAverage",
        ),
        (
            "head.bottleneck.1.running_var",
            "head/tpu_batch_normalization/moving_variance/ExponentialMovingAverage",
        ),
        # classifier
        ("head.classifier.weight", "head/dense/kernel/ExponentialMovingAverage"),
        ("head.classifier.bias", "head/dense/bias/ExponentialMovingAverage"),
        ("\\.(\\d+)\\.", lambda x: f"_{int(x.group(1))}/"),
    ]

    for name, param in list(model.named_parameters()) + list(model.named_buffers()):
        for pattern, sub in name_convertor:
            name = re.sub(pattern, sub, name)
        if "dense/kernel" in name and list(param.shape) not in [
            [1000, 1280],
            [21843, 1280],
        ]:
            continue
        if "dense/bias" in name and list(param.shape) not in [[1000], [21843]]:
            continue
        if "num_batches_tracked" in name:
            continue
        param.data.copy_(npz_dim_convertor(name, weight.get(name)))


def get_efficientnet_v2_structure(model_name):
    if "efficientnet_v2_s" in model_name:
        return [
            # expand ratio, kernel size, stride, in_filters, out_filters, layers, use_squeeze_excite, use_fused_mbconv
            (1, 3, 1, 24, 24, 2, False, True),
            (4, 3, 2, 24, 48, 4, False, True),
            (4, 3, 2, 48, 64, 4, False, True),
            (4, 3, 2, 64, 128, 6, True, False),
            (6, 3, 1, 128, 160, 9, True, False),
            (6, 3, 2, 160, 256, 15, True, False),
        ]
    elif "efficientnet_v2_m" in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 24, 24, 3, False, True),
            (4, 3, 2, 24, 48, 5, False, True),
            (4, 3, 2, 48, 80, 5, False, True),
            (4, 3, 2, 80, 160, 7, True, False),
            (6, 3, 1, 160, 176, 14, True, False),
            (6, 3, 2, 176, 304, 18, True, False),
            (6, 3, 1, 304, 512, 5, True, False),
        ]
    elif "efficientnet_v2_l" in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 32, 32, 4, False, True),
            (4, 3, 2, 32, 64, 7, False, True),
            (4, 3, 2, 64, 96, 7, False, True),
            (4, 3, 2, 96, 192, 10, True, False),
            (6, 3, 1, 192, 224, 19, True, False),
            (6, 3, 2, 224, 384, 25, True, False),
            (6, 3, 1, 384, 640, 7, True, False),
        ]
    elif "efficientnet_v2_xl" in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 32, 32, 4, False, True),
            (4, 3, 2, 32, 64, 8, False, True),
            (4, 3, 2, 64, 96, 8, False, True),
            (4, 3, 2, 96, 192, 16, True, False),
            (6, 3, 1, 192, 256, 24, True, False),
            (6, 3, 2, 256, 512, 32, True, False),
            (6, 3, 1, 512, 640, 8, True, False),
        ]


class ConvBNAct(nn.Sequential):
    """Convolution-Normalization-Activation Module"""

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        groups,
        norm_layer,
        act,
        conv_layer=nn.Conv2d,
    ):
        super(ConvBNAct, self).__init__(
            conv_layer(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_channel),
            act(),
        )


class SEUnit(nn.Module):
    """Squeeze-Excitation Unit
    paper: https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper
    """

    def __init__(
        self,
        in_channel,
        reduction_ratio=4,
        act1=partial(nn.SiLU, inplace=True),
        act2=nn.Sigmoid,
    ):
        super(SEUnit, self).__init__()
        hidden_dim = in_channel // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channel, hidden_dim, (1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, in_channel, (1, 1), bias=True)
        self.act1 = act1()
        self.act2 = act2()

    def forward(self, x):
        return x * self.act2(self.fc2(self.act1(self.fc1(self.avg_pool(x)))))


class StochasticDepth(nn.Module):
    """StochasticDepth
    paper: https://link.springer.com/chapter/10.1007/978-3-319-46493-0_39
    :arg
        - prob: Probability of dying
        - mode: "row" or "all". "row" means that each row survives with different probability
    """

    def __init__(self, prob, mode):
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x):
        if self.prob == 0.0 or not self.training:
            return x
        else:
            shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == "row" else [1]
            return x * torch.empty(shape).bernoulli_(self.survival).div_(self.survival).to(x.device)


class MBConvConfig:
    """EfficientNet Building block configuration"""

    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        in_ch: int,
        out_ch: int,
        layers: int,
        use_se: bool,
        fused: bool,
        act=nn.SiLU,
        norm_layer=nn.BatchNorm2d,
    ):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_layers = layers
        self.act = act
        self.norm_layer = norm_layer
        self.use_se = use_se
        self.fused = fused

    @staticmethod
    def adjust_channels(channel, factor, divisible=8):
        new_channel = channel * factor
        divisible_channel = max(divisible, (int(new_channel + divisible / 2) // divisible) * divisible)
        divisible_channel += divisible if divisible_channel < 0.9 * new_channel else 0
        return divisible_channel


class MBConv(nn.Module):
    """EfficientNet main building blocks
    :arg
        - c: MBConvConfig instance
        - sd_prob: stochastic path probability
    """

    def __init__(self, c, sd_prob=0.0):
        super(MBConv, self).__init__()
        inter_channel = c.adjust_channels(c.in_ch, c.expand_ratio)
        block = []

        if c.expand_ratio == 1:
            block.append(
                (
                    "fused",
                    ConvBNAct(
                        c.in_ch,
                        inter_channel,
                        c.kernel,
                        c.stride,
                        1,
                        c.norm_layer,
                        c.act,
                    ),
                )
            )
        elif c.fused:
            block.append(
                (
                    "fused",
                    ConvBNAct(
                        c.in_ch,
                        inter_channel,
                        c.kernel,
                        c.stride,
                        1,
                        c.norm_layer,
                        c.act,
                    ),
                )
            )
            block.append(
                (
                    "fused_point_wise",
                    ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity),
                )
            )
        else:
            block.append(
                (
                    "linear_bottleneck",
                    ConvBNAct(c.in_ch, inter_channel, 1, 1, 1, c.norm_layer, c.act),
                )
            )
            block.append(
                (
                    "depth_wise",
                    ConvBNAct(
                        inter_channel,
                        inter_channel,
                        c.kernel,
                        c.stride,
                        inter_channel,
                        c.norm_layer,
                        c.act,
                    ),
                )
            )
            block.append(("se", SEUnit(inter_channel, 4 * c.expand_ratio)))
            block.append(
                (
                    "point_wise",
                    ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity),
                )
            )

        self.block = nn.Sequential(OrderedDict(block))
        self.use_skip_connection = c.stride == 1 and c.in_ch == c.out_ch
        self.stochastic_path = StochasticDepth(sd_prob, "row")

    def forward(self, x):
        out = self.block(x)
        if self.use_skip_connection:
            out = x + self.stochastic_path(out)
        return out


class EfficientNetV2(nn.Module):
    """Pytorch Implementation of EfficientNetV2
    paper: https://arxiv.org/abs/2104.00298
    - reference 1 (pytorch): https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
    - reference 2 (official): https://github.com/google/automl/blob/master/efficientnetv2/effnetv2_configs.py
    :arg
        - layer_infos: list of MBConvConfig
        - out_channels: bottleneck channel
        - nlcass: number of class
        - dropout: dropout probability before classifier layer
        - stochastic depth: stochastic depth probability
    """

    def __init__(
        self,
        layer_infos=None,
        out_channels=1280,
        latent_dims=0,
        dropout=0.2,
        stochastic_depth=0.0,
        segmentation=False,
        block=MBConv,
        act_layer=nn.SiLU,
        norm_layer=nn.BatchNorm2d,
        version=None,
    ):
        super(EfficientNetV2, self).__init__()

        if layer_infos is None:
            if version is None:
                version = "efficientnet_v2_s"
            layer_infos = [MBConvConfig(*layer_config) for layer_config in get_efficientnet_v2_structure(version)]

        self.layer_infos = layer_infos
        self.norm_layer = norm_layer
        self.act = act_layer
        self.input_channels = 3
        self.in_channel = layer_infos[0].in_ch
        self.final_stage_channel = layer_infos[-1].out_ch
        self.out_channels = out_channels
        self.cur_block = 0
        self.num_block = sum(stage.num_layers for stage in layer_infos)
        self.stochastic_depth = stochastic_depth
        self.segmentation = segmentation

        if self.segmentation:
            self.number_of_skip_connections = 4
            self.skip_connection_dims = []
            self.c_list = []

        self.stem = ConvBNAct(self.input_channels, self.in_channel, 3, 2, 1, self.norm_layer, self.act)
        self.blocks = nn.Sequential(*self.make_stages(layer_infos, block))

    def make_stages(self, layer_infos, block):
        return [layer for layer_info in layer_infos for layer in self.make_layers(copy.copy(layer_info), block)]

    def make_layers(self, layer_info, block):
        layers = []
        for i in range(layer_info.num_layers):
            layers.append(block(layer_info, sd_prob=self.get_sd_prob()))
            layer_info.in_ch = layer_info.out_ch
            layer_info.stride = 1
        return layers

    def get_sd_prob(self):
        sd_prob = self.stochastic_depth * (self.cur_block / self.num_block)
        self.cur_block += 1
        return sd_prob

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.
        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            self.input_channels = in_channels
            self.stem = ConvBNAct(self.input_channels, self.in_channel, 3, 2, 1, self.norm_layer, self.act)

    def forward(self, x):
        if self.segmentation:
            num_blocks = int(len(self.blocks))

            x = self.stem(x)

            for i in range(num_blocks):
                prev_x, last_shape = x, x.shape[2]
                x = self.blocks[i](x)

                if last_shape != x.shape[2]:
                    self.c_list.append(prev_x)
                    self.skip_connection_dims.append(prev_x.shape[1])

            self.c_list.append(x)  # get final layer
            self.skip_connection_dims.append(x.shape[1])

            while self.number_of_skip_connections < len(self.c_list):
                del self.c_list[0]
                del self.skip_connection_dims[0]

            self.c_list = self.c_list[::-1]
            self.skip_connection_dims = self.skip_connection_dims[::-1]
            return x

        else:
            return self.blocks(self.stem(x))

    def change_dropout_rate(self, p):
        self.head[-2] = nn.Dropout(p=p, inplace=True)

    @classmethod
    def from_name(
        cls,
        version,
        input_channels=3,
        latent_dims=0,
        dropout=0.1,
        stochastic_depth=0.2,
        segmentation=False,
        **kwargs,
    ):

        cls._check_model_name_is_valid(version)
        residual_config = [MBConvConfig(*layer_config) for layer_config in get_efficientnet_v2_structure(version)]
        model = cls(
            residual_config,
            1280,
            latent_dims,
            dropout=dropout,
            stochastic_depth=stochastic_depth,
            block=MBConv,
            act_layer=nn.SiLU,
            segmentation=segmentation,
        )

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

        model._change_in_channels(input_channels)
        return model

    @classmethod
    def from_pretrained(
        cls,
        version,
        input_channels=3,
        latent_dims=0,
        dropout=0.1,
        stochastic_depth=0.2,
        segmentation=False,
        **override_params,
    ):
        """Create an efficientnet model according to name.
        Args:
            version (str): Name for efficientnet.

            input_channels (int): Input data's channel number.

            latent_dims (int):
                Number of categories for classification.
                It controls the output size for final linear layer.

            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'
        Returns:
            A pretrained efficientnet model.
        """
        cls._check_model_name_is_valid(version)
        residual_config = [MBConvConfig(*layer_config) for layer_config in get_efficientnet_v2_structure(version)]
        model = cls(
            residual_config,
            1280,
            latent_dims,
            dropout=dropout,
            stochastic_depth=stochastic_depth,
            block=MBConv,
            act_layer=nn.SiLU,
            segmentation=segmentation,
        )

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

        load_from_zoo(model, version)
        model._change_in_channels(input_channels)
        return model

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.
        Args:
            model_name (str): Name for efficientnet.
        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError("model_name should be one of: " + ", ".join(VALID_MODELS))


class EfficientNetV2_AE(nn.Module):
    def __init__(
        self,
        im_h,
        im_w,
        im_c,
        recon_c,
        latent_dims,
        version="efficientnet_v2_s",
        device="cuda",
        pretrained=False,
    ):
        super(EfficientNetV2_AE, self).__init__()

        if pretrained:
            self.encoder = EfficientNetV2.from_pretrained(
                version=version,
                input_channels=im_c,
                latent_dims=latent_dims,
                segmentation=False,
            )
        else:
            self.encoder = EfficientNetV2.from_name(
                version=version,
                input_channels=im_c,
                latent_dims=latent_dims,
                segmentation=False,
            )

        em_shape = self.encoder(torch.ones((1, im_c, im_h, im_w))).shape[1:]
        h_dim = int(np.prod(em_shape))

        self.flatten = nn.Sequential(nn.Flatten(), nn.Linear(h_dim, latent_dims), nn.Tanh())

        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, h_dim),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=em_shape),
            nn.ConvTranspose2d(em_shape[0], 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, recon_c, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Sigmoid(),
        )
        self.to(device)

    def disable_encoder_gradients(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.flatten.parameters():
            param.requires_grad = False

    def encode(self, x):
        return self.flatten(self.encoder(x))

    def forward(self, x):
        return self.decoder(self.flatten(self.encoder(x)))

    @torch.no_grad()
    def predict(self, x):
        return self.forward(x)

    def loss(
        self,
        actual,
        recon,
    ):
        recons_loss = F.mse_loss(recon, actual)
        return recons_loss


class EfficientNetV2_VAE(nn.Module):
    def __init__(
        self,
        im_h,
        im_w,
        im_c,
        recon_c,
        latent_dims,
        version="efficientnet_v2_s",
        device="cuda",
        pretrained=False,
    ):
        super(EfficientNetV2_VAE, self).__init__()

        if pretrained:
            self.encoder = EfficientNetV2.from_pretrained(
                version=version,
                input_channels=im_c,
                latent_dims=latent_dims,
                segmentation=False,
            )
        else:
            self.encoder = EfficientNetV2.from_name(
                version=version,
                input_channels=im_c,
                latent_dims=latent_dims,
                segmentation=False,
            )

        em_shape = self.encoder(torch.ones((1, im_c, im_h, im_w))).shape[1:]
        h_dim = int(np.prod(em_shape))

        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(h_dim, latent_dims)
        self.fc_logvar = nn.Sequential(
            nn.Linear(h_dim, latent_dims), nn.Tanh()
        )  # Sigmoid or Tanh dat shiz or you get NaN's
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, h_dim),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=em_shape),
            nn.ConvTranspose2d(em_shape[0], 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, recon_c, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Sigmoid(),
        )
        self.to(device)

    def encode(self, x):
        x = self.flatten(self.encoder(x))
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = self.N.sample(mu.shape)
        return eps * std + mu

    def forward(self, x):
        x = self.flatten(self.encoder(x))
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = self.N.sample(mu.shape)
        z = eps * std + mu
        return self.decoder(z), mu, logvar

    def predict(self, x):
        x, _, _ = self.forward(x)
        return x

    def loss(self, actual, recon, mu, logvar, kld_weight=1.0):
        recons_loss = F.mse_loss(recon, actual)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        return recons_loss + kld_loss * kld_weight


class EfficientNetV2_FPN_Segmentation(nn.Module):
    def __init__(
        self,
        version,
        im_c,
        n_classes,
        loss=None,
        device="cuda",
        pretrained=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        if pretrained:
            self.encoder = EfficientNetV2.from_pretrained(
                version=version, input_channels=im_c, latent_dims=0, segmentation=True
            )
        else:
            self.encoder = EfficientNetV2.from_name(
                version=version, input_channels=im_c, latent_dims=0, segmentation=True
            )

        filter_depths = fpn_filter_depth_v2[version][::-1]

        self.decoder = FPNSegmentation(
            n_classes=n_classes,
            fpn_filter_depth=filter_depths,
            final_upscale_ratio=4,
            weighted=False,
        )
        if loss is None:
            self.seg_loss = DiceLoss()
        else:
            self.seg_loss = loss

    def loss(self, gt, pr):
        return self.seg_loss(gt, pr)

    def forward(self, x):
        self.encoder(x)
        x = self.decoder(self.encoder.c_list)
        return x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)


class EfficientNetV2_AE_FPN_Segmentation(nn.Module):
    def __init__(
        self,
        version,
        im_c,
        im_h,
        im_w,
        recon_c,
        latent_dims,
        n_classes,
        seg_loss=None,
        device="cuda",
        pretrained=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        if pretrained:
            self.encoder = EfficientNetV2.from_pretrained(
                version=version,
                input_channels=im_c,
                latent_dims=latent_dims,
                segmentation=False,
            )
        else:
            self.encoder = EfficientNetV2.from_name(
                version=version,
                input_channels=im_c,
                latent_dims=latent_dims,
                segmentation=False,
            )

        em_shape = self.encoder(torch.ones((1, im_c, im_h, im_w))).shape[1:]
        h_dim = int(np.prod(em_shape))

        self.flatten = nn.Sequential(nn.Flatten(), nn.Linear(h_dim, latent_dims))

        filter_depths = fpn_filter_depth_v2[version][::-1]

        self.seg_decoder = FPNSegmentation(
            n_classes=n_classes,
            fpn_filter_depth=filter_depths,
            final_upscale_ratio=4,
            weighted=False,
        )

        self.ae_decoder = nn.Sequential(
            nn.Linear(latent_dims, h_dim),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=em_shape),
            nn.ConvTranspose2d(em_shape[0], 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, recon_c, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Sigmoid(),
        )
        if seg_loss is None:
            self.seg_loss_function = DiceLoss()
        else:
            self.seg_loss_function = seg_loss

        self.to(device)

    def encode(self, x):
        ae_x = self.flatten(self.encoder(x))
        seg_x = self.seg_decoder(self.encoder.c_list)
        return ae_x, seg_x

    def seg_loss(self, gt, pr):
        return self.seg_loss_function(gt, pr)

    def ae_loss(self, actual, recon):
        return F.mse_loss(recon, actual)

    def forward(self, x):
        ae_x = self.ae_decoder(self.flatten(self.encoder(x)))
        seg_x = self.seg_decoder(self.encoder.c_list)
        return ae_x, seg_x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)


class VAE(nn.Module):
    """Expects input of (batch_size, C, H, W)"""

    def __init__(self, im_w, im_h, im_c, recon_c, latent_dims=256, device="cuda"):
        super(VAE, self).__init__()

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)

        encoder_list = [
            nn.Conv2d(im_c, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        ]

        self.encoder = nn.Sequential(*encoder_list)
        sample_input = torch.zeros([1, im_c, im_h, im_w])
        em_shape = nn.Sequential(*encoder_list[:-1])(sample_input).shape[1:]
        h_dim = int(np.prod(em_shape))

        self.fc_mu = nn.Linear(h_dim, latent_dims)
        self.fc_logvar = nn.Sequential(nn.Linear(h_dim, latent_dims), nn.Sigmoid())  # Sigmoid dat shiz or you get NaN's

        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, h_dim),
            nn.Unflatten(1, em_shape),
            nn.ConvTranspose2d(
                em_shape[0],
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=(0, 0),
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, recon_c, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.to(device)

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = self.N.sample(mu.shape)
        return eps * std + mu

    def forward(self, x):
        x = self.encoder(x)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = self.N.sample(mu.shape)
        z = eps * std + mu
        return self.decoder(z), mu, logvar

    def predict(self, x):
        (
            x,
            _,
            _,
        ) = self.forward(x)
        return x

    def loss(self, actual, recon, mu, logvar, kld_weight=1.0):
        recons_loss = F.mse_loss(recon, actual)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        return recons_loss + kld_loss * kld_weight


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
        latent_dims=None,
        device="cpu",
    ):
        super().__init__()

        obs_dim = observation_space.shape[0] if latent_dims is None else latent_dims
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.device = device
        self.to(device)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy() if self.device == "cpu" else a.cpu().numpy()


class MLPActor(nn.Module):
    def __init__(self, latent_dims, action_space, hidden_sizes=(256, 256), device="cpu"):
        super().__init__()

        # build policy and value functions
        self.fc = nn.Sequential(
            nn.Linear(latent_dims, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_space),
        )
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.fc(x)

    def act(self, obs):
        with torch.no_grad():
            a = self.fc(obs).squeeze()
            return a.numpy() if self.device == "cpu" else a.cpu().numpy()


class EfficientNetV2_AE_MLP(nn.Module):
    def __init__(
        self,
        im_h,
        im_w,
        im_c,
        recon_c,
        action_dim,
        latent_dims,
        version="efficientnet_v2_s",
        device="cuda",
        pretrained=False,
        hidden_sizes=(1024, 256),
    ):
        super(EfficientNetV2_AE_MLP, self).__init__()

        self.encoder = EfficientNetV2_AE(
            im_h=im_h,
            im_w=im_w,
            im_c=im_c,
            recon_c=recon_c,
            latent_dims=latent_dims,
            version=version,
            device=device,
            pretrained=pretrained,
        )
        # build policy and value functions

        self.actor = MLPActor(latent_dims=latent_dims, action_space=action_dim, hidden_sizes=hidden_sizes)

        self.device = device
        self.to(device)

    def disable_encoder_gradients(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encoder_checkpoint(self, file_path):
        self.encoder.load_state_dict(torch.load(file_path, map_location=self.device))
        self.disable_encoder_gradients()

    def forward(self, x):
        return self.actor(self.encoder.encode(x))

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

    def loss(self, gt, pr):
        return F.mse_loss(pr, gt)


class EfficientNetV2_AE_MLPAC(nn.Module):
    def __init__(
        self,
        im_h,
        im_w,
        im_c,
        recon_c,
        action_dim,
        latent_dims,
        act_limit=1.0,
        version="efficientnet_v2_s",
        device="cuda",
        pretrained=False,
        hidden_sizes=(1024, 256),
        activation=nn.ReLU,
    ):
        super(EfficientNetV2_AE_MLPAC, self).__init__()

        self.encoder = EfficientNetV2_AE(
            im_h=im_h,
            im_w=im_w,
            im_c=im_c,
            recon_c=recon_c,
            latent_dims=latent_dims,
            version=version,
            device=device,
            pretrained=pretrained,
        )
        # build policy and value functions

        self.pi = SquashedGaussianMLPActor(latent_dims, action_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(latent_dims, action_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(latent_dims, action_dim, hidden_sizes, activation)

        self.device = device
        self.to(device)

    def disable_encoder_gradients(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encoder_checkpoint(self, file_path):
        self.encoder.load_state_dict(torch.load(file_path, map_location=self.device))
        self.disable_encoder_gradients()

    def forward(self, x, deterministic=False):
        x = self.encoder.encode(x)
        x, _ = self.pi(x, deterministic, False)
        return x

    def encode(self, x):
        x = self.encoder.encode(x)
        return x

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy() if self.device == "cpu" else a.cpu().numpy()

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

    def loss(self, gt, pr):
        return F.mse_loss(pr, gt)


class PPOPolicyNetwork(nn.Module):
    def __init__(
        self,
        observation_dims,
        action_dims,
        hidden_sizes=(1024, 512),
        device="cuda",
        std=0.0,
    ):
        super(PPOPolicyNetwork, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(observation_dims, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_dims),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_dims, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(1, action_dims) * std)
        self.to(device)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


class PPO:
    def __init__(
        self,
        observation_dims,
        action_dims,
        hidden_sizes=(2048, 512),
        device="cuda",
        checkpoint=None,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lamda=0.95,
        ppo_epsilon=0.2,
        critic_discount=0.5,
        entropy_beta=0.001,
        ppo_steps=128,
        mini_batch_size=64,
        ppo_epochs=100,
        test_epochs=10,
        num_tests=5,
        max_test_length=100,
        model_save_path=None,
    ):

        self.device = device
        self.action_dims = action_dims
        self.observation_dims = observation_dims
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lamda = gae_lamda
        self.ppo_epsilon = ppo_epsilon
        self.critic_discount = critic_discount
        self.entropy_beta = entropy_beta
        self.ppo_steps = ppo_steps
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.test_epochs = test_epochs
        self.num_tests = num_tests
        self.early_stop = False
        self.new_observation_and_actions = False
        self.new_test_observation_and_actions = False
        self.max_test_length = max_test_length

        self.observation = torch.zeros(1, dtype=torch.float, device=self.device)
        self.actions = torch.zeros(1, dtype=torch.float, device=self.device)
        self.done = 0

        self.test_observation = torch.zeros(1, dtype=torch.float, device=self.device)
        self.test_actions = torch.zeros(1, dtype=torch.float, device=self.device)
        self.test_done = 0

        self.model_save_path = model_save_path
        self.model = PPOPolicyNetwork(
            observation_dims=observation_dims,
            action_dims=action_dims,
            hidden_sizes=hidden_sizes,
            device=device,
        )
        self.test_model = None

        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint, map_location=device))

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        self.frame_idx = 0
        self.train_epoch = 0
        self.best_reward = None
        self.test_reward = 0

        self.training_thread = None
        self.testing_thread = None
        self.read_lock = threading.Lock()
        self.running = False
        self.test_running = False
        self.test_environment_on = False

    def start_thread(self):
        # Start up a separate thread to do the training
        self.training_thread = threading.Thread(target=self.training_loop)
        self.training_thread.start()
        self.running = True
        return self

    def start_test_thread(self):
        # Start up a separate thread to do the training
        self.test_environment_on = True
        self.testing_thread = threading.Thread(target=self.test_env)
        self.testing_thread.start()
        self.test_running = False
        return self

    def stop_thread(self):
        self.running = False
        self.training_thread.join()

    def stop_test_thread(self):
        self.test_environment_on = False
        self.test_running = False
        self.testing_thread.join()

    def compute_gae(self, next_value, rewards, masks, values):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lamda * masks[step] * gae
            # prepend to get correct order back
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(self, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        # generates random mini-batches until we have covered the full batch
        for _ in range(batch_size // self.mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[
                rand_ids, :
            ]

    def normalize(self, x):
        x -= x.mean()
        x /= x.std() + 1e-8
        return x

    def ppo_update(self, states, actions, log_probs, returns, advantages):
        # print("ppo update")

        count_steps = 0
        sum_returns = 0.0
        sum_advantage = 0.0
        sum_loss_actor = 0.0
        sum_loss_critic = 0.0
        sum_entropy = 0.0
        sum_loss_total = 0.0

        # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
        for _ in range(self.ppo_epochs):
            # grabs random mini-batches several times until we have covered all data
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(
                states, actions, log_probs, returns, advantages
            ):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * advantage

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = self.critic_discount * critic_loss + actor_loss - self.entropy_beta * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # track statistics
                sum_returns += return_.mean()
                sum_advantage += advantage.mean()
                sum_loss_actor += actor_loss
                sum_loss_critic += critic_loss
                sum_loss_total += loss
                sum_entropy += entropy

                count_steps += 1

    def ready_for_update(self):
        return not self.new_observation_and_actions

    def ready_for_test_update(self):
        ready = self.test_running and not self.new_test_observation_and_actions
        return ready

    def update_environment(self, observation, actions, done):
        with self.read_lock:
            self.observation, self.actions = deepcopy(observation), deepcopy(actions)
            self.done = deepcopy(done)
            self.new_observation_and_actions = True

    def update_test_environment(self, observation, actions, done):
        with self.read_lock:
            self.test_observation, self.test_actions = (
                deepcopy(observation),
                deepcopy(actions),
            )
            self.test_done = deepcopy(done)
            self.new_test_observation_and_actions = True

    def step_environment(self, actions):
        while not self.new_observation_and_actions:
            time.sleep(0.01)

        with self.read_lock:
            observation, expected_action = (
                np.expand_dims(np.array(deepcopy(self.observation)), axis=0),
                deepcopy(self.actions),
            )
            done = deepcopy(self.done)
            self.new_observation_and_actions = False

        if actions is None:
            return observation, 0, done, 0
        else:
            # error = np.abs(expected_action - actions)
            error = np.sum(np.abs(expected_action - actions))
            # reward = torch.as_tensor(np.full((1, self.action_dims), fill_value=2) - error, dtype=torch.float32, device=self.device)
            reward = torch.as_tensor(4 - error, dtype=torch.float32, device=self.device)
            done = torch.as_tensor(done, device=self.device)
            return observation, reward, done, 0

    def step_test_environment(self, actions):
        while not self.new_test_observation_and_actions:
            time.sleep(0.01)

        with self.read_lock:
            observation, expected_action = (
                np.expand_dims(np.array(deepcopy(self.test_observation)), axis=0),
                deepcopy(self.test_actions),
            )
            done = deepcopy(self.test_done)
            self.new_test_observation_and_actions = False

        if actions is None:
            return observation, 0, done, 0
        else:
            # error = np.abs(expected_action - actions)
            error = np.sum(np.abs(expected_action - actions))
            # reward = torch.as_tensor(np.full((1, self.action_dims), fill_value=2) - error, dtype=torch.float32, device=self.device)
            reward = torch.as_tensor(4 - error, dtype=torch.float32, device=self.device)
            done = torch.as_tensor(done, device=self.device)
            return observation, reward, done, 0

    def test_env(self, deterministic=True):
        while self.test_environment_on:
            while not self.test_running:
                time.sleep(0.1)

            print("Starting Tests")
            test_rewards = np.zeros(self.num_tests)

            for i in range(self.num_tests):
                print("Test", i)
                state, _, _, _ = self.step_test_environment(None)
                total_reward = 0

                for j in range(self.max_test_length):
                    state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                    dist, _ = self.test_model(state_t)
                    action = dist.mean.detach().cpu().numpy() if deterministic else dist.sample().cpu().numpy()
                    next_state, reward, done, _ = self.step_test_environment(action)
                    state = next_state
                    total_reward += reward.detach().cpu().numpy()

                    if done:
                        break

                test_rewards[i] = total_reward

            self.test_reward = np.mean(test_rewards)
            print("Frame %s. reward: %s" % (self.frame_idx, self.test_reward))
            # Save a checkpoint every time we achieve a best reward
            if self.best_reward is None or self.best_reward < self.test_reward:
                if self.best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (self.best_reward, self.test_reward))
                    path_name = f"{self.model_save_path}/ppo_{self.frame_idx}.pth"
                    torch.save(self.test_model, path_name)
                self.best_reward = self.test_reward

            self.test_running = False

    def training_loop(self):
        while not self.early_stop:
            print("train epoch", self.train_epoch)

            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []

            state, _, _, _ = self.step_environment(None)

            for b in range(self.ppo_steps):
                state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                dist, value = self.model(state_t)

                action = dist.sample()
                next_state, reward, done, _ = self.step_environment(action.cpu().numpy())
                log_prob = dist.log_prob(action)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                masks.append((1 - done).unsqueeze(-1))

                states.append(torch.as_tensor(state, dtype=torch.float32, device=self.device))
                actions.append(action)

                state = next_state
                self.frame_idx += 1

            next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
            _, next_value = self.model(next_state_t)

            returns = self.compute_gae(next_value, rewards, masks, values)

            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            advantage = returns - values
            advantage = self.normalize(advantage)

            self.ppo_update(states, actions, log_probs, returns, advantage)
            self.train_epoch += 1

            if self.train_epoch % self.test_epochs == 0 and not self.test_running:
                with self.read_lock:
                    self.test_model = deepcopy(self.model)
                    self.test_running = True
                    self.new_test_observation_and_actions = False


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    im_h, im_w, im_c, batch, n_classes, latent_dims, recon_c = 256, 512, 3, 1, 2, 256, 1
    x = torch.randn((batch, im_c, im_h, im_w)).to(device)

    model = EfficientNetV2_VAE(im_h=im_h, im_w=im_w, im_c=im_c, latent_dims=latent_dims, device=device)

    assert model.encode(x).shape == (batch, latent_dims)

    time_now = time.time()
    vae, _, _ = model.forward(x)
    print(f"EfficientNetV2_VAE inference (s) {time.time() - time_now:.4f}")

    assert vae.shape == (batch, im_c, im_h, im_w)

    model = EfficientNetV2_AE(im_h=im_h, im_w=im_w, im_c=im_c, latent_dims=latent_dims, device=device)

    assert model.encode(x).shape == (batch, latent_dims)

    time_now = time.time()
    ae = model.forward(x)
    print(f"EfficientNetV2_AE inference (s) {time.time() - time_now:.4f}")

    assert ae.shape == (batch, im_c, im_h, im_w)

    model = VAE(im_h=im_h, im_w=im_w, im_c=im_c, latent_dims=latent_dims, device=device)

    assert model.encode(x).shape == (batch, latent_dims)

    time_now = time.time()
    vae, _, _ = model.forward(x)
    print(f"VAE inference (s) {time.time() - time_now:.4f}")

    assert vae.shape == (batch, im_c, im_h, im_w)

    model = EfficientNetV2_FPN_Segmentation(version="efficientnet_v2_s", im_c=im_c, n_classes=n_classes).to(device)

    time_now = time.time()
    seg = model.forward(x)
    print(f"EfficientNetV2_FPN_Segmentation inference (s) {time.time() - time_now:.4f}")

    assert seg.shape == (batch, n_classes, im_h, im_w)

    model = EfficientNetV2_AE_FPN_Segmentation(
        version="efficientnet_v2_s",
        im_c=im_c,
        im_h=im_h,
        im_w=im_w,
        recon_c=recon_c,
        n_classes=n_classes,
        latent_dims=latent_dims,
    ).to(device)

    assert model.encode(x)[0].shape == (batch, latent_dims)
    assert model.encode(x)[1].shape == (batch, n_classes, im_h, im_w)

    time_now = time.time()
    ae, seg = model.forward(x)
    print(f"EfficientNetV2_AE_FPN_Segmentation inference (s) {time.time() - time_now:.4f}")

    assert ae.shape == (batch, recon_c, im_h, im_w)
    assert seg.shape == (batch, n_classes, im_h, im_w)
