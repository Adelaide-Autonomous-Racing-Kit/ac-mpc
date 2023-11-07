"""
    RefineNet-LightWeight

    RefineNet-LigthWeight PyTorch for non-commercial purposes

    Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
    THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch.nn as nn


def batchnorm(in_planes: int) -> nn.Module:
    "batch norm 2d"
    return nn.BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 1,
    bias: bool = False,
) -> nn.Module:
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )


def conv1x1(
    in_planes: int, out_planes: int, stride: int = 1, bias: bool = False
) -> nn.Module:
    "1x1 convolution"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=bias,
    )


def convbnrelu(
    in_planes: int,
    out_planes: int,
    kernel_size: int,
    stride: int = 1,
    groups: int = 1,
    activation: bool = True,
) -> nn.Sequential:
    "convolution, batchnorm, relu"
    layers = [
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size,
            stride=stride,
            padding=int(kernel_size / 2.0),
            groups=groups,
            bias=False,
        ),
        batchnorm(out_planes),
    ]
    if activation:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)
