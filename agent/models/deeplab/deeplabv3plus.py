from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_decoder import EncoderDecoder
from .utils.layer_factory import conv1x1, convbnrelu

os_to_rates = {
    8: [12, 24, 36],
    16: [6, 12, 18],
    32: [3, 6, 9],
}


class DeepLabV3plus(EncoderDecoder):
    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int,
        classification_head: nn.Module = None,
        verbose_sizes: bool = False,
        interpolation_mode: str = "bilinear",
    ):
        super(DeepLabV3plus, self).__init__()
        self._encoder = backbone
        atrous_rates = os_to_rates[backbone._output_stride]
        self._aspp = ASPP(self._aspp_in_channels, 256, atrous_rates)
        self._decoder = Decoder()
        # Optional to add custom classification head
        if classification_head is not None:
            self._classification = classification_head
        else:
            self._classification = conv1x1(256, n_classes)

        self._interpolation_mode = interpolation_mode
        self._low_level_reducer = self._build_dim_reducer()
        self._verbose_sizes = verbose_sizes

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor, grad_chk: bool = False) -> torch.Tensor:
        if self._verbose_sizes:
            print("input.size()", x.shape)

        _, l2, _, l4 = self._encoder(x, grad_chk)
        l2, l4 = l2[1], l4[1]  # Strip labels

        if self._verbose_sizes:
            print("1/4_logits.size()", l2.shape)
            print("1/8 - 1/16 logits", l4.shape)

        l2 = self._low_level_reducer(l2)
        aspp_features = self._aspp(l4)

        if self._verbose_sizes:
            print("aspp_features.size()", aspp_features.shape)
            print("x.size()", x.shape)

        decoder_out = self._decoder(l2, aspp_features)

        if self._verbose_sizes:
            print("decoder_out.size()", decoder_out.shape)

        out = self._classification(decoder_out)

        if self._verbose_sizes:
            print("class.size()", out.shape)

        return F.interpolate(out, mode=self._interpolation_mode, size=x.shape[2:])

    def _build_dim_reducer(self):
        encoder_channel_sizes = self._encoder_channels
        return conv1x1(encoder_channel_sizes[1][1], 48, bias=False)

    @property
    def _aspp_in_channels(self):
        channel_sizes = self._encoder_channels
        return channel_sizes[-1][1]


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # 48 channels from low level features onto 256 from ASPP
        self._conv_block = nn.Sequential(
            convbnrelu(304, 256, 3), convbnrelu(256, 256, 3)
        )

    def forward(
        self, low_level_features: torch.Tensor, aspp_features: torch.Tensor
    ) -> torch.Tensor:
        low_level_features = F.interpolate(
            low_level_features, mode="bilinear", size=aspp_features.shape[2:]
        )
        x = torch.cat([low_level_features, aspp_features], axis=1)
        return self._conv_block(x)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: List[int]):
        super(ASPP, self).__init__()

        aspp_convs = nn.ModuleList(
            self.build_aspp_convs(in_channels, out_channels, atrous_rates)
        )
        # build aspp convs
        self.aspp_convs = aspp_convs

        self.project = nn.Sequential(
            nn.Conv2d(len(aspp_convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def build_aspp_convs(self, in_channels, out_channels, rates):
        # 1x1 Conv
        convs = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]

        # 3x3 Conv's at rates
        convs.extend([ASPPConv(in_channels, out_channels, rate) for rate in rates])

        # image pooling
        convs.append(ASPPPooling(in_channels, out_channels))

        return convs

    def forward(self, x):
        res = []
        for conv in self.aspp_convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x
