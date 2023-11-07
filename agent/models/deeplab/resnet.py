from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision
from loguru import logger
from torch.utils.checkpoint import checkpoint

"""
    ResNet wrapper of pytorch's implementation
        Enables gradient checkpointing and intermediate
        feature representations to be returned in the
        forward pass to multi-scale decoder networks
"""


class ResnetEncoder(nn.Module):
    def __init__(self, model: nn.Module, output_stride: int = 32):
        super().__init__()
        self._output_stride = output_stride

        self._level1 = Ignore2ndArg(
            nn.Sequential(
                *list(model.children())[0:4], *list(model.layer1.children())
            )
        )
        self._level2 = Ignore2ndArg(
            nn.Sequential(*list(model.layer2.children()))
        )
        self._level3 = Ignore2ndArg(
            nn.Sequential(*list(model.layer3.children()))
        )
        self._level4 = Ignore2ndArg(
            nn.Sequential(*list(model.layer4.children()))
        )
        # Dummy Tensor so that checkpoint can be used on first conv block
        self._dummy = torch.ones(1, requires_grad=True)
        self._deeplab_surgery()

    # Returns intermediate representations for use in decoder
    # representation spatial size depends on output stride [32,16,8]
    def forward(
        self, x: torch.Tensor, gradient_chk: bool = False
    ) -> List[Tuple[str, torch.Tensor]]:
        if gradient_chk:
            dummy = self._dummy
            l1 = checkpoint(self._level1, x, dummy)  # 1/4
            l2 = checkpoint(self._level2, l1, dummy)  # 1/8
            l3 = checkpoint(self._level3, l2, dummy)  # 1/16 - 1/8
            l4 = checkpoint(self._level4, l3, dummy)  # 1/32 - 1/16 - 1/8
        else:
            l1 = self._level1(x)  # 1/4
            l2 = self._level2(l1)  # 1/8
            l3 = self._level3(l2)  # 1/16 - 1/8
            l4 = self._level4(l3)  # 1/32 - 1/16 - 1/8

        return [("level1", l1), ("level2", l2), ("level3", l3), ("level4", l4)]

    # Network surgery for use in deeplabv3+
    def _deeplab_surgery(self):
        # Handles Resnet 18 & 32
        block_type = self._level4.module[0].__class__.__name__
        conv = "conv2" if block_type == "Bottleneck" else "conv1"

        if self._output_stride in {8, 16}:
            self._level4.module[0].downsample[0].stride = (1, 1)
            getattr(self._level4.module[0], conv).stride = (1, 1)

        if self._output_stride == 8:
            self._level3.module[0].downsample[0].stride = (1, 1)
            getattr(self._level3.module[0], conv).dilation = (2, 2)
            getattr(self._level3.module[0], conv).padding = (2, 2)
            getattr(self._level3.module[0], conv).stride = (1, 1)
            getattr(self._level4.module[0], conv).dilation = (4, 4)
            getattr(self._level4.module[0], conv).padding = (4, 4)

        if self._output_stride == 16:
            getattr(self._level4.module[0], conv).dilation = (2, 2)
            getattr(self._level4.module[0], conv).padding = (2, 2)


# Ignores dummy tensor in checkpointed modules
class Ignore2ndArg(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(
        self, x: torch.Tensor, dummy_arg: torch.Tensor = None
    ) -> torch.Tensor:
        return self.module(x)


"""
    Returns the specified variant of ResNet, optionally loaded with Imagenet
        pretrained weights
"""


def build(variant: str = "50", imagenet: bool = False) -> nn.Module:
    weights = None
    if imagenet:
        weights = "IMAGENET1K_V1" if variant == "18" else "IMAGENET1K_V2"
        logger.info("Initialising with imagenet pretrained weights")
    if variant in model_dict.keys():
        model = model_dict[variant](weights=weights)
    else:
        print("Invalid or unimplemented ResNet Variant")
        print("Valid options are: '18', '32', '50', '101', '152'")
    # Convert to Encoder-Decoder integrable version
    return model


model_dict = {
    "18": torchvision.models.resnet18,
    "34": torchvision.models.resnet34,
    "50": torchvision.models.resnet50,
    "101": torchvision.models.resnet101,
    "152": torchvision.models.resnet152,
}
