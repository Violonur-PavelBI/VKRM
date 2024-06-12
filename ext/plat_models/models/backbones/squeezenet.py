import torch

from typing import Any
from ..core.module import Module
from ..core.container import Sequential
from ..core import (
    torch_init as init,
    Conv2d,
    ReLU,
    MaxPool2d,
)
from ..core.abstract import BACKBONES, Backbone


class Fire(Module):
    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
    ) -> None:
        super().__init__()
        self.input_channels = inplanes
        self.squeeze = Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = ReLU(inplace=True)
        self.expand1x1 = Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = ReLU(inplace=True)
        self.expand3x3 = Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1
        )
        self.expand3x3_activation = ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x)),
            ],
            1,
        )


class SqueezeNet(Backbone):
    def __init__(self, input_channels: int = 3, version: str = "1_0") -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = 512
        self.inner_features = dict()
        if version == "1_0":
            self.features = Sequential(
                Conv2d(self.input_channels, 96, kernel_size=7, stride=2, padding=3),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
            self.inner_features["10"] = [
                f"features.{10}",
                8,
                64,
            ]
            self.inner_features["12"] = [
                f"features.{12}",
                16,
                64,
            ]
        elif version == "1_1":
            self.features = Sequential(
                Conv2d(self.input_channels, 64, kernel_size=3, stride=2),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
            self.inner_features["10"] = [
                f"features.{10}",
                8,
                32,
            ]
            self.inner_features["12"] = [
                f"features.{12}",
                16,
                64,
            ]
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError(
                f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected"
            )

        self.downsample_factor = 16

        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        return x


def _squeezenet_backbone(
    pretrained: bool = False, progress: bool = True, version: str = "1_0", **kwargs: Any
) -> SqueezeNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = SqueezeNet(**kwargs)
    # FIXME: add loading of pretrained weights,
    # either from our acessible storage (as described in format, platform `workdirs/pmodels`), or else.

    # if pretrained:
    #     path = f"/app/pmodels/squeezenet/squeezenet_{version}.pth"
    #     backbone.load_state_dict(torch.load(path))

    return backbone

def b_squeezenet_1_0(
    **kwargs: Any
) -> SqueezeNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = _squeezenet_backbone(version="1_0", **kwargs)

    return backbone

def b_squeezenet_1_1(
    **kwargs: Any
) -> SqueezeNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = _squeezenet_backbone(version="1_1", **kwargs)
    # FIXME: add loading of pretrained weights,
    # either from our acessible storage (as described in format, platform `workdirs/pmodels`), or else.

    # if pretrained:
    #     path = "/app/pmodels/squeezenet/squeezenet_1_1.pth"
    #     backbone.load_state_dict(torch.load(path))

    return backbone


BACKBONES.register_module("b_squeezenet_1_0", module=b_squeezenet_1_0)
BACKBONES.register_module("b_squeezenet_1_1", module=b_squeezenet_1_1)
