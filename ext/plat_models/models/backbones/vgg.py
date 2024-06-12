import torch

from typing import Any
from ..core.module import Module
from ..core.abstract import BACKBONES
from ..core import BatchNorm2d, Sequential, Conv2d, MaxPool2d, ReLU


cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
    ],
}


class VGG(Module):
    def __init__(
        self,
        cfg,
        in_channels: int = 3,
    ):
        super().__init__()
        self.downsample_factor = 1
        self.features = self._make_layers(in_channels, cfg)
        # self.classifier = Linear(512, num_cls)

    def _make_layers(self, in_channels, cfg):
        self.input_channels = in_channels
        self.output_channels = 1
        layers = []
        for x in cfg:
            if x == "M":
                layers += [MaxPool2d(kernel_size=3, stride=2, padding=1)]
                self.downsample_factor *= 2
            else:
                layers += [
                    Conv2d(in_channels, x, kernel_size=3, padding=1),
                    BatchNorm2d(x),
                    ReLU(inplace=True),
                ]
                in_channels = x
                self.output_channels = x
        return Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        return out


def VGG11():
    return VGG(cfg["VGG11"])


def VGG13():
    return VGG(cfg["VGG13"])


def VGG16():
    return VGG(cfg["VGG16"])


def VGG19():
    return VGG(cfg["VGG19"])


def b_vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG11:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = VGG11(**kwargs)
    # FIXME: add loading of pretrained weights,
    # either from our acessible storage (as described in format, platform `workdirs/pmodels`), or else.

    if pretrained:
        path = "/app/pmodels/vgg/vgg11.pth"
        backbone.load_state_dict(torch.load(path))

    return backbone


def b_vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG13:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = VGG13(**kwargs)
    # FIXME: add loading of pretrained weights,
    # either from our acessible storage (as described in format, platform `workdirs/pmodels`), or else.

    if pretrained:
        path = "/app/pmodels/vgg/vgg13.pth"
        backbone.load_state_dict(torch.load(path))

    return backbone


def b_vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG16:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = VGG16(**kwargs)
    # FIXME: add loading of pretrained weights,
    # either from our acessible storage (as described in format, platform `workdirs/pmodels`), or else.

    if pretrained:
        path = "/app/pmodels/vgg/vgg16.pth"
        backbone.load_state_dict(torch.load(path))

    return backbone


def b_vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG19:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = VGG19(**kwargs)
    # FIXME: add loading of pretrained weights,
    # either from our acessible storage (as described in format, platform `workdirs/pmodels`), or else.

    if pretrained:
        path = "/app/pmodels/vgg/vgg19.pth"
        backbone.load_state_dict(torch.load(path))

    return backbone


BACKBONES.register_module("b_vgg11", module=b_vgg11)
BACKBONES.register_module("b_vgg13", module=b_vgg13)
BACKBONES.register_module("b_vgg16", module=b_vgg16)
BACKBONES.register_module("b_vgg19", module=b_vgg19)
