from ...backbones import VGG11, VGG13, VGG16, VGG19
from .attacher import ClassificationNetwork
from ...core.abstract import CLASMODELS
from typing import Any


def vgg11(
    pretrained: bool = False, progress: bool = True, in_channels = 3, num_classes=1000, **kwargs: Any
) -> VGG11:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = VGG11(**kwargs)
    # FIXME: add loading of pretrained weights,
    # either from our acessible storage (as described in format, platform `workdirs/pmodels`), or else.

    model = ClassificationNetwork(backbone, num_classes=num_classes, dropout_p=0.2)
    return model

def vgg13(
    pretrained: bool = False, progress: bool = True, in_channels = 3, num_classes=1000, **kwargs: Any
) -> VGG13:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = VGG13(**kwargs)
    # FIXME: add loading of pretrained weights,
    # either from our acessible storage (as described in format, platform `workdirs/pmodels`), or else.

    model = ClassificationNetwork(backbone, num_classes=num_classes, dropout_p=0.2)
    return model

def vgg16(
    pretrained: bool = False, progress: bool = True, in_channels = 3, num_classes=1000, **kwargs: Any
) -> VGG16:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = VGG16(**kwargs)
    # FIXME: add loading of pretrained weights,
    # either from our acessible storage (as described in format, platform `workdirs/pmodels`), or else.

    model = ClassificationNetwork(backbone, num_classes=num_classes, dropout_p=0.2)
    return model

def vgg19(
    pretrained: bool = False, progress: bool = True, in_channels = 3, num_classes=1000, **kwargs: Any
) -> VGG19:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = VGG19(**kwargs)
    # FIXME: add loading of pretrained weights,
    # either from our acessible storage (as described in format, platform `workdirs/pmodels`), or else.

    model = ClassificationNetwork(backbone, num_classes=num_classes, dropout_p=0.2)
    return model

CLASMODELS.register_module("vgg11", module=vgg11)
CLASMODELS.register_module("vgg13", module=vgg13)
CLASMODELS.register_module("vgg16", module=vgg16)
CLASMODELS.register_module("vgg19", module=vgg19)