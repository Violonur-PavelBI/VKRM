from ...backbones.squeezenet import _squeezenet_backbone
from .attacher import ClassificationNetwork
from ...core.abstract import CLASMODELS, AbsClass2DModel
from typing import Any


def _squeezenet(
    num_classes=1000, **kwargs: Any
) -> AbsClass2DModel:
    """

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = _squeezenet_backbone(**kwargs)
    model = ClassificationNetwork(backbone, num_classes=num_classes, dropout_p=0.2)
    return model

def squeezenet_1_0(
    pretrained: bool = False, progress: bool = True, num_classes=1000, **kwargs: Any
) -> ClassificationNetwork:
    """

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet(
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        version = "1_0",
        **kwargs,
    )

def squeezenet_1_1(
    pretrained: bool = False, progress: bool = True, num_classes=1000, **kwargs: Any
) -> ClassificationNetwork:
    """

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet(
        pretrained = pretrained,
        progress = progress,
        num_classes=num_classes,
        version = "1_1",
        **kwargs,
    )


CLASMODELS.register_module("squeezenet_1_0", module=squeezenet_1_0)
CLASMODELS.register_module("squeezenet_1_1", module=squeezenet_1_1)
