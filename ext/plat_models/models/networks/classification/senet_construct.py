from ...backbones import b_se_resnext101_32x4d, b_se_resnext50_32x4d
from ...core.abstract import CLASMODELS
from .attacher import ClassificationNetwork


def se_resnext101_32x4d(num_classes: int, pretrained=False):
    return ClassificationNetwork(
        b_se_resnext101_32x4d(pretrained=pretrained),
        num_classes,
        0,
    )


def se_resnext50_32x4d(num_classes: int, pretrained=False):
    return ClassificationNetwork(
        b_se_resnext50_32x4d(pretrained=pretrained),
        num_classes,
        0,
    )


CLASMODELS.register_module("se_resnext101_32x4d", module=se_resnext101_32x4d)
CLASMODELS.register_module("se_resnext50_32x4d", module=se_resnext50_32x4d)
__all__ = ["se_resnext50_32x4d", "se_resnext101_32x4d"]
