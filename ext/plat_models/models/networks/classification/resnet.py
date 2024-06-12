from ...core.abstract import AbsClass2DModel, CLASMODELS
from ...baseblocks.resnet_basic import Bottleneck, BasicBlock
from ...backbones.resnet import _resnet_backbone
from .attacher import ClassificationNetwork

from ...core import Module
from typing import Union, Tuple

# FIXME: add loading pretrained weights /storage/3010/Repin/pmodels (storage: storage_labs on all clusters except 1.m),
# but better will be when using environment variable


def _resnet(
    arch,
    block: Union[BasicBlock, Bottleneck],
    layers: Tuple[int, int, int, int],
    pretrained: bool,
    progress: bool,
    num_classes: int,
    **kwargs
) -> AbsClass2DModel:
    # class _ResNet(ResNet):
    # downsample_factor = 32

    kwargs.update({"pretrained": pretrained})

    backbone = _resnet_backbone(arch, block, layers, progress=progress, **kwargs)
    model = ClassificationNetwork(backbone, num_classes=num_classes, dropout_p=0)
    return model


def resnet18(num_classes: int = 1000, pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        num_classes=num_classes,
        **kwargs
    )


def resnet34(num_classes: int = 1000, pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet34",
        BasicBlock,
        [3, 4, 6, 3],
        pretrained,
        progress,
        num_classes=num_classes,
        **kwargs
    )


def resnet50(num_classes: int = 1000, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        num_classes=num_classes,
        **kwargs
    )


def resnet101(num_classes: int = 1000, pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet101",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        num_classes=num_classes,
        **kwargs
    )


def resnet152(num_classes: int = 1000, pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet152",
        Bottleneck,
        [3, 8, 36, 3],
        pretrained,
        progress,
        num_classes=num_classes,
        **kwargs
    )


def resnext50_32x4d(num_classes: int = 1000, pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        num_classes=num_classes,
        **kwargs
    )


def resnext101_32x8d(
    num_classes: int = 1000, pretrained=False, progress=True, **kwargs
):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        num_classes=num_classes,
        **kwargs
    )


def wide_resnet50_2(num_classes: int = 1000, pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        num_classes=num_classes,
        **kwargs
    )


def wide_resnet101_2(
    num_classes: int = 1000, pretrained=False, progress=True, **kwargs
):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        num_classes=num_classes,
        **kwargs
    )


CLASMODELS.register_module("resnet18", module=resnet18)
CLASMODELS.register_module("resnet34", module=resnet34)
CLASMODELS.register_module("resnet50", module=resnet50)
CLASMODELS.register_module("resnet101", module=resnet101)
CLASMODELS.register_module("resnet152", module=resnet152)
CLASMODELS.register_module("resnext50_32x4d", module=resnext50_32x4d)
CLASMODELS.register_module("resnext101_32x8d", module=resnext101_32x8d)
CLASMODELS.register_module("wide_resnet50_2", module=wide_resnet50_2)
CLASMODELS.register_module("wide_resnet101_2", module=wide_resnet101_2)
