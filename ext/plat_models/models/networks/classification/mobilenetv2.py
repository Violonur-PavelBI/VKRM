from ...backbones import MobileNetV2
from .attacher import ClassificationNetwork
from ...core.abstract import CLASMODELS
from typing import Any


def mobilenet_v2(
    pretrained: bool = False, progress: bool = True, num_classes=1000, **kwargs: Any
) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = MobileNetV2(**kwargs)
    # FIXME: add loading of pretrained weights,
    # either from our acessible storage (as described in format, platform `workdirs/pmodels`), or else.

    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    model = ClassificationNetwork(backbone, num_classes=num_classes, dropout_p=0.2)
    return model


CLASMODELS.register_module("mobilenet_v2", module=mobilenet_v2)
