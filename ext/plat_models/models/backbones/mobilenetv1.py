from typing import Any
from ..core.abstract import Backbone, BACKBONES
from ..core import (
    BatchNorm2d,
    Sequential,
    Conv2d,
    ReLU
)
from ..core.functional import adaptive_avg_pool2d, flatten


class MobileNetV1(Backbone):
    def __init__(
            self, 
            input_channels: int = 3, 
            num_classes: int = 1000
            ):
        super(MobileNetV1, self).__init__()
        self.downsample_factor = 1
        self.inner_features = dict()
        self.counter_layers = 0
        def conv_bn(self, inp, oup, stride):
            self.input_channels = inp
            self.downsample_factor *= stride
            self.output_channels = oup
            self.inner_features[self.counter_layers] = [
                f"{self.counter_layers}",
                self.downsample_factor,
                oup,
            ]
            self.counter_layers += 1
            return Sequential(
                Conv2d(inp, oup, 3, stride, 1, bias=False),
                BatchNorm2d(oup),
                ReLU(inplace=True)
                )

        def conv_dw(self, inp, oup, stride):
            self.input_channels = inp
            self.downsample_factor *= stride
            self.output_channels = oup
            self.inner_features[self.counter_layers] = [
                f"model.{self.counter_layers}",
                self.downsample_factor,
                oup,
            ]
            self.counter_layers += 1
            return Sequential(
                # dw
                Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                BatchNorm2d(inp),
                ReLU(inplace=True),

                # pw
                Conv2d(inp, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
                ReLU(inplace=True),
                )

        self.model = Sequential(
            conv_bn(self, input_channels, 32, 2),
            conv_dw(self, 32, 64, 1),
            conv_dw(self, 64, 128, 2),
            conv_dw(self, 128, 128, 1),
            conv_dw(self, 128, 256, 2),
            conv_dw(self, 256, 256, 1),
            conv_dw(self, 256, 512, 2),
            conv_dw(self, 512, 512, 1),
            conv_dw(self, 512, 512, 1),
            conv_dw(self, 512, 512, 1),
            conv_dw(self, 512, 512, 1),
            conv_dw(self, 512, 512, 1),
            conv_dw(self, 512, 1024, 2),
            conv_dw(self, 1024, 1024, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def b_mobilenetv1(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> MobileNetV1:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = MobileNetV1(**kwargs)
    # FIXME: add loading of pretrained weights,
    # either from our acessible storage (as described in format, platform `workdirs/pmodels`), or else.

    # if pretrained:
    #     path = "/app/pmodels/mobilenetv1/mobilenetv1.pth"
    #     backbone.load_state_dict(torch.load(path))

    return backbone

BACKBONES.register_module("b_mobilenetv1", module=b_mobilenetv1)