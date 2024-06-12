# import torch
# from torch import nn
# from torch import Tensor
# from .utils import load_state_dict_from_url
from typing import Callable, Optional, List

from ..core.container import Sequential

from ..core.module import Module

from ..core.abstract import Node
from ..core import Conv2d, BatchNorm2d, ReLU6
from ..core import Tensor

model_urls = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., Module]] = None,
        activation_layer: Optional[Callable[..., Module]] = None,
        dilation: int = 1,
        # downsample_factor: int = 1
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = BatchNorm2d
        if activation_layer is None:
            activation_layer = ReLU6
        super().__init__(
            Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            activation_layer(inplace=True),
        )
        self.out_channels = out_planes
        # self.output_channels = out_planes
        # self.input_channels = in_planes
        # self.downsample_factor = downsample_factor


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(Node):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., Module]] = None,
        downsample_factor: int = 1,
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.output_channels = inp
        self.input_channels = oup
        self.downsample_factor = downsample_factor
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer)
            )
        layers.extend(
            [
                # dw
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                ),
                # pw-linear
                Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
