from typing import Optional, List, Callable, Any

from ..core.container import Sequential

from ..core.module import Module

from ..core.abstract import Backbone, BACKBONES
from ..baseblocks.mobilenetv2_basic import InvertedResidual, _make_divisible, ConvBNReLU
from ..core import (
    torch_init as init,
    Tensor,
    BatchNorm2d,
    Dropout,
    Linear,
    Conv2d,
    GroupNorm,
)
from ..core.functional import adaptive_avg_pool2d, flatten
import torch


class MobileNetV2(Backbone):
    def __init__(
        self,
        num_classes: int = 1000,
        input_channels: int = 3,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., Module]] = None,
        norm_layer: Optional[Callable[..., Module]] = None,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()
        self.downsample_factor = 2
        self.input_channels = input_channels
        self.inner_features = dict()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                "or a 4-element list, got {}".format(inverted_residual_setting)
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.output_channels = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        features: List[Module] = [
            ConvBNReLU(self.input_channels, input_channel, stride=2, norm_layer=norm_layer)
        ]
        # building inverted residual blocks
        num_layer = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            self.downsample_factor *= s
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                        downsample_factor=self.downsample_factor,
                    )
                )
                input_channel = output_channel
                if i == n - 1:
                    self.inner_features[num_layer] = [
                        f"features.{num_layer}",
                        self.downsample_factor,
                        output_channel,
                    ]
                num_layer += 1
        # building last several layers
        features.append(
            ConvBNReLU(
                input_channel,
                self.output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
            )
        )
        self.inner_features[num_layer] = [
            f"features.{num_layer+1}",
            self.downsample_factor,
            self.output_channels,
        ]
        # make it Sequential
        self.features = Sequential(*features)

        # weight initialization
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, (BatchNorm2d, GroupNorm)):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, Linear):
                init.normal_(m.weight, 0, 0.01)
                init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def b_mobilenetv2(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
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

    if pretrained:
        path = "/app/pmodels/mobilenetv2/mobilenetv2.pth"
        backbone.load_state_dict(torch.load(path))

    return backbone


BACKBONES.register_module("b_mobilenetv2", module=b_mobilenetv2)
