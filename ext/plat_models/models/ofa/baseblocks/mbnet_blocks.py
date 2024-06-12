import copy
from typing import Union

import torch

from ...baseblocks.mobilenetv2_basic import _make_divisible as make_divisible

from ..primitives.func import val2list, get_net_device, adjust_bn_according_to_idx
from ..primitives.static import MyModule, MyNetwork, IdentityLayer

from ..primitives.utils import LayerRegistry, set_layer_from_config

from .conv_blocks import ConvBlock, DynamicConvBlock
from .se_blocks import SEModule, DynamicSE


@LayerRegistry.registry
class MBConvBlock(MyModule):
    """
    Базовый блок для MobileNet сетей: pointwise conv + depthwise conv (+ SE) + pointwise conv.

    В случае равенства входных и выходных размеров имеет skip connection.

    Промежуточное количество каналов устанавливается в зависимости от параметра `expand_ratio`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=6,
        mid_channels=None,
        act_func="relu6",
        use_se=False,
        **kwargs
    ):
        super(MBConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        self.inverted_bottleneck: Union[ConvBlock, None]
        # TODO: Refactor изачально тут было self.expand_ratio == 1
        # Возможно это можно вообще выкинуть и всегда делать 3 свёртки
        # Изменение связано с поддержкой прунинга
        if self.expand_ratio is None:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = ConvBlock(
                in_channels=self.in_channels,
                out_channels=feature_dim,
                kernel_size=1,
                stride=1,
                dilation=1,
                bias=False,
                use_bn=True,
                act_func=self.act_func,
            )

        self.depth_conv: ConvBlock = ConvBlock(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=1,
            separable=True,
            bias=False,
            use_bn=True,
            act_func=self.act_func,
        )
        if self.use_se:
            self.se: SEModule = SEModule(feature_dim)

        self.point_linear: ConvBlock = ConvBlock(
            in_channels=feature_dim,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            use_bn=True,
            act_func=None,
        )

        if self.stride == 1 and self.in_channels == self.out_channels:
            self.shortcut = IdentityLayer()
        else:
            self.shortcut = None

    def forward(self, x):
        if self.shortcut:
            residual = self.shortcut(x)

        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.point_linear(x)

        if self.shortcut:
            x = x + residual

        return x

    def update_state(self) -> None:
        """
        Pruning critical support. TODO: delete.

        Обновляет внутренние состояния
        на основе параметров внутренних примитивов.
        """

        if self.inverted_bottleneck:
            self.in_channels = self.inverted_bottleneck.conv.in_channels
        else:
            self.in_channels = self.depth_conv.conv.in_channels
        self.out_channels = self.point_linear.conv.out_channels
        self.mid_channels = self.depth_conv.conv.in_channels
        self.expand_ratio = self.mid_channels / self.in_channels
        self.kernel_size = self.depth_conv.conv.kernel_size[0]
        self.stride = self.depth_conv.conv.stride[0]
        self.padding = self.depth_conv.conv.padding[0]

    @property
    def config(self):
        self.update_state()
        return {
            "name": MBConvBlock.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
            "mid_channels": self.mid_channels,
            "act_func": self.act_func,
            "use_se": self.use_se,
        }


class DynamicMBConvBlock(MyModule):
    """
    Базовый динамический блок для MobileNet сетей: pointwise conv + depthwise conv (+ SE) + pointwise conv.

    В случае равенства входных и выходных размеров имеет skip connection.

    Промежуточное количество каналов устанавливается в зависимости от параметра `expand_ratio`.

    Подбираемые параметры: `expand_ratio` и `kernel_size` поканальной свёртки.
    """

    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size_list=3,
        expand_ratio_list=6,
        stride=1,
        act_func="relu6",
        use_se=False,
    ):
        super(DynamicMBConvBlock, self).__init__()

        self.in_channel_list = sorted(in_channel_list)
        self.out_channel_list = sorted(out_channel_list)

        self.kernel_size_list = val2list(kernel_size_list)
        self.expand_ratio_list = val2list(expand_ratio_list)

        self.mid_channels_list = sorted(
            [
                make_divisible(round(in_ch * expand), MyNetwork.CHANNEL_DIVISIBLE)
                for in_ch in self.in_channel_list
                for expand in self.expand_ratio_list
            ]
        )

        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se

        self.inverted_bottleneck: Union[DynamicConvBlock, None]
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = DynamicConvBlock(
                in_channel_list=self.in_channel_list,
                out_channel_list=self.mid_channels_list,
                kernel_size_list=[1],
                stride=1,
                dilation=1,
                bias=False,
                use_bn=True,
                act_func=self.act_func,
            )

        self.depth_conv: DynamicConvBlock = DynamicConvBlock(
            in_channel_list=self.mid_channels_list,
            out_channel_list=self.mid_channels_list,
            kernel_size_list=self.kernel_size_list,
            stride=self.stride,
            dilation=1,
            separable=True,
            bias=False,
            use_bn=True,
            act_func=self.act_func,
        )
        if self.use_se:
            self.se: DynamicSE = DynamicSE(max(self.mid_channels_list))

        self.point_linear: DynamicConvBlock = DynamicConvBlock(
            in_channel_list=self.mid_channels_list,
            out_channel_list=self.out_channel_list,
            kernel_size_list=[1],
            stride=1,
            dilation=1,
            bias=False,
            use_bn=True,
            act_func=None,
        )

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_in_channel = max(self.in_channel_list)
        self.active_out_channel = max(self.out_channel_list)

        if self.stride == 1 and self.in_channels == self.out_channels:
            self.shortcut = IdentityLayer()
        else:
            self.shortcut = None

    def forward(self, x):
        if self.shortcut:
            residual = self.shortcut(x)

        in_channel = x.size(1)

        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.active_out_channel = make_divisible(
                round(in_channel * self.active_expand_ratio),
                MyNetwork.CHANNEL_DIVISIBLE,
            )
        self.depth_conv.active_kernel_size = self.active_kernel_size
        self.point_linear.active_out_channel = self.active_out_channel

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.point_linear(x)

        if self.shortcut:
            x = x + residual

        return x

    @property
    def config(self):
        return {
            "name": DynamicMBConvBlock.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size_list": self.kernel_size_list,
            "expand_ratio_list": self.expand_ratio_list,
            "stride": self.stride,
            "act_func": self.act_func,
            "use_se": self.use_se,
        }

    ############################################################################################

    @property
    def in_channels(self):
        return self.active_in_channel

    @property
    def out_channels(self):
        return self.active_out_channel

    @property
    def active_middle_channels(self):
        return make_divisible(
            round(self.in_channels * self.active_expand_ratio),
            MyNetwork.CHANNEL_DIVISIBLE,
        )

    ############################################################################################

    def set_active_subnet(self, active_kernel_size, active_expand_ratio, w=None):
        self.active_kernel_size = active_kernel_size
        self.active_expand_ratio = active_expand_ratio

        if self.inverted_bottleneck:
            self.inverted_bottleneck.active_in_channel = self.in_channels
            self.inverted_bottleneck.active_out_channel = self.active_middle_channels

        self.depth_conv.active_in_channel = self.active_middle_channels
        self.depth_conv.active_out_channel = self.active_middle_channels
        self.depth_conv.active_kernel_size = self.active_kernel_size

        self.point_linear.active_in_channel = self.active_middle_channels
        self.point_linear.active_out_channel = self.out_channels

    def get_active_subnet(self):
        sub_layer: MBConvBlock = set_layer_from_config(self.get_active_subnet_config())
        sub_layer = sub_layer.to(get_net_device(self))

        middle_channel = self.active_middle_channels
        if sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck = self.inverted_bottleneck.get_active_subnet()

        sub_layer.depth_conv = self.depth_conv.get_active_subnet()

        if self.use_se:
            sub_layer.se = self.se.get_active_subnet(middle_channel)

        sub_layer.point_linear = self.point_linear.get_active_subnet()

        return sub_layer

    def get_active_subnet_config(self):
        return {
            "name": MBConvBlock.__name__,
            "in_channels": self.active_in_channel,
            "out_channels": self.active_out_channel,
            "kernel_size": self.active_kernel_size,
            "stride": self.stride,
            "expand_ratio": self.active_expand_ratio,
            "mid_channels": self.active_middle_channels,
            "act_func": self.act_func,
            "use_se": self.use_se,
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        importance = torch.sum(
            torch.abs(self.point_linear.conv.conv.weight.data), dim=(0, 2, 3)
        )
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(
                    round(max(self.in_channel_list) * expand),
                    MyNetwork.CHANNEL_DIVISIBLE,
                )
                for expand in sorted_expand_list
            ]

            right = len(importance)
            base = -len(target_width_list) * 1e5
            for i in range(expand_ratio_stage + 1):
                left = target_width_list[i]
                importance[left:right] += base
                base += 1e5
                right = left

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.point_linear.conv.conv.weight.data = torch.index_select(
            self.point_linear.conv.conv.weight.data, 1, sorted_idx
        )

        adjust_bn_according_to_idx(self.depth_conv.bn.bn, sorted_idx)
        self.depth_conv.conv.conv.weight.data = torch.index_select(
            self.depth_conv.conv.conv.weight.data, 0, sorted_idx
        )

        if self.use_se:
            self.se.reorganize_weights(sorted_idx)

        if self.inverted_bottleneck is not None:
            adjust_bn_according_to_idx(self.inverted_bottleneck.bn.bn, sorted_idx)
            self.inverted_bottleneck.conv.conv.weight.data = torch.index_select(
                self.inverted_bottleneck.conv.conv.weight.data,
                0,
                sorted_idx,
            )
            return None
        else:
            return sorted_idx
