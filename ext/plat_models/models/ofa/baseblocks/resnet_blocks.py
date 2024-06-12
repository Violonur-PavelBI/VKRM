import copy
from typing import OrderedDict, Union

import torch

from ... import core as c

from ...baseblocks.mobilenetv2_basic import _make_divisible as make_divisible

from ..primitives.func import val2list, get_net_device, adjust_bn_according_to_idx
from ..primitives.static import (
    build_activation,
    MyModule,
    MyNetwork,
    IdentityLayer,
)
from ..baseblocks.conv_blocks import ConvBlock, DynamicConvBlock
from ..primitives.utils import LayerRegistry, set_layer_from_config


@LayerRegistry.registry
class ResNetBottleneckBlock(MyModule):
    """
    Базовый блок для ResNet сетей:
        - pointwise conv + conv + pointwise conv
        - residual = Id | Conv | AvgPool_Conv | MaxPool_Conv

    Промежуточное количество каналов устанавливается в зависимости от параметра `expand_ratio`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=0.25,
        mid_channels=None,
        act_func="relu",
        downsample_mode="avgpool_conv",
    ):
        super(ResNetBottleneckBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func

        self.downsample_mode = downsample_mode

        if self.mid_channels is None:
            feature_dim = round(self.out_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        feature_dim = make_divisible(feature_dim, MyNetwork.CHANNEL_DIVISIBLE)
        self.mid_channels = feature_dim

        self.conv1: ConvBlock = ConvBlock(
            in_channels=self.in_channels,
            out_channels=feature_dim,
            kernel_size=1,
            stride=1,
            bias=False,
            use_bn=True,
            act_func=self.act_func,
        )
        self.conv2: ConvBlock = ConvBlock(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            use_bn=True,
            act_func=self.act_func,
        )
        self.conv3: ConvBlock = ConvBlock(
            in_channels=feature_dim,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            use_bn=True,
            act_func="none",
        )

        if stride == 1 and in_channels == out_channels:
            self.downsample = IdentityLayer()
        elif self.downsample_mode == "conv":
            self.downsample = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                use_bn=True,
                act_func="none",
            )
        elif self.downsample_mode in ("avgpool_conv", "maxpool_conv"):
            if stride == 1:
                downsampler = IdentityLayer()
            elif self.downsample_mode == "avgpool_conv":
                downsampler = c.AvgPool2d(
                    kernel_size=stride, stride=stride, padding=0, ceil_mode=True
                )
            else:
                downsampler = c.MaxPool2d(
                    kernel_size=stride, stride=stride, padding=0, ceil_mode=True
                )
            self.downsample = c.Sequential(
                OrderedDict(
                    [
                        ("pool", downsampler),
                        (
                            "conv",
                            ConvBlock(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                bias=False,
                                use_bn=True,
                                act_func="none",
                            ),
                        ),
                    ]
                )
            )
        else:
            raise NotImplementedError

        self.final_act = build_activation(self.act_func, inplace=True)

    def forward(self, x):
        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + residual
        x = self.final_act(x)
        return x

    def update_state(self) -> None:
        """Обновляет внутренние состояния
        на основе параметров внутренних примитивов"""

        self.in_channels = self.conv1.conv.in_channels
        self.out_channels = self.conv3.conv.out_channels
        self.kernel_size = self.conv2.conv.kernel_size[0]
        self.stride = self.conv2.conv.stride[0]
        self.expand_ratio = self.mid_channels / self.out_channels
        self.mid_channels = self.conv2.conv.out_channels

    @property
    def config(self):
        self.update_state()
        return {
            "name": self.__class__.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.expand_ratio,
            "mid_channels": self.mid_channels,
            "act_func": self.act_func,
            "downsample_mode": self.downsample_mode,
        }


class DynamicResNetBottleneckBlock(MyModule):
    """
    Базовый динамический блок для ResNet сетей:
        - pointwise conv + conv + pointwise conv
        - residual = Id | Conv | AvgPool_Conv | MaxPool_Conv

    Промежуточное количество каналов устанавливается в зависимости от параметра `expand_ratio`.

    Подбираемые параметры: `expand_ratio` и `active_out_channels` (через `active_width_mult`).
    """

    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        expand_ratio_list=0.25,
        kernel_size=3,
        stride=1,
        act_func="relu",
        downsample_mode="avgpool_conv",
    ):
        super(DynamicResNetBottleneckBlock, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.expand_ratio_list = val2list(expand_ratio_list)

        self.kernel_size = kernel_size
        self.stride = stride
        self.act_func = act_func
        self.downsample_mode = downsample_mode

        self.mid_channels_list = sorted(
            [
                make_divisible(round(out_ch * expand), MyNetwork.CHANNEL_DIVISIBLE)
                for out_ch in self.out_channel_list
                for expand in self.expand_ratio_list
            ]
        )

        self.conv1: DynamicConvBlock = DynamicConvBlock(
            in_channel_list=self.in_channel_list,
            out_channel_list=self.mid_channels_list,
            kernel_size_list=[1],
            stride=1,
            bias=False,
            use_bn=True,
            act_func=self.act_func,
        )
        self.conv2: DynamicConvBlock = DynamicConvBlock(
            in_channel_list=self.mid_channels_list,
            out_channel_list=self.mid_channels_list,
            kernel_size_list=[kernel_size],
            stride=stride,
            bias=False,
            use_bn=True,
            act_func=self.act_func,
        )
        self.conv3: DynamicConvBlock = DynamicConvBlock(
            in_channel_list=self.mid_channels_list,
            out_channel_list=self.out_channel_list,
            kernel_size_list=[1],
            stride=1,
            bias=False,
            use_bn=True,
            act_func="none",
        )

        if self.stride == 1 and self.in_channel_list == self.out_channel_list:
            self.downsample = IdentityLayer()
        elif self.downsample_mode == "conv":
            self.downsample: DynamicConvBlock = DynamicConvBlock(
                in_channel_list=self.in_channel_list,
                out_channel_list=self.out_channel_list,
                kernel_size_list=[1],
                stride=stride,
                bias=False,
                use_bn=True,
                act_func="none",
            )

        elif self.downsample_mode in ("avgpool_conv", "maxpool_conv"):
            if stride == 1:
                downsampler = IdentityLayer()
            elif self.downsample_mode == "avgpool_conv":
                downsampler = c.AvgPool2d(
                    kernel_size=stride, stride=stride, padding=0, ceil_mode=True
                )
            else:
                downsampler = c.MaxPool2d(
                    kernel_size=stride, stride=stride, padding=0, ceil_mode=True
                )
            self.downsample = c.Sequential(
                OrderedDict(
                    [
                        ("pool", downsampler),
                        (
                            "conv",
                            DynamicConvBlock(
                                in_channel_list=self.in_channel_list,
                                out_channel_list=self.out_channel_list,
                                kernel_size_list=[1],
                                stride=1,
                                bias=False,
                                use_bn=True,
                                act_func="none",
                            ),
                        ),
                    ]
                )
            )
        else:
            raise NotImplementedError

        self.final_act = build_activation(self.act_func, inplace=True)

        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        feature_dim = self.active_middle_channels

        self.conv1.active_out_channel = feature_dim
        self.conv2.active_out_channel = feature_dim
        self.conv3.active_out_channel = self.active_out_channel
        if isinstance(self.downsample, DynamicConvBlock):
            self.downsample.active_out_channel = self.active_out_channel
        elif isinstance(self.downsample, c.Sequential):
            self.downsample.conv.active_out_channel = self.active_out_channel

        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + residual
        x = self.final_act(x)
        return x

    @property
    def config(self):
        return {
            "name": self.__class__.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "expand_ratio_list": self.expand_ratio_list,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "act_func": self.act_func,
            "downsample_mode": self.downsample_mode,
        }

    ############################################################################################

    @property
    def in_channels(self):
        return max(self.in_channel_list)

    @property
    def out_channels(self):
        return max(self.out_channel_list)

    @property
    def active_middle_channels(self):
        feature_dim = round(self.active_out_channel * self.active_expand_ratio)
        feature_dim = make_divisible(feature_dim, MyNetwork.CHANNEL_DIVISIBLE)
        return feature_dim

    ############################################################################################

    def set_active_subnet(self, active_expand_ratio, active_width_mult):
        self.active_expand_ratio = active_expand_ratio
        self.active_out_channel = self.out_channel_list[active_width_mult]

        # TODO: устанавливать self.conv1.active_in_channel здесь, а не в forward?
        self.conv1.active_out_channel = self.active_middle_channels

        self.conv2.active_in_channel = self.active_middle_channels
        self.conv2.active_out_channel = self.active_middle_channels

        self.conv3.active_in_channel = self.active_middle_channels
        self.conv3.active_out_channel = self.active_out_channel

        if isinstance(self.downsample, DynamicConvBlock):
            self.downsample.active_out_channel = self.active_out_channel
        elif isinstance(self.downsample, c.Sequential):
            self.downsample.conv.active_out_channel = self.active_out_channel

    def get_active_subnet(self, in_channel):
        sub_layer: ResNetBottleneckBlock = set_layer_from_config(
            self.get_active_subnet_config(in_channel)
        )
        sub_layer = sub_layer.to(get_net_device(self))

        sub_layer.conv1 = self.conv1.get_active_subnet()
        sub_layer.conv2 = self.conv2.get_active_subnet()
        sub_layer.conv3 = self.conv3.get_active_subnet()

        if isinstance(self.downsample, DynamicConvBlock):
            sub_layer.downsample = self.downsample.get_active_subnet()
        elif isinstance(self.downsample, c.Sequential):
            sub_layer.downsample.conv = self.downsample.conv.get_active_subnet()

        return sub_layer

    def get_active_subnet_config(self, in_channel):
        self.conv1.active_in_channel = in_channel
        if isinstance(self.downsample, DynamicConvBlock):
            self.downsample.active_in_channel = in_channel
        elif isinstance(self.downsample, c.Sequential):
            self.downsample.conv.active_in_channel = in_channel

        return {
            "name": ResNetBottleneckBlock.__name__,
            "in_channels": in_channel,
            "out_channels": self.active_out_channel,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "expand_ratio": self.active_expand_ratio,
            "mid_channels": self.active_middle_channels,
            "act_func": self.act_func,
            "downsample_mode": self.downsample_mode,
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        # conv3 -> conv2
        importance = torch.sum(
            torch.abs(self.conv3.conv.conv.weight.data), dim=(0, 2, 3)
        )
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(
                    round(max(self.out_channel_list) * expand),
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
        self.conv3.conv.conv.weight.data = torch.index_select(
            self.conv3.conv.conv.weight.data, 1, sorted_idx
        )
        adjust_bn_according_to_idx(self.conv2.bn.bn, sorted_idx)
        self.conv2.conv.conv.weight.data = torch.index_select(
            self.conv2.conv.conv.weight.data, 0, sorted_idx
        )

        # conv2 -> conv1
        importance = torch.sum(
            torch.abs(self.conv2.conv.conv.weight.data), dim=(0, 2, 3)
        )
        if expand_ratio_stage > 0:
            sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
            sorted_expand_list.sort(reverse=True)
            target_width_list = [
                make_divisible(
                    round(max(self.out_channel_list) * expand),
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

        self.conv2.conv.conv.weight.data = torch.index_select(
            self.conv2.conv.conv.weight.data, 1, sorted_idx
        )
        adjust_bn_according_to_idx(self.conv1.bn.bn, sorted_idx)
        self.conv1.conv.conv.weight.data = torch.index_select(
            self.conv1.conv.conv.weight.data, 0, sorted_idx
        )

        return None
