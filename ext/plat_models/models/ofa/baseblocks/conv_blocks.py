from typing import List, Union

from ... import core as c

from ..primitives.func import val2list, get_same_padding, get_net_device, copy_bn
from ..primitives.static import MyModule, MyNetwork, build_activation
from ..primitives.dynamic import DynamicConv2d, DynamicBatchNorm2d
from ..primitives.utils import LayerRegistry, set_layer_from_config


@LayerRegistry.registry
class ConvBlock(MyModule):
    """
    Блок из свёртки, батч нормализации и активации.
    Свёртка может быть раздельной по каналам.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        separable: bool = False,
        bias: bool = False,
        use_bn: bool = True,
        act_func: str = "relu",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.separable = separable
        self.groups = self.in_channels if self.separable else 1
        self.bias = bias
        self.use_bn = use_bn
        self.act_func = act_func

        padding = get_same_padding(self.kernel_size)
        self.conv = c.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
        )

        if self.use_bn:
            self.bn = c.BatchNorm2d(out_channels)

        # inplace = self.use_bn ???
        self.act = build_activation(self.act_func)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x

    def update_state(self) -> None:
        """
        Pruning critical support. TODO: delete.

        Обновляет внутренние состояния
        на основе параметров внутренних примитивов.
        """

        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size[0]
        self.stride = self.conv.stride[0]
        self.dilation = self.conv.dilation[0]
        self.groups = self.in_channels if self.separable else 1

    @property
    def config(self):
        self.update_state()
        return {
            "name": ConvBlock.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "separable": self.separable,
            "bias": self.bias,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
        }


class DynamicConvBlock(MyNetwork):
    """
    Динамический блок из свёртки, батч нормализации и активации.
    Свёртка может быть раздельной по каналам.
    """

    def __init__(
        self,
        in_channel_list: List[int],
        out_channel_list: List[int],
        kernel_size_list: List[int] = [3],
        stride: int = 1,
        dilation: int = 1,
        separable: bool = False,
        bias: bool = False,
        use_bn: bool = True,
        act_func: str = "relu6",
    ):
        super(DynamicConvBlock, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.kernel_size_list = val2list(kernel_size_list)
        self.active_in_channel = max(self.in_channel_list)
        self.active_out_channel = max(self.out_channel_list)
        self.active_kernel_size = max(self.kernel_size_list)
        self.stride = stride
        self.dilation = dilation
        self.separable = separable
        self.use_bn = use_bn
        self.act_func = act_func
        self.bias = bias

        self.conv = DynamicConv2d(
            max_in_channels=self.active_in_channel,
            max_out_channels=self.active_out_channel,
            kernel_size_list=self.kernel_size_list,
            stride=self.stride,
            dilation=self.dilation,
            separable=self.separable,
            bias=self.bias,
        )
        if self.use_bn:
            self.bn = DynamicBatchNorm2d(self.active_out_channel)
        self.act = build_activation(self.act_func)

    def forward(self, x: c.Tensor):
        self.active_in_channel = x.size(1)
        self.conv.active_out_channel = self.active_out_channel
        self.conv.active_kernel_size = self.active_kernel_size

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x

    @property
    def config(self):
        return {
            "name": DynamicConvBlock.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size_list": self.kernel_size_list,
            "stride": self.stride,
            "dilation": self.dilation,
            "separable": self.separable,
            "bias": self.bias,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
        }

    def get_active_subnet(self):
        sub_layer: ConvBlock = set_layer_from_config(self.get_active_subnet_config())
        sub_layer = sub_layer.to(get_net_device(self))

        sub_layer.conv.weight.data.copy_(
            self.conv.get_active_filter(
                self.active_out_channel, self.active_in_channel, self.active_kernel_size
            ).data
        )
        if self.bias:
            sub_layer.conv.bias.data.copy_(
                self.conv.conv.bias[: self.active_out_channel].data
            )
        if self.use_bn:
            copy_bn(sub_layer.bn, self.bn.bn)

        return sub_layer

    def get_active_subnet_config(self):
        return {
            "name": ConvBlock.__name__,
            "in_channels": self.active_in_channel,
            "out_channels": self.active_out_channel,
            "kernel_size": self.active_kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "separable": self.separable,
            "bias": self.bias,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
        }
