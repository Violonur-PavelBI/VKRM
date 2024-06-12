import copy
from typing import List, Union

from torch import Tensor


from .... import core as c
from ....baseblocks.mobilenetv2_basic import _make_divisible as make_divisible

from ...abstract.ofa_abstract import BasicModule, DynamicModule
from ...baseblocks.conv_blocks import ConvBlock, DynamicConvBlock


class ZeroSegmentationHead(BasicModule):
    upsample_modes_list = ["nearest", "bilinear", "bicubic"]

    def __init__(
        self,
        in_channels: List[int],
        n_classes: int,
        use_bias: bool = False,
        upsample_mode: str = "nearest",
        upsample_factor: int = 2,
        final_conv_kernel: int = 1,
        act: None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = [make_divisible(in_channels[0], self.CHANNEL_DIVISIBLE)]
        self.n_classes = n_classes
        self.use_bias = use_bias
        self.upsample_mode = upsample_mode.lower()
        if self.upsample_mode not in self.upsample_modes_list:
            raise ValueError(f"{upsample_mode = } not in {self.upsample_modes_list = }")
        self.upsample_factor = upsample_factor
        self.final_conv_kernel_size = final_conv_kernel

        self.up = c.Upsample(mode=self.upsample_mode, scale_factor=self.upsample_factor)
        self.conv = ConvBlock(
            in_channels=self.in_channels[0],
            out_channels=self.n_classes,
            kernel_size=self.final_conv_kernel_size,
            bias=self.use_bias,
            use_bn=False,
            act_func=None,
        )
        self.act = act

    @property
    def config(self):
        return {
            "name": self.__class__.__name__,
            "in_channels": self.in_channels,
            "n_classes": self.n_classes,
            "use_bias": self.use_bias,
            "upsample_mode": self.upsample_mode,
            "upsample_factor": self.upsample_factor,
            "final_conv_kernel": self.final_conv_kernel_size,
        }

    @classmethod
    def build_from_config(cls, config: dict):
        return cls(**config)

    def forward(self, input_features: List[Tensor]) -> Tensor:
        output = self.conv(input_features[0])
        output = self.up(output)
        if self.act:
            output = self.act(output)
        return output


class DynamicZeroSegmentationHead(ZeroSegmentationHead, DynamicModule):
    SAMPLE_MODULE_CLS = ZeroSegmentationHead

    def __init__(
        self,
        in_channels: List[int],
        width_list: List[float],
        n_classes: int,
        use_bias: bool = False,
        upsample_mode: str = "nearest",
        upsample_factor: int = 2,
        final_conv_kernel: int = 1,
        act: None = None,
        **kwargs,
    ) -> None:
        super(ZeroSegmentationHead, self).__init__()
        self.in_channels = in_channels[0]
        self.width_list = sorted([round(w, 2) for w in width_list])
        self.n_classes = n_classes
        self.use_bias = use_bias
        self.upsample_mode = upsample_mode.lower()
        if self.upsample_mode not in self.upsample_modes_list:
            raise ValueError(f"{upsample_mode = } not in {self.upsample_modes_list = }")
        self.upsample_factor = upsample_factor
        self.final_conv_kernel_size = final_conv_kernel

        in_channels_combinations = [
            make_divisible(w * self.in_channels, self.CHANNEL_DIVISIBLE)
            for w in self.width_list
        ]

        self.up = c.Upsample(mode=self.upsample_mode, scale_factor=self.upsample_factor)
        self.conv = DynamicConvBlock(
            in_channel_list=in_channels_combinations,
            out_channel_list=[self.n_classes],
            kernel_size_list=[self.final_conv_kernel_size],
            bias=self.use_bias,
            use_bn=False,
            act_func=None,
        )
        self.act = act

    @property
    def config(self):
        config = super().config
        config["in_channels"] = [self.in_channels]
        config["width_list"] = self.width_list
        return config

    def set_active_subnet(self, active_in_channels: List[int]) -> None:
        self.conv.active_in_channel = active_in_channels[0]

    def set_max_net(self) -> None:
        w = max(self.width_list)
        active_in_channels = [
            make_divisible(w * self.in_channels, self.CHANNEL_DIVISIBLE)
        ]
        self.set_active_subnet(active_in_channels=active_in_channels)

    def set_min_net(self) -> None:
        w = min(self.width_list)
        active_in_channels = [
            make_divisible(w * self.in_channels, self.CHANNEL_DIVISIBLE)
        ]
        self.set_active_subnet(active_in_channels=active_in_channels)

    def get_active_subnet(self) -> ZeroSegmentationHead:
        subnet_config = self.get_active_subnet_config()
        subnet = ZeroSegmentationHead.build_from_config(subnet_config)

        subnet.up = copy.deepcopy(self.up)
        subnet.conv = self.conv.get_active_subnet()

        return subnet

    def get_active_subnet_config(self):
        subnet_config = self.config
        subnet_config.pop("width_list")
        subnet_config["name"] = ZeroSegmentationHead.__name__
        subnet_config["in_channels"] = [self.conv.active_in_channel]
        return subnet_config
