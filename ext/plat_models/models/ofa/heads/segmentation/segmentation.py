import copy
from typing import List, Union

from torch import Tensor


from .... import core as c
from ....baseblocks.mobilenetv2_basic import _make_divisible as make_divisible

from ...abstract.ofa_abstract import BasicModule, DynamicModule
from ...baseblocks.conv_blocks import ConvBlock, DynamicConvBlock
from ...baseblocks.merge_blocks import MergeBlock


class SegmentationHead(BasicModule):
    upsample_modes_list = [
        "deconv",
        "nearest",
        "bilinear",
        "bicubic",
    ]

    def __init__(
        self,
        levels: int,
        in_channels: List[int],
        mid_channels: int,
        n_classes: int,
        use_bias: bool = False,
        upsample_mode: str = "nearest",
        upsample_factor: int = 2,
        final_conv_kernel: int = 1,
        merge_policy: str = "sum",
        act: None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.levels = levels
        self.in_channels = [
            make_divisible(in_ch, self.CHANNEL_DIVISIBLE) for in_ch in in_channels
        ]
        self.mid_channels = make_divisible(mid_channels, self.CHANNEL_DIVISIBLE)
        self.n_classes = n_classes
        self.use_bias = use_bias
        self.upsample_mode = upsample_mode.lower()
        if self.upsample_mode not in self.upsample_modes_list:
            raise ValueError(f"{upsample_mode = } not in {self.upsample_modes_list = }")
        self.upsample_factor = upsample_factor
        self.final_conv_kernel_size = final_conv_kernel
        self.merge_policy = merge_policy

        self.convs1x1 = c.ModuleList(
            [
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=self.mid_channels,
                    kernel_size=1,
                    bias=self.use_bias,
                    use_bn=True,
                    act_func="relu",
                )
                for in_ch in self.in_channels
            ]
        )
        self.merge_block = MergeBlock(
            levels=self.levels,
            channels=self.mid_channels,
            use_bias=self.use_bias,
            upsample_mode=self.upsample_mode,
            merge_policy=self.merge_policy,
        )

        if self.upsample_mode == "deconv":
            self.final_up = c.ConvTranspose2d(
                in_channels=self.mid_channels,
                out_channels=self.mid_channels,
                kernel_size=3,
                stride=self.upsample_factor,
                padding=1,
                output_padding=1,
                bias=self.use_bias,
            )
        else:
            self.final_up = c.Upsample(
                mode=self.upsample_mode, scale_factor=self.upsample_factor
            )
        self.final_conv = ConvBlock(
            in_channels=self.mid_channels,
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
            "levels": self.levels,
            "in_channels": self.in_channels,
            "mid_channels": self.mid_channels,
            "n_classes": self.n_classes,
            "use_bias": self.use_bias,
            "upsample_mode": self.upsample_mode,
            "upsample_factor": self.upsample_factor,
            "final_conv_kernel": self.final_conv_kernel_size,
            "merge_policy": self.merge_policy,
        }

    @classmethod
    def build_from_config(cls, config: dict):
        return cls(**config)

    def forward(self, input_features: List[Tensor]) -> Tensor:
        for i in range(self.levels):
            input = input_features[i]
            conv1x1 = self.convs1x1[i]

            input = conv1x1(input)
            input_features[i] = input

        output = self.merge_block(input_features)
        output = self.final_conv(output)
        output = self.final_up(output)
        if self.act:
            output = self.act(output)
        return output


class DynamicSegmentationHead(SegmentationHead, DynamicModule):
    SAMPLE_MODULE_CLS = SegmentationHead

    def __init__(
        self,
        levels: int,
        in_channels: List[int],
        width_list: List[float],
        mid_channels: int,
        n_classes: int,
        use_bias: bool = False,
        upsample_mode: str = "nearest",
        upsample_factor: int = 2,
        final_conv_kernel: int = 1,
        merge_policy: str = "sum",
        act: None = None,
        **kwargs,
    ) -> None:
        super(SegmentationHead, self).__init__()
        self.levels = levels
        self.in_channels = in_channels
        self.width_list = sorted([round(w, 2) for w in width_list])
        self.mid_channels = make_divisible(mid_channels, self.CHANNEL_DIVISIBLE)
        self.n_classes = n_classes
        self.use_bias = use_bias
        self.upsample_mode = upsample_mode.lower()
        if self.upsample_mode not in self.upsample_modes_list:
            raise ValueError(f"{upsample_mode = } not in {self.upsample_modes_list = }")
        self.upsample_factor = upsample_factor
        self.final_conv_kernel_size = final_conv_kernel
        self.merge_policy = merge_policy

        in_channels_combinations = [
            [
                make_divisible(w * base_ch, self.CHANNEL_DIVISIBLE)
                for w in self.width_list
            ]
            for base_ch in self.in_channels
        ]
        self.convs1x1: Union[c.ModuleList, List[DynamicConvBlock]]
        self.convs1x1 = c.ModuleList(
            [
                DynamicConvBlock(
                    in_channel_list=in_ch,
                    out_channel_list=[self.mid_channels],
                    kernel_size_list=[1],
                    bias=self.use_bias,
                    use_bn=True,
                    act_func="relu",
                )
                for in_ch in in_channels_combinations
            ]
        )
        self.merge_block = MergeBlock(
            levels=self.levels,
            channels=self.mid_channels,
            use_bias=self.use_bias,
            upsample_mode=self.upsample_mode,
            merge_policy=self.merge_policy,
        )

        if self.upsample_mode == "deconv":
            self.final_up = c.ConvTranspose2d(
                in_channels=self.mid_channels,
                out_channels=self.mid_channels,
                kernel_size=3,
                stride=self.upsample_factor,
                padding=1,
                output_padding=1,
                bias=self.use_bias,
            )
        else:
            self.final_up = c.Upsample(
                mode=self.upsample_mode, scale_factor=self.upsample_factor
            )
        self.final_conv = ConvBlock(
            in_channels=self.mid_channels,
            out_channels=self.n_classes,
            kernel_size=self.final_conv_kernel_size,
            bias=self.use_bias,
            use_bn=False,
            act_func=None,
        )
        self.act = act

    @property
    def config(self):
        config = super().config
        config["width_list"] = self.width_list
        return config

    def set_active_subnet(self, active_in_channels: List[int]) -> None:
        for conv, active_in_ch in zip(self.convs1x1, active_in_channels):
            conv: DynamicConvBlock
            conv.active_in_channel = active_in_ch

    def set_max_net(self) -> None:
        w = max(self.width_list)
        active_in_channels = [
            make_divisible(w * base_ch, self.CHANNEL_DIVISIBLE)
            for base_ch in self.in_channels
        ]
        self.set_active_subnet(active_in_channels=active_in_channels)

    def set_min_net(self) -> None:
        w = min(self.width_list)
        active_in_channels = [
            make_divisible(w * base_ch, self.CHANNEL_DIVISIBLE)
            for base_ch in self.in_channels
        ]
        self.set_active_subnet(active_in_channels=active_in_channels)

    def get_active_subnet(self) -> SegmentationHead:
        subnet_config = self.get_active_subnet_config()
        subnet = SegmentationHead.build_from_config(subnet_config)

        subnet.convs1x1 = c.ModuleList(
            [conv.get_active_subnet() for conv in self.convs1x1]
        )
        subnet.merge_block = copy.deepcopy(self.merge_block)
        subnet.final_up = copy.deepcopy(self.final_up)
        subnet.final_conv = copy.deepcopy(self.final_conv)

        return subnet

    def get_active_subnet_config(self):
        subnet_config = self.config
        subnet_config.pop("width_list")
        subnet_config["name"] = SegmentationHead.__name__
        subnet_config["in_channels"] = [
            conv.active_in_channel for conv in self.convs1x1
        ]
        return subnet_config
