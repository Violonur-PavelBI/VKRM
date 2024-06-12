import copy
import random
from typing import Dict, List, Literal, Sequence, Tuple, Union

import torch
from torch import Tensor

from ... import core as c
from ...baseblocks.mobilenetv2_basic import _make_divisible as make_divisible

from ..abstract.ofa_abstract import DynamicModule, NASModule, Neck
from ..baseblocks.conv_blocks import ConvBlock, DynamicConvBlock


class FPN(c.Module, Neck):
    """Класс FPN с фиксированным количеством каналов на разных ветках FPN

    авторы у себя в статье написали, что ReLU не дали пользы им, поэтому их так мало
    https://arxiv.org/pdf/1612.03144.pdf В конце третьего раздела (и чё?!)
    """

    CHANNEL_DIVISIBLE = 8

    upsample_modes_list = [
        "deconv",
        "nearest",
        "bilinear",
        "bicubic",
    ]

    def __init__(
        self,
        levels: int,
        single_out: bool,
        in_channels: List[int],
        mid_channels: int,
        convs: List[List[Tuple[int, int]]],
        merge_policy: Literal["sum", "cat"] = "sum",
        act_func: str = "relu",
        use_bias: bool = False,
        upsample_mode: str = "nearest",
        **kwargs,
    ) -> None:
        """
        levels: int -- Количество входов в FPN.
        single_out: bool -- Только один выход в FPN.
        in_channels: list[int] -- Список с количеством каналов на каждом входе FPN.
        mid_channels: int -- Промежуточное количество каналов в свёртках FPN.
        convs: list[list[tuple, int, int]] -- Описание свёрток после слияния карт.
        merge_policy: str -- Политика слияния (sum | cat).
        act_func: str -- Функция активации.
        upsample_mode: str -- метод увеличения при слиянии карт.
        use_bias: bool -- использовать ли смещение в свёртках (во всех сразу).
        """
        super().__init__()
        self.levels = levels
        self.single_out = single_out
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.convs = convs
        self.merge_policy = merge_policy
        self.act_func = act_func
        self.use_bias = use_bias
        self.upsample_mode = upsample_mode
        if self.upsample_mode not in self.upsample_modes_list:
            raise ValueError(f"{upsample_mode = } not in {self.upsample_modes_list = }")

        self.pre_convs = c.ModuleList(
            [
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=self.mid_channels,
                    kernel_size=1,
                    bias=self.use_bias,
                    use_bn=False,
                    act_func=None,
                )
                for in_ch in self.in_channels
            ]
        )
        self.ups = c.ModuleList(
            [
                c.ConvTranspose2d(
                    in_channels=self.mid_channels,
                    out_channels=self.mid_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=self.use_bias,
                )
                if self.upsample_mode == "deconv"
                else c.Upsample(mode=self.upsample_mode, scale_factor=2)
                for _ in range(self.levels - 1)
            ]
        )
        if self.merge_policy == "cat":
            self.mid_convs = c.ModuleList(
                [
                    ConvBlock(
                        in_channels=self.mid_channels * 2,
                        out_channels=self.mid_channels,
                        kernel_size=1,
                        bias=self.use_bias,
                        use_bn=False,
                        act_func=None,
                    )
                    for _ in range(self.levels - 2)
                ]
            )
            final_in_channels = [self.mid_channels * 2] * (self.levels - 1) + [
                self.in_channels[-1]
            ]
        else:
            final_in_channels = [self.mid_channels] * (self.levels - 1) + [
                self.in_channels[-1]
            ]
        self.post_convs: Union[c.ModuleList, List[c.Sequential], List[List[ConvBlock]]]
        self.post_convs = c.ModuleList([c.Sequential() for _ in range(self.levels)])
        for level, conv_list in enumerate(self.convs):
            in_ch = final_in_channels[level]
            for out_ch, kernel in conv_list:
                self.post_convs[level].append(
                    ConvBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        bias=self.use_bias,
                        use_bn=True,
                        act_func=self.act_func,
                    )
                )
                in_ch = out_ch

    @property
    def config(self):
        return {
            "name": self.__class__.__name__,
            "levels": self.levels,
            "in_channels": self.in_channels,
            "mid_channels": self.mid_channels,
            "convs": self.convs,
            "merge_policy": self.merge_policy,
            "act_func": self.act_func,
            "use_bias": self.use_bias,
            "upsample_mode": self.upsample_mode,
            "single_out": self.single_out,
        }

    @classmethod
    def build_from_config(cls, config: dict):
        return cls(**config)

    def forward(self, input_features: List[Tensor]) -> List[Tensor]:
        if not self.single_out:
            output_features = [None] * self.levels
        else:
            output_features = [None]

        input = input_features[-1]

        if not self.single_out:
            output = self.post_convs[-1](input)
            output_features[-1] = output

        input = self.pre_convs[-1](input)
        for i in range(self.levels - 2, -1, -1):
            skip_input = input_features[i]
            pre_conv = self.pre_convs[i]
            upsample = self.ups[i]
            if not self.single_out or i == 0:
                post_conv = self.post_convs[i]

            input = upsample(input)
            skip_input = pre_conv(skip_input)
            if self.merge_policy == "sum":
                input = input + skip_input
            elif self.merge_policy == "cat":
                input = torch.cat([input, skip_input], dim=1)
            if not self.single_out or i == 0:
                output = post_conv(input)
                output_features[i] = output
            if self.merge_policy == "cat" and i > 0:
                input = self.mid_convs[i - 1](input)

        return output_features


class DynamicFPN(FPN, DynamicModule):
    """Расширение FPN для работы с энкодерами, которые имеют варьируемое количество каналов
    адаптируется к количеству каналов на входе

    Смотри docsting класса FPN!
    """

    upsample_modes_list = [
        "deconv",
        "nearest",
        "bilinear",
        "bicubic",
    ]
    SAMPLE_MODULE_CLS = FPN

    def __init__(
        self,
        levels: int,
        single_out: bool,
        in_channels: List[Union[List[int], int]],
        mid_channels: int,
        convs: List[List[Tuple[int, int]]],
        merge_policy: Literal["sum", "cat"] = "sum",
        act_func: str = "relu",
        use_bias: bool = False,
        upsample_mode: str = "nearest",
        **kwargs,
    ) -> None:
        super(FPN, self).__init__()
        self.levels = levels
        self.single_out = single_out
        self.in_channels: List[List[int]] = [
            x if isinstance(x, Sequence) else [x] for x in in_channels
        ]
        # TODO: вызывать make_divisible или сразу правильно писать в конфиге?
        self.mid_channels = make_divisible(mid_channels, self.CHANNEL_DIVISIBLE)
        self.convs = [
            [
                (make_divisible(channels, self.CHANNEL_DIVISIBLE), kernel)
                for channels, kernel in convs_level
            ]
            for convs_level in convs
        ]
        self.merge_policy = merge_policy
        self.act_func = act_func
        self.use_bias = use_bias
        self.upsample_mode = upsample_mode
        if self.upsample_mode not in self.upsample_modes_list:
            raise ValueError(f"{upsample_mode = } not in {self.upsample_modes_list = }")

        self.pre_convs: Union[c.ModuleList, List[DynamicConvBlock]]
        self.pre_convs = c.ModuleList(
            [
                DynamicConvBlock(
                    in_channel_list=in_ch_list,
                    out_channel_list=[self.mid_channels],
                    kernel_size_list=[1],
                    bias=use_bias,
                    use_bn=False,
                    act_func=None,
                )
                for in_ch_list in self.in_channels
            ]
        )
        self.ups = c.ModuleList(
            [
                c.ConvTranspose2d(
                    in_channels=self.mid_channels,
                    out_channels=self.mid_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=self.use_bias,
                )
                if self.upsample_mode == "deconv"
                else c.Upsample(mode=self.upsample_mode, scale_factor=2)
                for _ in range(self.levels - 1)
            ]
        )
        if self.merge_policy == "cat":
            self.mid_convs = c.ModuleList(
                [
                    ConvBlock(
                        in_channels=self.mid_channels * 2,
                        out_channels=self.mid_channels,
                        kernel_size=1,
                        bias=self.use_bias,
                        use_bn=False,
                        act_func=None,
                    )
                    for _ in range(self.levels - 2)
                ]
            )
            final_in_channels = [self.mid_channels * 2] * (self.levels - 1) + [
                self.in_channels[-1]
            ]
        else:
            final_in_channels = [self.mid_channels] * (self.levels - 1) + [
                self.in_channels[-1]
            ]
        self.post_convs: Union[
            c.ModuleList,
            List[c.Sequential],
            List[List[Union[ConvBlock, DynamicConvBlock]]],
        ]
        self.post_convs = c.ModuleList([c.Sequential() for _ in range(self.levels)])
        for level, conv_list in enumerate(self.convs):
            in_ch = final_in_channels[level]
            for out_ch, kernel in conv_list:
                if isinstance(in_ch, int):
                    post_conv = ConvBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        bias=self.use_bias,
                        use_bn=True,
                        act_func=self.act_func,
                    )
                elif isinstance(in_ch, list):
                    post_conv = DynamicConvBlock(
                        in_channel_list=in_ch,
                        out_channel_list=[out_ch],
                        kernel_size_list=[kernel],
                        bias=self.use_bias,
                        use_bn=True,
                        act_func=self.act_func,
                    )
                self.post_convs[level].append(post_conv)
                in_ch = out_ch

    def set_active_subnet(self, active_in_channels: List[int]) -> None:
        for conv, active_in_ch in zip(self.pre_convs, active_in_channels):
            conv: DynamicConvBlock
            conv.active_in_channel = active_in_ch
        conv: DynamicConvBlock = self.post_convs[-1][0]
        conv.active_in_channel = active_in_channels[-1]

    def set_max_net(self) -> None:
        active_in_channels = [max(x) for x in self.in_channels]
        self.set_active_subnet(active_in_channels)

    def set_min_net(self) -> None:
        active_in_channels = [min(x) for x in self.in_channels]
        self.set_active_subnet(active_in_channels)

    def get_active_subnet(self) -> FPN:
        pre_convs = c.ModuleList([conv.get_active_subnet() for conv in self.pre_convs])
        ups = c.ModuleList([copy.deepcopy(up) for up in self.ups])
        if self.merge_policy == "cat":
            mid_convs = c.ModuleList([copy.deepcopy(conv) for conv in self.mid_convs])
        post_convs = c.ModuleList(
            [copy.deepcopy(conv) for conv in self.post_convs[:-1]]
            + [
                c.Sequential(
                    self.post_convs[-1][0].get_active_subnet(),
                    *[copy.deepcopy(conv) for conv in self.post_convs[-1][1:]],
                )
            ]
        )

        subnet_config = self.get_active_subnet_config()
        subnet: FPN = FPN.build_from_config(subnet_config)
        subnet.pre_convs = pre_convs
        subnet.ups = ups
        if self.merge_policy == "cat":
            subnet.mid_convs = mid_convs
        subnet.post_convs = post_convs

        return subnet

    def get_active_subnet_config(self):
        subnet_config = self.config
        subnet_config["name"] = FPN.__name__
        subnet_config["in_channels"] = [
            conv.active_in_channel for conv in self.pre_convs
        ]
        return subnet_config


class NASFPN(DynamicFPN, NASModule):
    """
    FPN с подбором следующих параметров через NAS:
        - промежуточное количество каналов
        - для каждого уровня:
            - количество блоков
            - для каждого блока:
                - выходное количество каналов
                - размер ядра свёртки

    Активное количество каналов = базовое количество каналов * подбираемый множитель.
    Базовое количество каналов различно для промежуточных свёрток и для каждого уровня конечных свёрток.
    Список множителей общий для всей шеи, но каждое значение подбирается независимо.

    Смотри docsting класса FPN!
    """

    upsample_modes_list = [
        "nearest",
        "bilinear",
        "bicubic",
    ]
    SAMPLE_MODULE_CLS = FPN

    def __init__(
        self,
        levels: int,
        single_out: bool,
        in_channels: List[Union[List[int], int]],
        mid_channels: int,
        out_channel_list: List[int],
        depth_list: List[int],
        kernel_list: List[int],
        width_list: List[float],
        merge_policy: Literal["sum"] = "sum",
        act_func: str = "relu",
        use_bias: bool = False,
        upsample_mode: str = "nearest",
        **kwargs,
    ) -> None:
        super(FPN, self).__init__()
        self.levels = levels
        self.single_out = single_out
        self.in_channels: List[List[int]] = [
            x if isinstance(x, Sequence) else [x] for x in in_channels
        ]

        self.mid_channels = mid_channels
        self.out_channel_list = out_channel_list

        self.depth_list = sorted(depth_list)
        self.kernel_list = sorted(kernel_list)
        self.width_list = sorted([round(w, 2) for w in width_list])

        self.merge_policy = merge_policy
        self.act_func = act_func
        self.use_bias = use_bias
        self.upsample_mode = upsample_mode
        if self.upsample_mode not in self.upsample_modes_list:
            raise ValueError(f"{upsample_mode = } not in {self.upsample_modes_list = }")

        mid_ch_list = [
            make_divisible(w * self.mid_channels, self.CHANNEL_DIVISIBLE)
            for w in self.width_list
        ]
        self.pre_convs: Union[c.ModuleList, List[DynamicConvBlock]]
        self.pre_convs = c.ModuleList(
            [
                DynamicConvBlock(
                    in_channel_list=in_ch_list,
                    out_channel_list=mid_ch_list,
                    kernel_size_list=[1],
                    bias=use_bias,
                    use_bn=False,
                    act_func=None,
                )
                for in_ch_list in self.in_channels
            ]
        )
        self.ups = c.ModuleList(
            [
                c.Upsample(mode=self.upsample_mode, scale_factor=2)
                for _ in range(self.levels - 1)
            ]
        )

        final_in_channels = [mid_ch_list] * (self.levels - 1) + [self.in_channels[-1]]
        post_convs = []
        for i in range(self.levels):
            out_ch_list = [
                make_divisible(w * self.out_channel_list[i], self.CHANNEL_DIVISIBLE)
                for w in self.width_list
            ]
            level_convs = [
                DynamicConvBlock(
                    in_channel_list=final_in_channels[i],
                    out_channel_list=out_ch_list,
                    kernel_size_list=self.kernel_list,
                    bias=self.use_bias,
                    use_bn=True,
                    act_func=self.act_func,
                )
            ]
            for j in range(1, max(self.depth_list)):
                level_convs.append(
                    DynamicConvBlock(
                        in_channel_list=out_ch_list,
                        out_channel_list=out_ch_list,
                        kernel_size_list=self.kernel_list,
                        bias=self.use_bias,
                        use_bn=True,
                        act_func=self.act_func,
                    )
                )
            post_convs.append(c.ModuleList(level_convs))
        self.post_convs: Union[c.ModuleList, List[List[DynamicConvBlock]]]
        self.post_convs = c.ModuleList(post_convs)
        self.runtime_depth = [len(level_convs) for level_convs in self.post_convs]

    def forward(self, input_features: List[Tensor]) -> List[Tensor]:
        if not self.single_out:
            output_features = [None] * self.levels
        else:
            output_features = [None]

        input = input_features[-1]

        if not self.single_out:
            depth = self.runtime_depth[-1]
            post_conv = self.post_convs[-1]

            output = input
            for j in range(depth):
                output = post_conv[j](output)
            output_features[-1] = output

        input = self.pre_convs[-1](input)
        for i in range(self.levels - 2, -1, -1):
            skip_input = input_features[i]
            pre_conv = self.pre_convs[i]
            upsample = self.ups[i]
            if not self.single_out or i == 0:
                depth = self.runtime_depth[i]
                post_conv = self.post_convs[i]

            input = upsample(input)
            skip_input = pre_conv(skip_input)
            input = input + skip_input

            if not self.single_out or i == 0:
                output = input
                for j in range(depth):
                    output = post_conv[j](output)
                output_features[i] = output

        return output_features

    def sample_active_subnet(self, active_in_channels: List[int]) -> dict:
        mid = random.choice(self.width_list)
        active_depth = [random.choice(self.depth_list) for _ in range(self.levels)]
        active_kernel = [
            [random.choice(self.kernel_list) for _ in range(max(self.depth_list))]
            for __ in range(self.levels)
        ]
        active_width = [
            [random.choice(self.width_list) for _ in range(max(self.depth_list))]
            for __ in range(self.levels)
        ]
        out = {"d": active_depth, "k": active_kernel, "w": active_width}
        arch_config = {
            "active_in_channels": active_in_channels,
            "mid": mid,
            "up": self.upsample_mode,
            "out": out,
        }
        self.set_active_subnet(**arch_config)
        return arch_config

    def get_active_arch_desc(self) -> dict:
        def find_coef(value: int, base_value: int, coefs: list, divisor: int):
            for k in coefs:
                candidate_value = make_divisible(k * base_value, divisor)
                if candidate_value == value:
                    return k
            raise ValueError(f"{value = }")

        active_in_channels = [conv.active_in_channel for conv in self.pre_convs]
        mid = find_coef(
            self.pre_convs[0].active_out_channel,
            self.mid_channels,
            self.width_list,
            self.CHANNEL_DIVISIBLE,
        )
        active_depth = self.runtime_depth
        active_kernel = [
            [conv.active_kernel_size for conv in conv_list]
            for conv_list in self.post_convs
        ]
        active_width = [
            [
                find_coef(
                    conv.active_out_channel,
                    self.out_channel_list[i],
                    self.width_list,
                    self.CHANNEL_DIVISIBLE,
                )
                for conv in conv_list
            ]
            for i, conv_list in enumerate(self.post_convs)
        ]

        out = {"d": active_depth, "k": active_kernel, "w": active_width}
        arch_config = {
            "active_in_channels": active_in_channels,
            "mid": mid,
            "up": self.upsample_mode,
            "out": out,
        }
        return copy.deepcopy(arch_config)

    def get_subnets_grid(self) -> List[dict]:
        depth = max(self.depth_list)
        neck_desc_list = []
        for w_n in self.width_list:
            for d_n in self.depth_list:
                for k_n in self.kernel_list:
                    neck_desc = {
                        "mid": w_n,
                        "out": {
                            "d": [d_n] * self.levels,
                            "k": [[k_n] * depth] * self.levels,
                            "w": [[w_n] * depth] * self.levels,
                        },
                    }
                    neck_desc_list.append(neck_desc)
        return neck_desc_list

    def set_active_subnet(
        self,
        active_in_channels: List[int],
        mid: float,
        out: Dict[str, Union[List[int], List[List[int]]]],
        **kwargs,
    ) -> None:
        active_mid_channels = make_divisible(
            mid * self.mid_channels, self.CHANNEL_DIVISIBLE
        )
        final_active_in_channels = [active_mid_channels] * (self.levels - 1) + [
            active_in_channels[-1]
        ]
        for i in range(self.levels):
            conv: DynamicConvBlock = self.pre_convs[i]
            conv.active_in_channel = active_in_channels[i]
            conv.active_out_channel = active_mid_channels

            self.runtime_depth[i] = out["d"][i]

            in_ch = final_active_in_channels[i]
            for j in range(max(self.depth_list)):
                conv: DynamicConvBlock = self.post_convs[i][j]
                kernel = out["k"][i][j]
                out_ch = make_divisible(
                    out["w"][i][j] * self.out_channel_list[i], self.CHANNEL_DIVISIBLE
                )
                conv.active_kernel_size = kernel
                conv.active_in_channel = in_ch
                conv.active_out_channel = out_ch
                in_ch = out_ch

    def set_max_net(self) -> None:
        active_in_channels = [max(x) for x in self.in_channels]
        mid = max(self.width_list)
        active_depth = [max(self.depth_list) for _ in range(self.levels)]
        active_kernel = [
            [max(self.kernel_list) for _ in range(max(self.depth_list))]
            for __ in range(self.levels)
        ]
        active_width = [
            [max(self.width_list) for _ in range(max(self.depth_list))]
            for __ in range(self.levels)
        ]
        out = {"d": active_depth, "k": active_kernel, "w": active_width}
        arch_config = {
            "active_in_channels": active_in_channels,
            "mid": mid,
            "out": out,
        }
        self.set_active_subnet(**arch_config)

    def set_min_net(self) -> None:
        active_in_channels = [min(x) for x in self.in_channels]
        mid = min(self.width_list)
        active_depth = [min(self.depth_list) for _ in range(self.levels)]
        active_kernel = [
            [min(self.kernel_list) for _ in range(max(self.depth_list))]
            for __ in range(self.levels)
        ]
        active_width = [
            [min(self.width_list) for _ in range(max(self.depth_list))]
            for __ in range(self.levels)
        ]
        out = {"d": active_depth, "k": active_kernel, "w": active_width}
        arch_config = {
            "active_in_channels": active_in_channels,
            "mid": mid,
            "out": out,
        }
        self.set_active_subnet(**arch_config)

    def get_active_subnet(self) -> FPN:
        pre_convs = c.ModuleList([conv.get_active_subnet() for conv in self.pre_convs])
        ups = c.ModuleList([copy.deepcopy(up) for up in self.ups])
        post_convs = c.ModuleList(
            [
                c.Sequential(
                    *[
                        self.post_convs[i][j].get_active_subnet()
                        for j in range(self.runtime_depth[i])
                    ]
                )
                for i in range(self.levels)
            ]
        )

        subnet_config = self.get_active_subnet_config()
        subnet: FPN = FPN.build_from_config(subnet_config)
        subnet.pre_convs = pre_convs
        subnet.ups = ups
        subnet.post_convs = post_convs

        return subnet

    def get_active_subnet_config(self) -> dict:
        return {
            "name": FPN.__name__,
            "levels": self.levels,
            "in_channels": [conv.active_in_channel for conv in self.pre_convs],
            "mid_channels": self.pre_convs[0].active_out_channel,
            "convs": [
                [
                    (
                        self.post_convs[i][j].active_out_channel,
                        self.post_convs[i][j].active_kernel_size,
                    )
                    for j in range(self.runtime_depth[i])
                ]
                for i in range(self.levels)
            ],
            "merge_policy": self.merge_policy,
            "act_func": self.act_func,
            "use_bias": self.use_bias,
            "upsample_mode": self.upsample_mode,
            "single_out": self.single_out,
        }

    @property
    def config(self):
        return {
            "name": self.__class__.__name__,
            "levels": self.levels,
            "in_channels": self.in_channels,
            "mid_channels": self.mid_channels,
            "depth_list": self.depth_list,
            "out_channel_list": self.out_channel_list,
            "kernel_list": self.kernel_list,
            "width_list": self.width_list,
            "merge_policy": self.merge_policy,
            "act_func": self.act_func,
            "use_bias": self.use_bias,
            "upsample_mode": self.upsample_mode,
            "single_out": self.single_out,
        }

    @property
    def active_out_channels(self):
        return [
            conv_list[d - 1].active_out_channel
            if d > 0
            else self.pre_convs[0].active_out_channel
            for conv_list, d in zip(self.post_convs, self.runtime_depth)
        ]
