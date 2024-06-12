import copy
import random
from typing import List, Union

from ... import core as c
from ...baseblocks.mobilenetv2_basic import _make_divisible as make_divisible

from ..primitives.func import val2list

from ..abstract.ofa_abstract import NASModule

from ..baseblocks.conv_blocks import DynamicConvBlock
from ..baseblocks.resnet_blocks import DynamicResNetBottleneckBlock

from .ofa_resnet import ResNet

__all__ = ["OFAResNet"]


class OFAResNet(ResNet, NASModule):
    DOWNSAMPLE_MODES = ("avgpool_conv", "maxpool_conv", "conv")
    SAMPLE_MODULE_CLS = ResNet

    def __init__(
        self,
        bn_param=(0.1, 1e-5),
        depth_list=(0, 1, 2),
        expand_ratio_list=(0.2, 0.25, 0.35),
        width_mult_list=(0.65, 0.8, 1.0),
        base_depth_list=[2, 2, 4, 2],
        downsample_mode="avgpool_conv",
        **kwargs,
    ):
        self.depth_list = val2list(depth_list)
        self.expand_ratio_list = val2list(expand_ratio_list)
        self.width_mult_list = val2list(width_mult_list)
        # sort
        self.depth_list.sort()
        self.expand_ratio_list.sort()
        self.width_mult_list.sort()

        self.base_depth_list = base_depth_list
        self.STAGE_WIDTH_LIST = [256, 512, 1024, 2048]

        input_channel = [
            make_divisible(64 * width_mult, self.CHANNEL_DIVISIBLE)
            for width_mult in self.width_mult_list
        ]
        mid_input_channel = [
            make_divisible(channel // 2, self.CHANNEL_DIVISIBLE)
            for channel in input_channel
        ]

        stage_width_list: List[List[int]] = self.STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = [
                make_divisible(width * width_mult, self.CHANNEL_DIVISIBLE)
                for width_mult in self.width_mult_list
            ]

        n_block_list = [
            base_depth + max(self.depth_list) for base_depth in self.base_depth_list
        ]
        stride_list = [1, 2, 2, 2]

        input_stem = [
            DynamicConvBlock(
                val2list(3),
                mid_input_channel,
                3,
                stride=2,
                use_bn=True,
                act_func="relu",
                bias=True,
            ),
            DynamicConvBlock(
                mid_input_channel,
                mid_input_channel,
                3,
                stride=1,
                use_bn=True,
                act_func="relu",
            ),
            DynamicConvBlock(
                mid_input_channel,
                input_channel,
                3,
                stride=1,
                use_bn=True,
                act_func="relu",
            ),
        ]
        if downsample_mode not in self.DOWNSAMPLE_MODES:
            raise ValueError(f"{downsample_mode=} must be in {self.DOWNSAMPLE_MODES}")
        blocks = []
        for d, width, s in zip(n_block_list, stage_width_list, stride_list):
            for i in range(d):
                stride = s if i == 0 else 1
                bottleneck_block = DynamicResNetBottleneckBlock(
                    input_channel,
                    width,
                    expand_ratio_list=self.expand_ratio_list,
                    kernel_size=3,
                    stride=stride,
                    act_func="relu",
                    downsample_mode=downsample_mode,
                )
                blocks.append(bottleneck_block)
                input_channel = width
        super().__init__(input_stem, blocks)
        self.input_stem: Union[c.ModuleList, List[DynamicConvBlock]]
        self.blocks: Union[c.ModuleList, List[DynamicResNetBottleneckBlock]]

        self.set_bn_param(*bn_param)
        self.input_stem_skipping = False
        self.runtime_depth = [0] * len(n_block_list)

    @property
    def ks_list(self):
        return [3]

    @staticmethod
    def name():
        return "OFAResNet"

    def forward(self, x):
        for i, layer in enumerate(self.input_stem):
            if self.input_stem_skipping and i == 1:
                pass
            elif i == 1:
                x = layer(x) + x
            else:
                x = layer(x)
        x = self.max_pooling(x)
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                x = self.blocks[idx](x)
        return x

    @property
    def config(self):
        return {
            "name": self.__class__.__name__,
            "bn": self.get_bn_param(),
            "input_stem": [layer.config for layer in self.input_stem],
            "blocks": [block.config for block in self.blocks],
        }

    @classmethod
    def build_from_config(cls, config):
        raise ValueError("do not support this function")

    def set_max_net(self):
        self.set_active_subnet(
            d=max(self.depth_list),
            e=max(self.expand_ratio_list),
            w=len(self.width_mult_list) - 1,
        )

    def set_min_net(self):
        self.set_active_subnet(
            d=min(self.depth_list),
            e=min(self.expand_ratio_list),
            w=0,
        )

    def set_active_subnet(self, d=None, e=None, w=None, **kwargs):
        depth = val2list(d, len(self.base_depth_list) + 1)
        expand_ratio = val2list(e, len(self.blocks))
        width_mult = val2list(w, len(self.base_depth_list) + 2)

        if width_mult[0] is not None:
            channels = self.input_stem[0].out_channel_list[width_mult[0]]
            self.input_stem[0].active_out_channel = channels
            self.input_stem[1].active_in_channel = channels
            self.input_stem[1].active_out_channel = channels
            self.input_stem[2].active_in_channel = channels
        if width_mult[1] is not None:
            channels = self.input_stem[2].out_channel_list[width_mult[1]]
            self.input_stem[2].active_out_channel = channels

        if depth[0] is not None:
            self.input_stem_skipping = depth[0] != max(self.depth_list)
        for stage_id, (block_idx, d, w) in enumerate(
            zip(self.grouped_block_index, depth[1:], width_mult[2:])
        ):
            if d is not None:
                self.runtime_depth[stage_id] = max(self.depth_list) - d
            if w is not None:
                for idx in block_idx:
                    block: DynamicResNetBottleneckBlock = self.blocks[idx]
                    block.set_active_subnet(expand_ratio[idx], w)

            if stage_id == len(self.grouped_block_index) - 1:
                self.layers_to_hook.pop()
                skip_last = self.runtime_depth[stage_id]
                block_n = block_idx[len(block_idx) - skip_last - 1]
                self.layers_to_hook.append(
                    {"module": f"blocks.{block_n}", "hook_type": "forward"}
                )

    def sample_active_subnet(self):
        # sample expand ratio
        expand_setting = []
        for block in self.blocks:
            expand_setting.append(random.choice(block.expand_ratio_list))

        # sample depth
        depth_setting = [random.choice([max(self.depth_list), min(self.depth_list)])]
        for stage_id in range(len(self.base_depth_list)):
            depth_setting.append(random.choice(self.depth_list))

        # sample width_mult
        width_mult_setting = [
            random.choice(list(range(len(self.input_stem[0].out_channel_list)))),
            random.choice(list(range(len(self.input_stem[2].out_channel_list)))),
        ]
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            stage_first_block = self.blocks[block_idx[0]]
            width_mult_setting.append(
                random.choice(list(range(len(stage_first_block.out_channel_list))))
            )

        arch_config = {
            "d": depth_setting,
            "e": expand_setting,
            "w": width_mult_setting,
        }
        self.set_active_subnet(**arch_config)
        return arch_config

    def get_active_arch_desc(self) -> dict:
        depth_setting = [
            min(self.depth_list) if self.input_stem_skipping else max(self.depth_list)
        ]
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_setting.append(
                len(block_idx)
                - self.runtime_depth[stage_id]
                - self.base_depth_list[stage_id]
            )

        expand_setting = [block.active_expand_ratio for block in self.blocks]

        width_mult_setting = [
            self.input_stem[0].out_channel_list.index(
                self.input_stem[0].active_out_channel
            ),
            self.input_stem[2].out_channel_list.index(
                self.input_stem[2].active_out_channel
            ),
        ]
        for block_idx in self.grouped_block_index:
            block = self.blocks[block_idx[0]]
            width_mult = block.out_channel_list.index(block.active_out_channel)
            width_mult_setting.append(width_mult)

        arch_config = {
            "d": depth_setting,
            "e": expand_setting,
            "w": width_mult_setting,
        }
        return copy.deepcopy(arch_config)

    def get_subnets_grid(self) -> List[dict]:
        n_blocks = len(self.blocks)
        n_stages = len(self.grouped_block_index)
        backbone_desc_list = []
        for d in self.depth_list:
            for e in self.expand_ratio_list:
                for w in range(len(self.width_mult_list)):
                    backbone_desc = {
                        "d": [d] * (n_stages + 1),
                        "e": [e] * n_blocks,
                        "w": [w] * (n_stages + 2),
                    }
                    if d != max(self.depth_list):
                        backbone_desc["d"][0] = min(self.depth_list)
                    backbone_desc_list.append(backbone_desc)
        return backbone_desc_list

    @property
    def active_out_channels(self):
        """
        Список активных выходных каналов бэкбона
        для установления активных входных каналов следующих блоков.
        """

        blocks_channels = {}
        in_channels = self.input_stem[-1].active_out_channel
        for i in range(len(self.blocks)):
            out_channels = self.blocks[i].active_out_channel
            blocks_channels[i] = {"in": in_channels, "out": out_channels}
            in_channels = out_channels

        # first hook is the output of input stem, the rest are from blocks
        active_out_channels = [self.input_stem[-1].active_out_channel]
        for hook_info in self.layers_to_hook[1:]:
            block_idx = int(hook_info["module"].split(".")[1])
            if hook_info["hook_type"] == "pre":
                active_out_channels.append(blocks_channels[block_idx]["in"])
            else:
                active_out_channels.append(blocks_channels[block_idx]["out"])

        return active_out_channels

    def get_active_subnet(self):
        input_stem = [self.input_stem[0].get_active_subnet()]
        if not self.input_stem_skipping:
            input_stem.append(self.input_stem[1].get_active_subnet())
        input_stem.append(self.input_stem[2].get_active_subnet())
        input_channel = self.input_stem[2].active_out_channel

        blocks = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                block: DynamicResNetBottleneckBlock = self.blocks[idx]
                blocks.append(block.get_active_subnet(input_channel))
                input_channel = block.active_out_channel
        subnet = ResNet(input_stem, blocks)

        subnet.set_bn_param(**self.get_bn_param())
        return subnet

    def get_active_net_config(self):
        input_stem_config = [self.input_stem[0].get_active_subnet_config(3)]
        input_mid_channel = self.input_stem[0].active_out_channel
        if not self.input_stem_skipping:
            input_stem_config.append(
                self.input_stem[1].get_active_subnet_config(input_mid_channel)
            )
        input_stem_config.append(
            self.input_stem[2].get_active_subnet_config(input_mid_channel)
        )
        input_channel = self.input_stem[2].active_out_channel

        blocks_config = []
        for stage_id, block_idx in enumerate(self.grouped_block_index):
            depth_param = self.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                blocks_config.append(
                    self.blocks[idx].get_active_subnet_config(input_channel)
                )
                input_channel = self.blocks[idx].active_out_channel
        return {
            "name": ResNet.__name__,
            "bn": self.get_bn_param(),
            "input_stem": input_stem_config,
            "blocks": blocks_config,
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks:
            block.re_organize_middle_weights(expand_ratio_stage)
