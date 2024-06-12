import copy
import random
from typing import List, Union

from ... import core as c
from ...baseblocks.mobilenetv2_basic import _make_divisible as make_divisible

from ..primitives.func import val2list

from ..abstract.ofa_abstract import NASModule

from ..baseblocks.conv_blocks import ConvBlock
from ..baseblocks.mbnet_blocks import MBConvBlock, DynamicMBConvBlock

from .ofa_mobilenetv3 import MobileNet


class OFAMobileNet(MobileNet, NASModule):
    """Перенос из оригинальной репы с изменениями."""

    SAMPLE_MODULE_CLS = MobileNet

    def __init__(
        self,
        bn_param=(0.1, 1e-5),
        width_mult_list=[1.0],
        ks_list=(3, 5),
        expand_ratio_list=(3, 4, 6),
        depth_list=(2, 3, 4),
        use_se=False,
        act_func="relu",
        n_stages=5,
        base_stage_width = [16, 16, 24, 40, 80, 112, 160],
        **kwargs,
    ):
        super(MobileNet, self).__init__()
        self.width_mult = width_mult_list[0]
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)

        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        self.use_se = use_se
        self.act_func = act_func
        self.n_stages = n_stages

        self.base_stage_width = base_stage_width[: 2 + self.n_stages]
        n_block_list = [1] + [max(self.depth_list)] * 5
        width_list = []
        for base_width in self.base_stage_width:
            width = make_divisible(base_width * self.width_mult, self.CHANNEL_DIVISIBLE)
            width_list.append(width)

        input_channel, first_block_dim = width_list[0], width_list[1]
        first_conv = ConvBlock(
            3,
            input_channel,
            kernel_size=3,
            stride=2,
            act_func=self.act_func,
            bias=True,
        )
        first_block = MBConvBlock(
            in_channels=input_channel,
            out_channels=first_block_dim,
            kernel_size=3,
            stride=self.stride_stages[0],
            expand_ratio=1,
            act_func=self.act_stages[0],
            use_se=self.se_stages[0],
        )

        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1
        feature_dim = first_block_dim

        for width, n_block, s, act_func, use_se in zip(
            width_list[2:],
            n_block_list[1:],
            self.stride_stages[1:],
            self.act_stages[1:],
            self.se_stages[1:],
        ):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                block = DynamicMBConvBlock(
                    in_channel_list=val2list(feature_dim),
                    out_channel_list=val2list(output_channel),
                    kernel_size_list=ks_list,
                    expand_ratio_list=expand_ratio_list,
                    stride=stride,
                    act_func=act_func,
                    use_se=use_se,
                )
                blocks.append(block)
                feature_dim = output_channel

        super().__init__(first_conv, blocks)
        self.blocks: Union[c.ModuleList, List[Union[MBConvBlock, DynamicMBConvBlock]]]

        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

    @property
    def stride_stages(self):
        stride_stages = [1, 2, 2, 2, 1, 2]
        stride_stages = stride_stages[: 1 + self.n_stages]
        return stride_stages

    @property
    def act_stages(self):
        act_stages = [
            "relu",
            "relu",
            "relu",
            self.act_func,
            self.act_func,
            self.act_func,
        ]
        act_stages = act_stages[: 1 + self.n_stages]
        return act_stages

    @property
    def se_stages(self):
        if self.use_se:
            se_stages = [False, False, True, False, True, True]
        else:
            se_stages = [False, False, False, False, False, False]
        se_stages = se_stages[: 1 + self.n_stages]
        return se_stages

    @staticmethod
    def name():
        return "OFAMobileNet"

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks[0](x)
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)
        return x

    @property
    def config(self):
        return {
            "name": self.__class__.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": [block.config for block in self.blocks],
        }

    @classmethod
    def build_from_config(cls, config):
        raise ValueError("does not support this function")

    @property
    def grouped_block_index(self):
        return self.block_group_info

    def set_max_net(self):
        arch_config = {
            "ks": max(self.ks_list),
            "e": max(self.expand_ratio_list),
            "d": max(self.depth_list),
        }
        self.set_active_subnet(**arch_config)
        return arch_config

    def set_min_net(self):
        arch_config = {
            "ks": min(self.ks_list),
            "e": min(self.expand_ratio_list),
            "d": min(self.depth_list),
        }
        self.set_active_subnet(**arch_config)
        return arch_config

    def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):
        ks = val2list(ks, len(self.blocks) - 1)
        expand_ratio = val2list(e, len(self.blocks) - 1)
        depth = val2list(d, len(self.block_group_info))

        for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
            block.set_active_subnet(k, e)

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

        self.layers_to_hook.pop()
        block_idx = self.block_group_info[-1]
        skip_last = len(block_idx) - self.runtime_depth[-1]
        block_n = block_idx[len(block_idx) - skip_last - 1]
        self.layers_to_hook.append(
            {"module": f"blocks.{block_n}", "hook_type": "forward"}
        )

    def set_constraint(self, include_list, constraint_type="depth"):
        if constraint_type == "depth":
            self.__dict__["_depth_include_list"] = include_list.copy()
        elif constraint_type == "expand_ratio":
            self.__dict__["_expand_include_list"] = include_list.copy()
        elif constraint_type == "kernel_size":
            self.__dict__["_ks_include_list"] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__["_depth_include_list"] = None
        self.__dict__["_expand_include_list"] = None
        self.__dict__["_ks_include_list"] = None

    def sample_active_subnet(self):
        ks_candidates = (
            self.ks_list
            if self.__dict__.get("_ks_include_list", None) is None
            else self.__dict__["_ks_include_list"]
        )
        expand_candidates = (
            self.expand_ratio_list
            if self.__dict__.get("_expand_include_list", None) is None
            else self.__dict__["_expand_include_list"]
        )
        depth_candidates = (
            self.depth_list
            if self.__dict__.get("_depth_include_list", None) is None
            else self.__dict__["_depth_include_list"]
        )

        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [
                depth_candidates for _ in range(len(self.block_group_info))
            ]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        arch_config = {
            "ks": ks_setting,
            "e": expand_setting,
            "d": depth_setting,
        }
        self.set_active_subnet(**arch_config)
        return arch_config

    def get_active_arch_desc(self) -> dict:
        ks_setting = [block.active_kernel_size for block in self.blocks[1:]]
        expand_setting = [block.active_expand_ratio for block in self.blocks[1:]]
        depth_setting = self.runtime_depth
        arch_config = {
            "ks": ks_setting,
            "e": expand_setting,
            "d": depth_setting,
        }
        return copy.deepcopy(arch_config)

    def get_subnets_grid(self) -> List[dict]:
        n_blocks = len(self.blocks) - 1
        backbone_desc_list = []
        for k in self.ks_list:
            for e in self.expand_ratio_list:
                for d in self.depth_list:
                    backbone_desc = {
                        "ks": [k] * n_blocks,
                        "e": [e] * n_blocks,
                        "d": [d] * self.n_stages,
                    }
                    backbone_desc_list.append(backbone_desc)
        return backbone_desc_list

    @property
    def active_out_channels(self):
        """
        Cписок активных выходных каналов бэкбона
        для установления активных входных каналов следующих блоков.
        """

        active_out_channels = []
        for hook_info in self.layers_to_hook:
            block_idx = int(hook_info["module"].split(".")[1])
            block = self.blocks[block_idx]
            if hook_info["hook_type"] == "pre":
                active_out_channels.append(block.in_channels)
            else:
                active_out_channels.append(block.out_channels)

        return active_out_channels

    def get_active_subnet(self):
        first_conv = copy.deepcopy(self.first_conv)
        blocks = [copy.deepcopy(self.blocks[0])]

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                block: DynamicMBConvBlock = self.blocks[idx]
                stage_blocks.append(block.get_active_subnet())
            blocks += stage_blocks

        _subnet = MobileNet(first_conv, blocks)
        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    def get_active_net_config(self):
        first_conv_config = self.first_conv.config
        first_block_config = self.blocks[0].config

        block_config_list = [first_block_config]
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                block: DynamicMBConvBlock = self.blocks[idx]
                stage_blocks.append(block.get_active_subnet_config())
            block_config_list += stage_blocks

        return {
            "name": MobileNet.__name__,
            "bn": self.get_bn_param(),
            "first_conv": first_conv_config,
            "blocks": block_config_list,
        }

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks[1:]:
            block.re_organize_middle_weights(expand_ratio_stage)
