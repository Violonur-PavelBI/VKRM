import copy
from typing import List, Union

from ... import core as c

from ..abstract.ofa_typing import HookInfo

from ..primitives.utils import set_layer_from_config
from ..primitives.func import set_bn_param, get_bn_param
from ..primitives.static import MyNetwork

from ..baseblocks.conv_blocks import ConvBlock
from ..baseblocks.mbnet_blocks import MBConvBlock

__all__ = ["MobileNet"]


class MobileNet(MyNetwork):
    """Реализация сети из OFA

    HSwish используется из pytorch"""

    def __init__(self, first_conv: ConvBlock, blocks):
        super().__init__()
        self.first_conv = first_conv
        self.blocks: Union[c.ModuleList, List[MBConvBlock]] = c.ModuleList(blocks)

        self.layers_to_hook: List[HookInfo] = []
        hook_info: HookInfo
        for name, module in self.blocks.named_modules():
            if len(name.split(".")) > 1:
                continue
            s = getattr(module, "stride", None)
            if s == 2:
                hook_info = {"module": f"blocks.{name}", "hook_type": "pre"}
                self.layers_to_hook.append(hook_info)
        hook_info = {"module": f"blocks.{len(self.blocks) - 1}", "hook_type": "forward"}
        self.layers_to_hook.append(hook_info)

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
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
        first_conv = set_layer_from_config(config["first_conv"])

        blocks = []
        for block_config in config["blocks"]:
            blocks.append(MBConvBlock.build_from_config(block_config))

        net = cls(first_conv, blocks)
        if "bn" in config:
            net.set_bn_param(**config["bn"])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-5)

        return net

    def set_bn_param(self, momentum, eps, gn_channel_per_group=None, **kwargs):
        set_bn_param(self, momentum, eps, gn_channel_per_group, **kwargs)

    def get_bn_param(self):
        return get_bn_param(self)

    def get_parameters(self, keys=None, mode="include"):
        if keys is None:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    yield param
        elif mode == "include":
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and param.requires_grad:
                    yield param
        elif mode == "exclude":
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and param.requires_grad:
                    yield param
        else:
            raise ValueError("do not support: %s" % mode)

    def weight_parameters(self):
        return self.get_parameters()

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, MBConvBlock) and m.shortcut:
                m.point_linear.bn.weight.data.zero_()

    @property
    def grouped_block_index(self):
        info_list = []
        block_index_list = []
        for i, block in enumerate(self.blocks[1:], 1):
            if block.shortcut is None and len(block_index_list) > 0:
                info_list.append(block_index_list)
                block_index_list = []
            block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list
