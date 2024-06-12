from typing import List, Union

from ... import core as c

from ..abstract.ofa_typing import HookInfo
from ..primitives.utils import set_layer_from_config
from ..primitives.static import IdentityLayer, MyNetwork
from ..baseblocks.conv_blocks import ConvBlock
from ..baseblocks.resnet_blocks import ResNetBottleneckBlock

__all__ = ["ResNet"]


class ResNet(MyNetwork):
    def __init__(self, input_stem, blocks):
        super().__init__()

        self.input_stem: Union[c.ModuleList, List[ConvBlock]] = c.ModuleList(input_stem)
        self.max_pooling = c.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.blocks: Union[
            c.ModuleList,
            List[ResNetBottleneckBlock],
        ] = c.ModuleList(blocks)

        self.layers_to_hook: List[HookInfo] = []
        hook_info: HookInfo
        hook_info = {
            "module": f"input_stem.{len(self.input_stem) - 1}",
            "hook_type": "forward",
        }
        self.layers_to_hook.append(hook_info)
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
        for i, layer in enumerate(self.input_stem):
            if len(self.input_stem) == 3 and i == 1:
                x = layer(x) + x
            else:
                x = layer(x)
        x = self.max_pooling(x)
        for block in self.blocks:
            x = block(x)
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
        input_stem = []
        for layer_config in config["input_stem"]:
            input_stem.append(set_layer_from_config(layer_config))
        blocks = []
        for block_config in config["blocks"]:
            blocks.append(set_layer_from_config(block_config))

        net = cls(input_stem, blocks)
        if "bn" in config:
            net.set_bn_param(**config["bn"])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-5)

        return net

    def zero_last_gamma(self):
        # TODO разобраться зачем это нужно
        for m in self.modules():
            if isinstance(m, ResNetBottleneckBlock) and isinstance(
                m.downsample, IdentityLayer
            ):
                m.conv3.bn.weight.data.zero_()

    @property
    def grouped_block_index(self):
        info_list = []
        block_index_list = []
        for i, block in enumerate(self.blocks):
            if (
                not isinstance(block.downsample, IdentityLayer)
                and len(block_index_list) > 0
            ):
                info_list.append(block_index_list)
                block_index_list = []
            block_index_list.append(i)
        if len(block_index_list) > 0:
            info_list.append(block_index_list)
        return info_list
