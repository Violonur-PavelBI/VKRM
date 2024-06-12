from typing import TypedDict, Union
import torch
import torch.nn as nn
from torch import Tensor

from ... import core as c

from .utils import LayerRegistry
from .func import set_bn_param, get_bn_param


from models.core.converter import ConverterTorch2Plat
from models.core.custom import _CustomLeafModule
from models.core.utils.convert_bases import convert_base, get_inputs
import torch
from torch.fx.node import Node
import torchvision


def build_activation(act_func, inplace=True):
    """Не всё смог взять из core"""
    # TODO: выпилить ненужные
    if act_func == "relu":
        return c.ReLU(inplace=inplace)
    elif act_func == "leaky_relu":
        return c.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_func == "relu6":
        return c.ReLU6(inplace=inplace)
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "h_tanh":
        return nn.Hardtanh
    elif act_func == "sigmoid":
        return c.Sigmoid()
    elif act_func == "h_swish":
        return Hswish(inplace=inplace)
    elif act_func == "h_sigmoid":
        return Hsigmoid(inplace=inplace)
    elif act_func is None or act_func == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


class MyModule(c.Module):
    def forward(self, x):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @classmethod
    def build_from_config(cls, config: dict):
        return cls(**config)


class MyNetwork(MyModule):
    CHANNEL_DIVISIBLE = 8

    def zero_last_gamma(self):
        raise NotImplementedError

    @property
    def grouped_block_index(self):
        raise NotImplementedError

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


class Hsigmoid(c.Module):
    """Переписано на использование Relu6 из Core"""

    def __init__(self, inplace=True) -> None:
        super().__init__()
        self.relu = c.ReLU6(inplace)

    def forward(self, x):
        return self.relu(x + 3.0) * 0.166666666

    def __repr__(self):
        return "Hsigmoid()"


class Hswish(c.Module):
    """Переписано на использование Relu6 из Core

    в теории можно унаследоваться от Hsigmoid,
    но может быть просадка по скорости"""

    def __init__(self, inplace=True) -> None:
        super().__init__()
        self.relu = c.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu(x + 3.0) * 0.166666666

    def __repr__(self):
        return "Hswish()"


@LayerRegistry.registry
class IdentityLayer(MyModule):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

    @property
    def config(self):
        return {"name": IdentityLayer.__name__}


class MergeLayer(c.Module):
    policies_list = ["cat", "sum"]

    def __init__(self, policy: str = "sum") -> None:
        super().__init__()
        if policy not in self.policies_list:
            raise ValueError
        self.policy = policy

    def forward(self, x, y):
        if self.policy == "sum":
            return x + y
        elif self.policy == "cat":
            return torch.cat([x, y], dim=1)
        else:
            raise ValueError


class Flatten(c.Module):
    """В платформе нет нормальной обработки торчовского Flatten."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        return x


class Argmax(_CustomLeafModule):
    layer_platform_name = "argmax"

    class PlatDict(TypedDict):
        dim: Union[int, None]
        input: str
        type: str
        output: str

    def __init__(self, dim: Union[int, None] = None) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = False

    def forward(self, input: Tensor) -> Tensor:
        return torch.argmax(input, dim=self.dim, keepdim=self.keepdim)

    def toPlatform(self, fx_node: Node, Converter: ConverterTorch2Plat):
        layer = convert_base(fx_node, Converter)
        layer: Argmax.PlatDict
        layer["type"] = "argmax"
        layer["dim"] = self.dim

        inputs = get_inputs(fx_node)
        layer["input"] = inputs[0]
        return layer

    @classmethod
    def fromPlatform(cls, layer: PlatDict, binary):
        return Argmax(dim=layer["dim"])
