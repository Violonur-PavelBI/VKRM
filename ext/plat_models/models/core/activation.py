import torch
import torch.nn as nn

from .utils import wrapcls
from .base_primitive import _BasePrimitiveConvertationInterface as BPCI


@wrapcls
class Sigmoid(nn.Sigmoid, BPCI, subgroup="module"):
    layer_platform_name = "sigmoid"

    @classmethod
    def fromPlatform(cls, layer, binary):
        sigmoid = cls()
        return sigmoid


@wrapcls
class ReLU(nn.ReLU, BPCI, subgroup="module"):
    layer_platform_name = "relu"

    @classmethod
    def fromPlatform(cls, layer, binary):
        relu = cls()
        return relu


@wrapcls
class LeakyReLU(nn.LeakyReLU, BPCI, subgroup="module"):
    layer_platform_name = "leakyrelu"

    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer["negative_slope"] = self.negative_slope
        return layer

    @classmethod
    def fromPlatform(cls, layer, binary):
        relu = cls(negative_slope=layer["negative_slope"])
        return relu


@wrapcls
class _ReLUT(nn.Module, BPCI, subgroup="module"):
    layer_platform_name = "relut"
    _subclasses = {}
    thr: int

    def __init__(self, inplace=False):
        super().__init__()
        self.thr = self.__class__.thr

    def __init_subclass__(cls):
        cls._subclasses[cls.thr] = cls

    def forward(self, x):
        return torch.clamp(x, min=0, max=self.thr)

    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer["thr"] = self.thr
        return layer

    @classmethod
    def fromPlatform(cls, layer, binary):
        subclass = cls._subclasses[layer["thr"]]
        return subclass()


@wrapcls
class ReLU6(_ReLUT):
    thr = 6
