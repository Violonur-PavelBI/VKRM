import torch.nn as nn

from .base_primitive import _BasePrimitiveConvertationInterface as BPCI
from .utils import wrapcls


@wrapcls
class ReplicationPad2d(nn.ReplicationPad2d, BPCI, subgroup="module"):
    layer_platform_name = "replication_pad"

    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer["padding"] = list(self.padding)
        return layer

    @classmethod
    def fromPlatform(cls, layer, binary):
        return nn.ReflectionPad2d(padding=layer["padding"])
