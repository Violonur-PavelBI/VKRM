import torch.nn as nn

from .base_primitive import _BasePrimitiveConvertationInterface as BPCI
from .utils import wrapcls


@wrapcls
class Dropout(nn.Dropout, BPCI, subgroup="module"):
    layer_platform_name = "dropout"

    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer["p"] = self.p
        return layer

    @classmethod
    def fromPlatform(cls, layer, binary):
        return nn.Dropout(p=layer["p"])
