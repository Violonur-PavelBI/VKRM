import torch.nn as nn

from .base_primitive import _BasePrimitiveConvertationInterface as BPCI
from .utils import wrapcls
from .functional import Interpolate


@wrapcls
class Upsample(nn.Upsample, BPCI, subgroup="module"):
    layer_platform_name = "upsample"

    def _as_integer_ratio(self, scale_factor):
        return Interpolate._as_integer_ratio(scale_factor)

    @classmethod
    def _from_integer_ratio(cls, scale_factor):
        return Interpolate._from_integer_ratio(scale_factor)

    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer["mode"] = self.mode
        if not self.align_corners is None:
            layer["align_corners"] = self.align_corners
        if self.scale_factor:
            layer["scale_factor"] = self._as_integer_ratio(self.scale_factor)
        else:
            layer["size"] = self.size
        return layer

    @classmethod
    def fromPlatform(cls, layer, binary):
        scale_factor = cls._from_integer_ratio(layer.get("scale_factor"))
        upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode=layer.get("mode"),
            align_corners=layer.get("align_corners"),
        )
        return upsample
