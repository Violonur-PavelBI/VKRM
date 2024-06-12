import numpy as np

import torch
import torch.nn as nn

from .base_primitive import _BasePrimitiveConvertationInterface as BPCI
from .utils import wrapcls
from .utils._types import NPTYPES
from .utils.convert_bases import get_weights, get_biases


@wrapcls
class Linear(nn.Linear, BPCI, subgroup="module"):
    layer_platform_name = "dense"

    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer = get_weights(layer, self, Converter)
        layer = get_biases(layer, self, Converter)
        layer["in_features"] = self.in_features
        layer["out_features"] = self.out_features
        layer["relu"] = False
        return layer

    @classmethod
    def fromPlatform(cls, layer, binary):
        in_features = layer.get("in_features")
        out_features = layer.get("out_features")
        bias = layer.get("use_biases")
        dense = nn.Linear(in_features, out_features, bias)
        bin_start = layer.get("weights")[0]
        bin_end = bin_start + layer.get("weights")[1]
        weight = torch.from_numpy(
            np.frombuffer(
                binary[bin_start:bin_end], dtype=NPTYPES.get(layer.get("weights_type"))
            )
        )
        dense.weight = nn.Parameter(
            weight.view(out_features, in_features), requires_grad=True
        )
        if bias:
            bin_start = layer.get("biases")[0]
            bin_end = bin_start + layer.get("biases")[1]
            dense.bias = nn.Parameter(
                torch.from_numpy(
                    np.frombuffer(
                        binary[bin_start:bin_end],
                        dtype=NPTYPES.get(layer.get("biases_type")),
                    )
                ),
                requires_grad=True,
            )
        return dense
