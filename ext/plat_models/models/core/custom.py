from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from torch.fx.node import Node
    from .converter import ConverterTorch2Plat

from .utils import wrapcls
from .utils._types import NPTYPES, TTYPES
from .utils.convert_bases import (
    update_binary_file,
    convert_base,
    get_inputs,
)
import warnings

from .module import _CustomLeafModule


class LinearOpt(_CustomLeafModule):
    layer_platform_name = "dense_opt"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        """Линейный слой с вытягиванием в вектор в начале форварда"""
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias, device, dtype)

    def forward(self, x: torch.Tensor):
        out = x.view(x.size(0), -1)
        # print(out.shape)
        out = self.linear(out)
        # print(out.shape)
        # print(self.linear.weight.shape)
        return out

    def toPlatform(self, fx_node: Node, Converter: ConverterTorch2Plat):
        layer = convert_base(fx_node, Converter)
        inputs = get_inputs(fx_node)
        layer["type"] = "dense"
        layer["type_version"] = type(self).layer_platform_name
        layer["input"] = inputs[0]
        layer["in_features"] = self.linear.in_features
        layer["out_features"] = self.linear.out_features

        weight = self.linear.weight
        weight_pointers = update_binary_file(data=weight, Converter=Converter)
        layer["weights"] = weight_pointers
        layer["weights_type"] = TTYPES[weight.dtype]

        bias = self.linear.bias
        if not bias is None:
            bias_pointers = update_binary_file(data=bias, Converter=Converter)
            layer["use_biases"] = True
            layer["biases"] = bias_pointers
            layer["biases_type"] = TTYPES[bias.dtype]
        else:
            layer["use_biases"] = False
        layer["relu"] = False
        return layer

    @classmethod
    def fromPlatform(cls, layer, binary):
        in_features = layer.get("in_features")
        out_features = layer.get("out_features")
        bias = layer.get("use_biases")
        bin_start = layer.get("weights")[0]
        bin_end = bin_start + layer.get("weights")[1]
        weight = torch.from_numpy(
            np.frombuffer(
                binary[bin_start:bin_end], dtype=NPTYPES.get(layer.get("weights_type"))
            )
        )
        dense = cls(in_features, out_features, bias)
        dense.linear.weight = nn.Parameter(
            weight.view(out_features, in_features), requires_grad=True
        )
        if bias:
            bin_start = layer.get("biases")[0]
            bin_end = bin_start + layer.get("biases")[1]
            dense.linear.bias = nn.Parameter(
                torch.from_numpy(
                    np.frombuffer(
                        binary[bin_start:bin_end],
                        dtype=NPTYPES.get(layer.get("biases_type")),
                    )
                ),
                requires_grad=True,
            )
        return dense
