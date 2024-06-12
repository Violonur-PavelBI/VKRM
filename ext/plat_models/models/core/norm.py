from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.fx.node import Node
    from .converter import ConverterTorch2Plat
from .base_primitive import _BasePrimitiveConvertationInterface as BPCI
from .utils import wrapcls
from .utils._types import NPTYPES, TTYPES
from .utils.convert_bases import (
    get_norm_params,
    update_binary_file,
)


@wrapcls
class BatchNorm2d(nn.BatchNorm2d, BPCI, subgroup="module"):
    # TODO Rename all Norm2d to BatchNorm2d in /models/
    layer_platform_name = "batchnorm2d"

    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer = get_norm_params(layer, self, Converter)
        mean = self.running_mean
        mean_pointers = update_binary_file(mean, Converter)
        layer["mean"] = mean_pointers
        layer["mean_type"] = TTYPES[mean.dtype]
        var = self.running_var
        var_pointers = update_binary_file(var, Converter)
        layer["var"] = var_pointers
        layer["var_type"] = TTYPES[var.dtype]
        layer["type"] = "batchnorm2d"
        layer["num_features"] = self.num_features
        layer["training"] = self.training
        layer["momentum"] = self.momentum
        layer["keep_stats"] = self.track_running_stats
        layer["relu"] = False
        return layer

    @classmethod
    def fromPlatform(cls, layer, binary):
        num_features = layer.get("num_features")
        eps = layer.get("eps")
        momentum = layer.get("momentum")
        affine = layer.get("training")
        track_running_stats = layer.get("keep_stats")
        batchnorm2d = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        if affine is not True:
            bin_start = layer.get("gamma")[0]
            bin_end = bin_start + layer.get("gamma")[1]
            batchnorm2d.weight = nn.Parameter(
                torch.from_numpy(
                    np.frombuffer(
                        binary[bin_start:bin_end],
                        dtype=NPTYPES[layer.get("gamma_type")],
                    )
                ),
                requires_grad=True,
            )
            bin_start = layer.get("beta")[0]
            bin_end = bin_start + layer.get("beta")[1]
            batchnorm2d.bias = nn.Parameter(
                torch.from_numpy(
                    np.frombuffer(
                        binary[bin_start:bin_end], dtype=NPTYPES[layer.get("beta_type")]
                    )
                ),
                requires_grad=True,
            )
        if track_running_stats:
            bin_start = layer.get("mean")[0]
            bin_end = bin_start + layer.get("mean")[1]
            batchnorm2d.running_mean = nn.Parameter(
                torch.from_numpy(
                    np.frombuffer(
                        binary[bin_start:bin_end], dtype=NPTYPES[layer.get("mean_type")]
                    )
                ),
                requires_grad=False,
            )
            bin_start = layer.get("var")[0]
            bin_end = bin_start + layer.get("var")[1]
            batchnorm2d.running_var = nn.Parameter(
                torch.from_numpy(
                    np.frombuffer(
                        binary[bin_start:bin_end], dtype=NPTYPES[layer.get("var_type")]
                    )
                ),
                requires_grad=False,
            )
        # TODO : batchnorm2d.stage.eval
        return batchnorm2d


class BatchNorm1d(nn.BatchNorm1d, BatchNorm2d, subgroup="module"):
    layer_platform_name = "batchnorm1d"

@wrapcls
class InstanceNorm2d(nn.InstanceNorm2d, BPCI, subgroup="module"):
    layer_platform_name = None

    def toPlatform(self, fx_node, Converter):
        raise NotImplementedError

    @classmethod
    def fromPlatform(cls, layer, binary):
        raise NotImplementedError


@wrapcls
class GroupNorm(nn.GroupNorm):
    layer_platform_name = None

    def toPlatform(self, fx_node, Converter):
        raise NotImplementedError

    @classmethod
    def fromPlatform(cls, layer, binary):
        raise NotImplementedError
