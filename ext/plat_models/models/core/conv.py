import numpy as np

import torch
import torch.nn as nn

from .base_primitive import _BasePrimitiveConvertationInterface as BPCI
from .utils import wrapcls
from .utils.convert_bases import get_filters, get_biases
from .utils._types import NPTYPES


@wrapcls
class Conv1d(nn.Conv1d, BPCI, subgroup="module"):
    layer_platform_name = "conv1d"

    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer = get_filters(layer, self, Converter)
        layer = get_biases(layer, self, Converter)
        layer["in_channels"] = self.in_channels
        layer["out_channels"] = self.out_channels
        layer["group"] = self.groups
        layer["kernel_lenght"] = self.kernel_size[0]
        layer["stride_lenght"] = self.stride[0]
        layer["pad_left"] = self.padding[0]
        layer["pad_right"] = self.padding[0]
        layer["dilation_lenght"] = self.dilation[0]
        return layer

    @classmethod
    def fromPlatform(cls, layer, binary):
        in_channels = layer.get("in_channels")
        out_channels = layer.get("out_channels")
        kernel_size = layer.get("kernel_lenght")
        stride = layer.get("stride_lenght")
        padding = (layer.get("pad_left"), layer.get("pad_right"))
        dilation = layer.get("dilation_lenght")
        groups = layer.get("group")
        bias = layer.get("use_biases")
        conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding[0],
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        bin_start = layer.get("filters")[0]
        bin_end = bin_start + layer.get("filters")[1]
        weight = torch.from_numpy(
            np.frombuffer(
                binary[bin_start:bin_end], dtype=NPTYPES.get(layer.get("filters_type"))
            )
        )
        conv1d.weight = nn.Parameter(
            weight.view(out_channels, int(in_channels / groups), kernel_size),
            requires_grad=True,
        )
        if bias:
            bin_start = layer.get("biases")[0]
            bin_end = bin_start + layer.get("biases")[1]
            conv1d.bias = nn.Parameter(
                torch.from_numpy(
                    np.frombuffer(
                        binary[bin_start:bin_end],
                        dtype=NPTYPES.get(layer.get("biases_type")),
                    )
                ),
                requires_grad=True,
            )
        # TODO : conv2d.stage.eval, fix padding: (cant, eat tuplr padding)
        return conv1d


@wrapcls
class Conv2d(nn.Conv2d, BPCI, subgroup="module"):
    layer_platform_name = "conv2d"

    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer = get_filters(layer, self, Converter)
        layer = get_biases(layer, self, Converter)
        layer["in_channels"] = self.in_channels
        layer["out_channels"] = self.out_channels
        layer["group"] = self.groups
        layer["kernel_height"] = self.kernel_size[0]
        layer["kernel_width"] = self.kernel_size[1]
        layer["stride_height"] = self.stride[0]
        layer["stride_width"] = self.stride[1]
        layer["pad_up"] = self.padding[0]
        layer["pad_left"] = self.padding[1]
        layer["pad_down"] = self.padding[0]
        layer["pad_right"] = self.padding[1]
        layer["dilation_height"] = self.dilation[0]
        layer["dilation_width"] = self.dilation[1]
        return layer

    @classmethod
    def fromPlatform(cls, layer, binary):
        in_channels = layer["in_channels"]
        out_channels = layer["out_channels"]
        kernel_size = (layer["kernel_height"], layer["kernel_width"])
        stride = (layer["stride_height"], layer["stride_width"])
        padding = (layer["pad_up"], layer["pad_left"])
        dilation = (layer["dilation_height"], layer["dilation_width"])
        groups = layer["group"]
        bias = layer["use_biases"]
        conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        bin_start = layer["filters"][0]
        bin_end = bin_start + layer["filters"][1]
        weight = torch.from_numpy(
            np.frombuffer(
                binary[bin_start:bin_end], dtype=NPTYPES[layer["filters_type"]]
            )
        )
        conv2d.weight = nn.Parameter(
            weight.view(
                out_channels, int(in_channels / groups), kernel_size[0], kernel_size[1]
            ),
            requires_grad=True,
        )
        if bias:
            bin_start = layer.get("biases")[0]
            bin_end = bin_start + layer.get("biases")[1]
            conv2d.bias = nn.Parameter(
                torch.from_numpy(
                    np.frombuffer(
                        binary[bin_start:bin_end],
                        dtype=NPTYPES.get(layer["biases_type"]),
                    )
                ),
                requires_grad=True,
            )
        # TODO : conv2d.stage.eval
        return conv2d


@wrapcls
class ConvTranspose2d(nn.ConvTranspose2d, BPCI, subgroup="module"):
    layer_platform_name = "deconv2d"

    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer = get_filters(layer, self, Converter)
        layer = get_biases(layer, self, Converter)
        layer["in_channels"] = self.in_channels
        layer["out_channels"] = self.out_channels
        layer["group"] = self.groups
        layer["kernel_height"] = self.kernel_size[0]
        layer["kernel_width"] = self.kernel_size[1]
        layer["stride_height"] = self.stride[0]
        layer["stride_width"] = self.stride[1]
        layer["pad_up"] = self.padding[1]
        layer["pad_left"] = self.padding[0]
        layer["pad_down"] = self.padding[1]
        layer["pad_right"] = self.padding[0]
        layer["output_padr"] = self.output_padding[0]
        layer["output_padd"] = self.output_padding[1]
        layer["dilation_height"] = self.dilation[0]
        layer["dilation_width"] = self.dilation[1]
        layer["relu"] = False
        return layer

    @classmethod
    def fromPlatform(cls, layer, binary):
        in_channels = layer.get("in_channels")
        out_channels = layer.get("out_channels")
        kernel_size = (layer.get("kernel_height"), layer.get("kernel_width"))
        stride = (layer.get("stride_height"), layer.get("stride_width"))
        padding = (layer.get("pad_up"), layer.get("pad_left"))
        out_padding = (layer.get("output_padr"), layer.get("output_padd"))
        dilation = (layer.get("dilation_height"), layer.get("dilation_width"))
        groups = layer.get("group")
        bias = layer.get("use_biases")
        deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=out_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        bin_start = layer.get("filters")[0]
        bin_end = bin_start + layer.get("filters")[1]
        weight = torch.from_numpy(
            np.frombuffer(
                binary[bin_start:bin_end], dtype=NPTYPES.get(layer.get("filters_type"))
            )
        )
        deconv.weight = nn.Parameter(
            weight.view(
                in_channels, int(out_channels / groups), kernel_size[0], kernel_size[1]
            ),
            requires_grad=True,
        )
        if bias:
            bin_start = layer.get("biases")[0]
            bin_end = bin_start + layer.get("biases")[1]
            deconv.bias = nn.Parameter(
                torch.from_numpy(
                    np.frombuffer(
                        binary[bin_start:bin_end],
                        dtype=NPTYPES.get(layer.get("biases_type")),
                    )
                ),
                requires_grad=True,
            )
        return deconv


# __all__ = [
#     obj.__name__
#     for obj in [
#         Conv1d,
#         Conv2d
#     ]
# ]
