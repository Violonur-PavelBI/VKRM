import torch.nn as nn

from .base_primitive import _BasePrimitiveConvertationInterface as BPCI
from .utils import wrapcls


class BPCIPool1d(BPCI):
    layer_platform_name = "pool1d"

    @classmethod
    def fromPlatform(cls, layer, binary):
        if layer["global"] is False:
            layer_keys = layer.keys()
            kernel_size = layer["kernel_length"]
            stride = layer["stride_length"]
            padding = (layer["pad_left"], layer["pad_right"])
            dilation = layer["dilation"] if "dilation" in layer_keys else 1
            return_indices = False
            ceil_mode = layer["ceil_mode"]
            if layer["operation"] == "max":
                pool = nn.MaxPool1d(
                    kernel_size, stride, padding, dilation, return_indices, ceil_mode
                )
            elif layer["operation"] == "avg":
                pool = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode)
        else:
            if layer["operation"] == "max":
                pool = nn.AdaptiveMaxPool1d(1)
            elif layer["operation"] == "avg":
                pool = nn.AdaptiveAvgPool1d(1)
        return pool


@wrapcls
class MaxPool1d(nn.MaxPool1d, BPCIPool1d, subgroup="module"):

    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer["kernel_lenght"] = self.kernel_size
        layer["stride_lenght"] = self.stride
        layer["pad_left"] = self.padding
        layer["pad_right"] = self.padding
        layer["ceil_mode"] = self.ceil_mode
        layer["operation"] = "max"
        layer["global"] = False
        return layer


@wrapcls
class AvgPool1d(nn.AvgPool1d, BPCIPool1d, subgroup="module"):

    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer["kernel_lenght"] = self.kernel_size
        layer["stride_lenght"] = self.stride
        layer["pad_left"] = self.padding
        layer["pad_right"] = self.padding
        layer["ceil_mode"] = self.ceil_mode
        layer["operation"] = "avg"
        layer["global"] = False
        return layer


@wrapcls
class AdaptiveAvgPool1d(nn.AdaptiveAvgPool1d, BPCIPool1d, subgroup="module"):

    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer["operation"] = "avg"
        layer["global"] = True
        return layer


@wrapcls
class AdaptiveMaxPool1d(nn.AdaptiveMaxPool1d, BPCIPool1d, subgroup="module"):

    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer["operation"] = "max"
        layer["global"] = True
        return layer



class BPCIPool2d(BPCI):
    layer_platform_name = "pool2d"

    @classmethod
    def fromPlatform(cls, layer, binary):
        if layer["global"] is False:
            layer_keys = layer.keys()
            kernel_size = (layer["kernel_height"], layer["kernel_width"])
            stride = (layer["stride_height"], layer["stride_width"])
            padding = (layer["pad_up"], layer["pad_left"])
            dilation = layer["dilation"] if "dilation" in layer_keys else 1
            return_indices = False
            ceil_mode = layer["ceil_mode"]
            if layer["operation"] == "max":
                pool = nn.MaxPool2d(
                    kernel_size, stride, padding, dilation, return_indices, ceil_mode
                )
            elif layer["operation"] == "avg":
                pool = nn.AvgPool2d(kernel_size, stride, padding, ceil_mode)
        else:
            if layer["operation"] == "max":
                pool = nn.AdaptiveMaxPool2d(1)
            elif layer["operation"] == "avg":
                pool = nn.AdaptiveAvgPool2d(1)
        return pool


@wrapcls
class MaxPool2d(nn.MaxPool2d, BPCIPool2d, subgroup="module"):
    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer["kernel_height"] = self.kernel_size
        layer["kernel_width"] = self.kernel_size
        layer["stride_height"] = self.stride
        layer["stride_width"] = self.stride
        layer["pad_left"] = self.padding
        layer["pad_right"] = self.padding
        layer["pad_up"] = self.padding
        layer["pad_down"] = self.padding
        layer["ceil_mode"] = self.ceil_mode
        layer["operation"] = "max"
        layer["global"] = False
        return layer


@wrapcls
class AvgPool2d(nn.AvgPool2d, BPCIPool2d, subgroup="module"):
    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer["kernel_height"] = self.kernel_size
        layer["kernel_width"] = self.kernel_size
        layer["stride_height"] = self.stride
        layer["stride_width"] = self.stride
        layer["pad_left"] = self.padding
        layer["pad_right"] = self.padding
        layer["pad_up"] = self.padding
        layer["pad_down"] = self.padding
        layer["ceil_mode"] = self.ceil_mode
        layer["operation"] = "avg"
        layer["global"] = False
        return layer


@wrapcls
class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, BPCIPool2d, subgroup="module"):
    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer["operation"] = "avg"
        layer["global"] = True
        return layer


@wrapcls
class AdaptiveMaxPool2d(nn.AdaptiveMaxPool2d, BPCIPool2d, subgroup="module"):
    def toPlatform(self, fx_node, Converter):
        layer = super().toPlatform(fx_node, Converter)
        layer["operation"] = "max"
        layer["global"] = True
        return layer
