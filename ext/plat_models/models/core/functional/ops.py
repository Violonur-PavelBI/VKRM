from models.core.registry.misc import REFSTR
from dataclasses import dataclass
import torch
from ..base_primitive import (
    LAYER_PLATFORM_NAME,
    _BasePrimitiveConvertationInterface as BPCI,
)
from ..utils.convert_bases import convert_base, get_inputs, thruth_getitem_and_getattr


from functools import wraps
from typing import Union

from torch.nn.functional import interpolate


class Concat(BPCI, subgroup="functional"):
    layer_platform_name = "concat"
    alternative_fx_name = "_VariableFunctionsClass.cat"

    @classmethod
    def toPlatform(cls, fx_node, Converter):
        layer = super(Concat, cls).toPlatform(fx_node, Converter)
        inputs = get_inputs(fx_node)
        layer["input"] = inputs
        layer["implicit"] = False
        return layer

    @classmethod
    def fromPlatform(cls, layer, tensors):
        inputs = list(map(lambda item: tensors[item], layer["input"]))
        # FIXME
        return torch.cat(inputs, dim=1)

    __call__ = staticmethod(torch.cat)
    # @wraps(torch.cat)
    # def __call__(cls, tensors, dim: int, out: Union[torch.Tensor, None] = None):
    #     return torch.cat(tensors, dim, out)


class Flatten(BPCI, subgroup="functional"):
    layer_platform_name = "flatten"

    __call__ = staticmethod(torch.flatten)

    @classmethod
    def fromPlatform(cls, layer, binary):
        raise NotImplementedError


class Abs(BPCI, subgroup="functional"):
    layer_platform_name = "abs"


class Add(BPCI, subgroup="functional"):
    layer_platform_name = "add"

    @classmethod
    def toPlatform(cls, fx_node, Converter):
        layer = super(Add, cls).toPlatform(fx_node, Converter)
        inputs = get_inputs(fx_node)
        layer.pop("input")
        layer["input1"] = inputs[0]
        layer["input2"] = inputs[1]
        layer["a1"] = 1
        layer["a2"] = 1
        return layer

    @classmethod
    def fromPlatform(cls, layer, tensors):
        # FIXME
        a1 = layer["a1"]
        a2 = layer["a2"]
        input1 = tensors[layer["input1"]]
        input2 = tensors[layer["input2"]]
        return torch.add(input1 * a1, input2 * a2)


class Mul(BPCI, subgroup="functional"):
    layer_platform_name = "mul"

    @classmethod
    def toPlatform(cls, fx_node, Converter):
        layer = super(Mul, cls).toPlatform(fx_node, Converter)
        inputs = get_inputs(fx_node)
        layer.pop("input")
        layer["input1"] = inputs[0]
        layer["input2"] = inputs[1]
        layer["relu"] = False
        return layer

    @classmethod
    def fromPlatform(cls, layer, tensors):
        # FIXME
        input1 = tensors[layer["input1"]]
        input2 = tensors[layer["input2"]]
        return torch.mul(input1, input2)


class Max(BPCI, subgroup="functional"):
    layer_platform_name = "max"

    @dataclass
    class MaxDataclass:
        input1: REFSTR
        input2: REFSTR
        type: LAYER_PLATFORM_NAME

    @classmethod
    def toPlatform(cls, fx_node, Converter):
        # TODO Исправить метод для случая одного инпута
        layer = super(Max, cls).toPlatform(fx_node, Converter)
        inputs = get_inputs(fx_node)
        layer.pop("input")
        layer["input1"] = inputs[0]
        if len(inputs) > 1:
            layer["input2"] = inputs[1]
        # TODO: layer definition by pydantic to validate
        # Example:
        # return MaxDataclass(**layer).to_dict()
        return layer

    @classmethod
    def fromPlatform(cls, layer, tensors):
        input1 = tensors[layer["input1"]]
        input2 = tensors.get(layer.get("input2"))
        # FIXME
        if input2 is None:
            return torch.max(input1)
        else:
            return torch.max(input1, input2)


class Interpolate(BPCI, subgroup="functional"):
    layer_platform_name = "interpolate"

    @classmethod
    def _as_integer_ratio(cls, scale_factor):
        if scale_factor:
            if isinstance(scale_factor, (tuple, list)):
                new_scale_factor = list()
                for item in scale_factor:
                    new_scale_factor.append(list(item.as_integer_ratio()))
            else:
                new_scale_factor = [scale_factor.as_integer_ratio(), scale_factor.as_integer_ratio()]
            return new_scale_factor
        else:
            return scale_factor

    @classmethod
    def _from_integer_ratio(cls, scale_factor):
        if scale_factor:
            if isinstance(scale_factor[0], list):
                new_scale_factor = list()
                for item in scale_factor:
                    new_scale_factor.append(item[0]/item[1])
                new_scale_factor = tuple(new_scale_factor)
            else:
                new_scale_factor = scale_factor[0]/scale_factor[1]
            return new_scale_factor
        else:
            return scale_factor

    @classmethod
    def toPlatform(cls, fx_node, Converter):
        layer = super(Interpolate, cls).toPlatform(fx_node, Converter)
        layer["mode"] = fx_node.kwargs.get("mode")
        if not fx_node.kwargs.get("align_corners") is None:
            layer["align_corners"] = fx_node.kwargs.get("align_corners")
        if fx_node.kwargs.get("scale_factor"):
            layer["scale_factor"] = cls._as_integer_ratio(fx_node.kwargs.get("scale_factor"))
        if type(fx_node.kwargs.get("size")) in [int, tuple]:
            size = fx_node.kwargs.get("size")
            if isinstance(size, tuple):
                size = list(size)
            layer["size"] = fx_node.kwargs.get("size")
        elif type(fx_node.kwargs.get("size")) is type(None):
            pass
        else:
            size_input_node = thruth_getitem_and_getattr(fx_node.kwargs.get("size"))
            if not isinstance(size_input_node, str):
                layer["ref_size"] = size_input_node.meta["name"]
            else:
                # if already string
                layer["ref_size"] = size_input_node
        return layer

    @classmethod
    def fromPlatform(cls, layer, tensors):
        input_ = tensors[layer["input"]]
        scale_factor = None
        if "ref_size" in layer.keys():
            # TODO change to use RefNode
            # HARDCODE then getattr is shape by hxw
            size = tensors[layer["ref_size"]].shape[-2:]
        elif "size" in layer.keys():
            size = layer.get("size")
        elif "scale_factor" in layer.keys():
            size = None
            scale_factor = cls._from_integer_ratio(layer.get("scale_factor"))
        return interpolate(
            input=input_,
            size=size,
            scale_factor=scale_factor,
            mode=layer["mode"],
            align_corners=layer.get("align_corners"),
        )
