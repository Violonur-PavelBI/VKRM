"""This file contains bases layers construct for different type of layers"""
from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor
    from torch.fx.node import Node
    from torch.nn.modules.batchnorm import _NormBase

    from .. import Module
    from ..converter import ConverterTorch2Plat

from .symbols import DUBLICATE_SEP as DS
from ._types import TTYPES


def update_binary_file(data: Tensor, Converter: ConverterTorch2Plat) -> List[int, int]:
    """Записывает данные в файл, возвращает позиции указателей, между которым лежат данные

    Это вообще, по-моему, должно быть методом конвертера тут вообще по непонятной причине
    мы делаем запись руками, а потом обновляем состояния конвертера, если он может сделать это сам
    тем более, что конвертер умеет сам создавать этот бинарный файл
    """
    Converter.binary_file = open(Converter.binary_name, "a")
    data: np.ndarray = data.detach().numpy()
    data.tofile(Converter.binary_file)
    output = [
        Converter.binary_counter,
        Converter.binary_file.tell() - Converter.binary_counter,
    ]
    Converter.binary_counter = Converter.binary_file.tell()
    Converter.binary_file.close()
    return output


def convert_base(fx_node, Converter):
    ## Assigning name to use later in convertation
    ## Name must be unique, so renaming and changing handling of
    ## inputs of next layers, using "output" key to get inputs for
    ## next layers then we get them.
    layer = OrderedDict()
    name = fx_node.meta["name"]
    if fx_node.meta["name"] not in Converter.unique_names.keys():
        layer["name"] = name
        Converter.unique_names[name] = 0
    else:
        Converter.unique_names[name] += 1
        count = Converter.unique_names[name]
        layer["name"] = DS.join([name, str(count)])
    layer["output"] = layer["name"]
    fx_node.meta["output"] = layer["name"]  # to use by next layers
    return layer


def get_inputs(fx_node):
    input_nodes = fx_node.all_input_nodes
    inputs = list()
    for input_node in input_nodes:
        inputs.append(
            input_node.meta["output"]
        )  # using "output" list assignet by `convert_base`
    return inputs


def thruth_getitem_and_getattr(fx_node):
    cur_input = fx_node.all_input_nodes
    assert len(cur_input) == 1
    cur_input = cur_input[0]
    if cur_input.op == "call_function" and cur_input.meta["type"] in [
        "getitem",
        "getattr",
    ]:
        return thruth_getitem_and_getattr(cur_input)
    elif cur_input.op == "placeholder":
        if not cur_input.meta["pass_inputs"]:
            return cur_input
        else:
            assert len(cur_input.meta["pass_inputs"]) == 1
            return cur_input.meta["pass_inputs"][0]
    else:
        return cur_input


def get_weights(layer, module, Converter):
    weights = module.weight
    weights = update_binary_file(data=weights, Converter=Converter)
    layer["weights"] = weights
    layer["weights_type"] = TTYPES[weights.dtype]
    return layer


def get_filters(layer: OrderedDict, module: Module, Converter: ConverterTorch2Plat):
    weight = module.weight
    filters = update_binary_file(data=weight, Converter=Converter)
    layer["filters"] = filters
    layer["filters_type"] = TTYPES[weight.dtype]
    return layer


def get_biases(layer: OrderedDict, module: Module, Converter: ConverterTorch2Plat):
    if not module.bias is None:
        layer["use_biases"] = True
        bias = module.bias
        biases = update_binary_file(data=bias, Converter=Converter)
        layer["biases"] = biases
        layer["biases_type"] = TTYPES[bias.dtype]
    else:
        layer["use_biases"] = False
    return layer


def get_norm_params(
    layer: OrderedDict, module: _NormBase, Converter: ConverterTorch2Plat
):
    layer["eps"] = module.eps
    gamma = module.weight
    gamma = update_binary_file(gamma, Converter)
    layer["gamma"] = gamma
    layer["gamma_type"] = TTYPES[module.weight.dtype]
    beta = module.bias
    beta = update_binary_file(beta, Converter)
    layer["beta"] = beta
    layer["beta_type"] = TTYPES[module.bias.dtype]
    return layer
