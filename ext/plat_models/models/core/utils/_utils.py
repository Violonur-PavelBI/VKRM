from collections import OrderedDict

import torch.fx
from torch.nn import Module as TorchModule
from inspect import getdoc


class LayerAttrTypeError(Exception):
    __module__ = Exception.__module__
    typing = (bool, int, float, str, list, type(None), slice)

    def __init__(self, layer, key, value):
        name = layer["name"]
        type_ = layer["type"]
        value_type = type(value)
        message = f"OrderedDict have bad type of item.\n"
        message += f"   Layer name:         {name}\n"
        message += f"   Layer type:         {type_}\n"
        message += f"   Arreibute key:      {key}\n"
        message += f"   Attribute type:     {value_type}\n"
        super().__init__(message)


def wrapcls(cls, original_cls=None):

    if original_cls is None:
        original_cls = cls.mro()[0]
    doc = getdoc(original_cls)
    cls.__doc__ = f"""Wraps {original_cls.__name__}\n""" + doc
    return cls
