import torch
from models.core.base_primitive import _BasePrimitiveConvertationInterface as BPCI
from ..base_primitive import _BasePrimitiveConvertationInterface as BPCI
from ..utils.convert_bases import convert_base, get_inputs


class GetItem(BPCI, subgroup="functional"):
    layer_platform_name = "getitem"

    @classmethod
    def toPlatform(cls, fx_node, Converter):
        layer = super(GetItem, cls).toPlatform(fx_node, Converter)
        layer["index"] = str(fx_node.args[1])
        return layer


class GetAttr(BPCI, subgroup="functional"):
    layer_platform_name = "getattr"


class View(BPCI, subgroup="functional"):
    __call__ = staticmethod(torch.Tensor.view)  # FIXME: TEST and FIX if not working
    layer_platform_name = "view"

    @classmethod
    def toPlatform(cls, fx_node, Converter):
        layer = super(View, BPCI).toPlatform(fx_node, Converter)
        raise NotImplementedError("View not implemented!")
