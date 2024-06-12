from ..base_primitive import _BasePrimitiveConvertationInterface as BPCI
from ..utils.convert_bases import convert_base, get_inputs


class Input(BPCI, subgroup="functional"):
    layer_platform_name = "model_input"

    @classmethod
    def toPlatform(cls, fx_node, Converter):
        layer = convert_base(fx_node, Converter)
        layer["type"] = fx_node.meta["type"]
        layer["input"] = Converter.pass_inputs
        # To pass info between convertations of childs
        fx_node.meta["pass_inputs"] = Converter.pass_inputs
        # ....
        Converter.pass_inputs = []
        return layer


class Output(BPCI, subgroup="functional"):
    layer_platform_name = "model_output"

    @classmethod
    def toPlatform(cls, fx_node, Converter):
        layer = convert_base(fx_node, Converter)
        inputs = get_inputs(fx_node)
        layer["input"] = inputs
        layer["type"] = fx_node.meta["type"]
        return layer
