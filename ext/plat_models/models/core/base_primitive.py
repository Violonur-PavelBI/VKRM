import torch.nn as nn
from abc import ABC, abstractclassmethod
from .utils.convert_bases import convert_base, get_inputs
from .utils import wrapcls
import inspect
from typing import Type, Union, Dict


SIMPLE = Union[str, float, int, bool]
LAYER_PLATFORM_NAME = str  # Имя класса\функции в формате платформы
PRIMITIVE_REGISTRY = {}
REVERSED_REGISTRY = {"functional": {}, "module": {}}


class _BasePrimitiveConvertationInterface(ABC):
    layer_platform_name: str

    def __init_subclass__(cls, subgroup=None) -> None:
        if inspect.ismethod(cls.toPlatform) and cls.toPlatform.__self__ is cls:
            if hasattr(cls, "alternative_fx_name"):
                key = cls.alternative_fx_name
            else:
                key = cls.layer_platform_name
            PRIMITIVE_REGISTRY[key] = cls.toPlatform
            if subgroup is not None:
                REVERSED_REGISTRY[subgroup][cls.layer_platform_name] = cls.fromPlatform
        else:
            """This meants that toPlatform is method, so we only need to register cls
            in reverse_registry
            """
            if subgroup == "module":

                REVERSED_REGISTRY[subgroup][cls.layer_platform_name] = cls.fromPlatform

            else:
                pass
                # raise NotImplementedError("must be passed subgroup in inheritance")
            return super().__init_subclass__()

    @classmethod
    # @abstractclassmethod
    def toPlatform(
        cls, fx_node, Converter
    ) -> Dict[str, Union[SIMPLE, Dict[str, SIMPLE]]]:
        layer = convert_base(fx_node, Converter)
        inputs = get_inputs(fx_node)
        layer["type"] = cls.layer_platform_name
        layer["input"] = inputs[0]
        return layer

    @abstractclassmethod
    def fromPlatform(cls, layer, binary):
        pass
