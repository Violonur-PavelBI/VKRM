from abc import abstractmethod
from typing import Dict, Sequence, Union
from .base_primitive import _BasePrimitiveConvertationInterface as BPCI
from .meta import _CoreInterface, ABCEnforcedProperties
import torch.nn as nn
from torch import Tensor
from torch.nn import Module as TorchModule
from .mappings import Plat2Torch_fromJSON, Torch2Plat_fromFX


class Module(_CoreInterface, metaclass=ABCEnforcedProperties):
    @abstractmethod
    def __init__(self) -> None:
        """Must be implemented in child class"""
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Must be implemented in child class"""
        pass


class _CustomLeafModule(Module, BPCI, subgroup="custom"):
    layer_platform_name: str  # Имя слоя "type" в формате платформы

    def __init_subclass__(cls) -> None:
        assert hasattr(cls, "layer_platform_name"), (
            "Custom layer platform name must be defined"
            "in class description by attribute `layer_platform_name`"
        )
        mkey = cls.layer_platform_name
        assert mkey not in Torch2Plat_fromFX.keys()
        assert mkey not in Plat2Torch_fromJSON["module"].keys()
        Torch2Plat_fromFX[mkey] = cls.toPlatform
        Plat2Torch_fromJSON["module"][mkey] = cls.fromPlatform
        return super().__init_subclass__()

    @abstractmethod
    def toPlatform(
        self,
        model_path: str,
        input_example: Union[Tensor, Sequence[Tensor], Dict[str, Tensor]],
        output_example: Union[Tensor, Sequence[Tensor], Dict[str, Tensor]],
        verbose=False,
    ):
        layer = convert_base(fx_node, Converter)
        inputs = get_inputs(fx_node)
        layer["type"] = layer_platform_name
        layer["input"] = inputs[0]
        return super().toPlatform(model_path, input_example, output_example, verbose)
