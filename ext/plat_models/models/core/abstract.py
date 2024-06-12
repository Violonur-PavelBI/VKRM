from .module import Module
from .tensor import Tensor
from .registry import Registry, build_from_cfg
from ._partial_loading import _LoadingInterface
from abc import abstractmethod
from typing import Literal, Optional, Union, Tuple, List

# RGB = "RGB"
# BGR = "BGR"
# RGBA = "RGBA"
# SWIR = "SWIR"
# # IR1 = "IR1"
# SPECTRAL = "SPECTRAL"
# GRAYSCALE = "GRAYSCALE"

NODES = Registry("nodes", build_from_cfg)
NETWORKS = Registry("networks", build_from_cfg, locations=["models.networks"])
BACKBONES = Registry(
    "backbones", parent=NETWORKS, locations=["models.backbones"], scope="backbones"
)
HEADS = Registry("heads", parent=NETWORKS, scope="heads", locations=["models.heads"])
NECKS = Registry("necks", parent=NETWORKS, scope="necks")
CLASMODELS = Registry(
    "classification",
    parent=NETWORKS,
    scope="classification",
    locations=["models.networks.classification"],
)
SEGMMODELS = Registry(
    "segmentation",
    parent=NETWORKS,
    scope="segmentation",
    locations=["models.networks.segmentation"],
)
DETMODELS = Registry(
    "object_detection",
    parent=NETWORKS,
    scope="object_detection",
    locations=["models.networks.object_detection"],
)
IMGMODELS = Registry("imgmodel", parent=NETWORKS, scope="img")


class Node(Module):
    r"""Класс, определяющий необходимые атрибуты, для описания узлов (базовых блоков)
    Атрибуты:
    - `output_channels` - глубина выходной карты признаков.
    - `downsample_factor` - показатель отношения $ \frac{HW_{in}}{HW_{out}} $ , для ряда декодеров,
    желательно чтобы был целой степенью двойки.
    - `input_channels` -  глубина входной карты признаков.
    """
    output_channels: int
    downsample_factor: int
    input_channels: int

    def __init_subclass__(cls, register=True) -> None:
        if register:
            NODES.register_module(module=cls)
        return super().__init_subclass__()


class Backbone(Module):
    r"""Класс, определяющий необходимые атрибуты и реализующий общие методы для описания бэкбонов

    Атрибуты:
    - `output_channels` - глубина выходной карты признаков.
    - `downsample_factor` - показатель отношения $ \frac{HW_{in}}{HW_{out}} $ , для ряда декодеров,
    желательно чтобы был целой степенью двойки.
    - `input_channels` -  глубина входной карты признаков.
    """
    output_channels: int
    downsample_factor: int
    input_channels: int
    _trained_additional_meta: dict(channel_format=Literal["RGB", "IR", "RGBA", "BGR"])

    def __init_subclass__(cls, register=True) -> None:
        if register:
            BACKBONES.register_module(module=cls)
        return super().__init_subclass__()

    @abstractmethod
    def __init__(self) -> None:
        """Must be implemented in child class"""
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Must be implemented in child class"""
        pass


class Head(Module):
    r"""Класс, определяющий необходимые атрибуты и реализующий общие методы для описания хедов. NOTE: WORK IN PROGRESS
    Атрибуты:
    - `scale_adaptive` -  адаптивна к размеру входа или нет, также нужно выставить `lup_factor` либо `out_spatial_sizes`.
    - `out_spatial_sizes` - опциональный, выставляется в том случае если нельзя указать `lup_factor` и голова адаптивная, например имеет AdaptiveAvgPool.
    - `lup_factor` - 'local upsample factor', опциональный, во сколько раз повышает дискретизацию, показатель отношения
    $ \frac{HW_{out}}{HW_{in}} $ где HW_{out} и HW_{in} локальные, для ряда декодеров, указывать явно в том случае если не адаптивный,
    тоесть без интерполяции к произвольному размеру, желательно чтобы был целой степенью двойки.
    - `num_classes` - опциональный,число классов, если голова для семантической задачи. (классификация, сегм-1D, 2D, ..., идентификация)
    - `output_channels` - опциональный, глубина выходной карты признаков.
    """
    # TODO: FIRST ADD CONCEPT OF NECKS
    # TODO: refactor heads. Maybe split to 2 types or more of heads.
    scale_adaptive: bool
    out_spatial_sizes: Optional[Tuple[int]]
    lup_factor: Optional[int]
    num_classes: Optional[int]
    output_channels: Optional[Union[List[int], int]]


class AtomicNetwork(Module):
    """
    Абстрактный класс моделей, которые нельзя разделить на декодер\энкодер,
    либо статейные модели.
    """

    num_classes: Optional[int]
    input_channels: int

    def __init_subclass__(cls, register=True) -> None:
        if register:
            NETWORKS.register_module(module=cls)
        return super().__init_subclass__()


class AbsSegm2DModel(_LoadingInterface, Module):
    """Класс, отображающий сеть сегментации.
    Определяет атрибуты, шаблона сети собранной из backbone и head сегментации

    Атрибуты:
    - `backbone`: сеть - кодировщик (энкодер).
    - `head`: - прикрепленный декодер
    - `num_classes`: - число классов. Является выходной глубиной.
    - `pretrained`: - булев флаг, для использовани предобученных весов.

    """

    backbone: Backbone
    head: Module
    num_classes: int
    pretrained: Optional[bool] = False
    graph_module_order = {"backbone": ["head"]}

    def __init_subclass__(cls, register=True) -> None:
        if register:
            SEGMMODELS.register_module(module=cls)
        return super().__init_subclass__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class AbsDetect2DModel(_LoadingInterface, Module):
    """Класс, отображающий абстрактный head (декодер) для задач классификации
    Атрибуты:
    - `backbone`: сеть - кодировщик (энкодер).
    - `num_classes`: - число классов. Является выходной глубиной.
    - `pretrained`: - булев флаг, для использовани предобученных весов.
    - `head`: - прикрепленный декодер
    """

    backbone: Backbone
    num_classes: int
    pretrained: Optional[bool] = False
    head: Module
    graph_module_order = {"backbone": ["head"]}

    def __init_subclass__(cls, register=True) -> None:
        if register:
            DETMODELS.register_module(module=cls)
        return super().__init_subclass__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class AbsClass2DModel(_LoadingInterface, Module):
    """Класс, отображающий абстрактный head (декодер) для задач классификации
    Атрибуты:
    - `backbone`: сеть - кодировщик (энкодер).
    - `num_classes`: - число классов. Является выходной глубиной.
    - `pretrained`: - булев флаг, для использовани предобученных весов.
    - `head`: - прикрепленный декодер
    """

    backbone: Backbone
    num_classes: int
    pretrained: Optional[bool] = False
    head: Module
    graph_module_order = {"backbone": ["head"]}

    def __init_subclass__(cls, register=True) -> None:
        if register:
            CLASMODELS.register_module(module=cls)
        return super().__init_subclass__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass
