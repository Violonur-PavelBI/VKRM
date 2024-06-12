"""В данном файле объявлены классы, которые должны являться базовыми для всех, являющиеся частью OFA
данные классы задают интерфейс, а также будут использоваться в проверках, когда нужно определить
насколько динамичным (В контексте алгоритма OFA) является объект

Также в данном файле можно найти базовые классы, относящиеся к структуре Backbone, Neck, Head
относительно них нужно решить насколько они нужны

возможно данный файл стоит переместить
"""


from abc import ABC, abstractmethod
from typing import List, Union

from torch import Tensor
from torch.nn import Module

from ..primitives.static import MyNetwork
from .ofa_typing import HookInfo


class BasicModule(MyNetwork, ABC):
    """Абстрактный класс для статических модулей, использующихся в OFA
    этот класс должен заменить класс MyNetwork
    в этом классе нужно пересмотреть то, как объявляется интерфейс, т.е.
    заменить коснтрукции `raise NotImplemented` на использование `@abstractmethod`
    """


class DynamicModule(BasicModule):
    """Абстрактный класс модуля, параметры которого не подбираются NAS-ом,
    но зависят от других компонент и потому динамические.

    Под зависимостью подразумевается, что они подстраиваются под вход, форма которого может меняться
    меняется она адаптивно во время "forward"
    """

    SAMPLE_MODULE_CLS: BasicModule

    @abstractmethod
    def get_active_subnet(self) -> BasicModule:
        """Функция возвращает активную подсеть"""
        pass

    @abstractmethod
    def set_active_subnet(self, **kwargs) -> None:
        """Функция устанавливает активную подсеть"""
        pass

    @abstractmethod
    def set_max_net(self) -> None:
        """Функция устанавливает максимальную подсеть"""
        pass

    @abstractmethod
    def set_min_net(self) -> None:
        """Функция устанавливает минимальную подсеть"""
        pass


class NASModule(DynamicModule):
    """Абстрактный класс модуля, параметры которой подбираются NAS-ом

    данный модуль умеет менять свою конфигурацию"""

    @abstractmethod
    def sample_active_subnet(self, **kwargs) -> dict:
        """Функция сэмплирует случайную конфигурацию архитектуры,
        в соответсвии с ней устанавливает активную подсеть
        и возвращает словарь с описанием конфигурации"""
        pass

    @abstractmethod
    def get_active_arch_desc(self) -> dict:
        """Функция возвращает словарь с описанием конфигурации активной подсети."""
        pass

    @abstractmethod
    def get_subnets_grid(self) -> List[dict]:
        """Функция возвращает словарь с описаниями конфигураций основных подсетей."""
        pass


# TODO Переписать докстринги, нормально реализовать, начать использовать
#  Можно сюда ещё и registry вкорячить, чтобы в этих классах регистрировались


# ? Абстрактный класс Backbone есть в models.backbones.abstract, можно сделать от него наследование
class Backbone(ABC):
    """Абстрактный класс для энкодера
    Возможно тут можно отоборазить, нужно решить стоит ли отображать, что он состоит из
    Сети и Штуки, которая забирает промежуточные выходы в Neck

    Объект данного класса должен содержать список слоёв с которых можно забрать фичи при помощи хуков
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self.layers_to_hook: List[HookInfo]


class WrapperModule(BasicModule):
    """Класс для модулей которые, например, забирают фичи с Backbone

    дело в том, что в классе, который является итоговой сетью CompositeNet
      необходимо произовдить проверку?"""

    def __init__(self, module: BasicModule) -> None:
        super().__init__()
        self.module = module

    def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
        """Производит прокидывание к своему подмодулю в случае неудачной попытки доступа

        Возможно это не правильно (потому что фактически интерфейс шире, чем у данного класса
        и не парсится в дебагере), зато очень гибко и абстрактно"""
        try:
            attribute = super().__getattr__(name)
        except AttributeError as e:
            attribute = getattr(self.module, name)
        return attribute


class Neck(ABC):
    """Часть, которая стоит после энкодера, FPN и иже с ними"""


class Head(ABC):
    """Модуль для целевой задачи"""
