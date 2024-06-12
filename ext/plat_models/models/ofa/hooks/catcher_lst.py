import copy
from typing import Tuple, List, Dict, Union

from torch.utils.hooks import RemovableHandle
from torch.nn import Module
from ...core import Tensor

from ..abstract.ofa_abstract import (
    WrapperModule,
    Backbone,
    DynamicModule,
    NASModule,
)

from ..backbones import backbone_name2class


class HookCatcherLst(WrapperModule):
    def __init__(self, module: Backbone, catch_counter: Union[None, int]) -> None:
        """Забирает карты признаков в модели при помощи хуков и упаковывает в лист

        В модель предоставляет информацию о том с каких слоёв и каким образом забирать карты признаков.
        Карты признаков могут забираться с входа или выхода.
        Собирает метаданные о модулях к которым цепляется

        Args:
            module - Backbone, модуль с которого будут забираться промежуточные карты
            catch_counter - количество карт признаков, с которых будут забираться фичи
            если None, то со всех, иначе последних n карт из списка в модели.
        """
        super().__init__(module)
        self.catch_counter = catch_counter
        self.hooks_list: List[Tensor] = []
        self.delete_handlers: List[RemovableHandle] = []
        self._update_hook_info()
        self._attach_hooks()
        self.module: Backbone

    @property
    def config(self):
        return {
            "name": self.__class__.__name__,
            "module": self.module.config,
            "catch_counter": self.catch_counter,
        }

    @classmethod
    def build_from_config(cls, config: dict):
        module_cls = backbone_name2class[config["module"]["name"]]
        catch_counter = config["catch_counter"]
        module = module_cls.build_from_config(config["module"])
        return cls(module, catch_counter)

    def forward(self, x: Tensor) -> List[Tensor]:
        self.hooks_list.clear()
        _ = self.module(x)
        return self.hooks_list

    def _update_hook_info(self):
        """Обновляет информацию о хуках на основе актуальной информации из модуля

        обрезает под количество хуков для забора"""
        self.hooks_info = self.module.layers_to_hook
        if self.catch_counter is not None:
            full_size = len(self.module.layers_to_hook)
            if full_size < self.catch_counter:
                raise ValueError(
                    f"catch_counter ({self.catch_counter}) must be less than \
                                 len of catch_keys ({full_size})"
                )
            catch_start_index = full_size - self.catch_counter
            self.hooks_info = self.hooks_info[catch_start_index:]

    def _attach_hooks(self):
        hooks_list = self.hooks_list
        for info in self.hooks_info:
            hook_module: Module = self.module.get_submodule(info["module"])
            info["module_conf"] = hook_module.config
            if info["hook_type"] == "pre":

                def _forward_pre_hook(self, input: Tuple[Tensor]) -> Tensor:
                    hooks_list.append(input[0])

                self.delete_handlers.append(
                    hook_module.register_forward_pre_hook(_forward_pre_hook)
                )

            elif info["hook_type"] == "forward":

                def _forward_hook(self, input: Tuple[Tensor], result) -> Tensor:
                    hooks_list.append(result)

                self.delete_handlers.append(
                    hook_module.register_forward_hook(_forward_hook)
                )
            else:
                raise NotImplementedError(
                    f"catch hook with {info['hook_type']} type not implemented"
                )

    def _clear_hooks(self):
        for h in self.delete_handlers:
            h.remove()
        self.delete_handlers.clear()


class DynamicHookCatcherLst(HookCatcherLst, NASModule):
    """Класс, который адекватно обрабатывает взятие подсети

    По идее он должен работать с динамическими модулями, но пока он может работать
    со всем"""

    SAMPLE_MODULE_CLS = HookCatcherLst

    def __init__(self, module: Backbone, catch_counter: Union[None, int]) -> None:
        super().__init__(module, catch_counter)
        self.module: Union[Backbone, DynamicModule, NASModule]

    def get_active_subnet(self) -> HookCatcherLst:
        """Этому методу не нужен конфиг подсети

        снимает хуки на время снятия"""

        self._clear_hooks()
        if hasattr(self.module, "get_active_subnet"):
            submodule = self.module.get_active_subnet()
        else:
            submodule = copy.deepcopy(self.module)
        self._attach_hooks()

        subnet = HookCatcherLst(submodule, self.catch_counter)
        return subnet

    def sample_active_subnet(self) -> Dict[str, List]:
        self._clear_hooks()
        subnet_config = self.module.sample_active_subnet()
        self._update_hook_info()
        self._attach_hooks()
        return subnet_config

    def get_active_arch_desc(self) -> dict:
        return copy.deepcopy(self.module.get_active_arch_desc())
    
    def get_subnets_grid(self) -> List[dict]:
        return self.module.get_subnets_grid()

    @property
    def active_out_channels(self):
        active_out_channels = self.module.active_out_channels
        if self.catch_counter is not None:
            active_out_channels = active_out_channels[-self.catch_counter :]
        return active_out_channels

    def set_active_subnet(self, subnet_config: Dict[str, List]) -> None:
        self._clear_hooks()
        self.module.set_active_subnet(**subnet_config)
        self._update_hook_info()
        self._attach_hooks()

    def set_max_net(self) -> None:
        self._clear_hooks()
        self.module.set_max_net()
        self._update_hook_info()
        self._attach_hooks()

    def set_min_net(self) -> None:
        self._clear_hooks()
        self.module.set_min_net()
        self._update_hook_info()
        self._attach_hooks()
