import copy
from typing import Any, List, Dict, Mapping, Union

from ... import core as c

from ..abstract.ofa_abstract import BasicModule, DynamicModule, NASModule

from ..primitives.static import MyNetwork

from ..hooks import (
    HookCatcherLst,
    DynamicHookCatcherLst,
    backbone_hooks_name2class,
)
from ..necks import neck_name2class
from ..heads import head_name2class


class CompositeSubNet(c.Sequential, c.Module):
    #! TODO Refactor, класс должен наследоваться от MyNetwork
    get_parameters = MyNetwork.get_parameters
    get_bn_param = MyNetwork.get_bn_param
    weight_parameters = MyNetwork.weight_parameters

    def __init__(self, backbone_hooks: HookCatcherLst, neck, head) -> None:
        super().__init__()
        self.backbone_hooks: HookCatcherLst = backbone_hooks
        self.neck: Union[None, MyNetwork] = neck
        self.head: MyNetwork = head

    def __getattr__(self, name: str):
        """Временный метод для прокидывания запроса атрибутов к суперсети"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.backbone_hooks.module, name)

    @property
    def config(self):
        return {
            "name": self.__class__.__name__,
            "backbone_hooks": self.backbone_hooks.config,
            "neck": self.neck.config if self.neck is not None else None,
            "head": self.head.config,
        }

    @classmethod
    def build_from_config(cls, config: dict):
        backbone_hooks_cls: HookCatcherLst = backbone_hooks_name2class[
            config["backbone_hooks"]["name"]
        ]
        backbone_hooks = backbone_hooks_cls.build_from_config(config["backbone_hooks"])

        if config["neck"] is not None:
            neck_name = config["neck"]["name"]
            neck_cls: MyNetwork = neck_name2class[neck_name]
            neck = neck_cls.build_from_config(config["neck"])
        else:
            neck = None
        head_cls: MyNetwork = head_name2class[config["head"]["name"]]
        head = head_cls.build_from_config(config["head"])

        return cls(backbone_hooks, neck, head)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        current_keys = self.state_dict().keys()
        incoming_keys = list(state_dict.keys())
        for key in incoming_keys:
            if key not in current_keys:
                # AttrPassDDP
                if key[:7] == "module.":
                    modified_key = key[7:]
                else:
                    modified_key = "module." + key
                if modified_key in current_keys:
                    state_dict[modified_key] = state_dict.pop(key)

        return super().load_state_dict(state_dict, strict)

    def set_params(self, params: dict):
        if "head" in params and hasattr(self.head, "set_params"):
            self.head.set_params(params["head"])


class CompositeSuperNet(CompositeSubNet, NASModule):
    SAMPLE_MODULE_CLS = CompositeSubNet

    def __init__(self, backbone_hooks, neck, head) -> None:
        super().__init__(backbone_hooks, neck, head)
        self.backbone_hooks: DynamicHookCatcherLst
        self.neck: Union[None, DynamicModule, NASModule]
        self.head: Union[BasicModule, DynamicModule]

    def sample_active_subnet(self) -> Dict[str, dict]:
        backbone_config = self.backbone_hooks.sample_active_subnet()
        active_in_channels = self.backbone_hooks.active_out_channels

        if isinstance(self.neck, NASModule):
            neck_config = self.neck.sample_active_subnet(active_in_channels)
            active_in_channels = self.neck.active_out_channels
        elif isinstance(self.neck, DynamicModule):
            self.neck.set_active_subnet(active_in_channels)
            neck_config = None
        else:
            neck_config = None

        if isinstance(self.head, DynamicModule):
            self.head.set_active_subnet(active_in_channels)

        composite_arch_config = {
            "backbone": backbone_config,
            "neck": neck_config,
            "head": None,
        }
        return composite_arch_config

    def sample_active_backbone(self) -> Dict[str, dict]:
        """
        Метод ресемплирует случайную конфигурацию бэкбона,
        в соответсвии с которой устанавливает активную подсеть бэкбона и активное число входных каналов шеи,
        и возвращает словарь с описанием конфигурации активной композитной подсети.
        Метод поддерживается только в случае, если шея является NAS модулем.
        """
        if not isinstance(self.neck, NASModule):
            raise AssertionError("Neck must be a NASModule for this method.")

        backbone_config = self.backbone_hooks.sample_active_subnet()
        active_in_channels = self.backbone_hooks.active_out_channels

        neck_config = self.neck.get_active_arch_desc()
        neck_config["active_in_channels"] = active_in_channels
        self.neck.set_active_subnet(**neck_config)

        composite_arch_config = {
            "backbone": backbone_config,
            "neck": neck_config,
            "head": None,
        }
        return composite_arch_config

    def sample_active_neck(self) -> Dict[str, dict]:
        """
        Метод ресемплирует случайную конфигурацию шеи,
        в соответсвии с которой устанавливает активную подсеть шеи,
        и возвращает словарь с описанием конфигурации активной композитной подсети.
        Метод поддерживается только в случае, если шея является NAS модулем.
        """
        if not isinstance(self.neck, NASModule):
            raise AssertionError("Neck must be a NASModule for this method.")

        backbone_config = self.backbone_hooks.get_active_arch_desc()
        active_in_channels = self.backbone_hooks.active_out_channels

        neck_config = self.neck.sample_active_subnet(active_in_channels)

        composite_arch_config = {
            "backbone": backbone_config,
            "neck": neck_config,
            "head": None,
        }
        return composite_arch_config

    def get_active_arch_desc(self) -> dict:
        backbone_config = self.backbone_hooks.get_active_arch_desc()

        if isinstance(self.neck, NASModule):
            neck_config = self.neck.get_active_arch_desc()
        else:
            neck_config = None

        composite_arch_config = {
            "backbone": backbone_config,
            "neck": neck_config,
            "head": None,
        }
        return copy.deepcopy(composite_arch_config)

    def get_subnets_grid(self) -> List[dict]:
        original_arch_desc = self.get_active_arch_desc()

        backbone_desc_list = self.backbone_hooks.get_subnets_grid()
        if isinstance(self.neck, NASModule):
            neck_desc_list = self.neck.get_subnets_grid()
        else:
            neck_desc_list = [None]
        subnet_desc_list = []
        for backbone_desc in backbone_desc_list:
            for neck_desc in neck_desc_list:
                subnet_desc = {"backbone": backbone_desc, "neck": neck_desc}
                self.set_active_subnet(subnet_desc)
                subnet_desc = self.get_active_arch_desc()
                subnet_desc_list.append(subnet_desc)

        self.set_active_subnet(original_arch_desc)
        return subnet_desc_list

    def set_active_subnet(self, arch_config: Dict[str, dict]) -> None:
        self.backbone_hooks.set_active_subnet(arch_config["backbone"])
        active_in_channels = self.backbone_hooks.active_out_channels

        if isinstance(self.neck, NASModule):
            arch_config["neck"]["active_in_channels"] = active_in_channels
            self.neck.set_active_subnet(**arch_config["neck"])
            active_in_channels = self.neck.active_out_channels
        elif isinstance(self.neck, DynamicModule):
            self.neck.set_active_subnet(active_in_channels)

        if isinstance(self.head, DynamicModule):
            self.head.set_active_subnet(active_in_channels)

    def set_max_net(self) -> None:
        self.backbone_hooks.set_max_net()
        if self.neck is not None:
            self.neck.set_max_net()
        if isinstance(self.head, DynamicModule):
            self.head.set_max_net()

    def set_min_net(self) -> None:
        self.backbone_hooks.set_min_net()
        if self.neck is not None:
            self.neck.set_min_net()
        if isinstance(self.head, DynamicModule):
            self.head.set_min_net()

    def _get_submodule_subnet(self, module) -> MyNetwork:
        # TODO вынести в другое место, чтобы другие могли использовать
        if module is None:
            return None
        if isinstance(module, DynamicModule):
            return module.get_active_subnet()
        return copy.deepcopy(module)

    def get_active_subnet(self) -> CompositeSubNet:
        active_backbone_hooks = self._get_submodule_subnet(self.backbone_hooks)
        neck = self._get_submodule_subnet(self.neck)
        head = self._get_submodule_subnet(self.head)

        active_subnet = self.SAMPLE_MODULE_CLS(active_backbone_hooks, neck, head)
        return active_subnet
