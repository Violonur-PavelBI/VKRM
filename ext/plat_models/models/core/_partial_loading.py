from .module import Module
from typing import Any, Mapping, Optional, Union, Tuple, List, Dict
from copy import copy
from collections import OrderedDict


def _dict_flatten(
    dictionary: Dict[str, Union[Dict, List]], parent_key="", separator="."
) -> List[str]:
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, Mapping):
            items.extend(_dict_flatten(value, new_key, separator=separator))
        elif isinstance(value, list):
            items.extend([new_key + separator + leaf_key for leaf_key in value])
        else:
            raise TypeError(f"Wrong type of _load_order, {type(dictionary)}")
    return items


def _one_key_check(key: str, variants: List[str], load_modules: List[str]) -> bool:
    for variant in variants:
        variant = variant.split(".")
        if key in variant:
            idx = variant.index(key)
            while idx != 0:
                idx -= 1
                if variant[idx] not in load_modules:
                    raise TypeError(f"Load module {key} is dependent on {variant[idx]}")
            return True
    return False


class _LoadingInterface(Module):
    """Класс, определяющий необходимые атрибуты и реализующий общие методы для описания готовых сетей.
    На данный момент необходим только лишь для шаблонирования загрузки.
    Атрибуты:
        graph_module_order (Dict[str, Union[Dict, List]]): иерархический словарь, содержащий имена блоков, (бакбоунов, хедов, неков и т.п.)
        которые применяются последовательно (тоесть вход второго это выход первого) во время прямого прохода, начинаются от входа, смотри пример.

    """

    graph_module_order: Mapping[str, Union[Mapping, List]]

    @staticmethod
    def __merge_state_dict(
        initial_state_dict: Mapping[str, Any],
        state_dict_to_merge: Mapping[str, Any],
        keep_keys: List[str],
    ) -> Mapping[str, Any]:
        non_existent_keys = set(keep_keys) - set(
            [key.split(".")[0] for key in initial_state_dict.keys()]
        )
        if non_existent_keys:
            raise KeyError(
                f"Keep Keys {non_existent_keys} didn't exist in initial_state_dict"
            )
        non_existent_keys = set(state_dict_to_merge.keys()) - set(
            initial_state_dict.keys()
        )
        if non_existent_keys:
            raise KeyError(
                f"Keys {non_existent_keys} from loaded state not exist in original state dict"
            )

        merged = OrderedDict()
        for okey, ovalue in initial_state_dict.items():
            assert okey in state_dict_to_merge.keys()
            top_block_name = okey.split(".")[0]
            if top_block_name in keep_keys:
                value = state_dict_to_merge[okey]
            else:
                value = ovalue
            merged[okey] = value

        return merged

    def __load_keys_validation(self, load_modules: List[str], _graph_module_order=None):
        _load_order = (
            self.graph_module_order
            if _graph_module_order is None
            else _graph_module_order
        )
        _load_order = copy(_load_order)

        load_variants: List[str]
        load_variants = _dict_flatten(_load_order)

        valid = all(
            [_one_key_check(key, load_variants, load_modules) for key in load_modules]
        )
        return valid

    def partial_state_load(
        self,
        state_dict: Mapping[str, Any],
        load_modules: List[str],
        strict: bool = True,
    ):
        """Загрузка словаря состояния `state_dict` частями, т.е. позволяющая загрузить часть весов,
        остальные веса останутся случайно инициализированными.

        Note:
            Загрузка отдельных блоков имеет смысл когда они идут последовательно в графе прямого прохода,
            т.е. грузить хед без нека (если нек в сети есть), или хед без бакбоуна почти бессмысленно.

        Атрибуты:
            graph_module_order (Dict[str, Union[Dict, List]]): иерархический словарь, содержащий имена блоков, (бакбоунов, хедов, неков и т.п.)
            которые применяются последовательно (тоесть вход второго это выход первого) во время прямого прохода, начинаются от входа, смотри пример.

        Аргументы:
            state_dict (Mapping[str, Any]): обычный `state_dict` torch'а.
            load_modules (List[str]): лист, содержащий имена блоков,(имена блоков должны быть описаны в шаблоне исходной сети,
            либо в классе самой сети, в атрибуте `graph_module_order`) которые нужно загрузить.
            strict (bool, optional): Атрибут строг ой проверки, смотри `torch.nn.Module.load_state_dict` Defaults to True.

        Пример использования::

            class MyTestNet(_LoadingInterface):
                 backbone: SomeBlock
                 neck1: SomeBlock
                 neck2: SomeBlock
                 head1: SomeBlock
                 head2: SomeBlock
                 headaux: SomeBlock
                 graph_module_order = {
                     "backbone": {
                         "neck1": ["head1","headaux"],
                         "neck2": ["head2"]
                     }
                 }
            ...
                def forward(self, x):
                   bf = self.backbone(x)
                   n1f = self.neck1(bf)
                   n2f = self.neck2(bf)
                   h1o = self.head1(n1f)
                   h2o = self.head2(n2f)
                   haux = self.headaux(n1f)
                   return h1o, h2o, haux

            net = MyNet()
            # Загрузит веса только "backbone",  "neck1", "head1"
            net.partial_state_load(state_dict, load_modules = ["backbone",  "neck1", "head1"])


        """
        self.__load_keys_validation(load_modules)
        state_dict = self.__merge_state_dict(
            self.state_dict(), state_dict, keep_keys=load_modules
        )
        return super().load_state_dict(state_dict, strict)
