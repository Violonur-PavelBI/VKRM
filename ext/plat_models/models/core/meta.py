from abc import ABCMeta
from typing import Dict, Union, Sequence, List, get_type_hints

from torch import Tensor
import torch
from .converter import ConverterPlat2Torch, ConverterTorch2Plat
from .registry.registry import METADATA_PK, TYPENAME_AK, INIT_ARG_AK, REGISTRY_MK


class ABCEnforcedProperties(ABCMeta):
    @staticmethod
    def get_templates_annotations(obj) -> Dict:
        all_anotations = {}
        all_anotations.update(get_type_hints(type(obj)))
        for _type in obj.__class__.mro():
            if not isinstance(_type, ABCEnforcedProperties):
                for key in get_type_hints(_type).keys():
                    all_anotations.pop(key)

                break
            else:
                pass
        all_anotations = {
            key: value
            for key, value in all_anotations.items()
            if not key.startswith("_")
        }
        return all_anotations

    @staticmethod
    def __check_annotations__(obj) -> None:
        all_anotations = ABCEnforcedProperties.get_templates_annotations(obj)
        for name, annotated_type in all_anotations.items():
            if not hasattr(obj, name):
                if hasattr(annotated_type, "__args__"):
                    if None.__class__ in annotated_type.__args__:
                        continue
                    else:
                        raise AttributeError(
                            f"required attribute {name} not present "
                            f"in {obj.__class__}"
                        )
                else:
                    raise AttributeError(
                        f"required attribute {name} not present " f"in {obj.__class__}"
                    )

    @staticmethod
    def __set_configuration_fields__(obj) -> None:
        all_annotations = ABCEnforcedProperties.get_templates_annotations(obj)
        obj._conf_fields = list(all_annotations.keys())

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        cls.__check_annotations__(obj)
        cls.__set_configuration_fields__(obj)
        return obj


# class MetadataDescriptor:
#     def __init__(self):
#         self._name = ''

#     def __set_name__(self, owner, name):
#         self._name = '__MyObj_' + name

#     def __set__(self, instance, value):
#         setattr(instance, self._name, value)

#     def __get__(self, instance, owner):
#         return getattr(instance, self._name, 0)


class _CoreInterface(torch.nn.Module):
    _conf_fields: List[str]
    __train_meta: Dict[str, Dict]

    def toPlatform(
        self,
        model_path: str,
        input_example: Union[Tensor, Sequence[Tensor], Dict[str, Tensor]],
        output_example: Union[Tensor, Sequence[Tensor], Dict[str, Tensor]],
        verbose=False,
    ):
        temp = ConverterTorch2Plat(
            torch_model=self,
            model_dirpath=model_path,
            input_example=input_example,
            output_example=output_example,
            verbose=verbose,
        )
        platform_graph = temp.convert()
        return platform_graph

    @classmethod
    def fromPlatform(cls, model_path: str, verbose: bool = False):
        temp = ConverterPlat2Torch(
            model_path=model_path,
            verbose=verbose,
        )
        return temp

    @property
    def metadata(self):
        return {**self.__train_meta}

    @metadata.setter
    def metadata(self, meta_data: Dict = None):
        """
        metadata (Dict): metadata of network, get next required fields:
        "task name": [classification, segmentation, detection]
        "init_args" : constructor parameters, Note: Will be set automatically by registry,
        "_registries": tuple(str,str) - name of root and child registries, Note: Will be set automatically by registry,
        "_typename": str - name of construction function in registry
        "dataset": info about dataset from training.json, (uuid выборки и или датасета,
            мета-инфрормация датасета (размеры картинок, число классов, имена классов, число объектов, среднийй размер объектов и т.д))
        "loss functions": [{ "name": - name of loss function from training.json }],
        "resolution": {
            "h": height from training.json or model input ,
            "w": width from training.json or model input
            }, or other type resolution (L), (D,H,W)
        "backbone":{
            "name": name of backbone from training.json or atomic(full) net ,
            "input_c": channels from training.json or model input,
            "channel descriptions" : "RGB" or "RGB" + "mask"
            },
        "head":{
            "name": name of head from training.json or none,
            "channel resolution":  ($ C_{out}, C_{num_classes}$),
            "spatial resolution":  "1", "H/2, W/2", "H/8, W/8",
            },
        "optimization params":{
            "lr": lr from training.json,
            "wd": weight_decay from training.json ,
            "optimization algorithm": name of optimization algorithm
        },
        "batch_size":batch_size lr from training.json,
        "memory on GPU": ,
        "forward time estimation without grad, ms": ,
        "forward time estimation with grad, ms":
        """
        _trained_model_meta = (
            {} if not hasattr(self, "metadata") else self.metadata
        )
        if not set(meta_data.keys()) - set([TYPENAME_AK, INIT_ARG_AK, REGISTRY_MK]):
            self.__train_meta = {**meta_data}
            return

        to_check_fields = []
        submodel_needed_fields = {}
        for _field in self._conf_fields:
            value = getattr(self, _field)
            if not isinstance(value, _CoreInterface):
                _trained_model_meta[_field] = value
            elif isinstance(value, torch.nn.Module) and not isinstance(
                value, _CoreInterface
            ):
                raise TypeError(
                    "All blocks of models must be child of models.core.Module"
                )
            else:
                to_check_fields += [_field]
                submodel = getattr(self, _field)
                submodel: _CoreInterface
                if hasattr(submodel, "__train_meta"):
                    submodel_needed_fields.update({_field: submodel.__train_meta})

        for _field in to_check_fields:
            if not _field in meta_data.keys():
                raise KeyError(f"This model need {to_check_fields} keys in metadata.")
        for submodule_name in submodel_needed_fields.keys():
            submodule_dict = meta_data[submodule_name]
            assert isinstance(submodule_dict, dict)
            key_set = set(submodule_dict.keys())
            needed_key_set = set(submodel_needed_fields[submodule_name].keys())
            assert len(needed_key_set - key_set) == 0
        _trained_model_meta.update(meta_data)
        self.__train_meta = _trained_model_meta

    def update_metadata(self, metadata: Dict):
        self.__train_meta.update(metadata)

    # metadata = property(get_metadata, set_metadata)
