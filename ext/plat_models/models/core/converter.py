# TODO: добавить операцию деление(как умножение)
from __future__ import annotations
import os, re
import json
from copy import deepcopy as dcopy
from typing import Union, List, Dict, Sequence, TYPE_CHECKING

import torch
import torch.onnx
import torch.fx as fx
import torch.nn as nn
from torch import Tensor

from .base_primitive import _BasePrimitiveConvertationInterface as BPCI
from .base_primitive import PRIMITIVE_REGISTRY, REVERSED_REGISTRY
from .utils._utils import LayerAttrTypeError
from .utils import convert_bases
from .utils.plat2torch.fix_plat2torch import FixPreviousLayers, FixNextLayers
from .utils.symbols import DEPTH_NAME_SEP, DUBLICATE_SEP

if TYPE_CHECKING:
    from .module import Module

DNP = DEPTH_NAME_SEP
DS = DUBLICATE_SEP
MASKING = False
DEFAULT_FIRMWARE_FOLDER_NAME = "model"


def _check_module_is_masked(m: Module):
    return MASKING and (hasattr(m, "__masked__") and (getattr(m, "__masked__")))


def _check_module_is_primitive(m: Module):
    return issubclass(type(m), BPCI)


class CoreTracer(fx.Tracer):
    @staticmethod
    def is_leaf_module(m: Module, module_qualified_name: str) -> bool:
        return _check_module_is_primitive(m) or _check_module_is_masked(m)


def core_symbolic_trace(root, concrete_args=None):
    tracer = CoreTracer()
    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    return fx.GraphModule(tracer.root, graph, name)


def custom_Create_Name(self: fx.graph._Namespace, candidate: str, obj) -> str:
    if obj is not None and obj in self._obj_to_name:
        return self._obj_to_name[obj]
    point = DNP
    num_point = DS
    _illegal_char_regex = re.compile("[^0-9a-zA-Z_·]+")
    candidate = _illegal_char_regex.sub(point, candidate)
    if candidate[0].isdigit():
        candidate = f"{point}{candidate}"
    match = self._name_suffix_regex.match(candidate)
    if match is None:
        base = candidate
        num = None
    else:
        base, num_str = match.group(1, 2)
        num = int(num_str)
    candidate = base if num is None else f"{base}_{num}"
    num = num if num else 0
    while candidate in self._used_names or self._is_illegal_name(candidate, obj):
        num += 1
        candidate = f"{base}{num_point}{num}"
    self._used_names.setdefault(candidate, 0)
    if obj is None:
        self._unassociated_names.add(candidate)
    else:
        self._obj_to_name[obj] = candidate
    return candidate


setattr(fx.graph._Namespace, "create_name", custom_Create_Name)


class HelperTree:
    def __init__(self, name: str, module: nn.Module) -> None:
        self._name = name
        self._module_name = module.__class__.__name__
        self.self_location = list()
        self.walk_location = list()
        self.child_list = list()
        self.child_names = list()
        self.lvl = 0

    def _get_pre_child(self, walk_location: list):
        node = self
        for location in walk_location:
            node = getattr(node, location)
        return node

    def _set_child(self, child) -> None:
        child.self_location = self.walk_location
        pre_child = self._get_pre_child(self.walk_location)
        setattr(pre_child, child._name, child)
        pre_child.child_list.append(child)
        pre_child.child_names.append(child._name)
        self.walk_location = self.walk_location + [child._name]

    def _make_child_custom(self, module):
        module._helper_tree = self
        for tree_child in self.child_list:
            child_module = getattr(module, tree_child._name)
            child_module.__masked__ = True
            child_module = tree_child._make_child_custom(child_module)
        return module

    def add_children(self, child_name: str, child_module: nn.Module) -> None:
        if not CoreTracer.is_leaf_module(child_module, None):
            child = HelperTree(child_name, child_module)
            self._set_child(child)

    def get_masked_module(self, module):
        module = self._make_child_custom(module)
        return module

    def build(self):
        lvl = self.lvl
        lvl += 1
        print(lvl * "|=>", self._name)
        for child in self.child_list:
            child.lvl = lvl
            child.build()


class ConverterTorch2Plat:
    def __init__(
        self,
        torch_model: nn.Module,
        model_dirpath: str,
        input_example: Union[None, Tensor, Sequence[Tensor], Dict[str, Tensor]],
        output_example: Union[None, Tensor, Sequence[Tensor], Dict[str, Tensor]],
        verbose: bool = False,
    ):
        os.makedirs(model_dirpath, exist_ok=True)
        self.convert_method_dict: dict = PRIMITIVE_REGISTRY
        self.verbose: bool = verbose
        self.binary_name: str = os.path.join(
            model_dirpath, DEFAULT_FIRMWARE_FOLDER_NAME + ".bin"
        )
        self.json_name: str = os.path.join(
            model_dirpath, DEFAULT_FIRMWARE_FOLDER_NAME + ".json"
        )
        # self.binary_name: str = model_dirpath + ".bin"
        # self.json_name: str = model_dirpath + ".json"
        self._create_binary_file(self.binary_name)
        self.torch_model: nn.Module = torch_model
        self.helper_tree = HelperTree(
            self.torch_model.__class__.__name__, self.torch_model
        )
        self.unique_names: dict = dict()
        self.pass_inputs = []
        self.pass_outputs = None
        self.input_example = input_example
        self.output_example = output_example

    def convert(self) -> List[dict]:
        self._grow_tree(self.torch_model)
        self.torch_model = self.helper_tree.get_masked_module(self.torch_model)
        plat_model = self._convert_module(self.torch_model)
        plat_model = self._sorted_layers_keys(plat_model)
        plat_model = self._fix_point(plat_model)
        self.plat_dict = self._to_dict(plat_model)
        plat_model = self._detached_graphs_fixing(plat_model)
        plat_model = self._fix_inputs_outputs(plat_model)
        with open(self.json_name, "w") as json_file:
            json.dump(plat_model, json_file, indent=4)
        return plat_model

    def _create_binary_file(self, binary_name: str) -> None:
        self.binary_counter: int = 0
        self.binary_name: str = binary_name
        self.binary_file = open(self.binary_name, "wb")
        self.binary_file.close()

    def _getattr(self, module: nn.Module, names: Union[str, List[str]]) -> nn.Module:
        if isinstance(names, str):
            if DNP in names:
                names = names.split(DNP)
                for attr in names:
                    if attr == "":
                        continue
                    if DS in attr:
                        attr = attr.split(DS)[0]
                    module = getattr(module, attr)
            else:
                if DS in names:
                    names = names.split(DS)[0]
                module = getattr(module, names)
        elif isinstance(names, list):
            for attr in names:
                if not DNP in attr:
                    module = getattr(module, attr)
                else:
                    for item in attr.split(DNP):
                        module = getattr(module, str(item))
        return module

    def _grow_tree(self, module: nn.Module):
        if issubclass(type(module), BPCI):
            return None
        for child_name, child_module in module.named_children():
            location = dcopy(self.helper_tree.walk_location)
            self.helper_tree.add_children(child_name, child_module)
            self._grow_tree(child_module)
            self.helper_tree.walk_location = location

    def _define_node_type(self, current_fx_node, module):
        if current_fx_node.op == "placeholder":
            current_fx_node.meta["type"] = "model_input"
        elif current_fx_node.op == "output":
            current_fx_node.meta["type"] = "model_output"
        elif current_fx_node.op == "call_function":
            current_fx_node.meta["type"] = current_fx_node.target.__qualname__
        elif current_fx_node.op == "call_method":
            current_fx_node.meta["type"] = current_fx_node.target
        else:
            fx_node_module = module.get_submodule(current_fx_node.target)
            if _check_module_is_masked(fx_node_module):
                # Assigning output for current node in current frame
                current_fx_node.meta["name"] += ".output"
                # Getting current frame input
                self.pass_inputs = convert_bases.get_inputs(current_fx_node)
            # fx_node.meta["type"] = "Module_not_used"

    def _append_layer(
        self,
        plat_model: List[Union[Dict[str, str], fx.node.Node]],
        layer: Dict[str, str],
    ) -> None:
        for key, value in layer.items():
            if not isinstance(value, LayerAttrTypeError.typing):
                raise LayerAttrTypeError(layer, key, value)
            if key.startswith("input"):
                if isinstance(value, list):
                    for item in value:
                        if not item.startswith(self.torch_model.__class__.__name__):
                            raise Exception(f"FIX layer: ({layer['type']})")
                else:
                    if not value.startswith(self.torch_model.__class__.__name__):
                        raise Exception(f"FIX layer: ({layer['type']})")
        if not any([key.startswith("input") for key in layer.keys()]):
            raise Exception(f"({layer['type']}) does not have input")
        return plat_model.append(layer)

    def _set_meta_to_node(self, fx_node, module):
        module_class_name = self.torch_model.__class__.__name__
        if fx_node.op in ["call_function", "call_method"]:
            name = fx_node.name
        else:
            name = fx_node.target
        new_name = (
            [module_class_name]
            + module._helper_tree.self_location
            + [module._helper_tree._name]
            + [name]
        )
        if new_name[1] == module_class_name:
            new_name = new_name[1:]
        new_name = ".".join(new_name)
        fx_node.meta["name"] = new_name
        fx_node.meta["helper_tree"] = module._helper_tree

    def _convert_call(self, plat_model, outer_fx_node, module):
        if outer_fx_node.op != "call_module":
            layer = self.convert_method_dict[outer_fx_node.meta["type"]](
                outer_fx_node, self
            )
            self._append_layer(plat_model, layer)
        else:
            child_module = self._getattr(module, outer_fx_node.name)
            if _check_module_is_primitive(child_module):
                layer = child_module.toPlatform(outer_fx_node, self)
                self._append_layer(plat_model, layer)
            else:
                list_module = self._convert_module(child_module)
                plat_model.extend(list_module)
                outer_fx_node.meta["output"] = outer_fx_node.meta["name"]

    def _convert_module(self, module):
        plat_model = list()
        fx_trace = core_symbolic_trace(module)
        for fx_node in fx_trace.graph.nodes:
            self._set_meta_to_node(fx_node, module)
            self._define_node_type(fx_node, module)
            self._convert_call(plat_model, fx_node, module)
        return plat_model

    def _sorted_layers_keys(self, plat_model: List[dict]) -> List[dict]:
        for layer in plat_model:
            if "input1" in layer.keys():
                continue
            key_list = ["name", "type", "input", "output"]
            for key in layer.keys():
                if key not in key_list:
                    key_list.append(key)
            for key in key_list:
                layer[key] = layer.pop(key)
        return plat_model

    def _to_dict(self, plat_model: List[dict]) -> Dict[str, dict]:
        plat_dict = dict()
        for layer in plat_model:
            if isinstance(layer, dict):
                plat_dict[layer["name"]] = layer
            else:
                raise Exception(f"Cant add to self.plat_dict this layer: {layer}")
        return plat_dict

    def _fix_inputs_outputs(self, plat_model: List[dict]) -> List[dict]:
        for i, layer in reversed(list(enumerate(plat_model))):
            if i == 0:
                continue
            right_layer = self._recursive_replace(layer)
            plat_model[i] = right_layer
        model_inputs = {"type": "input_data", "name": list()}
        for layer in plat_model:
            if layer["type"] == "model_input" and len(layer["name"].split(".")) == 2:
                model_inputs["name"].append(layer["name"])
                model_inputs[layer["name"]] = "NotImplemented"
        self._get_tensors_shape(model_inputs, type="input")
        model_outputs = {"type": "output_data", "name": plat_model[-1]["input"]}
        for name in model_outputs["name"]:
            model_outputs[name] = "NotImplemented"
        self._get_tensors_shape(model_outputs, type="output")
        for i, layer in reversed(list(enumerate(plat_model))):
            if layer["type"] in ["model_input", "model_output", "dropout"]:
                plat_model.pop(i)
        ## changing logic; placing model by key "model";
        ## adding version
        ## placing inputs by key "model_inputs"
        ## placing outputs by key "model_outputs"
        main = dict(
            version=2.0,
            model_inputs={key: model_inputs[key] for key in model_inputs["name"]},
            model_outputs={key: model_outputs[key] for key in model_outputs["name"]},
            model=plat_model,
        )
        # version 1
        if os.getenv("OLD_VERSION"):
            main = [model_inputs, model_outputs, *plat_model]
        # plat_model.insert(0, model_inputs)
        # plat_model.insert(1, model_outputs)
        return main

    def _seq_shape(self, dct, obj):
        for i, name in enumerate(dct["name"]):
            dct[name] = list(obj[i].shape)

    def _get_tensors_shape(self, dct, type: str):
        if type == "input":
            obj = self.input_example
        elif type == "output":
            obj = self.output_example
        else:
            raise Exception("Wrong type srt!!!!")
        if isinstance(obj, Tensor):
            dct[dct["name"][0]] = list(obj.shape)
        elif isinstance(obj, Sequence):
            self._seq_shape(dct, obj)
        elif isinstance(obj, dict):
            raise NotImplementedError

    def _fix_point(self, plat_model: List[dict]) -> List[dict]:
        for layer in plat_model:
            for key, value in layer.items():
                if key in ["name", "output", "ref_size"] or key.startswith("input"):
                    if isinstance(value, str):
                        layer[key] = value.replace(DS, ".")
                    else:
                        inputs = list()
                        for item in value:
                            inputs.append(item.replace(DS, "."))
                        layer[key] = inputs
        return plat_model

    def _recursive_replace(self, layer: dict) -> dict:
        for key_input, value_name in layer.items():
            if key_input.startswith("input"):
                if isinstance(value_name, str):
                    if self.plat_dict[value_name]["type"] in [
                        "model_input",
                        "model_output",
                        "dropout",
                    ]:
                        if isinstance(self.plat_dict[value_name]["input"], str):
                            layer[key_input] = self.plat_dict[value_name]["input"]
                            layer = self._recursive_replace(layer)
                            continue
                        else:
                            if len(self.plat_dict[value_name]["input"]) == 0:
                                continue
                            else:
                                layer[key_input] = self.plat_dict[value_name]["input"][
                                    0
                                ]
                                layer = self._recursive_replace(layer)
                                continue
                elif isinstance(value_name, list):
                    for i, _input_name in enumerate(value_name):
                        if self.plat_dict[_input_name]["type"] in [
                            "model_input",
                            "model_output",
                            "dropout",
                        ]:
                            layer[key_input][i] = self.plat_dict[_input_name]["input"]
                            new_input = list()
                            for item in layer[key_input]:
                                if isinstance(item, str):
                                    new_input.append(item)
                                else:
                                    new_input.extend(item)
                            layer[key_input] = new_input
                            layer = self._recursive_replace(layer)
                            continue
                else:
                    raise Exception("Poshel .....")
        return layer

    def _detached_graphs_fixing(self, plat_model):
        op_types = ["getitem", "getattr"]
        for layer in reversed(plat_model):
            for key, value in layer.items():
                if key.startswith("input"):
                    if isinstance(value, list):
                        for i, v in enumerate(value):
                            if self.plat_dict[v]["type"] in op_types:
                                layer[key][i] = self.plat_dict[v]["input"]
                    else:
                        if self.plat_dict[value]["type"] in op_types:
                            layer[key] = self.plat_dict[value]["input"]
        for i, layer in reversed(list(enumerate(plat_model))):
            if layer["type"] in op_types:
                plat_model.pop(i)
        return plat_model


class ConverterPlat2Torch(nn.Module):
    def __init__(self, model_path: str, verbose=False):
        super().__init__()
        self.convert_method_dict = REVERSED_REGISTRY
        self.verbose = verbose
        with open(os.path.join(model_path, "model.bin"), "rb") as f:
            self.binary = f.read()
        with open(os.path.join(model_path, "model.json"), "r") as read_file:
            plat_format = json.load(read_file)
        try:
            version = plat_format["version"]

            if float(version) < 2:
                raise ValueError("Version less than 2 didn't supported")
            self.plat_model = plat_format["model"]
        except (KeyError, TypeError) as e:
            if not os.getenv("OLD_VERSION"):
                print("Version not found!")
                raise e
            else:
                self.plat_model = plat_format
                self.plat_model = self.plat_model[2:]

        self.torch_model = nn.ModuleList(self._build_model())
        self.tensors = dict()

    def _build_model(self):
        model_counter = 0
        torch_model = list()
        convert_method_dict = self.convert_method_dict["module"]

        for layer in self.plat_model:
            type_version = "type_version" in layer.keys()
            type_name = layer["type_version"] if type_version else layer["type"]
            if type_name in convert_method_dict.keys():
                convert_method = convert_method_dict.get(type_name)
                torch_model.append(convert_method(layer, self.binary))
                layer["model_counter"] = model_counter
                model_counter += 1
        return torch_model

    def _get_input(self, x):
        tensors = dict()
        if len(x) == 1:
            x = x[0]
            tensors[self.plat_model[0]["input"]] = x
        else:
            tensor_names = [
                item
                for item in sorted(self.plat_model[0].keys())
                if item.startswith("input")
            ]
            for i, name in enumerate(tensor_names):
                tensors[self.plat_model[0][name]] = x[i]
        return tensors

    def _forward_layer(self, layer):
        if layer["type"] in FixPreviousLayers.keys():
            self.tensors = FixPreviousLayers[layer["type"]](
                layer=layer, tensors=self.tensors
            )
        output_name = layer["output"]
        if "model_counter" in layer.keys():
            if isinstance(layer["input"], (list)):
                self.tensors[output_name] = self.torch_model[layer["model_counter"]](
                    [self.tensors[name_input] for name_input in layer["input"]]
                )
            else:
                self.tensors[output_name] = self.torch_model[layer["model_counter"]](
                    self.tensors[layer["input"]]
                )
        else:
            ##### TODO clear code!
            convert_method_dict = self.convert_method_dict["functional"]
            try:
                convert_method = convert_method_dict[layer["type"]]
            except KeyError:
                if layer["type"] in self.convert_method_dict["module"]:
                    convert_method = self.convert_method_dict["module"][layer["type"]]
                else:
                    _type = layer["type"]
                    raise Exception(f"Not implemented convertation method {_type}")
            if isinstance(output_name, list):
                outputs = convert_method(layer, self.tensors)
                for i, name in enumerate(output_name):
                    self.tensors[name] = outputs[i]
            else:
                self.tensors[output_name] = convert_method(layer, self.tensors)
        if layer["type"] in FixNextLayers.keys():
            self.tensors = FixNextLayers[layer["type"]](
                layer=layer, tensors=self.tensors
            )

    def forward(self, *x):
        self.tensors = self._get_input(x)
        for layer in self.plat_model:
            print(
                "Convert --<<--Plat2Torch--<<--:", layer.get("name"), layer.get("type")
            ) if self.verbose else None
            self._forward_layer(layer)
        return self.tensors[layer["name"]], self.tensors
