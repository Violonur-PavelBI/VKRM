import os
import json

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.abstract import Backbone

from typing import Dict, List, Union

from torch.nn import ModuleList, Module
from dataclasses import dataclass


@dataclass
class NasParams:
    n_cell_stages: List[int]
    width_stages: List[int]
    stride_stages: List[int]
    nas_mode: str
    candidates: List[nn.Module]


class Fixed_net(Backbone):
    def __init__(
        self,
        nas_params: NasParams,
        output_layers: List[int],
        arch_mask: Dict[str, int],
        input_channels: int = 3,
        output_channels: Union[int, None] = None,
        SearchNet: Module = None,
        ZeroLayer: Module = None,
    ) -> None:
        """
        nas_params :
            {
                "n_cell_stages": List[int]
                "width_stages": List[int],
                "stride_stages": List[int],
                "nas_mode": str,
                "candidates": List[nn.Module],
            }
        output_layers - list of layer numbers whose output is used by the network head
        arch_mask - binary mask of the selected operations in each layer
        input_channels - number of input channels
        output_channels - number of output channels
        SearchNet - the class from nas submodule which initializes the network using "nas_params"
        ZeroLayer - the type of class layer to be removed from SearchNet
        """
        super(Fixed_net, self).__init__()

        if SearchNet is None or ZeroLayer is None:
            raise ValueError(
                "SearchNet and ZeroLayer is required classes for initializing network!!! Its must forward from nas submodule."
            )

        backbone = SearchNet(
            op_candidates=nas_params["candidates"],
            width_stages=nas_params["width_stages"],
            stride_stages=nas_params["stride_stages"],
            n_cell_stages=nas_params["n_cell_stages"],
            input_channels=input_channels,
            output_channels=output_channels,
            output_layers=output_layers,
        )
        self.arch_mask = arch_mask
        self.output_channels = output_channels
        self.downsample_factor = 1
        for i in nas_params["stride_stages"]:
            self.downsample_factor *= i
        self.input_channels = input_channels
        self.out_block = backbone.out_block
        self.num_out_block = [block["num"] for block in self.out_block]
        self.model = self.apply_fixed_net(backbone, ZeroLayer)

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.num_out_block:
                outputs.append(x)
        if self.last_conv is None:
            return outputs
        x = self.last_conv(x)
        return x

    def apply_fixed_net(self, backbone: Module, ZeroLayer: Module):
        """
        backbone - selected network by nas
        ZeroLayer - the type of class layer to be removed from SearchNet
        """

        fixed_model = ModuleList([backbone.first_conv])

        # with open(self.arch_path, 'r') as mask:
        #     arch_mask = json.load(mask)

        for i, (block, choise) in enumerate(
            zip(backbone._modules["blocks"], self.arch_mask.values())
        ):
            layer = block._modules["conv"]._modules[str(choise.index(True))]
            if not isinstance(layer, ZeroLayer):
                fixed_model.append(layer)
            else:
                self.num_out_block = [
                    num - 1 if num < i else num for num in self.num_out_block
                ]
        if backbone.last_conv is not None:
            self.last_conv = backbone.last_conv
        else:
            self.last_conv = None

        return fixed_model


def build_Fixednet(
    dir_net_params: str, SearchNet: Module = None, ZeroLayer: Module = None
) -> Fixed_net:
    """
    Function create searched architecture by NAS from the confuguration stored in the input directory

    Arguments:
        dir_net_params: str - directory consists binary mask net (*.json) and configuration net candidates (*.json)
        SearchNet - the class from nas submodule which initializes the network using "nas_params"
        ZeroLayer - the type of class layer to be removed from SearchNet

    Output:
        Fixed_net - the net that resulted from the architecture search
    """

    candidates = []

    nas_path = os.path.join(".", "project_info", "arch", dir_net_params[1:], "network_nas.json")

    # with open(binary_mask_path, 'r') as net_f, open(nas_params_path, 'r') as param_f:
    with open(nas_path, "r") as f:
        nas = json.load(f)
        nas_params = nas["params"]
        n_cell_stages = [-1] * len(nas_params["layers"])
        width_stages = [-1] * len(nas_params["layers"])
        stride_stages = [-1] * len(nas_params["layers"])

        for candidate, params in nas_params["candidates"].items():
            candidates.append(
                params["kernel_size"]
                + "_Conv"
                + params["stride"]
                + "!"
                + params["activation"]
                + "!"
                + params["normalization"]
            )

        for layer, params in nas_params["layers"].items():
            num_layer = int(layer[5:]) - 1
            n_cell_stages[num_layer] = params["n_cell_stage"]
            width_stages[num_layer] = params["width_stage"]
            stride_stages[num_layer] = params["stride_stage"]

        output_layers = nas_params["output_layers"]
        input_channels = nas_params["input_channels"]
        output_channels = nas_params["output_channels"]

        # with open(binary_mask_path, 'r') as net_f:
        arch_mask = nas["arch"]

    model = Fixed_net(
        nas_params={
            "n_cell_stages": n_cell_stages,
            "width_stages": width_stages,
            "stride_stages": stride_stages,
            "candidates": candidates,
        },
        output_layers=output_layers,
        input_channels=input_channels,
        output_channels=output_channels,
        arch_mask=arch_mask,
        SearchNet=SearchNet,
        ZeroLayer=ZeroLayer,
    )

    return model
