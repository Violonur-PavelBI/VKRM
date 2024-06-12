from ..abstract.ofa_typing import HookInfo
from ...backbones.vgg import VGG, cfg
from ..primitives.static import MyNetwork
from typing import List
from models.core import MaxPool2d

class _OFA_VGG(VGG, MyNetwork):
    layers_to_hook: List[HookInfo]

    def __init__(self,
                 cfg: dict,
                 in_channels: int = 3,
    ):
        super().__init__(cfg, in_channels)
        self._create_hook_info()

    @staticmethod
    def _create_single_hook_info(layer_name, pre=True):
        if pre:
            hook_type = "pre"
        else:
            hook_type = "forward"
        return {
            "module": layer_name,
            "hook_type": hook_type
        }


    def  _create_hook_info(self):
        nam_mods = [*self.features.named_modules()]
        layer_to_hook = []
        for idx, (layer_name, layer) in enumerate(nam_mods):

            if idx == len(nam_mods) -1:
                layer_to_hook+= [self._create_single_hook_info(
                    f"features.{layer_name}", False
                )]
                continue
            if isinstance(layer, MaxPool2d):
                layer_to_hook += [self._create_single_hook_info(
                    f"features.{layer_name}", True
                )]

        self.layers_to_hook = layer_to_hook


def OFA_VGG11():
    return _OFA_VGG(cfg["VGG11"])


def OFA_VGG13():
    return _OFA_VGG(cfg["VGG13"])


def OFA_VGG16():
    return _OFA_VGG(cfg["VGG16"])


def OFA_VGG19():
    return _OFA_VGG(cfg["VGG19"])
