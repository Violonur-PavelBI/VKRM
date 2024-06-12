from copy import deepcopy
from typing import Dict, Set

from ..core.module import Module
from ..core import ModuleDict, Tensor


class CatcherHooker(ModuleDict):
    def __init__(self, encoder, keys={}) -> None:
        """
        Some-like similar to torchvision `IntermediateLayerGetter`
        but uses hooks, that's why it didn't brake encoders forward.
        Also we didn't use bad asumptions, like in `torchvision._utils.IntermediateLayerGetter`

        Args:
            encoder - encoder or backbone, top level sub-layers would be observed
            catch_keys - set of names of child modules (nested in encoder), which outputs we need to catch,
        NOTE:
            if in keys would be "main", then main output of backbone also would be in outs
        """
        super().__init__()

        self.outs = {}
        self.catch_keys = keys
        self.encoder = encoder
        self.hook_handles = self._attach_hooks(self, self.catch_keys, self.outs)

    def forward(self, x):
        main_out = self.encoder(x)
        out = {}
        out.update(self.outs)
        if "main" in self.catch_keys:
            out.update({"main": main_out})
        
        self.outs.clear()

        return out

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self) ] =result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
            
        result._reattach_hooks(result, result.catch_keys, result.outs)
        return result
    
    @staticmethod
    def _place_output_to_list_via_hook(module: Module, name: str, outs: Dict[str, Tensor]):
        def _forward_hook(self, input: Tensor, output: Tensor):
            outs.update({name: output})
            return output

        hook_handles = module.register_forward_hook(_forward_hook)
        return hook_handles
    
    @staticmethod
    def _attach_hooks(parent_module, keys, outs):
        hook_handles = []
        for name, module in parent_module.named_modules():
            if name in keys:
                hook_handle = parent_module._place_output_to_list_via_hook(module, name, outs)
                hook_handles.append(hook_handle)
        return hook_handles

    @staticmethod
    def _reattach_hooks(module, keys, outs):
        for hook_handle in module.hook_handles:
            hook_handle.remove()
        
        module.hook_handles = module._attach_hooks(module, keys, outs)
        

class DecoderAttacher(CatcherHooker):
    def __init__(self, encoder, decoder):
        keys = decoder.wants
        super().__init__(encoder, keys)
        self.decoder = decoder

    def forward(self, x):
        features = super().forward(x)
        return self.decoder(features)
