from copy import deepcopy


class LayerRegistry:
    REGISTRY = {}

    @staticmethod
    def registry(layer_cls):
        LayerRegistry.REGISTRY[layer_cls.__name__] = layer_cls
        return layer_cls

    @staticmethod
    def get_layer_by_name(layer_name: str):
        return LayerRegistry.REGISTRY[layer_name]


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    layer_config = deepcopy(layer_config)
    layer_name = layer_config.pop("name")
    layer = LayerRegistry.get_layer_by_name(layer_name)
    return layer.build_from_config(layer_config)
