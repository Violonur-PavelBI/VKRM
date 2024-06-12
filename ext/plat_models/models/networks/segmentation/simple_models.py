from ...core import SEGMMODELS
from ...heads.segmentation import _segm_heads as _segm_heads
from .unet import unet
from itertools import product
from functools import wraps, partial
from ...backbones import BACKBONES


_backbones = BACKBONES._module_dict


def _create_constructor(backbone_name, backbone_cf, head_cls):
    @wraps(backbone_cf)
    def _construct(*args, num_classes, **kwargs):
        return head_cls(
            backbone_cf(*args, **kwargs),
            num_classes=num_classes,
            out_channels=num_classes,
        )

    func = _construct
    func.__name__ = backbone_name + "_" + head_cls.__name__
    return _construct


_models = [
    _create_constructor(backbone_name, backbone_cf, head_cls)
    for (backbone_name, backbone_cf), head_cls in product(
        _backbones.items(),
        _segm_heads,
    )
    if not isinstance(backbone_cf, type)
]
for constructor_f in _models:
    SEGMMODELS.register_module(name=constructor_f.__name__, module=constructor_f)
_atomic_nets = [partial(unet, bilinear=False), partial(unet, bilinear=True)]
a_unet_deconv = partial(unet, bilinear=False)
a_unet_deconv.__name__ = "UNet_deconv"
a_unet = partial(unet, bilinear=True)
a_unet.__name__ = "UNet"
_models += [a_unet_deconv, a_unet]

# __all__ = [model.__name__ for model in _models]
