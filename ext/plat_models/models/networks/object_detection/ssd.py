from typing import List, Union, Tuple, Dict, Optional, Literal
from collections import OrderedDict

from models.core.tensor import Tensor
from torch import Tensor

from ...core import (
    AdaptiveAvgPool2d,
    Conv2d,
    ReLU,
    Module,
    Tensor
)
from ...core.abstract import AbsDetect2DModel
from ...core.container import Sequential, ModuleList
from ...core.abstract import Backbone, BACKBONES
from ...backbones.resnet import (
    b_resnet18,
    b_resnet34,
    b_resnet50,
    b_resnet101,
    b_resnet152,
    b_resnext50_32x4d,
    b_resnext101_32x8d,
    b_wide_resnet50_2,
    b_wide_resnet101_2,
)

from ...backbones.mobilenetv2 import b_mobilenetv2
from ...backbones.mobilenetv1 import b_mobilenetv1
from ...backbones.senet import b_se_resnext50_32x4d, b_se_resnext101_32x4d
from ...backbones.squeezenet import b_squeezenet_1_0, b_squeezenet_1_1
from ...core import DETMODELS, BACKBONES

from ...heads.object_detection import SSD_head, NMS
from ...utils.catcher import CatcherHooker

_backbone_map = {
    "resnet18": b_resnet18,
    "resnet34": b_resnet34,
    "resnet50": b_resnet50,
    "resnet101": b_resnet101,
    "resnet152": b_resnet152,
    "resnext50_32x4d": b_resnext50_32x4d,
    "resnext101_32x8d": b_resnext101_32x8d,
    "wide_resnet50_2": b_wide_resnet50_2,
    "wide_resnet101_2": b_wide_resnet101_2,
    "mobilenet_v2": b_mobilenetv2,
    "mobilenet_v1": b_mobilenetv1,
    # "squeezenet_1_0": b_squeezenet_1_0,
    # "squeezenet_1_1": b_squeezenet_1_1,
    # "se_resnext50_32x4d": b_se_resnext50_32x4d,
    # "se_resnext101_32x4d": b_se_resnext101_32x4d,
}
BACKNAMES = Literal[
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "mobilenet_v1",
    "mobilenet_v2",
    # "squeezenet_1_0",
    # "squeezenet_1_1",
]


# @DETMODELS.register_module()
class SSD(Sequential, AbsDetect2DModel, register=False):
    def __init__(
        self,
        backname: BACKNAMES,
        input_channels: int,
        downsample_factors: List[int],
        aspect_ratios: List[List[int]],
        nms_config: Dict,
        size: Union[List[int], Tuple[int, int]] = (300, 300),
        num_classes: int = 1000,
        pretrained: bool = False
    ) -> None:
        """Class realizing the model of object detection on images - Single Shot Detector (SSD)

        Args:
            backname (str): backbone name
            input_channels (int): number of input channels
            downsample_factors (List[int]): list of feature map compression ratios. 
                The output of the network is taken from the last layer of the specified compression
            aspect_ratios (List[List[int]]): the ratios of height to width of a bounding box for each compression level
            nms_config (Dict): settings for initializing of the nms layer.
            size (Union[List[int], Tuple[int, int]]): size of the input image (width, height).
            num_classes (int): number of object classes.
            pretrained (bool): flag indicating the use of pre-trained backbone weights
        """
        super().__init__()
        backbone = get_backbone(backname)
        backbone = backbone(pretrained=pretrained, input_channels=input_channels)
        keys, base_channels, main_output = init_keys(backbone, downsample_factors)
        # keys.add("main")
        backbone = CatcherHooker(backbone, keys)
        backbone = BackboneWithExtras(
            backbone=backbone,
            size=size,
            main_output=main_output
        )
        
        head = SSD_head(
            size=size,
            aspect_ratios=aspect_ratios,
            num_classes=num_classes,
            base_channels=base_channels,
            extra_channels=backbone.extra_channels
        )

        ssd_nms = NMS(**nms_config)

        super().__init__(
            OrderedDict(
                [
                    ("backbone", backbone),
                    ("head", head),
                    ("ssd_nms", ssd_nms)
                ]
            )
        )
        self.num_classes = num_classes
    
def init_keys(
        backbone: Backbone, 
        downsample_factors: List[int]
    ) -> set:
        name_layers = set()
        base_channels = []
        used_ds_factors = []
        inner_features = sorted(
            backbone.inner_features.items(), key=lambda x: x[0], reverse=True
        )
        main_output = None
        for _, (name, ds_factor, ch) in inner_features:
            if ds_factor in downsample_factors and \
            ds_factor not in used_ds_factors:
                name_layers.add("encoder." + name)
                if main_output is None:
                    main_output = "encoder." + name
                base_channels.append(ch)
                used_ds_factors.append(ds_factor)

        return name_layers, base_channels[::-1], main_output


class BackboneWithExtras(Backbone):
    def __init__(
            self, 
            backbone: Backbone, 
            size: Union[List[int], Tuple[int, int]],
            main_output: str
        ) -> None:
        super().__init__()

        self.backbone = backbone
        self.extras, self.out_extras = self.make_extras(backbone.encoder.output_channels, size)
        self.extra_channels = self.get_extra_channels(self.extras)
        self.output_channels = self.extra_channels[-1]
        self.downsample_factor = backbone.encoder.downsample_factor * \
                                 2 ** (len(self.extra_channels)) 
        self.input_channels = backbone.encoder.output_channels
        self.main_output = main_output

    def forward(self, x: Tensor) -> List[Tensor]:
        sources = []
        output = self.backbone(x)
        x = output[self.main_output]
        sources = [v for k, v in output.items()]
        for i, v in enumerate(self.extras):
            x = v(x)
            if i in self.out_extras:
                sources.append(x)

        return sources

    @staticmethod
    def get_extra_channels(extra_layers: List):
        """Returns number of out channels for output extra layers"""
        return list(layer.out_channels for layer in extra_layers[2::4])

    @staticmethod
    def make_extras(base_out_channels: int, size: List[int] = 300) -> List:
        # Extra layers added after the backbone for feature scaling
        layers = []

        size = 'x'.join(map(str, size))

        # extras_dsf_cfg = {
        #     "300x300": [64, 128],
        #     "512x512": [64, 128, 256],
        #     "640x480": [64, 128, 256],
        # }
        # extras_downsample_factors = extras_dsf_cfg[size]

        extras_cfg = {  # encoded channels and strides of layers
            "300x300": [128, "S", 256, 128, "S", 256],
            "512x512": [128, "S", 256, 128, "S", 256, 128, "S", 256],
            "640x480": [128, "S", 256, 128, "S", 256, 128, "S", 256],
        }
        extras = extras_cfg[size]
        out_extras = []
        in_channels = base_out_channels
        kernel_size = 1
        for i, v in enumerate(extras):
            if in_channels != "S":
                if v == "S":
                    layers += [
                        Conv2d(
                            in_channels,
                            extras[i + 1],
                            kernel_size=kernel_size,
                            stride=2,
                            padding=1,
                        )
                    ]
                    out_extras.append(len(layers))
                else:
                    layers += [Conv2d(in_channels, v, kernel_size=kernel_size)]
                layers += [ReLU()]
                kernel_size = 3 if kernel_size == 1 else 1
            in_channels = v
        return Sequential(*layers), out_extras

def get_backbone(backname: str) -> Backbone:
    return _backbone_map[backname]

def ssd(
    backname: str,
    input_channels: int,
    size: Union[List[int], Tuple[int, int]],
    downsample_factors: List[int],
    aspect_ratios: List[List[int]],
    nms_config: Dict,
    num_classes: int = 1000,
    pretrained: bool = False
) -> SSD:
    """SSD builder function

    Args:
        backname (str): backbone name
        input_channels (int): number of input channels.
        size (int): size of the input image (width, height).
        downsample_factors (List[int]): list of feature map compression ratios. 
            The output of the network is taken from the last layer of the specified compression
        aspect_ratios (List[List[int]]): The ratios of height to width of a bounding box for each compression level
        nms_config (Dict): settings for initializing of the nms layer.
        num_classes (int): number of object classes.
        pretrained (bool): flag indicating the use of pre-trained backbone weights. 
            If value is True backbone uses pre-trained weights

    Returns:
        SSD: SSD model
    """

    model = SSD(
        backname=backname,
        input_channels=input_channels,
        size=size,
        downsample_factors=downsample_factors,
        aspect_ratios=aspect_ratios,
        nms_config=nms_config,
        num_classes=num_classes,
        pretrained=pretrained
    )

    return model

DETMODELS.register_module(name='ssd', force=False, module=ssd)