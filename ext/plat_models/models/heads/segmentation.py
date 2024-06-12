from ..core import (
    Module,
    Tensor,
    Conv2d,
)
from ..core import functional as F
from typing import Optional, Dict
from ..backbones import Backbone
from collections import OrderedDict
from ..baseblocks import (
    AbsImgDecoder,
    SDeconvUpsample,
    SqueezeSDeconvUpsample,
    SqueezeRDeconvUpsample,
)
from .deeplabv3 import DeepLabNeck
from ..core import AbsSegm2DModel

### TODO: Implement head and attached separately


class SegAdpHead(AbsSegm2DModel):
    # renamed UpsampleWrapper with 1 conv
    def __init__(
        self, backbone: Backbone, num_classes=100, use_softmax=False, **kwargs
    ):
        super(SegAdpHead, self).__init__()
        self.backbone = backbone
        self.use_softmax = use_softmax
        self.head = Conv2d(
            self.backbone.output_channels, num_classes, 1, padding=0, bias=False
        )
        self.num_classes = num_classes

    def forward(self, x):
        shape = x.shape[-2:]
        x = self.backbone(x)
        x = self.head(x)
        x = F.interpolate(x, size=shape, mode="bilinear")
        if self.use_softmax:
            x = F.softmax(x, dim=1)
        return x


class SegPrHead(AbsSegm2DModel):
    def __init__(
        self, backbone: Backbone, num_classes=100, use_softmax=False, **kwargs
    ):
        super(SegPrHead, self).__init__()
        self.backbone = backbone
        self.use_softmax = use_softmax
        self.head = Conv2d(
            self.backbone.output_channels, num_classes, 1, padding=0, bias=False
        )
        self.num_classes = num_classes

    def forward(self, x):
        x = self.backbone(x)

        x = self.head(x)
        x = F.interpolate(
            x,
            scale_factor=[
                self.backbone.downsample_factor,
                self.backbone.downsample_factor,
            ],
            mode="bilinear",
        )
        if self.use_softmax:
            x = F.softmax(x, dim=1)
        return x


class RCDDecoder(AbsSegm2DModel):
    _decoder_maker: AbsImgDecoder
    backbone: Backbone

    def __init__(self, backbone: Backbone, num_classes=100, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.head = self.__class__._decoder_maker(
            backbone.output_channels,
            num_classes,
            upsample_factor=backbone.downsample_factor,
        )
        self.num_classes = self.head.num_classes

    def forward(self, x: Tensor):
        shape = x.shape[2:]
        x = self.backbone(x)
        x = self.head(x)
        # if x.shape[2:]!=shape:
        #     x = F.interpolate(x, shape)
        return x


class SDU(RCDDecoder):
    _decoder_maker = SDeconvUpsample


class SqRDU(RCDDecoder):
    _decoder_maker = SqueezeRDeconvUpsample


class SqSDU(RCDDecoder):
    _decoder_maker = SqueezeSDeconvUpsample


class DeepLabV3(RCDDecoder):
    _decoder_maker = DeepLabNeck


class SegmWithAux(Module):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        head (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_head (nn.Module, optional): auxiliary classifier used during training
    """

    __constants__ = ["aux_head"]

    def __init__(
        self, backbone: Module, head: Module, aux_head: Optional[Module] = None
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.aux_head = aux_head

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.head(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if self.aux_head is not None:
            x = features["aux"]
            x = self.aux_head(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return result


_segm_heads = [
    SegAdpHead,
    SegPrHead,
    SDU,
    SqRDU,
    SqSDU,
    DeepLabV3,
]  # Classes of SegmentationHeads

__all__ = [x.__name__ for x in _segm_heads]
