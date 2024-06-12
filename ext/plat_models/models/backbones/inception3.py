import warnings
from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple

import torch

from ..core.module import Module
from ..core.abstract import BACKBONES
from ..core import (
    torch_init as init,
    Tensor,
    BatchNorm2d,
    MaxPool2d,
    Linear,
    Conv2d,
)
from ..baseblocks.inception3_basic import *


# __all__ = ["Inception3", "InceptionOutputs", "_InceptionOutputs"]


# InceptionOutputs = namedtuple("InceptionOutputs", ["logits", "aux_logits"])
# InceptionOutputs.__annotations__ = {"logits": Tensor, "aux_logits": Optional[Tensor]}

# # Script annotations failed with _GoogleNetOutputs = namedtuple ...
# # _InceptionOutputs set here for backwards compat
# _InceptionOutputs = InceptionOutputs


class Inception3(Module):
    def __init__(
        self,
        input_channels: int = 3,
        # num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        inception_blocks: Optional[List[Callable[..., Module]]] = None,
        init_weights: Optional[bool] = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.downsample_factor = 1
        self.input_channels = input_channels
        # min input image: 75*75
        self.output_channels = 32
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d,
                InceptionA,
                InceptionB,
                InceptionC,
                InceptionD,
                InceptionE,
                InceptionAux,
            ]
        if init_weights is None:
            warnings.warn(
                "The default weight initialization of inception_v3 will be changed in future releases of "
                "torchvision. If you wish to keep the old behavior (which leads to long initialization times"
                " due to scipy/scipy#11299), please set init_weights=True.",
                FutureWarning,
            )
            init_weights = True
        if len(inception_blocks) != 7:
            raise ValueError(
                f"length of inception_blocks should be 7 instead of {len(inception_blocks)}"
            )
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)  # 2
        self.output_channels = 32
        self.conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = MaxPool2d(kernel_size=3, stride=2)  # 4
        self.conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = MaxPool2d(kernel_size=3, stride=2)  # 8
        self.mixed_5b = inception_a(192, pool_features=32)  # 8
        self.mixed_5c = inception_a(256, pool_features=64)  # 8
        self.mixed_5d = inception_a(288, pool_features=64)  # 8
        self.mixed_6a = inception_b(288)  # 32
        self.mixed_6b = inception_c(768, channels_7x7=128)
        self.mixed_6c = inception_c(768, channels_7x7=160)
        self.mixed_6d = inception_c(768, channels_7x7=160)
        self.mixed_6e = inception_c(768, channels_7x7=192)
        self.AuxLogits: Optional[Module] = None
        # if aux_logits:
        #     self.AuxLogits = inception_aux(768, num_classes)
        self.mixed_7a = inception_d(768)  # 128
        self.mixed_7b = inception_e(1280)
        self.mixed_7c = inception_e(2048)

        self.output_channels = 2048
        self.downsample_factor = 128
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(p=dropout)
        # self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, Conv2d) or isinstance(m, Linear):
                    stddev = float(m.stddev) if hasattr(m, "stddev") else 0.1  # type: ignore
                    init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
                elif isinstance(m, BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 3 x 299 x 299
        x = self.conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.mixed_6e(x)
        # N x 768 x 17 x 17
        # aux: Optional[Tensor] = None
        # if self.AuxLogits is not None:
        #     if self.training:
        #         aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        # x = self.avgpool(x)
        # # N x 2048 x 1 x 1
        # x = self.dropout(x)
        # # N x 2048 x 1 x 1
        # x = torch.flatten(x, 1)
        # # N x 2048
        # x = self.fc(x)
        # # N x 1000 (num_classes)
        # return x, aux
        return x

    # @torch.jit.unused
    # def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> InceptionOutputs:
    #     if self.training and self.aux_logits:
    #         return InceptionOutputs(x, aux)
    #     else:
    #         return x  # type: ignore[return-value]

    # def forward(self, x: Tensor) -> InceptionOutputs:
    def forward(self, x: Tensor) -> Tensor:
        x = self._transform_input(x)
        # x, aux = self._forward(x)
        # aux_defined = self.training and self.aux_logits
        # if torch.jit.is_scripting():
        #     if not aux_defined:
        #         warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
        #     return InceptionOutputs(x, aux)
        # else:
        #     return self.eager_outputs(x, aux)
        x = self._forward(x)
        return x


def b_inception3(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> Inception3:
    """
    Inception v3 model architecture from
    `Rethinking the Inception Architecture for Computer Vision <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    backbone = Inception3(**kwargs)

    return backbone


# BACKBONES.register_module("b_inception3", module=b_inception3)
