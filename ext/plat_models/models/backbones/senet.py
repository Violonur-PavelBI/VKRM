from ..core.container import Sequential
from ..core.module import Module
from ..core import (
    Norm2d,
    NLF,
    Conv2d,
    Tensor,
    # nlf,
    ReLU,
    BatchNorm2d,
    GroupNorm,
    flatten,
    MaxPool2d,
    Linear,
    AvgPool2d,
    Dropout,
    AdaptiveAvgPool2d,
)
from collections import OrderedDict
from ..baseblocks.senet_basic import (
    SEBottleneck,
    SEResNetBottleneck,
    SEResNeXtBottleneck,
)
from typing import Union, List
from ..core.abstract import Backbone, BACKBONES
import torch


class SENet(Backbone):
    """
    Main Squeeze Excitation Network Module
    """

    def __init__(
        self,
        block: Union[SEBottleneck, SEResNetBottleneck, SEResNetBottleneck],
        layers: List[int],
        groups: int,
        reduction: int,
        dropout_p: float = 0.2,
        inplanes: int = 128,
        input_3x3: bool = True,
        downsample_kernel_size: int = 3,
        downsample_padding: int = 1,
    ):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        self.input_channels = 3
        self.downsample_factor = 4
        self.inner_features = dict()

        if input_3x3:
            layer0_modules = [
                ("conv1", Conv2d(3, 64, 3, stride=2, padding=1, bias=False)),
                ("bn1", Norm2d(64)),
                ("relu1", ReLU(inplace=True)),
                ("conv2", Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                ("bn2", Norm2d(64)),
                ("relu2", ReLU(inplace=True)),
                ("conv3", Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
                ("bn3", Norm2d(inplanes)),
                ("relu3", ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                (
                    "conv1",
                    Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                ),
                ("bn1", Norm2d(inplanes)),
                ("relu1", ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(("pool", MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0,
            num_layer=0,
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
            num_layer=1,
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=1,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
            num_layer=2,
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=1,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
            num_layer=3,
        )

        self.output_channels = 512 * block.expansion
        # self.downsample_factor = 8 # FIXME: Dynamic computation, dependent on configuration

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        groups,
        reduction,
        num_layer,
        stride=1,
        downsample_kernel_size=1,
        downsample_padding=0,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=downsample_kernel_size,
                    stride=stride,
                    padding=downsample_padding,
                    bias=False,
                ),
                Norm2d(planes * block.expansion),
            )

        self.downsample_factor *= stride  # stride

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                groups,
                reduction,
                stride,
                downsample,
                downsample_factor=self.downsample_factor,
            )
        )
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups,
                    reduction,
                    downsample_factor=self.downsample_factor,
                )
            )

        self.inner_features[num_layer] = [
            f"layer{num_layer + 1}",
            self.downsample_factor,
            layers[-1].output_channels,
        ]

        return Sequential(*layers)

    def features(self, x):
        """
        Forward Pass through the each layer of SE network
        """
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    # def logits(self, x):
    #     """
    #     AvgPool and Linear Layer
    #     """
    #     x = self.avg_pool(x)
    #     if self.dropout is not None:
    #         x = self.dropout(x)

    #     x = x.view(x.shape[:2], -1)
    #     x = self.last_linear(x)
    #     return x

    def forward(self, x):
        x = self.features(x)
        # x = self.logits(x)
        return x


def b_se_resnext50_32x4d(pretrained=False):
    """
    Defination For SE Resnext50
    """
    model = SENet(
        SEResNeXtBottleneck,
        [3, 4, 6, 3],
        groups=32,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
    )
    if pretrained:
        path = "/app/pmodels/senet/se_resnext50_32x4d.pth"
        model.load_state_dict(torch.load(path))
    return model


def b_se_resnext101_32x4d(pretrained=False):
    """
    Defination For SE Resnext101
    """

    model = SENet(
        SEResNeXtBottleneck,
        [3, 4, 23, 3],
        groups=32,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
    )
    if pretrained:
        path = "/app/pmodels/senet/se_resnext101_32x4d.pth"
        model.load_state_dict(torch.load(path))
    return model


BACKBONES.register_module("b_se_resnext50_32x4d", module=b_se_resnext50_32x4d)
BACKBONES.register_module("b_se_resnext101_32x4d", module=b_se_resnext101_32x4d)
