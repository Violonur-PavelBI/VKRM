from ..core.container import Sequential
from ..core.module import Module
from ..core import (
    Norm2d,
    NLF,
    Conv2d,
    Tensor,
    # nlf,
    AdaptiveAvgPool2d,
    Linear,
    BatchNorm2d,
    GroupNorm,
    flatten,
    MaxPool2d,
)
from ..core import torch_init as init
from ..baseblocks.resnet_basic import Bottleneck, BasicBlock, conv1x1, conv3x3
from typing import Union, Type, List, Optional, Callable, Tuple
from ..core.module import Module
from abc import ABC, abstractmethod
from ..core.abstract import Backbone, BACKBONES
import torch


class AbstractResnet(Backbone):
    r"""Абстрактный класс, имплементирующий логику имен блоков Resnet-о подобных сетей, без хедера

    Consist:
        layer0: Module

        layer1: Module

        layer2: Module

        layer3: Module

    """

    layer0: Module
    layer1: Module
    layer2: Module
    layer3: Module
    layer4: Module

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        pass

    @abstractmethod
    def forward(self, x):
        pass


class ResNet(AbstractResnet):
    input_channels = 3

    # FIXME: Change to dynamic comp, or move to initialization functions, bcz this is shit
    # inner_features = {
    #     0: ["layer1", 4, 64],
    #     1: ["layer2", 8, 128],
    #     2: ["layer3", 16, 256],
    #     3: ["layer4", 32, 512],
    # }  #  <- information to construct heads (<layer name>, cumulative spatial stride, outter channels)
    inner_features = dict()

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        input_channels: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[
            List[bool]
        ] = None,  # This used by decoders such DeeplabV3
        norm_layer: Optional[Callable[..., Module]] = None,
    ) -> None:
        r"""Конструктор резнето-подобных энкодеров, позволяет имплементировать семейства:
        ResNet, ResNext, WideResNet.

        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): Вариант базовых блоков, используемыз при конструировании
            layers (List[int]): Число блоков на кажом уровне, Должно быть всего 4 числа.
            num_classes (int, optional): Число классов, для хедера. Defaults to 1000.
            zero_init_residual (bool, optional): Использование . Defaults to False.
            groups (int, optional): _description_. Defaults to 1.
            width_per_group (int, optional): _description_. Defaults to 64.
            replace_stride_with_dilation (Optional[List[bool]], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = Norm2d
        self._norm_layer = norm_layer
        self.input_channels = input_channels
        self.zero_init_residual = zero_init_residual
        self.inplanes = 64
        self.dilation = 1
        self.downsample_factor = 4
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        self.layer0 = Sequential(
            Conv2d(self.input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes),
            NLF(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(block, 64, layers[0], num_layer=0)
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            num_layer=1,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            num_layer=2,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            num_layer=3,
        )

        self.output_channels = 512 * block.expansion
        # self.downsample_factor = 32

        self.init_weights()

    def init_weights(self):
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        num_layer: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        self.downsample_factor *= stride

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                downsample_factor=self.downsample_factor,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    downsample_factor=self.downsample_factor,
                )
            )

        self.inner_features[num_layer] = [
            f"layer{num_layer + 1}",
            self.downsample_factor,
            layers[-1].output_channels,
        ]

        return Sequential(*layers)

    def _forward_impl(self, x: Tensor, aux: Optional[List] = []) -> Tensor:
        # See note [TorchScript super()]
        aux_out = []
        x = self.layer0(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet_backbone(
    arch,
    block: Union[BasicBlock, Bottleneck],
    layers: Tuple[int, int, int, int],
    pretrained: bool,
    progress: bool,
    **kwargs,
) -> AbstractResnet:
    backbone = ResNet(block, layers, **kwargs)
    if pretrained:
        path = "/app/pmodels/resnet/" + arch + ".pth"
        backbone.load_state_dict(torch.load(path))

    return backbone


def b_resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_backbone(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs
    )


def b_resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_backbone(
        "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def b_resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_backbone(
        "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def b_resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_backbone(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def b_resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_backbone(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def b_resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet_backbone(
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def b_resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet_backbone(
        "resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def b_wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet_backbone(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def b_wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet_backbone(
        "wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


BACKBONES.register_module("b_resnet18", module=b_resnet18)
BACKBONES.register_module("b_resnet34", module=b_resnet34)
BACKBONES.register_module("b_resnet50", module=b_resnet50)
BACKBONES.register_module("b_resnet101", module=b_resnet101)
BACKBONES.register_module("b_resnet152", module=b_resnet152)
BACKBONES.register_module("b_resnext50_32x4d", module=b_resnext50_32x4d)
BACKBONES.register_module("b_resnext101_32x8d", module=b_resnext101_32x8d)
BACKBONES.register_module("b_wide_resnet50_2", module=b_wide_resnet50_2)
BACKBONES.register_module("b_wide_resnet101_2", module=b_wide_resnet101_2)
