from ..core.module import Module
from ..core import (
    Norm2d,
    NLF,
    Conv2d,
    Tensor,
    # nlf
)
from typing import Optional, Callable
from ..core.abstract import Node


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> Conv2d:
    """3x3 convolution with padding"""
    return Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> Conv2d:
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(Node):
    r"""Базовый блок резнет-подобных сетей

    Consist:

        {   Conv2d: 3x3
            NL2d
            NLF
            Conv2d: 3x3
            Nl2d
        } + Identity
        s
        Opt: Downsample Layer (Для Identity, если stride!=1)

    Args:
        inplanes (int): Число входных каналов (плоскостей),
        planes (int): Число внутренних каналов (плоскостей).
        stride (int, optional): Общий страйд блока. Defaults to 1.
        downsample (Optional[Module], optional): Блок, обеспечивающий снижение вычисление identity в случае не 1 страйда. Defaults to None.
        groups (int, optional): Число груп для блоков конволюций. Defaults to 1. Для данного блока изменения параметра не доступно.
        base_width (int, optional): _description_. Defaults to 64. Для данного блока изменения параметра не доступно.
        dilation (int, optional): _description_. Defaults to 1. Для данного блока изменения параметра не доступно.
        norm_layer (Optional[Callable[..., Module]], optional): _description_. Defaults to None. Слой нормализации (BatchNorm и подобные)

    Raises:
        ValueError: Число груп не равно 1, и\или базовая ширина не 64.
        NotImplementedError: Параметр дилатации отличный от 1 не поддерживается.
    """

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., Module]] = None,
        downsample_factor: int = 1,
    ) -> None:
        r"""Args:
            inplanes (int): Число входных каналов (плоскостей),
            planes (int): Число внутренних каналов (плоскостей).
            stride (int, optional): Общий страйд блока. Defaults to 1.
            downsample (Optional[Module], optional): Блок, обеспечивающий снижение вычисление identity в случае не 1 страйда. Defaults to None.
            groups (int, optional): Число груп для блоков конволюций. Defaults to 1. Для данного блока изменения параметра не доступно.
            base_width (int, optional): _description_. Defaults to 64. Для данного блока изменения параметра не доступно.
            dilation (int, optional): _description_. Defaults to 1. Для данного блока изменения параметра не доступно.
            norm_layer (Optional[Callable[..., Module]], optional): _description_. Defaults to None. Слой нормализации (BatchNorm и подобные)

        Raises:
            ValueError: Число груп не равно 1, и\или базовая ширина не 64.
            NotImplementedError: Параметр дилатации отличный от 1 не поддерживается."""
        super(BasicBlock, self).__init__()
        self.output_channels = planes
        self.input_channels = inplanes
        self.downsample_factor = downsample_factor
        if norm_layer is None:
            norm_layer = Norm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = NLF(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(Node):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., Module]] = None,
        downsample_factor: int = 1,
    ) -> None:
        super(Bottleneck, self).__init__()
        self.output_channels = planes * self.expansion  # added * self.expansion
        self.input_channels = inplanes
        self.downsample_factor = downsample_factor
        if norm_layer is None:
            norm_layer = Norm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = NLF(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
