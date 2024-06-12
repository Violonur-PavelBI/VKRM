from ..core.container import Sequential
from ..core.module import Module
from ..core import (
    Norm2d,
    Conv2d,
    Tensor,
    # nlf,
    ReLU,
    BatchNorm2d,
    GroupNorm,
    flatten,
    MaxPool2d,
    Sigmoid,
    AvgPool2d,
    Dropout,
    AdaptiveAvgPool2d,
)
from ..core.abstract import Node

import math


class SEModule(Node):
    """
    Sequeeze Excitation Module
    """

    def __init__(self, channels, reduction, downsample_factor):
        super(SEModule, self).__init__()
        self.input_channels = channels
        self.output_channels = channels
        self.downsample_factor = downsample_factor

        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BottleneckForward(Node):
    """
    Base class for se bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(BottleneckForward):
    """
    Bottleneck for SENet154.
    """

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        groups,
        reduction,
        stride=1,
        downsample=None,
        downsample_factor=1,
    ):
        super(SEBottleneck, self).__init__()
        self.output_channels = planes * 4
        self.input_channels = inplanes
        self.downsample_factor = downsample_factor

        self.conv1 = Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = Norm2d(planes * 2)
        self.conv2 = Conv2d(
            planes * 2,
            planes * 4,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = Norm2d(planes * 4)
        self.conv3 = Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = Norm2d(planes * 4)
        self.relu = ReLU(inplace=True)
        self.se_module = SEModule(
            planes * 4, reduction=reduction, downsample_factor=downsample_factor
        )
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(BottleneckForward):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        groups,
        reduction,
        stride=1,
        downsample=None,
        downsample_factor=1,
    ):
        super(SEResNetBottleneck, self).__init__()
        self.output_channels = planes * 4
        self.input_channels = inplanes
        self.downsample_factor = downsample_factor

        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = Norm2d(planes)
        self.conv2 = Conv2d(
            planes, planes, kernel_size=3, padding=1, groups=groups, bias=False
        )
        self.bn2 = Norm2d(planes)
        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = Norm2d(planes * 4)
        self.relu = ReLU(inplace=True)
        self.se_module = SEModule(
            planes * 4, reduction=reduction, downsample_factor=downsample_factor
        )
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(BottleneckForward):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        groups,
        reduction,
        stride=1,
        downsample=None,
        base_width=4,
        downsample_factor=1,
    ):
        self.output_channels = planes * 4
        self.input_channels = inplanes
        self.downsample_factor = downsample_factor
        super(SEResNeXtBottleneck, self).__init__()
        # self.output_channels = planes * 4
        # self.input_channels = inplanes
        # self.downsample_factor = downsample_factor

        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = Norm2d(width)
        self.conv2 = Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = Norm2d(width)
        self.conv3 = Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = Norm2d(planes * 4)
        self.relu = ReLU(inplace=True)
        self.se_module = SEModule(
            planes * 4, reduction=reduction, downsample_factor=downsample_factor
        )
        self.downsample = downsample
        self.stride = stride


# __all__ = [SEBottleneck, SEResNeXtBottleneck, SEResNetBottleneck, SEModule]
