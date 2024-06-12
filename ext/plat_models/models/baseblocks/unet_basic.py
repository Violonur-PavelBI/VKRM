from ..core.container import Sequential
from ..core.module import Module
from ..core import (
    Conv2d,
    BatchNorm2d,
    ReLU,
    MaxPool2d,
    Upsample,
    ConvTranspose2d,
)
from ..core.functional import cat


class DoubleConv(Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = Sequential(
            Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            BatchNorm2d(mid_channels),
            ReLU(inplace=True),
            Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = Sequential(
            MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
