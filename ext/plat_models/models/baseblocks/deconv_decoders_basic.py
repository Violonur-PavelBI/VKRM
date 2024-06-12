from ..core.container import Sequential
from ..core.module import Module
from ..core import ConvTranspose2d, BatchNorm2d, ReLU
from ..core import functional as F
from math import gcd


class DeconvDeepWise_Block(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DeconvDeepWise_Block, self).__init__()

        self.block = Sequential(
            BatchNorm2d(
                in_channels,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=(3, 3),
                padding=1,
                stride=1,
                output_padding=0,
                groups=in_channels,
            ),
            ReLU(),
            ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(2, 2),
                padding=0,
                stride=2,
                output_padding=0,
                groups=gcd(in_channels, out_channels),
            ),
            ReLU(),
        )

    def forward(self, x):
        x = self.block(x)

        return x


class SqueezeDeconvDeepWise_Block(Module):
    def __init__(
        self, in_channels: int, out_channels: int, depth_squeeze_factor: int = 4
    ):
        super(SqueezeDeconvDeepWise_Block, self).__init__()
        self.squueeze_factor = depth_squeeze_factor
        self.block = Sequential(
            BatchNorm2d(
                in_channels,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            ConvTranspose2d(
                in_channels,
                max(64, in_channels // self.squueeze_factor),
                kernel_size=(3, 3),
                padding=1,
                stride=1,
                output_padding=0,
                groups=gcd(in_channels, max(64, in_channels // self.squueeze_factor)),
            ),
            ConvTranspose2d(
                max(64, in_channels // self.squueeze_factor),
                out_channels,
                kernel_size=(2, 2),
                padding=0,
                stride=2,
                output_padding=0,
                groups=gcd(max(64, in_channels // self.squueeze_factor), out_channels),
            ),
            ReLU(),
        )

    def forward(self, x):
        x = self.block(x)

        return x
