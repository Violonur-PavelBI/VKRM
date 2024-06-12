from abc import abstractmethod, ABCMeta

from ..core.container import Sequential

from ..core.module import Module
from ..core import Tensor, Conv2d, BatchNorm2d, ReLU, Upsample
from ..core import functional as F
from .deconv_decoders_basic import (
    DeconvDeepWise_Block,
    SqueezeDeconvDeepWise_Block,
)
from math import gcd
import math


class AbsImgDecoder(Module, metaclass=ABCMeta):
    in_channels: int
    num_classes: int
    upsample_factor: int

    @abstractmethod
    def __init__(self, in_channels, num_classes, upsample_factor):
        super(AbsImgDecoder, self).__init__()


class SDeconvUpsample(Module):
    def __init__(
        self, in_channels: int, out_channels: int, upsample_factor: int = 2, **kwargs
    ):
        """_summary_
        Sequential Deconvolitional Upsample
        This head implements sequential stacking of such blocks:
            {BN2d - ConvTranspose2d - ReLU - ConvTranspose2d - ReLU - BN2d} * n,
            where n=log(upsample_factor)/log(2)
        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            upsample_factor (int, optional): _description_. Defaults to 2.
        """
        super(SDeconvUpsample, self).__init__()

        if upsample_factor == 1.0:
            self.block = Sequential(
                Conv2d(
                    in_channels, out_channels, kernel_size=(1, 1), padding=0, stride=1
                ),
                ReLU(),
                BatchNorm2d(
                    out_channels,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
            )
        else:
            assert (
                type(upsample_factor) is int
                and math.modf(math.log(upsample_factor, 2))[0] < 1e-10
            ), "upsample factor must be an integer power of two 2!"
            layers = [
                DeconvDeepWise_Block(in_channels, in_channels)
                for i in range(1, round(math.log(upsample_factor, 2)))
            ]
            layers.append(DeconvDeepWise_Block(in_channels, out_channels))
            self.block = Sequential(*layers)

        self.num_classes = out_channels

    def forward(self, x: Tensor):
        return self.block(x)


class _UpsampleResidual(Module):
    def __init__(self, block: Module, inpc: int, outc: int, upscale: int):
        super().__init__()
        self.block = block
        if inpc >= outc:
            self.upsample_res = Sequential(
                Conv2d(inpc, outc, 1, bias=False),
                Upsample(scale_factor=upscale),
            )
        else:
            self.upsample_res = Sequential(
                Upsample(scale_factor=upscale),
                Conv2d(inpc, outc, 1, bias=False),
            )

    def forward(self, x):
        return self.block(x) + self.upsample_res(x)


class SqueezeSDeconvUpsample(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        upsample_factor=2,
        each_depth_squeeze_factor=4,
        **kwargs
    ):
        super(SqueezeSDeconvUpsample, self).__init__()
        if upsample_factor == 1.0:
            self.block = Sequential(
                Conv2d(
                    in_channels, out_channels, kernel_size=(1, 1), padding=0, stride=1
                ),
                ReLU(),
                BatchNorm2d(
                    out_channels,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
            )
        else:
            assert (
                type(upsample_factor) is int
                and math.modf(math.log(upsample_factor, 2))[0] < 1e-10
            ), "upsample factor must be an integer power of two 2!"
            depths = []
            depths += [in_channels]
            for i in range(1, round(math.log(upsample_factor, 2))):
                depths += [max(depths[i - 1] // each_depth_squeeze_factor, 64)]
            layers = [
                DeconvDeepWise_Block(
                    depths[i - 1],
                    depths[i],
                )
                for i in range(1, round(math.log(upsample_factor, 2)))
            ]
            layers.append(SqueezeDeconvDeepWise_Block(depths[-1], out_channels))
            self.block = Sequential(*layers)
        self.num_classes = out_channels

    def forward(self, x: Tensor):
        return self.block(x)


class SqueezeRDeconvUpsample(Module):
    # FIXME
    def __init__(
        self,
        in_channels,
        out_channels,
        upsample_factor=2,
        each_depth_squeeze_factor=4,
        **kwargs
    ):
        super(SqueezeRDeconvUpsample, self).__init__()
        if upsample_factor == 1.0:
            self.block = Sequential(
                Conv2d(
                    in_channels, out_channels, kernel_size=(1, 1), padding=0, stride=1
                ),
                ReLU(),
                BatchNorm2d(
                    out_channels,
                    eps=1e-05,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=True,
                ),
            )
        else:
            assert (
                type(upsample_factor) is int
                and math.modf(math.log(upsample_factor, 2))[0] < 1e-10
            ), "upsample factor must be an integer power of two 2!"
            depths = []
            depths += [in_channels]
            for i in range(1, round(math.log(upsample_factor, 2))):
                depths += [max(depths[i - 1] // each_depth_squeeze_factor, 64)]
            layers = [
                _UpsampleResidual(
                    SqueezeDeconvDeepWise_Block(
                        depths[i - 1], depths[i], each_depth_squeeze_factor
                    ),
                    depths[i - 1],
                    depths[i],
                    2,
                )
                for i in range(1, round(math.log(upsample_factor, 2)))
            ]
            layers.append(
                _UpsampleResidual(
                    SqueezeDeconvDeepWise_Block(
                        depths[-1], out_channels, each_depth_squeeze_factor
                    ),
                    depths[-1],
                    out_channels,
                    2,
                )
            )
            self.block = Sequential(*layers)
        self.num_classes = out_channels

    def forward(self, x: Tensor):
        return self.block(x)
