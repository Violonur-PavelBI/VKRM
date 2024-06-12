from ..core.container import ModuleList, Sequential
from ..core.module import Module
from ..core import (
    Conv2d,
    BatchNorm2d,
    ReLU,
    Tensor,
    AdaptiveAvgPool2d,
    cat,
    Dropout,
)
from ..core import functional as F
from typing import List


class ASPPConv(Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            BatchNorm2d(out_channels),
            ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            AdaptiveAvgPool2d(1),
            Conv2d(in_channels, out_channels, 1, bias=False),
            BatchNorm2d(out_channels),
            ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(Module):
    def __init__(
        self, in_channels: int, atrous_rates: List[int], out_channels: int = 256
    ) -> None:
        super().__init__()
        modules = []
        modules.append(
            Sequential(
                Conv2d(in_channels, out_channels, 1, bias=False),
                BatchNorm2d(out_channels),
                ReLU(),
            )
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = ModuleList(modules)

        self.project = Sequential(
            Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            BatchNorm2d(out_channels),
            ReLU(),
            Dropout(0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = cat(_res, dim=1)
        return self.project(res)
