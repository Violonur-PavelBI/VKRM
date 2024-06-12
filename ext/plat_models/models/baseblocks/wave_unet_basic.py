from ..core.container import Sequential
from ..core.module import Module
from ..core import Conv1d, BatchNorm1d, LeakyReLU, Upsample


class ConvReluBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.model = Sequential(
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            BatchNorm1d(out_channels),
            LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class DownBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.convrelu = ConvReluBlock(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )

    def forward(self, x):
        skip_c = self.convrelu(x)
        out = skip[:, :, ::2]
        return out, skip


class UpBlock(Module):
    def __init__(self, in_channels, cat_channels, out_channels, kernel_size):
        super().__init__()
        self.convrelu = ConvReluBlock(
            in_channels=in_channels + cat_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )
        self.upsample = Upsample(scale_factor=2, mode="linear", align_corners=True)

    def forward(self, x, skip_c):
        out = torch.cat([x, skip_c], dim=1)
        return self.conv(out)


class FinalBlock(Module):
    def __init__(self, in_channels, cat_channels, out_channels):
        super().__init__()
        self.conv = Conv1d(
            in_channels=in_channels + cat_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, skip_c):
        out = torch.cat([x, skip_c], dim=1)
        return self.conv(out)
