from ...core.container import ModuleList
from ...baseblocks.wave_unet_basic import ConvReluBlock, DownBlock, UpBlock, FinalBlock
from ...core.abstract import AtomicNetwork


class WaveUNet(AtomicNetwork):
    def __init__(
        self, input_c=1, num_classes=1, num_blocks=12, extra_channels=24, **kwargs
    ):
        super().__init__()
        self.input_channels = input_c
        self.num_classes = num_classes
        ch_list = [1] + [i * extra_channels for i in range(1, num_blocks + 2)]
        # Encoder:
        self.encoder = ModuleList()
        for i in range(num_blocks):
            self.encoder.append(
                DownBlock(
                    in_channels=ch_list[i], out_channels=ch_list[i + 1], kernel_size=15
                )
            )
        # Bottleneck layer:
        self.middle_layer = ConvReluBlock(
            in_channels=ch_list[-2], out_channels=ch_list[-1], kernel_size=15
        )
        # Decoder:
        self.decoder = ModuleList()
        for i in range(num_blocks):
            self.decoder.append(
                UpBlock(
                    in_channels=ch_list[-1 - i],
                    cat_channels=ch_list[-2 - i],
                    out_channels=ch_list[-2 - i],
                    kernel_size=5,
                )
            )
        self.decoder.append(
            FinalBlock(
                in_channels=ch_list[1],
                cat_channels=ch_list[0],
                out_channels=num_classes,
            )
        )

    def forward(self, x):
        skip_c_list = list()
        skip_c_list.append(x)
        for down_layer in self.encoder:
            x, skip_c = down_layer(s)
            skip_c_list.append(skip_c)
        x = self.middle_layer(x)
        for i, up_layer in enumerate(self.decoder):
            x = up_layer(x, skip_c_list[-1 - i])
        return x
