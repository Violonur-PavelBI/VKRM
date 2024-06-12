from ...core.container import ModuleList
from ...baseblocks.demucs_basic import BiLSTM, DownBlock, UpBlock
from ...core.abstract import AtomicNetwork


class Demucs(AtomicNetwork):
    def __init__(self, input_c=1, num_classes=1, num_blocks=6):
        super().__init__()
        self.input_channels = input_c
        self.num_classes = num_classes
        ch_list = [1, 64, 128, 256, 512, 1024, 2048]
        # Encoder:
        self.encoder = ModuleList()
        for i in range(num_blocks):
            self.encoder.append(
                DownBlock(in_channels=ch_list[i], out_channels=ch_list[i + 1])
            )
        # Bottleneck layer:
        self.middle_layer = BiLSTM(ch_list[-1])
        # Decoder:
        ch_list[0] = 2
        self.decoder = ModuleList()
        for i in range(num_blocks):
            self.decoder.append(
                UpBlock(in_channels=ch_list[-1 - i], out_channels=ch_list[-2 - i])
            )

    def forward(self, x):
        skip_c_list = list()
        skip_c_list.append(x)
        for down_layer in self.encoder:
            x = down_layer(x)
            skip_c_list.append(x)
        x = self.middle_layer(x)
        for i, up_layer in enumerate(self.decoder):
            x = up_layer(x, skip_c_list[-1 - i])
        return x
