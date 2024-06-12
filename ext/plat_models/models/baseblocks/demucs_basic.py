from ..core.container import Sequential
from ..core.module import Module
from ..core import LSTM, Linear, Conv1d, ReLU, GLU, ConvTranspose1d


class BiLSTM(Module):
    def __init__(self, input_size, num_layers=2):
        super().__init__()
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=input_size,
            num_layers=num_layers,
            bidirectional=True,
        )
        self.fc = Linear(2 * input_size, input_size)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x.permute(1, 2, 0)


class DownBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = Sequential(
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=8,
                stride=4,
                padding=2,
            ),
            ReLU(inplace=True),
            Conv1d(
                in_channels=out_channels,
                out_channels=2 * out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            GLU(dim=1),
        )

    def forward(self, x):
        return self.model(x)


class UpBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = Sequential(
            Conv1d(
                in_channels=in_channels,
                out_channels=2 * in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            GLU(dim=1),
            ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=8,
                stride=4,
                padding=2,
            ),
            ReLU(inplace=True),
        )

    def forward(self, x, skip_c):
        return self.model(x + skip_c)
