from ..core.container import Sequential
from ..core.module import Module
from ..core import (
    Conv2d,
    BatchNorm2d,
    LeakyReLU,
    InstanceNorm2d,
    Sigmoid,
)
import functools


class Pixel(Module):
    def __init__(self, input_nc, ndf, norm_layer):
        super(Pixel, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == InstanceNorm2d
        else:
            use_bias = norm_layer == InstanceNorm2d
        self.net = [
            Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            LeakyReLU(0.2, True),
            Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            BatchNorm2d(ndf * 2),
            LeakyReLU(0.2, True),
            Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
            Sigmoid(),
        ]
        self.net = Sequential(*self.net)

    def forward(self, data):
        return self.net(data)


def PixelDiscriminator(num_input_channel, ndf=64, norm_layer=BatchNorm2d):
    model = Pixel(num_input_channel, ndf, norm_layer)
    return model
