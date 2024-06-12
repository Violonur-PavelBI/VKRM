from ...core.module import Module
from ...baseblocks.unet_basic import DoubleConv, Down, Up, OutConv
from ...core import Sigmoid
from ...core.abstract import AtomicNetwork


class UNet(AtomicNetwork):
    def __init__(self, input_c=3, num_classes=3, bilinear=False, **kwargs):
        super(UNet, self).__init__()
        self.input_channels = input_c
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(input_c, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def unet(num_input_channel=3, bilinear=False, num_classes=3, **kwargs):
    model = UNet(num_input_channel, num_classes, bilinear, **kwargs)
    return model
