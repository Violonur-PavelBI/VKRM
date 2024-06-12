from collections import OrderedDict

import torch
import torch.nn.functional as F

from ... import core as c

from ...baseblocks.mobilenetv2_basic import _make_divisible as make_divisible

from ..primitives.static import MyNetwork, Hsigmoid


class SEModule(c.Module):
    """Реализация через AdaptiveAvgPool2d

    замена h_sigmoid на sigmoid

    нет реализации метода config"""

    CHANNEL_DIVISIBLE = 8
    REDUCTION = 4

    def __init__(self, channel, reduction=None):
        super(SEModule, self).__init__()
        self.avg = c.AdaptiveAvgPool2d((1, 1))

        self.channel = channel
        self.reduction = SEModule.REDUCTION if reduction is None else reduction
        num_mid = make_divisible(
            self.channel // self.reduction, divisor=self.CHANNEL_DIVISIBLE
        )

        self.reduce = c.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)
        self.relu = c.ReLU(inplace=True)
        self.expand = c.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)
        self.sigmoid = Hsigmoid()

    def forward(self, x: c.Tensor):
        y = self.avg(x)

        y = self.reduce(y)
        y = self.relu(y)

        y = self.expand(y)
        y = self.sigmoid(y)

        return x * y

    def __repr__(self):
        return "SE(channel=%d, reduction=%d)" % (self.channel, self.reduction)


class DynamicSE(SEModule):
    def __init__(self, max_channel):
        super(DynamicSE, self).__init__(max_channel)

    def get_active_weight(self, conv: c.Conv2d, in_channels, out_channels):
        return conv.weight[:out_channels, :in_channels, :, :]

    def get_active_bias(self, conv: c.Conv2d, out_channels):
        return conv.bias[:out_channels] if conv.bias is not None else None

    def get_active_subnet(self, channel: int):
        sub_layer: SEModule = SEModule(channel, self.reduction)
        mid_channel = make_divisible(
            channel // self.reduction, divisor=self.CHANNEL_DIVISIBLE
        )

        sub_layer.reduce.weight.data.copy_(
            self.get_active_weight(self.reduce, channel, mid_channel).data
        )
        sub_layer.reduce.bias.data.copy_(
            self.get_active_bias(self.reduce, mid_channel).data
        )

        sub_layer.expand.weight.data.copy_(
            self.get_active_weight(self.expand, mid_channel, channel).data
        )
        sub_layer.expand.bias.data.copy_(
            self.get_active_bias(self.expand, channel).data
        )

        return sub_layer

    def reorganize_weights(self, sorted_idx: c.Tensor):
        # se expand: output dim 0 reorganize
        self.expand.weight.data = torch.index_select(
            self.expand.weight.data, 0, sorted_idx
        )
        self.expand.bias.data = torch.index_select(self.expand.bias.data, 0, sorted_idx)

        # se reduce: input dim 1 reorganize
        self.reduce = self.reduce
        self.reduce.weight.data = torch.index_select(
            self.reduce.weight.data, 1, sorted_idx
        )
        # middle weight reorganize
        se_importance = torch.sum(torch.abs(self.expand.weight.data), dim=(0, 2, 3))
        se_importance, se_idx = torch.sort(se_importance, dim=0, descending=True)

        self.expand.weight.data = torch.index_select(self.expand.weight.data, 1, se_idx)
        self.reduce.weight.data = torch.index_select(self.reduce.weight.data, 0, se_idx)
        self.reduce.bias.data = torch.index_select(self.reduce.bias.data, 0, se_idx)

    def forward(self, x: c.Tensor):
        in_channel = x.size(1)
        num_mid = make_divisible(
            in_channel // self.reduction, divisor=MyNetwork.CHANNEL_DIVISIBLE
        )

        y = x.mean(3, keepdim=True).mean(2, keepdim=True)

        reduce_filter = self.get_active_weight(
            self.reduce, in_channel, num_mid
        ).contiguous()
        reduce_bias = self.get_active_bias(self.reduce, num_mid)
        y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
        y = self.relu(y)

        expand_filter = self.get_active_weight(
            self.expand, num_mid, in_channel
        ).contiguous()
        expand_bias = self.get_active_bias(self.expand, in_channel)
        y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)
        y = self.sigmoid(y)

        return x * y
