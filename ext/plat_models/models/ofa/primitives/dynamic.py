from typing import List

import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from ... import core as c

from .func import get_same_padding, sub_filter_start_end


class DynamicConv2d(c.Module):
    """
    Свёртка с динамическим количеством входных и выходных каналов и размером ядра.
    Переход к меньшему размеру ядра осуществляется через slice и умножение на матрицу.
    """

    def __init__(
        self,
        max_in_channels: int,
        max_out_channels: int,
        kernel_size_list: List[int],
        stride: int = 1,
        dilation: int = 1,
        separable: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if separable and (max_in_channels != max_out_channels or bias):
            raise ValueError(
                f"Unacceptable arguments if {separable = }:\n"
                f"\t{max_in_channels = },\n"
                f"\t{max_out_channels = },\n"
                f"\t{bias = }.\n"
                "max_in_channels must be equal to max_out_channels, bias must be False."
            )

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size_list = sorted(set(kernel_size_list))
        self.stride = stride
        self.dilation = dilation
        self.separable = separable
        self.groups = max_in_channels if separable else 1
        self.bias = bias

        self.conv = c.Conv2d(
            self.max_in_channels,
            self.max_out_channels,
            max(self.kernel_size_list),
            stride=self.stride,
            groups=self.groups,
            bias=self.bias,
        )

        scale_params = {}
        for i in range(len(self.kernel_size_list) - 1):
            ks_small = self.kernel_size_list[i]
            ks_large = self.kernel_size_list[i + 1]
            param_name = f"{ks_large}to{ks_small}"
            scale_params[f"{param_name}_matrix"] = Parameter(torch.eye(ks_small**2))
        for name, param in scale_params.items():
            self.register_parameter(name, param)

        self.active_out_channel = self.max_out_channels
        self.active_kernel_size = max(self.kernel_size_list)

    def get_active_filter(self, out_channel, in_channel, kernel_size):
        max_kernel_size = max(self.kernel_size_list)

        filters = self.conv.weight[:out_channel, :in_channel, :, :]
        if kernel_size < max_kernel_size:
            source_ks = max_kernel_size
            for target_ks in self.kernel_size_list[-2::-1]:
                start, end = sub_filter_start_end(source_ks, target_ks)
                filters_small = filters[:, :, start:end, start:end]
                filters_small = filters_small.contiguous()

                filters_small = filters_small.view(-1, target_ks**2)
                filters_small = F.linear(
                    filters_small,
                    self.__getattr__("%dto%d_matrix" % (source_ks, target_ks)),
                )
                filters_small = filters_small.view(
                    filters.size(0), filters.size(1), target_ks, target_ks
                )

                filters = filters_small
                source_ks = target_ks
                if source_ks == kernel_size:
                    break
        return filters

    def forward(self, x, out_channel=None, kernel_size=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)

        filters = self.get_active_filter(out_channel, in_channel, kernel_size)
        filters = filters.contiguous()
        bias = self.conv.bias[:out_channel] if self.bias else None
        padding = get_same_padding(kernel_size)
        groups = in_channel if self.separable else 1

        y = F.conv2d(x, filters, bias, self.stride, padding, self.dilation, groups)
        return y


class DynamicBatchNorm2d(c.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm2d, self).__init__()

        self.max_feature_dim = max_feature_dim
        self.bn = c.BatchNorm2d(self.max_feature_dim)

    @staticmethod
    def bn_forward(x, bn: c.BatchNorm2d, feature_dim):
        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y


class DynamicLinear(c.Module):
    """Динамический линейный слой. Делает view во время forward"""

    def __init__(self, max_in_features: int, max_out_features: int, bias: bool = True):
        super(DynamicLinear, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = c.LinearOpt(
            self.max_in_features, self.max_out_features, self.bias
        )

        self.active_out_features = self.max_out_features

    def get_active_weight(self, out_features, in_features):
        return self.linear.linear.weight[:out_features, :in_features]

    def get_active_bias(self, out_features):
        return self.linear.linear.bias[:out_features] if self.bias else None

    def forward(self, x: c.Tensor, out_features: int = None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        weight = self.get_active_weight(out_features, in_features).contiguous()
        bias = self.get_active_bias(out_features)
        x = x.view(x.size(0), -1)
        y = F.linear(x, weight, bias)
        return y
