from typing import Union, Tuple

import numpy as np
import torch

from ... import core as c


# копипаста с оригинальной репы
# nn заменял на core
def min_divisible_value(n1: int, v1: int):
    """make sure n1 is divisible by v1, otherwise decrease v1

    Это из оригинальной репы OFA"""
    if v1 >= n1:
        return n1
    while n1 % v1 != 0:
        v1 -= 1
    return v1


def get_same_padding(kernel_size: Union[int, Tuple[int, int]]):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert (
        isinstance(kernel_size, int),
        "kernel size should be either `int` or `tuple`",
    )
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def set_bn_param(net, momentum, eps, gn_channel_per_group=None, ws_eps=None, **kwargs):
    for m in net.modules():
        if isinstance(m, c.BatchNorm2d):
            m.momentum = momentum
            m.eps = eps


def get_bn_param(net):
    # TODO: Разобраться с этой странной функцией, и особенно проверкой на instance
    ws_eps = None
    for m in net.modules():
        if isinstance(m, c.BatchNorm2d):
            return {
                "momentum": m.momentum,
                "eps": m.eps,
                "ws_eps": ws_eps,
            }
    return None


def get_net_device(net):
    """Я считаю, что это -- одна из самых гениальных функций от авторов OFA"""
    return net.parameters().__next__().device


def adjust_bn_according_to_idx(bn, idx):
    # TODO: Разобраться с этой странной функцией, и особенно проверкой на instance
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    if isinstance(bn, c.BatchNorm2d):
        bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
        bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)


def copy_bn(target_bn, src_bn):
    # TODO: Разобраться с этой странной функцией, и особенно проверкой на instance
    feature_dim = target_bn.num_features

    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    if isinstance(src_bn, c.BatchNorm2d):
        target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
        target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])


def sub_filter_start_end(kernel_size: int, sub_kernel_size: int):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def val2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]
