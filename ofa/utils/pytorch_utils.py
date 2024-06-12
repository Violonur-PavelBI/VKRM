import copy
import math
import time

import torch
from torch import nn
import torch.nn as nn
from PIL import Image
from .sam import SAM
from loguru import logger

__all__ = [
    "CHANNEL_DIVISIBLE",
    "rm_bn_from_net",
    "get_net_device",
    "count_parameters",
    "count_net_flops",
    "measure_net_latency",
    "get_net_info",
    "build_optimizer",
    "calc_learning_rate",
    "attach_palette_to_mask",
    "init_models",
]


CHANNEL_DIVISIBLE = 8


def rm_bn_from_net(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x


def get_net_device(net):
    return net.parameters().__next__().device


def count_parameters(net):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_params


def count_net_flops(net, data_shape=(1, 3, 224, 224)):
    from .flops_counter import profile

    if isinstance(net, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        _net = net.module
    else:
        _net = net
    # они не считаются нормально!
    # нужно перевести на thop
    flop, _ = profile(copy.deepcopy(_net).cpu(), data_shape)
    return flop


def measure_net_latency(
    _net, l_type="gpu8", fast=True, input_shape=(3, 224, 224), clean=False
):
    if isinstance(net, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        net = _net.module
    else:
        net = _net
    # remove bn from graph
    rm_bn_from_net(net)

    # return `ms`
    if "gpu" in l_type:
        l_type, batch_size = l_type[:3], int(l_type[3:])
    else:
        batch_size = 1

    data_shape = [batch_size] + list(input_shape)
    if l_type == "cpu":
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
        if get_net_device(net) != torch.device("cpu"):
            if not clean:
                logger.info("move net to cpu for measuring cpu latency")
            net = copy.deepcopy(net).cpu()
    elif l_type == "gpu":
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
    else:
        raise NotImplementedError
    images = torch.zeros(data_shape, device=get_net_device(net))

    measured_latency = {"warmup": [], "sample": []}
    net.eval()
    with torch.no_grad():
        for i in range(n_warmup):
            inner_start_time = time.time()
            net(images)
            used_time = (time.time() - inner_start_time) * 1e3  # ms
            measured_latency["warmup"].append(used_time)
            if not clean:
                logger.info(f"Warmup {i}: {used_time:.3f}")
        outer_start_time = time.time()
        for i in range(n_sample):
            net(images)
        total_time = (time.time() - outer_start_time) * 1e3  # ms
        measured_latency["sample"].append((total_time, n_sample))
    return total_time / n_sample, measured_latency


def get_net_info(
    _net, input_shape=(3, 224, 224), measure_latency=None, print_info=True
):
    net_info = {}
    if isinstance(_net, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        net = _net.module
    else:
        net = _net
    # parameters
    net_info["params"] = count_parameters(net) / 1e6

    # flops
    net_info["flops"] = count_net_flops(net, [1] + list(input_shape)) / 1e6

    # latencies
    latency_types = [] if measure_latency is None else measure_latency.split("#")
    for l_type in latency_types:
        latency, measured_latency = measure_net_latency(
            net, l_type, fast=False, input_shape=input_shape
        )
        net_info["%s latency" % l_type] = {"val": latency, "hist": measured_latency}

    if print_info:
        logger.info(f"Total training params: {net_info['params']:%.2f}M")
        logger.info(f"Total FLOPs: {net_info['flops']:%.2f}M")
        for l_type in latency_types:
            logger.info(
                "Estimated %s latency: %.3fms"
                % (l_type, net_info["%s latency" % l_type]["val"])
            )

    return net_info


def build_optimizer(
    net_params,
    opt_type,
    momentum,
    nesterov,
    init_lr,
    weight_decay,
    no_decay_keys,
    betas,
    use_sam,
):
    if no_decay_keys is not None:
        assert isinstance(net_params, list) and len(net_params) == 2
        net_params = [
            {"params": net_params[0], "weight_decay": weight_decay},
            {"params": net_params[1], "weight_decay": 0},
        ]
    else:
        net_params = [{"params": net_params, "weight_decay": weight_decay}]

    if opt_type == "sgd":
        if use_sam:
            parents_optimizer = torch.optim.SGD
            optimizer = SAM(
                net_params,
                parents_optimizer,
                lr=init_lr,
                momentum=momentum,
                nesterov=nesterov,
            )
        else:
            optimizer = torch.optim.SGD(
                net_params, init_lr, momentum=momentum, nesterov=nesterov
            )
    elif opt_type == "adam":
        if use_sam:
            parents_optimizer = torch.optim.AdamW
            optimizer = SAM(net_params, parents_optimizer, lr=init_lr, betas=betas)
        else:
            optimizer = torch.optim.AdamW(net_params, init_lr, betas)
    else:
        raise NotImplementedError
    return optimizer


def calc_learning_rate(
    epoch, init_lr, n_epochs, batch=0, nBatch=None, lr_schedule_type="cosine"
) -> float:
    if lr_schedule_type == "cosine":
        t_total = n_epochs * nBatch
        t_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError("do not support: %s" % lr_schedule_type)
    return lr


def attach_palette_to_mask(mask, dataset_classes: int, background: bool = False):
    res = mask.astype("uint8")
    res = Image.fromarray(res, mode="P")
    g = torch.Generator()
    g.manual_seed(42)
    # для фона цвет уже есть
    if background:
        dataset_classes -= 1
    colours = [0, 0, 0] if dataset_classes == 1 or background else []
    colours += list(torch.randint(0, 255, size=(dataset_classes * 3,), generator=g))
    res.putpalette(colours)
    return res


def init_models(net):
    """
    Conv2d,
    BatchNorm2d, BatchNorm1d, GroupNorm
    Linear,
    """
    if isinstance(net, list):
        for sub_net in net:
            init_models(sub_net)
        return
    for m in net.modules():
        if type(m) in [nn.Conv2d, nn.Linear]:
            if m.bias is not None:
                m.bias.data.zero_()
