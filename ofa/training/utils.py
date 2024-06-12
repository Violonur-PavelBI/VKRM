from typing import List, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ofa.training.strategies.base_strategy import BaseContext

from ofa.utils import AverageMeter, get_net_device, DistributedTensor

__all__ = ["set_running_statistics"]


# FIXME
def set_running_statistics(model, data_loader, distributed=False):
    """Функция для калибровки BN статистик сети под выборку

    Используется, когда необходимо оценить качество части динамической подсети
    в режиме валидации

    Временно в фукнции используется атрибут класса
    нужно переделать на проход в цикле (?)
    """
    bn_mean = {}
    bn_var = {}
    old_forwards = {}
    DynamicBatchNormCls = None
    # FIXME
    for name, m in model.named_modules():
        # FIXME
        if DynamicBatchNormCls is None and "DynamicBatchNorm" in m.__class__.__name__:
            DynamicBatchNormCls = m.__class__
        if isinstance(m, nn.BatchNorm2d):
            if distributed:
                bn_mean[name] = DistributedTensor(name + "#mean")
                bn_var[name] = DistributedTensor(name + "#var")
            else:
                bn_mean[name] = AverageMeter()
                bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = (
                        x.mean(0, keepdim=True)
                        .mean(2, keepdim=True)
                        .mean(3, keepdim=True)
                    )  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = (
                        batch_var.mean(0, keepdim=True)
                        .mean(2, keepdim=True)
                        .mean(3, keepdim=True)
                    )

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,
                        0.0,
                        bn.eps,
                    )

                return lambda_forward

            old_forwards[name] = m.forward
            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    if len(bn_mean) == 0:
        # skip if there is no batch normalization layers in the network
        return

    with torch.no_grad():
        DynamicBatchNormCls.SET_RUNNING_STATISTICS = True
        for data in data_loader:
            images = data["image"].to(get_net_device(model))
            model(images)
        DynamicBatchNormCls.SET_RUNNING_STATISTICS = False

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)
        if name in old_forwards:
            m.forward = old_forwards[name]


def convert_out(output: Union[torch.Tensor, Sequence[torch.Tensor]]) -> None:
    """Переводит выход в fp32. Нужно для подсчёта метрик иногда, если использовали fp16 для форварда"""
    if isinstance(output, torch.Tensor):
        output = output.float()
    elif isinstance(output, Sequence):
        output: Sequence[torch.Tensor]
        for i in range(len(output)):
            convert_out(output[i])
    else:
        raise TypeError(f"Not implemented for {type(output)}")
