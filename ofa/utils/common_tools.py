import csv
import json
import os
import pickle
from copy import deepcopy
from datetime import datetime, timedelta
from functools import cached_property, partial
from pathlib import Path
from typing import Dict, Literal, Union

import numpy as np
import yaml
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams

from .configs_dataclasses import (
    CommonConfig,
    Config,
    EvolutionStageConfig,
    FinetuneStageConfig,
    PredictorDatasetStageConfig,
    PredictorLearnStageConfig,
    SupernetLearnStageConfig,
    ThresholdSearchStageConfig,
)
from loguru import logger

__all__ = [
    "build_config_from_file",
    "sort_dict",
    "get_same_padding",
    "get_split_list",
    "list_mean",
    "list_join",
    "subset_mean",
    "sub_filter_start_end",
    "min_divisible_value",
    "val2list",
    "write_log",
    "write_metrics",
    "pairwise_accuracy",
    "accuracy",
    "AverageMeter",
    "MultiClassAverageMeter",
    "Logger",
    "TensorBoardLogger",
]


def _load_config_file(config_file=Union[str, Path]) -> dict:
    if isinstance(config_file, (str, Path)):
        if isinstance(config_file, str):
            file_ext = config_file.split(".")[-1]
        else:
            file_ext = config_file.suffix[1:]

        if file_ext == "json":
            load_config = json.load
        elif file_ext == "yaml":
            load_config = partial(yaml.load, Loader=yaml.Loader)
        else:
            raise NotImplemented

    with open(config_file) as f_in:
        config = load_config(f_in)
    return config


def build_config_from_file(config_file=Union[str, Path]) -> Config:
    config_dict = _load_config_file(config_file)
    raw_config = deepcopy(config_dict)
    config = Config(**config_dict)
    config.raw_config = raw_config

    return config


def sort_dict(src_dict, reverse=False, return_dict=True):
    output = sorted(src_dict.items(), key=lambda x: x[1], reverse=reverse)
    if return_dict:
        return dict(output)
    else:
        return output


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def get_split_list(in_dim, child_num, accumulate=False):
    in_dim_list = [in_dim // child_num] * child_num
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    if accumulate:
        for i in range(1, child_num):
            in_dim_list[i] += in_dim_list[i - 1]
    return in_dim_list


def list_mean(x):
    return sum(x) / len(x)


def list_join(val_list, sep="\t"):
    return sep.join([str(val) for val in val_list])


def subset_mean(val_list, sub_indexes):
    sub_indexes = val2list(sub_indexes, 1)
    return list_mean([val_list[idx] for idx in sub_indexes])


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def min_divisible_value(n1, v1):
    """make sure v1 is divisible by n1, otherwise decrease v1"""
    if v1 >= n1:
        return n1
    while n1 % v1 != 0:
        v1 -= 1
    return v1


def val2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


class TensorBoardLogger:
    @cached_property
    def tensorboard(self):
        return SummaryWriter(self.path)

    def __init__(self, args: Config, path: str):
        if args is None:
            return

        self.path = path
        self.stage_name = None
        self.calls = 0
        os.makedirs(self.path, exist_ok=True)

        # hparams
        # TODO: refactor for dataclasses
        self.hparams: Dict = {}
        if hasattr(args.common, "hparams"):
            for hparam_name in args.common.hparams:
                if hasattr(args.common, hparam_name):
                    hparam_value = getattr(args.common, hparam_name)
                    if isinstance(hparam_value, list):
                        hparam_value = " ".join(map(str, hparam_value))
                    self.hparams[hparam_name] = hparam_value
        self.hparams_metrics: Dict = {}
        if hasattr(args.common, "hparams_metrics"):
            self.hparams_metrics = {
                metric: None for metric in args.common.hparams_metrics
            }
        exp, ssi, sei = hparams(self.hparams, self.hparams_metrics)
        self.tensorboard.file_writer.add_summary(exp)
        self.tensorboard.file_writer.add_summary(ssi)
        self.tensorboard.file_writer.add_summary(sei)
        self.tensorboard.close()

    def set_stage(self, stage_name: str):
        self.calls = 0
        self.stage_name = stage_name

    def write_metrics(self, metrics: dict):
        metrics_numeric = {
            f"{self.stage_name}/{k}": v
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }

        for k, v in metrics_numeric.items():
            self.tensorboard.add_scalar(k, v, self.calls)

        self.calls += 1

    def close(self):
        if self.tensorboard is not None:
            self.tensorboard.close()


class Logger:
    def __init__(
        self,
        args: Config,
        path: str,
        filename: str = "log",
        lyambda: float = 0.2,
        n_pred_dataset_logs: int = 5,
        pred_dataset_iter_weight: float = 0.1,
        task: Literal[
            "classification", "segmentation", "detection", "keypoints"
        ] = "classification",
    ):
        if args is None:
            return

        self.task = task
        self.last_cur_acc = None
        if self.task == "detection":
            # recall; imprecision
            self.worst_acc = "-1.0;2.0"
        else:
            self.worst_acc = -1.0
        self.best_acc = self.worst_acc
        self.active = True
        self.lyambda = lyambda
        self.prev_time = -1.0

        self.validation_frequency = args.common.exp.validation_frequency
        self.normal_epoch_weight = int(1 / pred_dataset_iter_weight)

        self.epochs_left = 0
        self.evolution_time = 0
        for stage_name in args.execute:
            stage_config = args.stages[stage_name]
            if isinstance(
                stage_config, (SupernetLearnStageConfig, FinetuneStageConfig)
            ):
                train_epochs = (
                    stage_config.learn_config.n_epochs
                    + stage_config.learn_config.warmup_epochs
                )
                val_epochs = (train_epochs - 1) // self.validation_frequency + 1
                self.epochs_left += (
                    train_epochs * stage_config.learn_config.dynamic_batch_size
                    + val_epochs * self._get_stage_validation_weight(stage_config)
                ) * self.normal_epoch_weight

            elif isinstance(stage_config, PredictorDatasetStageConfig):
                pred_dataset_size = stage_config.build_acc_dataset.n_subnets
                self.epochs_left += pred_dataset_size
                self.pred_dataset_write_frequency = (
                    1 + pred_dataset_size // n_pred_dataset_logs
                )

            elif isinstance(stage_config, PredictorLearnStageConfig):
                self.evolution_time += stage_config.pred_learn.n_epochs
            elif isinstance(stage_config, EvolutionStageConfig):
                self.evolution_time += (
                    stage_config.evolution.generations
                    * stage_config.evolution.population_size
                    / 100
                )
            elif isinstance(stage_config, ThresholdSearchStageConfig):
                # TODO: нет учёта времени валидации,
                # но зато знаменатель слишком маленький (5)
                self.evolution_time += (
                    stage_config.threshold.num_thresholds
                    * stage_config.dataset.n_classes
                    / 5
                )

        self.metric_names = [
            "cur_acc",
            "best_acc",
            "time",
            "timestamp",
            "component",
            "task",
        ]
        self.metrics = {k: "" for k in self.metric_names}
        self.metrics["component"] = "nas.sh"
        self.metrics["task"] = self.task

        # TODO: Настройка пути в зависимости от ключа на парадигму #79
        self.path = os.path.join("./shared/results", f"{filename}.csv")
        self.path_tmp = os.path.join(path, f"{filename}.pickle")

    def reset_best_acc(self):
        self.best_acc = self.worst_acc

    def log(self, metrics: dict):
        self.stage_cur_epoch += 1
        if self.task == "detection":
            recall = metrics.get("recall/val", "")
            imprecision = metrics.get("imprecision/val", "")
            cur_acc = f"{recall};{imprecision}"
        else:
            cur_acc = metrics.get(f"{self.main_metric_name}/val", "")
        cur_time = metrics.get("epoch_time_consume", None)
        if isinstance(cur_time, timedelta):
            cur_time = cur_time.total_seconds()
        self._update_metrics(cur_acc, cur_time)
        self._write_metrics()

    def update_stage(self, stage_config: CommonConfig):
        if stage_config is None:
            return

        self._read_from_tmp()
        self.stage_type = getattr(stage_config, "stage_type", "")
        self.stage_cur_epoch = 0
        if not isinstance(
            stage_config, (SupernetLearnStageConfig, FinetuneStageConfig)
        ):
            return

        self.stage_dynamic_batch_size = stage_config.learn_config.dynamic_batch_size
        self.stage_train_epochs = (
            stage_config.learn_config.n_epochs + stage_config.learn_config.warmup_epochs
        )
        self.stage_validation_weight = self._get_stage_validation_weight(stage_config)

    def set_main_metric_name(self, main_metric: str):
        self.main_metric_name = main_metric

    def _get_stage_validation_weight(
        self, args: Union[SupernetLearnStageConfig, FinetuneStageConfig]
    ):
        stage_validation_weight = 1
        if not isinstance(args, SupernetLearnStageConfig):
            return stage_validation_weight

        for params_name in [
            "ks_list",
            "depth_list",
            "expand_ratio_list",
            "width_mult_list",
        ]:
            params_list = getattr(args.supernet_config.backbone, params_name, 0)
            if isinstance(params_list, list):
                stage_validation_weight *= min(2, len(params_list))
        return stage_validation_weight

    def _get_best_acc(self, cur_acc=None):
        if self.task == "detection" and cur_acc is not None and cur_acc != ";":
            recall_best, imprecision_best = map(float, self.best_acc.split(";"))
            precision_best = 1 - imprecision_best
            f1_best = (
                2 * recall_best * precision_best / (recall_best + precision_best + 1e-9)
            )
            recall_cur, imprecision_cur = map(float, cur_acc.split(";"))
            precision_cur = 1 - imprecision_cur
            f1_cur = (
                2 * recall_cur * precision_cur / (recall_cur + precision_cur + 1e-9)
            )
            if f1_cur > f1_best:
                recall_best = recall_cur
                imprecision_best = imprecision_cur
                self.best_acc = f"{recall_best};{imprecision_best}"
        elif self.task != "detection" and cur_acc is not None:
            self.best_acc = max(self.best_acc, cur_acc)
        return self.best_acc

    def _get_cur_acc(self, cur_acc=None):
        if cur_acc is not None:
            self.last_cur_acc = cur_acc
        return self.last_cur_acc

    def _running_mean(self, cur_time):
        return self.lyambda * cur_time + (1 - self.lyambda) * self.prev_time

    def _get_cur_epoch_weight(self):
        if self.stage_type == "ArchAccDatasetBuild":
            return 1

        cur_epoch_weight = self.stage_dynamic_batch_size
        if (
            self.stage_cur_epoch % self.validation_frequency == 0
            or self.stage_cur_epoch == self.stage_train_epochs
        ):
            cur_epoch_weight += self.stage_validation_weight
        cur_epoch_weight *= self.normal_epoch_weight
        return cur_epoch_weight

    def _get_time_left(self, cur_time):
        if cur_time is None:
            return 0

        cur_epoch_weight = self._get_cur_epoch_weight()
        if self.prev_time == -1:
            self.prev_time = cur_time / cur_epoch_weight
        self.epochs_left -= cur_epoch_weight
        stage_epoch_time = self._running_mean(cur_time / cur_epoch_weight)
        time_left = stage_epoch_time * self.epochs_left
        if self.stage_type not in ["FineTune", "EvolutionOnSupernet"]:
            time_left += self.evolution_time

        self.prev_time = stage_epoch_time

        return time_left

    def _update_metrics(self, cur_acc, cur_time):
        if self.stage_type == "ArchAccDatasetBuild":
            self.metrics["cur_acc"] = self._get_cur_acc()
            self.metrics["best_acc"] = self._get_best_acc()
        elif cur_acc != "":
            self.metrics["cur_acc"] = self._get_cur_acc(cur_acc)
            self.metrics["best_acc"] = self._get_best_acc(cur_acc)
        else:
            self.metrics["cur_acc"] = cur_acc
        self.metrics["time"] = self._get_time_left(cur_time)
        self.metrics["timestamp"] = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")

    def _round_floats(self):
        def round4(x):
            return round(float(x), 4)

        cur_acc = self.metrics["cur_acc"]
        best_acc = self.metrics["best_acc"]

        if cur_acc is None:
            cur_acc = best_acc
        elif self.task != "detection":
            cur_acc = round4(cur_acc)
            best_acc = round4(best_acc)
        else:
            recall_cur, imprecision_cur = map(round4, cur_acc.split(";"))
            recall_best, imprecision_best = map(round4, self.best_acc.split(";"))
            cur_acc = f"{recall_cur};{imprecision_cur}"
            best_acc = f"{recall_best};{imprecision_best}"

        self.metrics["cur_acc"] = cur_acc
        self.metrics["best_acc"] = best_acc

    def _write_metrics(self):
        if self.stage_type == "ArchAccDatasetBuild" and not (
            self.stage_cur_epoch == 10
            or self.stage_cur_epoch % self.pred_dataset_write_frequency == 0
            or self.epochs_left == 0
        ):
            return

        cur_acc = self.metrics["cur_acc"]
        if (self.task == "detection" and cur_acc == ";") or (
            self.task != "detection" and cur_acc == ""
        ):
            return

        if not os.path.exists(self.path):
            self._create_log_file()
        self._round_floats()
        with open(self.path, mode="a") as fout:
            metric_values = [self.metrics[m_name] for m_name in self.metric_names]
            wrt = csv.writer(fout)
            wrt.writerow(metric_values)
        self._write_to_tmp()

    def _create_log_file(self):
        dir_path, _ = os.path.split(self.path)
        os.makedirs(dir_path, exist_ok=True)

        header = ",".join(self.metric_names)
        with open(self.path, mode="w") as fout:
            fout.write(header + "\n")

    def _read_from_tmp(self):
        if os.path.exists(self.path_tmp):
            with open(self.path_tmp, "rb") as f:
                tmp = pickle.load(f)
                self.epochs_left = tmp["epochs_left"]
                self.last_cur_acc = tmp["last_cur_acc"]
                self.best_acc = tmp["best_acc"]
                self.prev_time = tmp["prev_time"]

    def _write_to_tmp(self):
        with open(self.path_tmp, "wb") as f:
            stats = {
                "epochs_left": self.epochs_left,
                "last_cur_acc": self._get_cur_acc(),
                "best_acc": self._get_best_acc(),
                "prev_time": self.prev_time,
            }
            pickle.dump(stats, f)
        if self.epochs_left == 0:
            if os.path.exists(self.path_tmp):
                os.remove(self.path_tmp)


def write_metrics(logs_path, metrics, prefix=""):
    """write metrics to csv file

    metrics: dict ['metric_name']
    mode: str in ("", train, val, test)
    """

    def dict_key_prior(key: str) -> int:
        if key.lower().startswith("log"):
            return 0
        elif key.endswith("train"):
            return 1
        elif key.endswith("val"):
            return 2
        else:
            return 3

    def format_metric(num, precision=10):
        if isinstance(num, float):
            # disable scientific notation
            return f"{num:.{precision}f}"
        else:
            return str(num)

    if prefix:
        prefix = f"_{prefix}"

    log_file_path = os.path.join(logs_path, f"{prefix}metrics.csv")
    metrics["Logging time"] = datetime.now()
    if not os.path.exists(log_file_path):
        os.makedirs(logs_path, exist_ok=True)
        header = ",".join(sorted(metrics.keys(), key=dict_key_prior))
        with open(log_file_path, mode="w") as fout:
            fout.write(header + "\n")

    with open(log_file_path, mode="a") as fout:
        metric_values = [
            metrics[m_name] for m_name in sorted(metrics.keys(), key=dict_key_prior)
        ]
        log_str = ",".join(map(format_metric, metric_values))
        fout.write(log_str + "\n")


def write_log(logs_path, log_str, prefix="valid", should_print=True, mode="a"):
    """Перегруженная непонятная функция"""
    if not os.path.exists(logs_path):
        os.makedirs(logs_path, exist_ok=True)
    """ prefix: valid, train, test """
    if prefix in ["valid", "test"]:
        with open(os.path.join(logs_path, "valid_console.txt"), mode) as fout:
            fout.write(log_str + "\n")
            fout.flush()
    if prefix in ["valid", "test", "train"]:
        with open(os.path.join(logs_path, "train_console.txt"), mode) as fout:
            if prefix in ["valid", "test"]:
                fout.write("=" * 10)
            fout.write(log_str + "\n")
            fout.flush()
    else:
        with open(os.path.join(logs_path, "%s.txt" % prefix), mode) as fout:
            fout.write(log_str + "\n")
            fout.flush()
    if should_print:
        logger.info(log_str)


def pairwise_accuracy(la, lb, n_samples=200000):
    n = len(la)
    assert n == len(lb)
    total = 0
    count = 0
    for _ in range(n_samples):
        i = np.random.randint(n)
        j = np.random.randint(n)
        while i == j:
            j = np.random.randint(n)
        if la[i] >= la[j] and lb[i] >= lb[j]:
            count += 1
        if la[i] < la[j] and lb[i] < lb[j]:
            count += 1
        total += 1
    return float(count) / total


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# TODO: delete?
class MultiClassAverageMeter:

    """Multi Binary Classification Tasks
    TODO: разобраться с этим"""

    def __init__(self, num_classes, balanced=False, **kwargs):
        super(MultiClassAverageMeter, self).__init__()
        self.num_classes = num_classes
        self.balanced = balanced

        self.counts = []
        for k in range(self.num_classes):
            self.counts.append(np.ndarray((2, 2), dtype=np.float32))

        self.reset()

    def reset(self):
        for k in range(self.num_classes):
            self.counts[k].fill(0)

    def add(self, outputs, targets):
        outputs = outputs.data.cpu().numpy()
        targets = targets.data.cpu().numpy()

        for k in range(self.num_classes):
            output = np.argmax(outputs[:, k, :], axis=1)
            target = targets[:, k]

            x = output + 2 * target
            bincount = np.bincount(x.astype(np.int32), minlength=2**2)

            self.counts[k] += bincount.reshape((2, 2))

    def value(self):
        mean = 0
        for k in range(self.num_classes):
            if self.balanced:
                value = np.mean(
                    (
                        self.counts[k]
                        / np.maximum(np.sum(self.counts[k], axis=1), 1)[:, None]
                    ).diagonal()
                )
            else:
                value = np.sum(self.counts[k].diagonal()) / np.maximum(
                    np.sum(self.counts[k]), 1
                )

            mean += value / self.num_classes * 100.0
        return mean
