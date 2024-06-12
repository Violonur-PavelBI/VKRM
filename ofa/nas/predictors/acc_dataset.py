from __future__ import annotations

import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.utils.data
import torch.distributed as dist
from torch import Tensor
from datetime import datetime
from loguru import logger

from typing import Dict, List, Tuple, TYPE_CHECKING, Union

from dltools.conf.detection import find_best_threshold
from ofa.utils import list_mean, PredictorDatasetStageConfig
from threshold_search import get_metric_cls

if TYPE_CHECKING:
    from ofa.run_manager import LearnManager
    from ..arch_encoders import ArchEncoder
    from .predictors import Predictor
    from models.ofa.networks import CompositeSuperNet

__all__ = ["net_setting2id", "net_id2setting", "AccuracyDataset"]


def net_setting2id(net_setting):
    return json.dumps(net_setting)


def net_id2setting(net_id):
    return json.loads(net_id)


def lat_data_fastest_net(learn_manager: LearnManager) -> Union[None, Dict]:
    """Пытается загрузить датасет для прогноза latency и вернуть наиболее быструю сеть"""
    fastest_net = None
    args = learn_manager.args
    if (
        args.predictors is not None
        and args.predictors.latency_predictor_path is not None
    ):
        p = Path(args.predictors.latency_predictor_path)
        device_name = p.parts[-3]
        net_name = p.parts[-2]
        workdir = Path(os.getcwd()) / "performance_dataset" / device_name / "MB_Latency"
        if learn_manager.is_root:
            logger.info("Поиск сети ведётся по конкретному разрешению")
        i_size = args.dataset.image_size
        pred_file = "_".join(
            (
                net_name,
                str(i_size[0]),
                str(i_size[1]),
                "latency",
            )
        )
        pred_file = (workdir / pred_file).with_suffix(".json")
        if not pred_file.exists():
            return None
        with open(pred_file) as fin:
            lat_dataset: Dict[str, float] = json.load(fin)

        lat_dataset = pd.DataFrame(
            [(k, v) for k, v in lat_dataset.items()], columns=["net", "latency"]
        )
        fastest_net = lat_dataset.min()["net"]
        fastest_net = json.loads(fastest_net)
        del fastest_net["r"]
    return fastest_net


class RegDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        super(RegDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)


class AccuracyDataset:
    def __init__(self, path):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    @property
    def net_id_path(self):
        return os.path.join(self.path, "net_id.dict")

    @property
    def dict_path(self):
        return os.path.join(self.path, "pred.dict")

    def build_dataset(
        self,
        learn_manager: LearnManager,
        ofa_network: CompositeSuperNet,  # AttrPassDDP
    ):
        args: PredictorDatasetStageConfig = learn_manager.args
        pred_dataset = args.build_acc_dataset
        image_size = args.dataset.image_size

        try:
            latency_pred: Predictor = torch.load(
                args.predictors.latency_predictor_path,
                map_location=torch.device(learn_manager.device),
            )
            latency_pred.device = learn_manager.device
        except Exception as e:
            logger.warning(f"Latency predictor cannot be loaded, got {e}.")
            latency_pred = None

        try:
            efficiency_pred: Predictor = torch.load(
                args.predictors.efficiency_predictor_path,
                map_location=torch.device(learn_manager.device),
            )
            efficiency_pred.device = learn_manager.device
        except Exception as e:
            logger.warning(f"Efficiency predictor cannot be loaded, got {e}.")
            efficiency_pred = None

        # make net_id_list, either completely randomly or partly deterministically
        if learn_manager.is_root:
            net_id_list = set()
            os.makedirs(self.path, exist_ok=True)
            fastest_net = lat_data_fastest_net(learn_manager)

            net_count = pred_dataset.n_subnets
            if fastest_net is not None:
                net_count -= 1
            if pred_dataset.det_grid:
                subnets_grid = ofa_network.get_subnets_grid()
                for net_setting in subnets_grid:
                    net_id = net_setting2id(net_setting)
                    net_id_list.add(net_id)

                if len(net_id_list) > net_count:
                    net_id_list = random.sample(net_id_list, k=net_count)
                    net_id_list = set(net_id_list)

            random.seed(learn_manager.args.exp.seed)
            while len(net_id_list) < net_count:
                net_setting = ofa_network.sample_active_subnet()
                net_id = net_setting2id(net_setting)
                net_id_list.add(net_id)
            net_id_list = list(net_id_list)
            net_id_list.sort()
            if fastest_net is not None:
                net_id = net_setting2id(fastest_net)
                net_id_list.insert(0, net_id)

            json.dump(net_id_list, open(self.net_id_path, "w"), indent=4)

        dist.barrier()
        net_id_list = json.load(open(self.net_id_path))

        if learn_manager.is_root:
            time = datetime.now()
            metrics_tensorboard: Dict[str, List] = {}
            metrics_logger = {}

        acc_dict = {}
        with tqdm(
            total=len(net_id_list),
            desc="Building Acc Dataset",
            disable=(not learn_manager.is_root),
        ) as t:
            metric_save_path = os.path.join(self.path, "metric.csv")
            metric_dataframe = pd.DataFrame(
                columns=[
                    "arch",
                    "acc",
                    "latency",
                    "efficiency",
                ]
            )
            # load val dataset into memory
            val_dataset = []
            dist_batch_size = (
                learn_manager.valid_loader.batch_size
                * learn_manager.run_manager._world_size
            )
            iterations = pred_dataset.n_data_samples // dist_batch_size
            i = 0
            for data in learn_manager.valid_loader:
                val_dataset.append(data)
                i += 1
                if i > iterations:
                    break

            # load existing acc dict
            if os.path.isfile(self.dict_path):
                existing_acc_dict = json.load(open(self.dict_path, "r"))
            else:
                existing_acc_dict = {}

            if learn_manager.is_root and args.build_acc_dataset.save_nets:
                save_path = Path(self.path) / "nets"
                save_path.mkdir()

            warn_latency, warn_efficiency = True, True
            for i, net_id in enumerate(net_id_list):
                net_setting = net_id2setting(net_id)
                net_id = net_setting2id({**net_setting, "r": image_size})
                if net_id in existing_acc_dict:
                    acc_dict[net_id] = existing_acc_dict[net_id]
                    t.set_postfix({"acc": acc_dict[net_id], "net_id": net_id})
                    t.update()
                    continue
                ofa_network.set_active_subnet(net_setting)
                learn_manager.reset_running_statistics(ofa_network)
                net_setting_str = ",".join(
                    [
                        "%s_%s"
                        % (
                            key,
                            "%.1f" % list_mean(val) if isinstance(val, list) else val,
                        )
                        for key, val in net_setting.items()
                    ]
                )
                if (
                    learn_manager.run_manager.task in ("detection", "keypoints")
                    and args.build_acc_dataset.threshold_calibrate
                ):
                    metric_cls = get_metric_cls(args)
                    best_thresholds, results, validate_metrics = find_best_threshold(
                        model=ofa_network,
                        dataloader=val_dataset,
                        metric_cls=metric_cls,
                        metrics_dict=args.threshold.goals,
                        threshold_min=args.threshold.min_conf,
                        threshold_max=args.threshold.max_conf,
                        threshold_num=args.threshold.num_thresholds,
                        device=learn_manager.run_manager.device,
                        verbose=False,
                        find_best_nms_iou=False,
                    )
                    # dltools выплёвывает нам метрики на cuda
                    for k, v in validate_metrics.items():
                        validate_metrics[k] = v.item()
                    if learn_manager.strategy.main_metric in validate_metrics:
                        acc = validate_metrics[learn_manager.strategy.main_metric]
                    else:
                        acc = validate_metrics["F1"]

                else:
                    _, validate_metrics = learn_manager.validate(
                        run_str=net_setting_str,
                        net=ofa_network,
                        data_loader=val_dataset,
                        no_logs=True,
                    )
                    acc = validate_metrics[learn_manager.strategy.main_metric]

                arch_dict = {**net_setting, "r": image_size}

                if learn_manager.is_root and args.build_acc_dataset.save_nets:
                    net_save_path = save_path / str(i)
                    net_save_path.mkdir()
                    metadata = {}
                    metadata["supernet_dict"] = net_id
                    with open(net_save_path / "ofa_meta.json", "w") as js_out:
                        json.dump(metadata, js_out)

                    subnet = ofa_network.get_active_subnet()
                    with open(
                        net_save_path / "result_model_config.json", "w"
                    ) as js_out:
                        json.dump(subnet.config, js_out)

                    result_path = net_save_path / "result_model.pt"
                    torch.save(subnet.state_dict(), result_path)

                latency, efficiency = None, None
                if latency_pred is not None:
                    try:
                        latency = latency_pred.predict(arch_dict).item()
                    except Exception as e:
                        if warn_latency:
                            logger.warning(f"Latency cannot be predicted, got {e}.")
                            warn_latency = False
                if efficiency_pred is not None:
                    try:
                        efficiency = efficiency_pred.predict(arch_dict).item()
                    except Exception as e:
                        if warn_efficiency:
                            logger.warning(f"Efficiency cannot be predicted, got {e}.")
                            warn_efficiency = False

                t.set_postfix({"acc": acc, "net_id": net_id})
                t.update()

                metric_dataframe.loc[len(metric_dataframe.index)] = [
                    net_id,
                    acc,
                    latency,
                    efficiency,
                ]
                acc_dict.update({net_id: acc})
                if learn_manager.is_root:
                    json.dump(acc_dict, open(self.dict_path, "w"), indent=4)
                    metric_dataframe.to_csv(metric_save_path, index=False)

                    for k, v in validate_metrics.items():
                        if k not in metrics_tensorboard:
                            metrics_tensorboard[k] = []
                        metrics_tensorboard[k].append(v)
                    for k, v_list in metrics_tensorboard.items():
                        v_mean = np.mean(v_list)
                        v_std = np.std(v_list)
                        validate_metrics[f"{k}_mean"] = v_mean
                        validate_metrics[f"{k}_std"] = v_std

                    learn_manager.run_manager.tensorboard.write_metrics(
                        validate_metrics
                    )

                    metrics_logger["epoch_time_consume"] = datetime.now() - time
                    learn_manager.run_manager.logger.log(metrics_logger)
                    time = datetime.now()

        return acc_dict

    def train_val_split(self, arch_encoder: ArchEncoder, n_train_samples: int = None):
        # load data
        acc_dict = json.load(open(self.dict_path))
        X_all = []
        Y_all = []
        with tqdm(total=len(acc_dict), desc="Loading data") as t:
            for k, v in acc_dict.items():
                dic = json.loads(k)
                dic["r"] = tuple(dic["r"])
                X_all.append(arch_encoder.arch2feature(dic))
                Y_all.append(v)  # range: 0 - 1  / 100.
                t.update()
        base_acc = np.mean(Y_all)

        # convert to torch tensor
        X_all = np.array(X_all, dtype=np.float32)
        Y_all = np.array(Y_all, dtype=np.float32)
        X_all = torch.tensor(X_all, dtype=torch.float)
        Y_all = torch.tensor(Y_all, dtype=torch.float)

        # random shuffle
        shuffle_idx = torch.randperm(len(X_all))
        X_all = X_all[shuffle_idx]
        Y_all = Y_all[shuffle_idx]

        # split data
        if n_train_samples is None:
            n_train_samples = len(X_all) // 5 * 4
        X_train, Y_train = X_all[:n_train_samples], Y_all[:n_train_samples]
        X_val, Y_val = X_all[n_train_samples:], Y_all[n_train_samples:]
        logger.info(f"Train Size: {len(X_train)}. Valid Size: {len(X_val)}.")

        train_data = X_train, Y_train
        val_data = X_val, Y_val
        return train_data, val_data, base_acc

    def build_data_loaders(
        self,
        train_data: Tuple[Tensor, Tensor],
        val_data: Tuple[Tensor, Tensor],
        batch_size: int = 256,
        n_workers: int = 2,
    ):
        X_train, Y_train = train_data
        X_val, Y_val = val_data

        train_dataset = RegDataset(X_train, Y_train)
        val_dataset = RegDataset(X_val, Y_val)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=n_workers,
            persistent_workers=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=n_workers,
            persistent_workers=True,
        )

        return train_loader, valid_loader
