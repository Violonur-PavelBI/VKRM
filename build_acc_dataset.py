import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml

from ofa.networks import build_composite_supernet_from_args
from ofa.training.progressive_shrinking import load_models
from ofa.run_manager import LearnManager, RunManager
from ofa.nas.predictors import AccuracyDataset
from ofa.utils import PredictorDatasetStageConfig, AttrPassDDP


def worker(run_manager: RunManager):
    if run_manager.device == "cuda":
        torch.cuda.set_device(run_manager._local_rank)
    args: PredictorDatasetStageConfig = run_manager.cur_stage_config

    latest_path = os.path.join(
        run_manager.experiment_path, "supernet_weights/checkpoint.pt"
    )
    best_path = os.path.join(
        run_manager.experiment_path, "supernet_weights/model_best.pt"
    )
    if os.path.exists(best_path):
        checkpoint_path = best_path
    elif os.path.exists(latest_path):
        checkpoint_path = latest_path
    else:
        checkpoint_path = None
    args.exp.supernet_checkpoint_path = checkpoint_path

    net = build_composite_supernet_from_args(args.supernet_config)
    if run_manager.device == "cuda":
        net.to(run_manager.device)
    net.eval()
    device_ids = [run_manager._local_rank] if run_manager.device == "cuda" else None
    net = AttrPassDDP(net, device_ids=device_ids, find_unused_parameters=True)
    learn_manager = LearnManager(net, run_manager=run_manager)

    load_models(
        learn_manager,
        net,
        model_path=args.exp.supernet_checkpoint_path,
    )

    acc_path = os.path.join(run_manager.experiment_path, "pred_dataset")
    acc_dataset = AccuracyDataset(acc_path)
    # TODO: есть ключ subset_size, его нужно явно из конфига передать
    # добавить в конфиг поле с количеством картинок для одной оценки подсети
    acc_dict = acc_dataset.build_dataset(learn_manager, net)

    dist.barrier()
    if run_manager.is_root:
        # histogram
        acc_list = list(acc_dict.values())
        mean, std = np.mean(acc_list), np.std(acc_list)
        fig, ax = plt.subplots(figsize=(8, 5))
        bin_values, _, _ = ax.hist(acc_list, bins=100, color="blue")
        max_bin = np.max(bin_values)
        ax.axvline(x=mean, c="red")
        ax.axvline(x=mean - 3 * std, c="red", linestyle="dotted")
        ax.axvline(x=mean + 3 * std, c="red", linestyle="dotted")
        ax.text(x=mean + 0.25 * std, y=max_bin, s="μ", c="red")
        ax.text(x=mean - 2.75 * std, y=max_bin, s="μ-3σ", c="red")
        ax.text(x=mean + 2.25 * std, y=max_bin, s="μ+3σ", c="red")
        ax.set_title(f"mean = {mean:.6f}    std = {std:.6f}")

        dump_metrics = {"pred_dataset": {"mean": mean.item(), "std": std.item()}}
        run_manager.update_run_metrics(dump_metrics)

        learn_manager.tensorboard.tensorboard.add_figure(
            f"{run_manager.cur_stage}/histogram_{learn_manager.strategy.main_metric}",
            fig,
        )

    run_manager.stage_end()
