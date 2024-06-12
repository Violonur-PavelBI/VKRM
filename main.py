import argparse
import os
import random
import traceback
from contextlib import contextmanager
from datetime import datetime
from types import FunctionType
from typing import List

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from loguru import logger

from build_acc_dataset import worker as db_build_worker
from evolution import worker as evolution_worker
from finetune import worker as finetune_worker
from ofa.run_manager import RunManager
from ofa.utils import (
    Config,
    EvolutionStageConfig,
    ExperimentConfig,
    FinetuneStageConfig,
    PredictorDatasetStageConfig,
    PredictorLearnStageConfig,
    SupernetLearnStageConfig,
    ThresholdSearchStageConfig,
    build_config_from_file,
)
from predictor_learn import worker as pred_learn_worker
from train_ofa_net import worker as ps_worker
from threshold_search import worker as threshold_worker


@contextmanager
def process_group_manager(device):
    """Контекст менеджер для группы распределённого обучения"""
    try:
        if device == "cuda":
            dist.init_process_group("nccl")
        elif device == "cpu":
            dist.init_process_group("gloo")
        yield
    finally:
        dist.destroy_process_group()


def worker_runner(gpu, worker_func, run_manager: RunManager):
    """Функция для запуска стадий написанных под torchrun через mp.spawn

    1) Происходит переопределение переменных, отвечающих за DDP
    2) Для каждой стадии создаётся process_group"""
    run_manager._rank = run_manager._rank * run_manager._local_world_size + gpu
    os.environ["RANK"] = str(run_manager._rank)
    os.environ["LOCAL_RANK"] = str(gpu)
    run_manager._local_rank = gpu
    run_manager.is_root = run_manager._rank == 0
    with process_group_manager(run_manager.device):
        worker_func(run_manager)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path")
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--master-addr", help="ip:port", default="localhost:28500")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    parsed = build_config_from_file(args.config_path)
    parsed.common.exp.resume = args.resume
    parsed.common.config_path = args.config_path
    parsed.common.world_size = args.world_size
    parsed.common.rank = args.rank
    parsed.common.master_addr_full = args.master_addr
    return parsed


def skip_completed_stages(stages: List[str], resume: bool, experiment_path: str):
    timings_path = os.path.join(experiment_path, "timings.yaml")
    if resume and os.path.exists(timings_path):
        with open(timings_path, "r") as f:
            completed_stages: List[dict] = yaml.load(f, Loader=yaml.Loader)
        if "script_start" in completed_stages[-1].keys():
            return []
        while len(completed_stages) > 0 and completed_stages[0]["stage"] == stages[0]:
            completed_stages.pop(0)
            stages.pop(0)
    return stages


def ensure_reproducibility(args: ExperimentConfig):
    """
    Потенциально эта штука не обеспечивает полную воспроизводимость.
    """

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True


def get_stage_func(stage_config_cls) -> FunctionType:
    stagetype_stagefunc = {
        SupernetLearnStageConfig: ps_worker,
        PredictorDatasetStageConfig: db_build_worker,
        PredictorLearnStageConfig: pred_learn_worker,
        EvolutionStageConfig: evolution_worker,
        FinetuneStageConfig: finetune_worker,
        ThresholdSearchStageConfig: threshold_worker,
    }

    return stagetype_stagefunc[stage_config_cls]


def main(args: Config):
    ensure_reproducibility(args.common.exp)
    cv2.setNumThreads(0)
    os.environ["OMP_NUM_THREADS"] = "1"
    if args.common.exp.dont_use_env:
        os.environ["RANK"] = str(args.common.rank)  # Node rank
        os.environ["LOCAL_RANK"] = str(args.common.rank)
        # считается, что каждый экземпляр скрипта видит одинаковое количество карт
        os.environ["WORLD_SIZE"] = str(args.common.world_size)
        if args.common.exp.device == "cuda":
            local_world_size_temp = torch.cuda.device_count()
        else:
            local_world_size_temp = args.common.exp.local_world_size_on_cpu
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size_temp)
        master_arrd = args.common.master_addr_full.split(":")
        os.environ["MASTER_ADDR"] = master_arrd[0]
        os.environ["MASTER_PORT"] = master_arrd[1]
    else:
        args.common.rank = int(os.environ["RANK"])
        args.common.world_size = int(os.environ["WORLD_SIZE"])

    if args.common.exp.device == "cuda":
        args.common.ngpus_per_node = torch.cuda.device_count()
    else:
        args.common.ngpus_per_node = args.common.exp.local_world_size_on_cpu

    args.common.world_size = args.common.ngpus_per_node * args.common.world_size
    os.environ["WORLD_SIZE"] = str(args.common.world_size)

    run_manager = RunManager(args)
    if run_manager.is_root:
        os.makedirs(run_manager.experiment_path, exist_ok=True)
        config_dump_path = os.path.join(run_manager.experiment_path, "config.yaml")
        run_manager.log_dump_path = os.path.join(run_manager.experiment_path, "log.log")
        logger.add(run_manager.log_dump_path, diagnose=True)
        logger.info(os.environ["MASTER_PORT"])
        logger.info(os.environ["WORLD_SIZE"])
        logger.info(os.environ["MASTER_ADDR"])
        logger.info(os.environ["RANK"])

        with open(config_dump_path, "w") as dump_config_stream:
            yaml.dump(run_manager.args.raw_config, dump_config_stream, sort_keys=False)

        global_start = datetime.now()

    stages = skip_completed_stages(
        run_manager.stages, args.common.exp.resume, run_manager.experiment_path
    )
    for stage_name in stages:
        ensure_reproducibility(args.common.exp)
        start_stage = datetime.now()
        run_manager.set_stage(stage_name)
        stage_config = run_manager.cur_stage_config
        stage_func = get_stage_func(stage_config.__class__)
        try:
            if isinstance(stage_config, ThresholdSearchStageConfig):
                threshold_worker(run_manager)
            elif run_manager._local_world_size > 1:
                mp.spawn(
                    worker_runner,
                    args=(
                        stage_func,
                        run_manager,
                    ),
                    nprocs=run_manager._local_world_size,
                    join=True,
                )
            else:
                worker_runner(0, stage_func, run_manager)
        except Exception as e:
            err_dump_path = os.path.join(run_manager.experiment_path, "err.log")
            with open(err_dump_path, "a") as err_f:
                print(f"{stage_name}", file=err_f)
                print(traceback.format_exc(), file=err_f)
                print("==" * 30, file=err_f)
                logger.exception(f"{stage_name}\n{traceback.format_exc()}")
            raise e

        if run_manager.is_root:
            end_stage = datetime.now()
            stage_consume = end_stage - start_stage
            time_dic = {
                "stage": stage_name,
                "stage_type": stage_config.stage_type,
                "start": str(start_stage),
                "end": str(end_stage),
                "time_consume": str(stage_consume),
            }
            timing_dump_path = os.path.join(run_manager.experiment_path, "timings.yaml")
            with open(timing_dump_path, "a") as dump_timings_stream:
                yaml.dump([time_dic], dump_timings_stream, sort_keys=False)

    if run_manager.is_root:
        global_end = datetime.now()
        time_consume = global_end - global_start
        time_dic = {
            "script_start": str(global_start),
            "script_end": str(global_end),
            "script_time_consume": str(time_consume),
        }
        timing_dump_path = os.path.join(run_manager.experiment_path, "timings.yaml")
        with open(timing_dump_path, "a") as dump_timings_stream:
            yaml.dump([time_dic], dump_timings_stream, sort_keys=False)

        run_manager.tensorboard.stage_name = "time"
        run_manager.tensorboard.write_metrics({"total": time_consume.total_seconds()})


if __name__ == "__main__":
    args = get_args()
    main(args)
