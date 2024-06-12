import os
import warnings
from datetime import datetime
from typing import Dict, Literal

from torchmetrics import Metric
import torch
import torch.distributed as dist
import yaml
from loguru import logger

from ofa.utils import (
    Logger,
    TensorBoardLogger,
    write_metrics,
    write_log,
)
from ofa.utils.configs_dataclasses import (
    Config,
    FinetuneStageConfig,
    SupernetLearnStageConfig,
    PredictorDatasetStageConfig,
    PredictorLearnStageConfig,
    EvolutionStageConfig,
)

from dltools.anchors import build_auto_anchors

__all__ = ["RunManager"]


class RunManager:
    """Объект для логирования и контроля стадий

    TODO: Refactor сейчас сделано "ленивое" объявление TensorBoardLogger
    для того, чтобы изолировать стадии

        metrics_dump_path = os.path.join(run_manager.experiment_path, "metrics.yaml")

        self.metrics_dump_path -- путь к yaml файлу в который записываются результаты стадий, эволюции и т.д
    """

    provider_task_mapping = {
        "CSVAnnotated": "classification",
        "Segmentation": "segmentation",
        "DetectionYoloV4": "detection",
        "KeypointsYoloV7": "keypoints",
    }

    def __init__(self, args: Config) -> None:
        self.cur_stage: str = None
        # TODO: Refactor
        self._world_size = int(os.environ["WORLD_SIZE"])
        self._rank = int(os.environ["RANK"])
        self._local_rank = None
        self.device = args.common.exp.device
        if self.device == "cuda":
            self._local_world_size = torch.cuda.device_count()
        else:
            self._local_world_size = args.common.exp.local_world_size_on_cpu
        self.is_root = self._rank == 0

        self.current_stage_path: str = None
        self.args = args
        exp_config = self.args.common.exp
        self.resume = exp_config.resume
        self.debug = exp_config.debug
        if self.debug:
            self._debug_setup()

        # TODO: REFACTOR! Это не в run_manager быть должно!!!
        head_config = args.common.supernet_config.head
        if (
            "anchors" in head_config.dict()
            and "auto_anchors" in head_config.dict()
            and head_config.auto_anchors
        ):
            anchors = build_auto_anchors(args.common.dataset, len(head_config.strides))
            head_config.anchors = anchors
            for stage_config in self.args.stages.values():
                stage_config.supernet_config.head.anchors = anchors
        self.task: Literal["classification", "segmentation", "detection", "keypoints"]
        if args.common.dataset is not None:
            self.task = self.provider_task_mapping[args.common.dataset.type]
        else:
            self.task = "predictors"

        self.experiment_path = exp_config.workdir
        if self.experiment_path is None:
            timestamp = datetime.now().timestamp()
            if dist.is_initialized():
                group = dist.new_group(backend="gloo")
                sync = torch.tensor(timestamp)
                group.broadcast(sync, 0)
                timestamp = sync.item()
                dist.destroy_process_group(group)
            suffix = datetime.fromtimestamp(timestamp)
            suffix = suffix.strftime("%d.%m.%y_%H.%M.%S.%f")

            exp_root_path = exp_config.exp_root_path
            exp_prefix = exp_config.exp_prefix
            self.experiment_path = os.path.join(exp_root_path, exp_prefix, suffix)
            self.args.raw_config["common"]["exp"]["workdir"] = self.experiment_path

        self._setup_clearml()

        # TODO: Refactor должно как-то управляться
        self.metrics_dsts = ("file", "log", "tensorboard")

        if self.is_root:
            self.logger = Logger(self.args, self.experiment_path, task=self.task)
            self.tensorboard = TensorBoardLogger(
                self.args, os.path.join(self.experiment_path, "tensorboard")
            )
            self.metrics_dump_path = os.path.join(self.experiment_path, "metrics.yaml")

    def _debug_setup(self):
        """Переопределяет параметры из конфига для отладки

        Происходит сокращение времени работы стадий"""
        for stage_config in self.args.stages.values():
            if stage_config.threshold is not None:
                stage_config.threshold.min_conf = 0.1
                stage_config.threshold.max_conf = 0.9
                stage_config.threshold.num_thresholds = 2
            if isinstance(stage_config, SupernetLearnStageConfig):
                stage_config.learn_config.n_epochs = 1
                if stage_config.learn_config.warmup_epochs:
                    stage_config.learn_config.warmup_epochs = 1
            elif isinstance(stage_config, PredictorDatasetStageConfig):
                stage_config.build_acc_dataset.n_subnets = 5
            elif isinstance(stage_config, PredictorLearnStageConfig):
                stage_config.pred_learn.n_epochs = 1
            elif isinstance(stage_config, EvolutionStageConfig):
                stage_config.evolution.generations = 10
                stage_config.evolution.population_size = 100
            elif isinstance(stage_config, FinetuneStageConfig):
                stage_config.learn_config.n_epochs = 1
                if stage_config.learn_config.warmup_epochs:
                    stage_config.learn_config.warmup_epochs = 1
            if self.device == "cpu":
                stage_config.dataset.test_batch_size = 2
                stage_config.dataset.train_batch_size = 2

    @property
    def stages(self):
        return self.args.execute

    def set_stage(self, stage_name: str):
        self.cur_stage = stage_name
        self.cur_stage_config = self.args.stages[stage_name]
        self.current_stage_path = os.path.join(self.experiment_path, stage_name)
        if self.is_root:
            self.tensorboard.set_stage(self.cur_stage)
            self.logger.update_stage(self.cur_stage_config)

    def _setup_clearml(self):
        """Пытается создать задачу в ClearML

        Временно в try происходит и импортирование, пока полноценно не перейдём на ClearML.
        """
        exp_config = self.args.common.exp

        self.clearml_project = exp_config.clearml_project
        if self.clearml_project is None:
            return

        self.clearml_task = exp_config.clearml_task
        if self.clearml_task is None:
            prefix = exp_config.exp_prefix
            suffix = os.path.basename(os.path.normpath(self.experiment_path))
            self.clearml_task = f"{prefix}_{suffix}"
            self.args.raw_config["common"]["exp"]["clearml_task"] = self.clearml_task

        try:
            from clearml import Task

            Task.init(project_name=self.clearml_project, task_name=self.clearml_task)
        except Exception as e:
            logger.warning(f"ClearML Task init failed, got\n{e}")

    def write_metrics(self, metrics: Dict[str, Metric], prefix=""):
        self.metrics_dsts
        if "file" in self.metrics_dsts:
            write_metrics(self.current_stage_path, metrics, prefix)
        if "log" in self.metrics_dsts:
            # TODO refactor
            try:
                self.logger.log(metrics)
            except FileNotFoundError as e:
                logger.warning(f"{e} not found paradigma stuff don`t use!")

        if "tensorboard" in self.metrics_dsts:
            self.tensorboard.write_metrics(metrics)

    def update_run_metrics(self, metrics: dict):
        """Работает с метриками, которые характерны для всего запуска, например точность итоговой сети
        Обновляет yaml файл с метриками. Считывает старые метрики и переписывает значения.
        """
        dump_metrics = {}
        if os.path.exists(self.metrics_dump_path):
            with open(self.metrics_dump_path, "r") as fin:
                old_metrics = yaml.safe_load(fin)
                dump_metrics.update(old_metrics)

        dump_metrics.update(metrics)
        with open(self.metrics_dump_path, "w") as f:
            yaml.dump(dump_metrics, f, sort_keys=False)

    def write_log(self, log_str, prefix="valid", should_print=True, mode="a"):
        if self.is_root:
            write_log(self.current_stage_path, log_str, prefix, should_print, mode)

    def stage_end(self):
        if self.is_root:
            self.tensorboard.close()
