import math
import os
from typing import Dict, List, Union, Callable

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torchmetrics import Metric
from torchmetrics import ConfusionMatrix
import json
from loguru import logger
from models.ofa.primitives.static import MyModule
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from dltools.data_providers import DataProvider, DataProvidersRegistry
from dltools.data_providers.classification import CSVAnnotatedProvider
from ofa.training.strategies import (
    BaseStrategy,
    ClassificationStrategy,
    get_strategy_class,
)
from ofa.training.utils import set_running_statistics
from ofa.utils import (
    DEBUG_BATCH_COUNT,
    DistributedMetric,
    StageConfigsUnion,
    build_optimizer,
    calc_learning_rate,
    init_models,
)
from ofa.utils.configs_dataclasses import ClassificationStrategyConfig

from ..training.strategies.base_strategy import BaseContext
from .run_manager import RunManager
from models.ofa.abstract.ofa_abstract import NASModule


__all__ = ["LearnManager"]
MetricDict = Dict[str, Metric]


class LearnManager:
    """Класс хранящий состояния и предоставляющий методы для обучения сети
    Осуществляет поддержку обучения суперсети
    """

    def __init__(
        self,
        net: Union[MyModule, nn.parallel.DistributedDataParallel],
        init=True,
        run_manager: RunManager = None,
    ):
        self.path = run_manager.experiment_path
        self.net: NASModule = net
        self.is_root = run_manager.is_root
        self.best_acc = -1.0
        self.start_epoch = 0
        self.use_amp = run_manager.args.common.exp.use_amp
        scaler_enabled = self.use_amp and run_manager.device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.run_manager = run_manager
        self.args: StageConfigsUnion = run_manager.cur_stage_config
        self.learn_config = self.args.learn_config
        self.dataset_config = self.args.dataset
        self.debug = run_manager.debug
        self.device = run_manager.device
        if self.device == "cpu":
            self.dataset_config.pin_memory = False
        self.teacher_model: Union[nn.Module, None] = None
        self.get_teacher_model: Callable[[LearnManager], Union[None, nn.Module]]

        os.makedirs(self.path, exist_ok=True)

        self.net.to(self.device)
        cudnn.benchmark = True
        if init and self.is_root:
            # Как это потом синхронизируется?
            init_models(self.net)

        if self.learn_config.weight_loss:
            self.build_loss_weights()

        StrategyClass = get_strategy_class(self.args.strategy)
        self.strategy: BaseStrategy = StrategyClass(self.args, self.device)

        # optimizer
        # TODO: Разобраться, что это и почему написанно так.
        if self.learn_config.no_decay_keys:
            # check
            keys = self.learn_config.no_decay_keys.split("#")
            net_params = [
                self.net.module.get_parameters(
                    keys, mode="exclude"
                ),  # parameters with weight decay
                self.net.module.get_parameters(
                    keys, mode="include"
                ),  # parameters without weight decay
            ]
        else:
            try:
                net_params = self.net.weight_parameters()
            except Exception:
                net_params = []
                for param in self.net.parameters():
                    if param.requires_grad:
                        net_params.append(param)
        self.optimizer = self.build_optimizer(net_params)

        # logging objects
        if self.is_root:
            self.logger = self.run_manager.logger
            self.logger.set_main_metric_name(self.strategy.main_metric)
            self.tensorboard = self.run_manager.tensorboard

    def _rescale_lr(self):
        """Увеличение learning rate при распределённом обучении."""

        self.args.learn_config.init_lr *= math.sqrt(self.run_manager._world_size * self.run_manager._local_world_size)

    @property
    def save_path(self) -> str:
        """Возвращает папку для сохранения моделей и чекпоинта"""
        if self.__dict__.get("_save_path", None) is None:
            save_path = os.path.join(self.path, "supernet_weights")
            os.makedirs(save_path, exist_ok=True)
            self.__dict__["_save_path"] = save_path
        return self.__dict__["_save_path"]

    def save_model(
        self, checkpoint=None, is_best=False, model_name=None, model=None
    ) -> None:
        if self.is_root:
            if checkpoint is None:
                checkpoint = {"state_dict": self.net.state_dict()}

            if model_name is None:
                model_name = "checkpoint.pt"

            model_path = os.path.join(self.save_path, model_name)
            torch.save(checkpoint, model_path)
            best_path = os.path.join(self.save_path, "model_best.pt")
            if is_best or not os.path.exists(best_path):
                self.args.exp.supernet_checkpoint_path = best_path
                torch.save({"state_dict": checkpoint["state_dict"]}, best_path)

                teacher_path = os.path.join(self.save_path, "teacher.pt")
                model.set_max_net()
                teacher_model = model.get_active_subnet()
                torch.save({"state_dict": teacher_model.state_dict()}, teacher_path)

    def load_model(self, model_fname=None) -> None:
        if self.is_root:
            try:
                model_fname = os.path.join(self.save_path, "checkpoint.pt")
                logger.info(f"=> loading checkpoint {model_fname}")
                checkpoint = torch.load(model_fname, map_location="cpu")
            except Exception:
                self.run_manager.write_log(
                    f"fail to load checkpoint from {self.save_path}", "valid"
                )
                return

            self.net.load_state_dict(checkpoint["state_dict"])
            if (
                "cur_stage" in checkpoint
                and self.run_manager.cur_stage == checkpoint["cur_stage"]
            ):
                if "epoch" in checkpoint:
                    self.start_epoch = checkpoint["epoch"] + 1
                if "best_acc" in checkpoint:
                    self.best_acc = checkpoint["best_acc"]
                if "optimizer" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])

            params_path = os.path.join(self.save_path, "params.json")
            if os.path.exists(params_path):
                with open(params_path, "r") as f:
                    params = json.load(f)
                self.net.set_params(params)

            self.run_manager.write_log(
                "=> loaded checkpoint '{}'".format(model_fname), "valid"
            )

    def broadcast(self) -> None:
        """Раскидывает по воркерам номер эпохи и лучшую точность

        Эта штука тянется из-за хоровода
        TODO: Refactor/Delete эти вещи можно в state_dict хранить, а не по сети кидать
        """
        self.start_epoch = (
            torch.LongTensor(1).fill_(self.start_epoch).to(self.device)[0]
        )
        self.best_acc = torch.Tensor(1).fill_(self.best_acc).to(self.device)[0]
        dist.broadcast(self.start_epoch, 0)
        dist.broadcast(self.best_acc, 0)

    def prepare_context(self) -> BaseContext:
        """Метод создаёт объект контекста и помещает туда первичные объекты

        модель/инфорацию о моделях
        модель учитель
        """
        context: BaseContext = {}
        context["model"] = self.net

        return context

    def reset_metrics(self, metric_dict: MetricDict) -> None:
        for _, metric in metric_dict.items():
            metric.reset()

    def validate(
        self,
        epoch=0,
        is_test=False,
        run_str="",
        net=None,
        data_loader=None,
        no_logs=False,
        conf_matr=False,
    ):
        """Валидация сети
        # TODO: Refactor Возможно данной функции тут не место
        """
        strat = self.strategy

        if net is None:
            net = self.net
        if data_loader is None:
            if is_test:
                data_loader = self.test_loader
            else:
                data_loader = self.valid_loader

        net.eval()

        # TODO: Refactor Есть привязка к конкретной стратегии
        self.images_logged: bool = False
        if not isinstance(self.strategy, ClassificationStrategy):
            conf_matr = False
        if conf_matr:
            cm = ConfusionMatrix(task="multiclass", num_classes=net.head.n_classes).to(
                self.device
            )

        losses = DistributedMetric("val_loss")

        runtime_metric_dict = strat.build_metrics_dict()
        epoch_metric_dict = strat.build_metrics_dict()
        with torch.no_grad():
            t = tqdm(
                total=len(data_loader),
                desc="Validate Epoch #{} {}".format(epoch + 1, run_str),
                disable=no_logs or not self.is_root,
            )
            for i, data in enumerate(data_loader):
                if self.debug and i >= DEBUG_BATCH_COUNT:
                    break

                context: BaseContext = {}
                context.update(data)
                context["model"] = net
                strat.prepare_batch(context)

                strat.compute_output(context)
                strat.compute_loss(context)
                # TODO Refactor -- это временно,
                # пока loss не будет интегрирован в расчёт метрик
                if isinstance(data_loader, DataLoader):
                    batch_size = data_loader.batch_size
                else:  # isinstance(data["image"], torch.Tensor)
                    batch_size = data["image"].size(0)

                losses.update(context["loss"].detach(), batch_size)
                strat.update_metric(runtime_metric_dict, context)
                strat.update_metric(epoch_metric_dict, context)
                metrics = strat.get_metric_vals(runtime_metric_dict)

                # strategy logs
                if conf_matr:
                    cm.update(preds=context["output"], target=context["target"])
                if self.is_root and not self.images_logged:
                    strat.visualise(
                        context,
                        self.data_provider,
                        [0, 1, 2],
                        self.tensorboard,
                        self.run_manager.cur_stage,
                        epoch,
                    )
                    self.images_logged = True

                t.set_postfix({"loss": losses.avg.item(), **metrics})
                t.update(1)

                self.reset_metrics(runtime_metric_dict)
        if conf_matr:
            return (
                losses.avg.item(),
                strat.get_metric_vals(epoch_metric_dict),
                cm.compute(),
            )

        return losses.avg.item(), strat.get_metric_vals(epoch_metric_dict)

    def reset_running_statistics(
        self, net=None, subset_size=2000, subset_batch_size=None, data_loader=None
    ):
        """Функция для обновления BN статистик сети

        # TODO: Refactor возможно данной функции тут не место
        """
        if subset_batch_size is None:
            subset_batch_size = self.data_provider.test_batch_size
        if net is None:
            net = self.net
        if data_loader is None:
            data_loader = self.random_sub_train_loader(
                subset_size,
                subset_batch_size,
                cache_loader=True,
                world_size=self.run_manager._world_size,
                rank=self.run_manager._rank,
            )
        # TODO: refactor?
        if hasattr(net.head, "disable_postprocess"):
            net.head.disable_postprocess()
        set_running_statistics(net, data_loader, True)
        if hasattr(net.head, "enable_postprocess"):
            net.head.enable_postprocess()

    def adjust_learning_rate(
        self, optimizer: torch.optim.Optimizer, epoch: int, batch=0, nBatch=None
    ) -> float:
        """adjust learning of a given optimizer and return the new learning rate
        # TODO: Refactor это может ломать Adam
        """
        new_lr = calc_learning_rate(
            epoch,
            self.learn_config.init_lr,
            self.learn_config.n_epochs,
            batch,
            nBatch,
            self.learn_config.lr_schedule_type,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def warmup_adjust_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
        T_total,
        nBatch,
        epoch,
        batch=0,
        warmup_lr=0,
    ) -> float:
        """
        # TODO: Refactor это может ломать Adam
        """
        T_cur = epoch * nBatch + batch + 1
        new_lr = T_cur / T_total * (self.learn_config.init_lr - warmup_lr) + warmup_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def random_sub_train_loader(
        self,
        n_images,
        batch_size,
        num_worker=None,
        world_size=None,
        rank=None,
        cache_loader=True,
    ):
        if self.debug:
            n_images = DEBUG_BATCH_COUNT * batch_size

        return self.data_provider.build_sub_train_loader(
            n_images, batch_size, num_worker, world_size, rank, cache_loader
        )

    def build_optimizer(self, net_params):
        self._rescale_lr()
        return build_optimizer(
            net_params,
            self.learn_config.opt_type,
            self.learn_config.momentum,
            self.learn_config.nesterov,
            self.learn_config.init_lr,
            self.learn_config.weight_decay,
            self.learn_config.no_decay_keys,
            self.learn_config.betas,
            self.learn_config.use_sam,
        )

    @property
    def train_loader(self):
        return self.data_provider.train_loader_builder()

    @property
    def valid_loader(self):
        return self.data_provider.valid_loader_builder()

    @property
    def test_loader(self):
        return self.data_provider.test_loader_builder()

    @property
    def data_provider(self) -> DataProvider:
        if self.__dict__.get("_data_provider", None) is None:
            DataProviderClass = DataProvidersRegistry.get_provider_by_name(
                self.dataset_config.type
            )
            self.__dict__["_data_provider"] = DataProviderClass(
                init_config=self.dataset_config,
                rank=self.run_manager._rank,
                world_size=self.run_manager._world_size,
            )
        return self.__dict__["_data_provider"]

    def set_epoch(self, epoch=0) -> None:
        """Устанавливает номер эпохи для внутренних объектов, поддерживающих
        распределённое обучение

        """
        if isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)

    def get_teacher_model(self):
        """Возвращает либо модель учител либо устанавливает суперсеть в режим максимальной подсети
        и вовзращает ссылку на динамическую подсеть

        Не происходит изъятие подсети из суперсети. Поэтому нужно вызывать прямо перед расчётом
        """
        if self.learn_config.teach_choice_strategy == "old":
            return self.teacher_model
        elif self.learn_config.teach_choice_strategy == "big":
            self.net.set_max_net()
            return self.net
        else:  # В теории можно сделать возврат None
            raise ValueError(
                f"wrong value for teach_choice_strategy,\
                              got {self.learn_config.teach_choice_strategy}"
            )

    def sample_active_subnet(self, subnet_idx=0, seed=0) -> None:
        """Генерирует активную подсеть для прохода по ней в пределах батча

        возможно это стоит делать через arch encoder

        проверяет количество подсетей, если оно недостаточно, то скатывается
        к старому варианту семплирования"""

        net_counts = self.learn_config.dynamic_batch_size
        sample_s = self.learn_config.subnet_sample_strategy

        if (
            subnet_idx == 0
            and sample_s in ("sandwich", "big_and_other")
            and net_counts > 1
        ):
            self.net.set_max_net()

        elif subnet_idx == 1 and sample_s == "sandwich" and net_counts > 2:
            self.net.set_min_net()

        else:
            arch_config = self.net.sample_active_subnet()

    def build_loss_weights(self) -> None:
        """Обновляет критерий в стратегии, если она это поддерживает на критерий с весами"""

        if isinstance(self.args.strategy, ClassificationStrategyConfig):
            if isinstance(self.data_provider, CSVAnnotatedProvider):
                w_save_path = os.path.join(self.path, "loss_weights.pth")
                if self.is_root:
                    df = pd.DataFrame(self.data_provider.train_ann)
                    weights = 1 - df[1].value_counts() / len(df)
                    weights = weights.sort_index().values.astype(np.float32)
                    weights.tofile(w_save_path)
                    logger.info(f"computed weights for loss save to {w_save_path}")
                self.args.strategy.weight_for_loss = w_save_path

        dist.barrier()
