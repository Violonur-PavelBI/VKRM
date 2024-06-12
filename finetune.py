import copy
import json
import os
import time
from datetime import datetime
from typing import Tuple, Union

import torch
import torch.distributed as dist
import yaml
from dltools.utils.train_utils import ConvergenceEarlyStopping
from torchmetrics import ConfusionMatrix
from loguru import logger
from models.ofa.networks import CompositeSubNet
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ofa.run_manager import LearnManager, RunManager
from ofa.training.strategies.base_strategy import BaseContext
from ofa.training.strategies.classification import ClassificationStrategy
from ofa.training.utils import convert_out
from ofa.utils import DEBUG_BATCH_COUNT, AttrPassDDP, AverageMeter, DistributedMetric
from ofa.utils.common_tools import list_mean
from utils import save_interface


def save_results(learn_manager: LearnManager):
    logger.info("Save model")
    best_state_dict_path = os.path.join(
        learn_manager.run_manager.current_stage_path, "checkpoint", "best.pt"
    )
    best_state_dict = torch.load(best_state_dict_path, map_location="cpu")
    learn_manager.net.module.load_state_dict(best_state_dict)

    metrics_dump_path = os.path.join(
        learn_manager.run_manager.experiment_path, "metrics.yaml"
    )
    with open(metrics_dump_path, "r") as f:
        interface_metrics = yaml.safe_load(f)

    result_path = os.path.join(
        learn_manager.run_manager.experiment_path, "result_model.pt"
    )
    torch.save(learn_manager.net.state_dict(), result_path)

    save_interface(
        learn_manager.net,
        learn_manager.args.supernet_config,
        interface_metrics,
        experiment_path=learn_manager.run_manager.experiment_path,
    )


def train_one_epoch(learn_manager: LearnManager, epoch) -> Tuple[torch.Tensor, dict]:
    net = learn_manager.net
    strat = learn_manager.strategy
    learn_config = learn_manager.learn_config

    net.train()
    learn_manager.set_epoch(epoch)

    nBatch = len(learn_manager.train_loader)

    data_time = AverageMeter()

    losses = DistributedMetric("train_loss")

    runtime_metric_dict = strat.build_metrics_dict()
    epoch_metric_dict = strat.build_metrics_dict()
    t = tqdm(
        total=nBatch,
        desc="Train Epoch #{}".format(epoch + 1),
        disable=not learn_manager.is_root,
    )
    end = time.time()
    for i, data in enumerate(learn_manager.train_loader):
        if learn_manager.debug and i >= DEBUG_BATCH_COUNT:
            break

        context: BaseContext = learn_manager.prepare_context()
        context.update(data)
        data_time.update(time.time() - end)
        if epoch < learn_config.warmup_epochs:
            new_lr = learn_manager.warmup_adjust_learning_rate(
                learn_manager.optimizer,
                learn_config.warmup_epochs * nBatch,
                nBatch,
                epoch,
                i,
                learn_config.warmup_lr,
            )
        else:
            new_lr = learn_manager.adjust_learning_rate(
                learn_manager.optimizer, epoch - learn_config.warmup_epochs, i, nBatch
            )
        strat.prepare_batch(context)

        # TODO: Протестировать установку градиентов в None
        net.zero_grad()

        loss_of_subnets = []
        for subnet_idx in range(learn_config.dynamic_batch_size):
            dtype = torch.float16 if learn_manager.device == "cuda" else torch.bfloat16
            with torch.autocast(
                device_type=learn_manager.device,
                dtype=dtype,
                enabled=learn_manager.use_amp,
            ):
                strat.compute_output(context)
                strat.compute_loss(context)

            if learn_config.iters_to_accumulate:
                context["loss"] /= learn_config.iters_to_accumulate

            learn_manager.scaler.scale(context["loss"]).backward()
            loss_of_subnets.append(context["loss"].detach())

            convert_out(context["output"])

            strat.update_metric(runtime_metric_dict, context)
            strat.update_metric(epoch_metric_dict, context)

        if (
            (learn_config.iters_to_accumulate is None)
            or ((i + 1) % learn_config.iters_to_accumulate == 0)
            or ((i + 1) == len(learn_manager.train_loader))
        ):
            grad_clip_value = learn_manager.learn_config.grad_clip_value
            grad_clip_norm = learn_manager.learn_config.grad_clip_norm

            if grad_clip_value is not None or grad_clip_norm is not None:
                learn_manager.scaler.unscale_(learn_manager.optimizer)
                if grad_clip_value is not None:
                    torch.nn.utils.clip_grad_value_(net.parameters(), grad_clip_value)
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_norm)
            learn_manager.scaler.step(learn_manager.optimizer)
            learn_manager.scaler.update()

        losses.update(list_mean(loss_of_subnets), learn_manager.train_loader.batch_size)
        metrics = strat.get_metric_vals(runtime_metric_dict)

        # TODO: Refactor -- тут часть полей не нужна, потому что не влезают все
        t.set_postfix(
            {
                "loss": losses.avg.item(),  # nan (`clown face`)
                **metrics,  # Возможно стоит исключить часть, типа сделать несколько
                "lr": new_lr,
            }
        )
        t.update(1)
        learn_manager.reset_metrics(runtime_metric_dict)

        end = time.time()
    loss, train_metrics = losses.avg.item(), strat.get_metric_vals(epoch_metric_dict)
    train_metrics["Loss"] = loss
    net.zero_grad()
    return loss, train_metrics


def validate(
    learn_manager: LearnManager,
    epoch: int,
    is_test: bool = False,
    conf_matr: bool = False,
) -> dict:
    strat = learn_manager.strategy
    model = learn_manager.net
    model.eval()

    dataloader = learn_manager.test_loader if is_test else learn_manager.valid_loader
    losses = DistributedMetric("test_loss" if is_test else "val_loss")

    runtime_metric_dict = strat.build_metrics_dict()
    epoch_metric_dict = strat.build_metrics_dict()
    n_epochs = learn_manager.learn_config.n_epochs
    if not isinstance(strat, ClassificationStrategy):
        conf_matr = False
    if conf_matr:
        cm = ConfusionMatrix(task="multiclass", num_classes=model.head.n_classes).to(
            learn_manager.device
        )

    with torch.no_grad():
        t = tqdm(
            total=len(dataloader),
            desc="Validate Epoch #{} {}".format(epoch + 1, n_epochs),
            disable=not learn_manager.is_root,
        )
        learn_manager.images_logged: bool = False
        for i, data in enumerate(dataloader):
            if learn_manager.debug and i >= DEBUG_BATCH_COUNT:
                break

            context: BaseContext = {}
            context.update(data)
            context["model"] = model
            strat.prepare_batch(context)

            strat.compute_output(context)
            strat.compute_loss(context)
            # TODO Refactor -- это временно,
            # пока loss не будет интегрирован в расчёт метрик
            if isinstance(dataloader, DataLoader):
                batch_size = dataloader.batch_size
            else:  # isinstance(data["image"], torch.Tensor)
                batch_size = data["image"].size(0)

            losses.update(context["loss"].detach(), batch_size)
            strat.update_metric(runtime_metric_dict, context)
            strat.update_metric(epoch_metric_dict, context)
            metrics = strat.get_metric_vals(runtime_metric_dict)

            # strategy logs
            if conf_matr:
                cm.update(preds=context["output"], target=context["target"])
            if learn_manager.is_root and not learn_manager.images_logged:
                strat.visualise(
                    context,
                    learn_manager.data_provider,
                    [0, 1, 2],
                    learn_manager.tensorboard,
                    learn_manager.run_manager.cur_stage,
                    epoch,
                )
                learn_manager.images_logged = True

            t.set_postfix({"loss": losses.avg.item(), **metrics})
            t.update(1)

            learn_manager.reset_metrics(runtime_metric_dict)
    if conf_matr:
        return (
            losses.avg.item(),
            strat.get_metric_vals(epoch_metric_dict),
            cm.compute(),
        )
    loss = losses.avg.item()
    val_metrics = strat.get_metric_vals(epoch_metric_dict)
    val_metrics["Loss"] = loss
    return loss, val_metrics


def create_model(run_manager: RunManager) -> Union[CompositeSubNet, None]:
    model_config_path = os.path.join(
        run_manager.experiment_path, "result_model_config.json"
    )
    if not os.path.exists(model_config_path):
        return None

    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    model = CompositeSubNet.build_from_config(model_config)

    model_state_dict_path = os.path.join(run_manager.experiment_path, "result_model.pt")
    model_state_dict = torch.load(model_state_dict_path, map_location="cpu")
    model.load_state_dict(model_state_dict)
    return model.to(run_manager.device)


def worker(run_manager: RunManager):
    if run_manager.device == "cuda":
        torch.cuda.set_device(run_manager._local_rank)

    run_manager.cur_stage_config.learn_config.kd_ratio = 0
    if run_manager.is_root:
        logger.info("kd_ratio set to 0. We can't do distillation here.")

    model = create_model(run_manager)
    if model is None:
        if run_manager.is_root:
            logger.warning("Finetune is skipped due to evolution failure.")
        dist.barrier()
        run_manager.stage_end()
        return

    device_ids = [run_manager._local_rank] if run_manager.device == "cuda" else None
    model = AttrPassDDP(model, device_ids=device_ids, find_unused_parameters=True)
    learn_manager = LearnManager(net=model, init=False, run_manager=run_manager)
    strat = learn_manager.strategy
    learn_config = learn_manager.learn_config

    num_epochs = learn_config.n_epochs + learn_config.warmup_epochs
    args = run_manager.cur_stage_config

    if learn_manager.args.learn_config.early_stopping:
        early_stop_manager = ConvergenceEarlyStopping(
            num_epochs=num_epochs,
            save_path=learn_manager.run_manager.experiment_path,
            lyambda=learn_manager.args.learn_config.early_stopping_lyambda,
            pre_stop_epochs=learn_manager.args.learn_config.pre_stop_epochs,
            pipe_width=learn_manager.args.learn_config.pipe_width,
        )

    # zero epoch validate, best_metric_init
    val_loss, validate_metrics = validate(
        learn_manager, 0, is_test=args.is_test, conf_matr=False
    )
    main_metric = strat.main_metric
    learn_manager.best_acc = validate_metrics[main_metric]
    if learn_manager.is_root:
        save_dir_path = os.path.join(run_manager.current_stage_path, "checkpoint")
        os.makedirs(save_dir_path, exist_ok=True)
        latest_path = os.path.join(save_dir_path, "latest.pt")
        torch.save(model.state_dict(), latest_path)

        model_path = os.path.join(
            run_manager.current_stage_path, "checkpoint", "best.pt"
        )
        torch.save(model.state_dict(), model_path)

        val_log = f"zero eval 'Loss'\t : {validate_metrics['Loss']} \t"
        val_log += f"\t{strat.main_metric}: {validate_metrics[strat.main_metric]}"
        logger.info(val_log)

        epoch_metrics = {}
        for metric in validate_metrics:
            metric_name = f"{metric}/val"
            epoch_metrics[metric_name] = validate_metrics[metric]
        dump_metrics = {"finetune": epoch_metrics}
        learn_manager.run_manager.update_run_metrics(dump_metrics)

    for epoch in range(num_epochs):
        is_best = False
        start_epoch_time = datetime.now()
        epoch_metrics = {}
        # train one epoch
        validate_metrics = None
        train_loss, train_metrics = train_one_epoch(learn_manager, epoch)
        if learn_manager.is_root:
            for name, value in train_metrics.items():
                name = f"{name}/train"
                if value is None:
                    value = ""
                epoch_metrics[name] = value

        # validate
        val_freq = learn_manager.run_manager.args.common.exp.validation_frequency
        if (epoch + 1) % val_freq == 0 or epoch == num_epochs - 1:
            val_loss, validate_metrics = validate(
                learn_manager, epoch, is_test=args.is_test, conf_matr=False
            )
            is_best = validate_metrics[main_metric] > learn_manager.best_acc
            learn_manager.best_acc = max(
                learn_manager.best_acc, validate_metrics[main_metric]
            )

            if learn_manager.is_root:
                for metric in validate_metrics:
                    metric_name = f"{metric}/val"
                    epoch_metrics[metric_name] = validate_metrics[metric]

                save_dir_path = os.path.join(
                    run_manager.current_stage_path, "checkpoint"
                )
                os.makedirs(save_dir_path, exist_ok=True)
                latest_path = os.path.join(save_dir_path, "latest.pt")
                torch.save(model.state_dict(), latest_path)

        if learn_manager.is_root:
            if validate_metrics is None:
                for metric in train_metrics:
                    metric_name = f"{metric}/val"
                    epoch_metrics[metric_name] = ""

            end_epoch_time = datetime.now()
            epoch_metrics["epoch_time_consume"] = end_epoch_time - start_epoch_time

            if epoch == num_epochs - 1:
                learn_manager.run_manager.logger.reset_best_acc()
            learn_manager.run_manager.write_metrics(epoch_metrics)

            if is_best:
                dump_metrics = {"finetune": copy.deepcopy(epoch_metrics)}
                dump_metrics["finetune"]["epoch_time_consume"] = epoch_metrics[
                    "epoch_time_consume"
                ].total_seconds()
                learn_manager.run_manager.update_run_metrics(dump_metrics)
                model_path = os.path.join(
                    run_manager.current_stage_path, "checkpoint", "best.pt"
                )
                torch.save(model.state_dict(), model_path)

            logger.info(f"epoch time consume {end_epoch_time - start_epoch_time}")
            # Подразумевается, что loss есть всегда
            train_log_loss = "Loss/train"
            val_log_loss = "Loss/val"
            epoch_metrics_log = f"{train_log_loss}: {epoch_metrics[train_log_loss]} \t {val_log_loss}: {epoch_metrics[val_log_loss]} \t"
            train_log_metric = f"{strat.main_metric}/train"
            val_log_metric = f"{strat.main_metric}/val"
            if train_log_metric in epoch_metrics:
                epoch_metrics_log += (
                    f"\t{train_log_metric}: {epoch_metrics[train_log_metric]}"
                )
            if val_log_metric in epoch_metrics:
                epoch_metrics_log += (
                    f"\t{val_log_metric}: {epoch_metrics[val_log_metric]}"
                )
            logger.info(epoch_metrics_log)

        if (
            learn_manager.args.learn_config.early_stopping
            and validate_metrics is not None
        ):
            early_stop_manager.add(validate_metrics[main_metric])
            if early_stop_manager.stop:
                if learn_manager.is_root:
                    logger.info(
                        f"Early stopping on epoch={epoch+1}/{num_epochs} with {main_metric}={validate_metrics[main_metric]}."
                    )
                break

    if learn_manager.is_root:
        save_results(learn_manager)

    dist.barrier()
    run_manager.stage_end()
