import copy
import itertools
import json
import os
import random
import time
from datetime import datetime
from typing import Dict
from loguru import logger

import numpy as np
import torch
from tqdm import tqdm

from ofa.networks import build_composite_supernet_from_args
from ofa.training.strategies.base_strategy import BaseContext
from ofa.run_manager import LearnManager
from ofa.training.utils import convert_out
from ofa.utils import (
    AverageMeter,
    list_mean,
    DEBUG_BATCH_COUNT,
    DistributedMetric,
    SupernetLearnStageConfig,
)

from models.ofa.necks import NASFPN
from dltools.utils.train_utils import ConvergenceEarlyStopping

__all__ = [
    "validate",
    "train_one_epoch",
    "train",
    "load_models",
    "train_elastic_depth",
    "train_elastic_expand",
    "train_elastic_width_mult",
]


def validate(
    learn_manager: LearnManager,
    epoch: int = 0,
    is_test: bool = False,
    validate_func_dict: dict = None,
    additional_setting=None,
):
    """Валидация суперсети, путём оценки нескольких подсетей"""

    dynamic_net = learn_manager.net
    dynamic_net.eval()

    validate_func_dict = validate_func_dict or {}

    backbone_dict = validate_func_dict.get("backbone", {})
    neck_dict = validate_func_dict.get("neck", {})

    ks_list = sorted(backbone_dict.get("ks_list", dynamic_net.ks_list))
    expand_ratio_list = sorted(
        backbone_dict.get("expand_ratio_list", dynamic_net.expand_ratio_list)
    )
    depth_list = sorted(backbone_dict.get("depth_list", dynamic_net.depth_list))
    width_mult_list = sorted(
        backbone_dict.get(
            "width_mult_list",
            list(range(len(dynamic_net.width_mult_list)))
            if "width_mult_list" in dynamic_net.__dict__
            else [0],
        )
    )

    backbone_combinations = list(
        itertools.product(ks_list, depth_list, expand_ratio_list, width_mult_list)
    )
    if len(backbone_combinations) > 1:
        backbone_combinations = [backbone_combinations[0], backbone_combinations[-1]]

    neck = dynamic_net.neck
    if isinstance(neck, NASFPN):
        if neck_dict != {}:
            raise NotImplementedError

        w, k, d = map(min, [neck.width_list, neck.kernel_list, neck.depth_list])
        max_depth = max(neck.depth_list)
        mid = w
        depth = [d] * neck.levels
        kernel = [[k] * max_depth] * neck.levels
        width = [[w] * max_depth] * neck.levels
        min_neck = {"mid": mid, "out": {"d": depth, "k": kernel, "w": width}}

        neck_combinations = [(min_neck, w, k, d)]

        if (
            len(neck.width_list) > 1
            or len(neck.kernel_list) > 1
            or len(neck.depth_list) > 1
        ):
            w, k, d = map(max, [neck.width_list, neck.kernel_list, neck.depth_list])
            max_depth = max(neck.depth_list)
            mid = w
            depth = [d] * neck.levels
            kernel = [[k] * max_depth] * neck.levels
            width = [[w] * max_depth] * neck.levels
            max_neck = {"mid": mid, "out": {"d": depth, "k": kernel, "w": width}}

            neck_combinations.append((max_neck, w, k, d))

    else:
        neck_combinations = [(None, None, None, None)]

    subnet_settings = []
    for (k, d, e, w), (neck_arch, neck_w, neck_k, neck_d) in itertools.product(
        backbone_combinations, neck_combinations
    ):
        backbone_arch = {"d": d, "e": e, "ks": k, "w": w}
        arch = {"r": learn_manager.data_provider.image_size, "backbone": backbone_arch}
        arch_desc = "K%s-D%s-E%s-W%s" % (k, d, e, w)
        if neck_arch is not None:
            arch["neck"] = neck_arch
            arch_desc += " K%s-D%s-W%s" % (neck_k, neck_d, neck_w)
        subnet_settings.append([arch, arch_desc])
    if additional_setting is not None:
        subnet_settings += additional_setting

    valid_log = ""
    if is_test:
        data_loader = learn_manager.test_loader
    else:
        data_loader = learn_manager.valid_loader

    validate_metrics = {}
    for setting, name in subnet_settings:
        learn_manager.run_manager.write_log(
            "-" * 30 + " Validate %s " % name + "-" * 30, "train", should_print=False
        )
        dynamic_net.set_active_subnet(setting)

        learn_manager.reset_running_statistics(dynamic_net)

        loss, subnet_metrics = learn_manager.validate(
            epoch=epoch,
            is_test=is_test,
            run_str=name,
            net=dynamic_net,
            data_loader=data_loader,
        )

        main_metric = learn_manager.strategy.main_metric
        valid_log += " %s (%.3f)," % (name, subnet_metrics[main_metric])
        subnet_metrics["Loss"] = loss

        for k, v in subnet_metrics.items():
            if k not in validate_metrics:
                validate_metrics[k] = 0
            validate_metrics[k] += v / len(subnet_settings)

    return validate_metrics, valid_log


def train_one_epoch(learn_manager: LearnManager, epoch):
    dynamic_net = learn_manager.net
    strat = learn_manager.strategy
    learn_config = learn_manager.learn_config

    dynamic_net.train()
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
        context["teacher_model"] = learn_manager.get_teacher_model()
        strat.compute_distil_output(context)

        # TODO: Протестировать установку градиентов в None
        dynamic_net.zero_grad()

        def closure():
            loss_of_subnets = []
            for subnet_idx in range(learn_config.dynamic_batch_size):
                # set random seed before sampling
                # TODO сделать вариант, где у каждого worker своя сеть
                subnet_seed = int("%d%.3d%.3d" % (epoch * nBatch + i, subnet_idx, 0))
                random.seed(subnet_seed)
                learn_manager.sample_active_subnet(subnet_idx, subnet_seed)
                dtype = (
                    torch.float16 if learn_manager.device == "cuda" else torch.bfloat16
                )
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

                # ? Насколько адекватно, что для одного батча считаются метрики на разных подсетях
                # ? и это аггрегируется в одну
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
                    torch.nn.utils.clip_grad_value_(
                        dynamic_net.parameters(), grad_clip_value
                    )
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        dynamic_net.parameters(), grad_clip_norm
                    )
                return context["loss"]

        loss_of_subnets = []
        for subnet_idx in range(learn_config.dynamic_batch_size):
            # set random seed before sampling
            # TODO сделать вариант, где у каждого worker своя сеть
            subnet_seed = int("%d%.3d%.3d" % (epoch * nBatch + i, subnet_idx, 0))
            random.seed(subnet_seed)
            learn_manager.sample_active_subnet(subnet_idx, subnet_seed)
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

            # ? Насколько адекватно, что для одного батча считаются метрики на разных подсетях
            # ? и это аггрегируется в одну
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
                torch.nn.utils.clip_grad_value_(
                    dynamic_net.parameters(), grad_clip_value
                )
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(dynamic_net.parameters(), grad_clip_norm)

            learn_manager.scaler.step(learn_manager.optimizer, closure)
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
    dynamic_net.zero_grad()
    return loss, train_metrics


def train(learn_manager: LearnManager, validate_func=None):
    strat = learn_manager.strategy
    if validate_func is None:
        validate_func = validate

    num_epochs = (
        learn_manager.learn_config.n_epochs + learn_manager.learn_config.warmup_epochs
    )
    if learn_manager.args.learn_config.early_stopping:
        early_stop_manager = ConvergenceEarlyStopping(
            num_epochs=num_epochs,
            save_path=learn_manager.run_manager.experiment_path,
            lyambda=learn_manager.args.learn_config.early_stopping_lyambda,
            pre_stop_epochs=learn_manager.args.learn_config.pre_stop_epochs,
            pipe_width=learn_manager.args.learn_config.pipe_width,
        )

    for epoch in range(learn_manager.start_epoch, num_epochs):
        is_best = False
        validate_metrics = None
        epoch_metrics = {}
        start_epoch_time = datetime.now()
        if learn_manager.is_root:
            logger.info(f"start epoch time is {time.ctime()}")
        train_loss, train_metrics = train_one_epoch(learn_manager, epoch)
        if learn_manager.is_root:
            learn_manager.save_model(
                {
                    "cur_stage": learn_manager.run_manager.cur_stage,
                    "epoch": epoch,
                    "best_acc": learn_manager.best_acc,
                    "optimizer": learn_manager.optimizer.state_dict(),
                    "state_dict": learn_manager.net.state_dict(),
                },
                is_best=False,
                model=learn_manager.net,
            )
            for name, value in train_metrics.items():
                name = f"{name}/train"
                if value is None:
                    value = ""
                epoch_metrics[name] = value

        val_freq = learn_manager.run_manager.args.common.exp.validation_frequency
        if (epoch + 1) % val_freq == 0 or epoch == num_epochs - 1:
            validate_metrics, _val_log = validate_func(
                learn_manager, epoch=epoch, is_test=False
            )
            # best_acc
            main_metric = strat.main_metric
            is_best = validate_metrics[main_metric] > learn_manager.best_acc
            learn_manager.best_acc = max(
                learn_manager.best_acc, validate_metrics[main_metric]
            )
            if learn_manager.is_root:
                val_log = "Valid [{0}/{1}] loss={2:.3f}".format(
                    epoch + 1 - learn_manager.learn_config.warmup_epochs,
                    learn_manager.learn_config.n_epochs,
                    validate_metrics["Loss"],
                )
                for name, value in validate_metrics.items():
                    if value is None:
                        continue
                    val_log += ", {0}={1:.3f}".format(name, value)
                val_log += ", Train loss {loss:.3f}\t".format(
                    loss=train_loss,
                )
                for name, value in train_metrics.items():
                    if value is None:
                        continue
                    val_log += ", {0}={1:.3f}".format(name, value)

                val_log += _val_log
                learn_manager.run_manager.write_log(
                    val_log, "valid", should_print=False
                )
                learn_manager.save_model(
                    {
                        "cur_stage": learn_manager.run_manager.cur_stage,
                        "epoch": epoch,
                        "best_acc": learn_manager.best_acc,
                        "optimizer": learn_manager.optimizer.state_dict(),
                        "state_dict": learn_manager.net.state_dict(),
                    },
                    is_best=is_best,
                    model=learn_manager.net,
                )

            for name, value in validate_metrics.items():
                name = f"{name}/val"
                epoch_metrics[name] = value

            if learn_manager.args.learn_config.early_stopping:
                early_stop_manager.add(validate_metrics[main_metric])
                if early_stop_manager.stop:
                    if learn_manager.is_root:
                        logger.info(
                            f"Early stopping on epoch={epoch+1}/{num_epochs} with {main_metric}={validate_metrics[main_metric]}."
                        )
                    break

        if learn_manager.is_root:
            if validate_metrics is None:
                for metric in train_metrics:
                    metric_name = f"{metric}/val"
                    epoch_metrics[metric_name] = ""

            end_epoch_time = datetime.now()
            epoch_metrics["epoch_time_consume"] = end_epoch_time - start_epoch_time

            learn_manager.run_manager.write_metrics(epoch_metrics)

            if is_best:
                dump_metrics = {"supernet_last_metrics": copy.deepcopy(epoch_metrics)}
                dump_metrics["supernet_last_metrics"][
                    "epoch_time_consume"
                ] = epoch_metrics["epoch_time_consume"].total_seconds()
                learn_manager.run_manager.update_run_metrics(dump_metrics)

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

    if learn_manager.is_root and learn_manager.args.learn_config.early_stopping:
        early_stop_manager.plot()

    if learn_manager.is_root:
        logger.info(f"end train time is {time.ctime()}")


def load_models(learn_manager: LearnManager, dynamic_net, model_path=None):
    # specify init path
    init = torch.load(model_path, map_location="cpu")["state_dict"]
    dynamic_net.load_state_dict(init)
    if learn_manager.is_root:
        learn_manager.run_manager.write_log("Loaded init from %s" % model_path, "valid")

    params_path = os.path.join(learn_manager.save_path, "params.json")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            params = json.load(f)
        dynamic_net.set_params(params)


task2parameter = {
    "kernel": "ks_list",
    "depth": "depth_list",
    "expand": "expand_ratio_list",
}


def prepare_for_elastic_task(learn_manager: LearnManager) -> Dict:
    args: SupernetLearnStageConfig = learn_manager.args
    learn_config = args.learn_config
    backbone_config = args.supernet_config.backbone

    dynamic_net = learn_manager.net

    if learn_config.kd_ratio > 0 and args.exp.teacher_path:
        teacher_model = build_composite_supernet_from_args(args.supernet_config)
        teacher_model.set_max_net()
        teacher_model = teacher_model.get_active_subnet()
        teacher_model.to(learn_manager.device)
        load_models(
            learn_manager,
            teacher_model,
            model_path=args.exp.teacher_path,
        )
        learn_manager.teacher_model = teacher_model

    if learn_manager.run_manager.resume:
        learn_manager.load_model()
        learn_manager.run_manager.logger.stage_cur_epoch += learn_manager.start_epoch
        learn_manager.run_manager.tensorboard.calls += learn_manager.start_epoch

    backbone_dict = {
        "ks_list": sorted(
            {
                min(backbone_config.ks_list),
                max(backbone_config.ks_list),
            }
        ),
        "expand_ratio_list": sorted(
            {
                min(backbone_config.expand_ratio_list),
                max(backbone_config.expand_ratio_list),
            }
        ),
        "depth_list": sorted(
            {
                min(backbone_config.depth_list),
                max(backbone_config.depth_list),
            }
        ),
    }

    parameter_name = task2parameter[learn_config.task]
    parameter_list: list = getattr(backbone_config, parameter_name)

    if learn_config.task != "kernel":
        parameter_list.sort(reverse=True)
        n_stages = len(parameter_list) - 1
        current_stage = n_stages - 1

        if learn_manager.start_epoch == 0 and not learn_manager.run_manager.resume:
            load_models(
                learn_manager,
                dynamic_net,
                model_path=args.exp.supernet_checkpoint_path,
            )
            if learn_config.task == "expand":
                # для task == width тоже, но оно пока не поддерживается
                dynamic_net.re_organize_middle_weights(expand_ratio_stage=current_stage)

        else:
            assert learn_manager.run_manager.resume

        learn_manager.run_manager.write_log(
            "-" * 30
            + "Supporting Elastic %s: %s -> %s"
            % (
                learn_config.task.title(),
                parameter_list[: current_stage + 1],
                parameter_list[: current_stage + 2],
            )
            + "-" * 30,
            "valid",
        )

        other_param_lens = [
            len(getattr(backbone_config, param_name))
            for task, param_name in task2parameter.items()
            if task != learn_config.task
        ]
        # валидация со всеми текущими параметрами, если других параметров мало
        if (np.array(other_param_lens) == 1).all():
            backbone_dict[parameter_name] = sorted(parameter_list)

    validate_func_dict = {"backbone": backbone_dict}

    return validate_func_dict
