import json
import os
import yaml
from typing import Dict, Union

import torch

from torch.utils.data import DataLoader
from torchmetrics import Metric
from loguru import logger
from ofa.run_manager import RunManager
from ofa.training.utils import set_running_statistics
from ofa.utils.configs_dataclasses import (
    EvolutionParams,
    EvolutionStageConfig,
    ThresholdSearchStageConfig,
    YoloV4DetectionHeadConfig,
    YoloV7KeypointsHeadConfig,
)
from ofa.networks import build_composite_supernet_from_args

from models.ofa.networks import CompositeSubNet
from models.ofa.heads import YoloV4DetectionHead, YoloV7KeypointsHead

from dltools.data_providers import DataProvidersRegistry
from dltools.metrics import ImprecisionRecall, KeypointAccuracy
from dltools.conf.detection import find_best_threshold, check_requirements

from utils import save_interface


def create_model(
    run_manager: RunManager, dataloader: DataLoader, search_on_subnet: bool
) -> Union[CompositeSubNet, None]:
    workdir = run_manager.experiment_path
    if search_on_subnet:
        model_config_path = os.path.join(workdir, "result_model_config.json")
        if not os.path.exists(model_config_path):
            return None

        with open(model_config_path, "r") as f:
            model_config = json.load(f)
        model = CompositeSubNet.build_from_config(model_config)

        state_dict_path = os.path.join(workdir, "result_model.pt")
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)

    else:
        supernet_config = run_manager.args.common.supernet_config
        supernet = build_composite_supernet_from_args(supernet_config)

        latest_path = os.path.join(workdir, "supernet_weights/checkpoint.pt")
        best_path = os.path.join(workdir, "supernet_weights/model_best.pt")
        if os.path.exists(best_path):
            checkpoint_path = best_path
        elif os.path.exists(latest_path):
            checkpoint_path = latest_path
        else:
            checkpoint_path = None
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            supernet.load_state_dict(state_dict)

        supernet.set_max_net()
        set_running_statistics(supernet, dataloader)
        model = supernet.get_active_subnet()

    model.to(run_manager.device)
    model.eval()
    return model


def create_dataloader(args: ThresholdSearchStageConfig) -> DataLoader:
    dataset_config = args.dataset
    DataProviderClass = DataProvidersRegistry.get_provider_by_name(dataset_config.type)
    dataprovider = DataProviderClass(init_config=dataset_config)
    dataloader = dataprovider.test_loader_builder()
    return dataloader


def get_metric_cls(args: ThresholdSearchStageConfig) -> Metric:
    metric_name = args.threshold.metric
    if metric_name == "ImprecisionRecall":
        return ImprecisionRecall
    elif metric_name == "KeypointAccuracy":
        return KeypointAccuracy


def save_thresholds_subnet(
    model: CompositeSubNet,
    thresholds: Dict[str, float],
    run_manager: RunManager,
    args: ThresholdSearchStageConfig,
    evolution: EvolutionParams,
    sat: bool,
):
    # update thresholds in model and save
    head: Union[YoloV4DetectionHead, YoloV7KeypointsHead] = model.head
    head.change_conf_thresholds(thresholds["conf_thresholds"])
    head.change_nms_iou_threshold(thresholds["nms_iou_threshold"])

    workdir = run_manager.experiment_path
    result_path = os.path.join(workdir, "result_model.pt")
    torch.save(model.state_dict(), result_path)

    model_config_path = os.path.join(workdir, "result_model_config.json")
    with open(model_config_path, "w") as f:
        json.dump(model.config, f)

    # update dataclass config
    head_config = args.supernet_config.head
    if isinstance(head_config, YoloV7KeypointsHeadConfig):
        head_config.conf_threshold = thresholds["conf_thresholds"][0]
    elif isinstance(head_config, YoloV4DetectionHeadConfig):
        head_config.conf_thresholds = thresholds["conf_thresholds"]
    head_config.nms_iou_threshold = thresholds["nms_iou_threshold"]

    # update thresholds in interface
    metrics_dump_path = os.path.join(workdir, "metrics.yaml")
    with open(metrics_dump_path, "r") as f:
        interface_metrics = yaml.safe_load(f)

    stop = False
    stop_hpo = False
    if evolution.optimize_val == "latency" and evolution.constraint_type == "accuracy":
        if sat:
            message = "The target metric has been reached. The HPO will not be applied"
            logger.info(message)
            stop_hpo = True

    if evolution.preservation_of_interface != "not":
        save_interface(
            model,
            args.supernet_config,
            interface_metrics,
            preservation_of_interface=evolution.preservation_of_interface,
            experiment_path=workdir,
            stop=stop,
            stop_hpo=stop_hpo,
        )


def save_thresholds_supernet(thresholds: Dict[str, float], workdir: str):
    params_path = os.path.join(workdir, "supernet_weights/params.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
    else:
        params = {}

    params: Dict[str, dict]
    if "head" not in params:
        params["head"] = thresholds
    else:
        params["head"].update(thresholds)

    with open(params_path, "w") as f:
        json.dump(params, f)


def worker(run_manager: RunManager):
    if not run_manager.is_root:
        return

    args: ThresholdSearchStageConfig = run_manager.cur_stage_config

    dataloader = create_dataloader(args)
    model = create_model(run_manager, dataloader, args.threshold.search_on_subnet)
    if model is None:
        logger.warning("Threshold search is skipped due to evolution failure.")
        run_manager.stage_end()
        return
    metric_cls = get_metric_cls(args)

    best_thresholds, results, final_metrics = find_best_threshold(
        model=model,
        dataloader=dataloader,
        metric_cls=metric_cls,
        metrics_dict=args.threshold.goals,
        threshold_min=args.threshold.min_conf,
        threshold_max=args.threshold.max_conf,
        threshold_num=args.threshold.num_thresholds,
        find_best_nms_iou=args.threshold.nms_iou_search,
        double_check=args.threshold.double_check,
        device=run_manager.device,
    )
    sat = check_requirements(final_metrics, args.threshold.goals, {})["all"]

    logger.info(f"Best thresholds: {best_thresholds}")
    final_metrics = {k: v.item() for k, v in final_metrics.items()}
    logger.info(f"Got: {final_metrics}")

    results["best_thresholds"] = best_thresholds
    results["sat"] = sat
    results["metrics"] = final_metrics
    os.makedirs(run_manager.current_stage_path, exist_ok=True)
    results_path = os.path.join(run_manager.current_stage_path, "search.json")
    with open(results_path, "w") as f:
        json.dump(results, f)

    final_metrics_val = {f"{k}/val": v for k, v in final_metrics.items()}

    run_manager.logger.set_main_metric_name(args.strategy.main_metric)
    if args.threshold.search_on_subnet:
        run_manager.logger.reset_best_acc()
    run_manager.write_metrics(final_metrics_val)

    run_metrics = final_metrics
    final_metrics.update(best_thresholds)
    run_manager.update_run_metrics({"threshold_search": run_metrics})

    if args.threshold.search_on_subnet:
        for stage_config in run_manager.args.stages.values():
            if isinstance(stage_config, EvolutionStageConfig):
                evolution = stage_config.evolution
        save_thresholds_subnet(
            model, best_thresholds, run_manager, args, evolution, sat
        )
    else:
        save_thresholds_supernet(best_thresholds, run_manager.experiment_path)

    run_manager.stage_end()
