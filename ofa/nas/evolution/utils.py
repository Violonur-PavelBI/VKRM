import copy
import csv
import json
import os
from typing import Callable, Dict, Literal, TypedDict, Union

import pandas as pd
import torch
import torch.distributed as dist
from loguru import logger
from models.ofa.networks import CompositeSubNet, CompositeSuperNet
from torch import Tensor

from ofa.nas.arch_encoders import (
    CompositeEncoder,
    CompositeEncoderCatboost,
    build_arch_encoder,
)
from ofa.nas.evolution.search import EvolutionarySearch
from ofa.nas.predictors import Predictor
from ofa.networks import build_composite_supernet_from_args
from ofa.run_manager import LearnManager, RunManager
from ofa.training.progressive_shrinking import load_models
from ofa.utils import AttrPassDDP, EvolutionParams, EvolutionStageConfig
from utils import save_interface


class PredictedValues(TypedDict):
    accuracy: float
    latency: float
    efficiency: float


class SubnetInfo(TypedDict):
    net_config: dict
    metrics: dict
    predicted: PredictedValues


class Params(TypedDict):
    """
    - evolution: Характеристики сети, найденной эволюцией.
    - accdict: Характеристики сети, выбранной из датасета для обучения предиктора качества.
    """

    arch_encoder: Union[CompositeEncoder, CompositeEncoderCatboost]
    predictor_methods: Dict[str, Callable]
    finder: EvolutionarySearch
    evolution: SubnetInfo
    accdict: SubnetInfo
    test_from: str
    test: SubnetInfo
    confusion_matrix: Tensor
    result_subnet: CompositeSubNet
    acceptable_increase: float
    greed_heur_search: bool
    choose_tiny: bool
    evolutuion_success: bool
    acc_dict_success: bool


metrics_list = ["accuracy", "latency", "efficiency"]


def compute_acceptable_increase(mean: float, std: float, magic=True) -> float:
    """Расчёт оценки приращения к качеству сети при использовании дообучения"""
    return (
        (std / 0.00175 + 1 - mean) / 100
        if magic
        else ((1 - (1 - mean) * (1 - std * 3 / mean)) - mean) / 0.45
    )


def evolution_preprocess(run_manager: RunManager) -> Params:
    args: EvolutionStageConfig = run_manager.cur_stage_config
    if isinstance(args.evolution.accuracy_constraint, list):
        if run_manager.is_root:
            logger.warning("Drop constraint to one value F1")
        recall, imprecision = args.evolution.accuracy_constraint
        precision = 1 - imprecision
        f1 = 2 * recall * precision / (recall + precision + 1e-9)
        args.evolution.accuracy_constraint = f1

    evolution_args = args.evolution
    arch_encoder = build_arch_encoder(args)
    predictor_methods = {}
    # TODO: refactor metrics_list
    for metric in metrics_list:
        predictor_path = getattr(args.predictors, f"{metric}_predictor_path", None)
        if predictor_path is None:
            predictor_path = os.path.join(
                run_manager.experiment_path, f"{metric}_predictor.pt"
            )
        try:
            pred: Predictor = torch.load(
                predictor_path, map_location=torch.device(run_manager.device)
            )
            pred.device = run_manager.device
            pred_method = pred.predict
        except Exception as e:
            message = f"{metric.capitalize()} predictor cannot be loaded, got {e}."
            if run_manager.is_root:
                logger.warning(message)
            pred_method = None
            # TODO: FIXME
            if metric != "efficiency":
                if run_manager.is_root:
                    logger.warning("Evolution params drop to accuracy")
                evolution_args.constraint_type = "accuracy"
                evolution_args.optimize_val = "accuracy"
        predictor_methods[metric] = pred_method

    # user threshold correction
    # TODO: add flag for this

    acc_path = os.path.join(run_manager.experiment_path, "pred_dataset/pred.dict")
    if os.path.exists(acc_path):
        with open(acc_path) as fin:
            data = json.load(fin)
        data = pd.DataFrame(
            [(k, v) for k, v in data.items()], columns=["net", "metric"]
        )
    acc_dict_mean = data["metric"].mean()
    acc_dict_std = data["metric"].std()
    acceptable_increase = compute_acceptable_increase(
        acc_dict_mean, acc_dict_std
    ).item()
    greed_heur_search = False
    choose_tiny = False
    evolution_args.evolution_accuracy_constraint = evolution_args.accuracy_constraint
    evolution_args.accdict_accuracy_constraint = evolution_args.accuracy_constraint

    if evolution_args.constraint_type == "accuracy":
        user_constraint = evolution_args.accuracy_constraint

        dataset_max = data["metric"].max()
        if (
            data["metric"].min() > user_constraint
            and evolution_args.optimize_val == "latency"
        ):
            choose_tiny = True
        if dataset_max < user_constraint:
            if run_manager.is_root:
                logger.warning(
                    "Maximum value in pred dataset less than user constraint!"
                )
            heur_threshold = user_constraint - acceptable_increase
            if dataset_max > heur_threshold:  # acc_dict contain netsheur nets
                evolution_args.accdict_accuracy_constraint = dataset_max - 0.00000001
                evolution_args.evolution_accuracy_constraint = (
                    dataset_max * evolution_args.predictor_err_compensation
                )
                if run_manager.is_root:
                    logger.info(
                        "Maximum value in pred dataset more than heuristic constraint!"
                    )
                    logger.warning(
                        f"Accuracy threshold correction for evolution: {user_constraint} -> {dataset_max}"
                    )
                greed_heur_search = True
            else:
                if run_manager.is_root:
                    logger.warning(
                        "Maximum value in pred dataset less than heuristic constraint!"
                    )
                    logger.warning(
                        f"Accuracy threshold correction for evolution: {user_constraint} -> {heur_threshold}"
                    )
                    evolution_args.evolution_accuracy_constraint = (
                        heur_threshold * evolution_args.predictor_err_compensation
                    )

            # TODO WRITE: write_heur_to_metrics
    elif evolution_args.constraint_type not in ["latency", "efficiency"]:
        raise NotImplementedError(
            f"Unknown constraint_type: {evolution_args.constraint_type}"
        )

    finder = EvolutionarySearch(run_manager, arch_encoder, predictor_methods)
    params: Params = {
        "arch_encoder": arch_encoder,
        "predictor_methods": predictor_methods,
        "finder": finder,
        "acceptable_increase": acceptable_increase,
        "greed_heur_search": greed_heur_search,
        "choose_tiny": choose_tiny,
    }
    return params


def gather(params: Params, best_info: SubnetInfo, device):
    # TODO: FIXME

    net_config = best_info["net_config"]
    finder = params["finder"]
    arch_encoder = params["arch_encoder"]
    metrics = finder.all_metrics
    optimize_position = finder.optimize_position - 1
    reverse_sort = finder.reverse_sort

    info = [best_info[m] for m in metrics] + arch_encoder.arch2feature(
        net_config
    ).tolist()
    info = torch.tensor(info).to(device)

    gather_to = [torch.zeros_like(info) for _ in range(int(os.environ["WORLD_SIZE"]))]
    dist.all_gather(gather_to, info)
    if reverse_sort:
        arg = torch.stack(gather_to)[:, optimize_position].argmax().item()
    else:
        arg = torch.stack(gather_to)[:, optimize_position].argmin().item()

    net_config = gather_to[arg][len(metrics) :].cpu()
    net_config = arch_encoder.feature2arch(net_config)
    best_info["net_config"] = net_config
    for i, m in enumerate(metrics):
        best_info[m] = gather_to[arg][i].item()

    return best_info


def get_predicted_values(
    net_info: SubnetInfo, predictor_methods: Dict[str, Callable], verbose=True
):
    net_config = net_info["net_config"]
    for metric in metrics_list:
        if metric in net_info["predicted"]:
            continue
        pred = predictor_methods[metric]
        # TODO: Refactor. Возможна ситуация, когда для сети нет предиктора
        try:
            value = pred([net_config]).item()
        except Exception as e:
            message = f"Cannot predict {metric}, value set to -1. Got {e}."
            if verbose:
                logger.warning(message)
            value = -1

        net_info["predicted"][metric] = value
    return net_info


def note_main_metric(metric: str, main_metric: str):
    if metric == main_metric:
        return f"{metric} (main metric)"
    return metric


def print_summary(
    evolution_args: EvolutionParams, net_info: SubnetInfo, name: str, main_metric: str
):
    constraints = [evolution_args.constraint_type]
    thresholds = {
        constraint: getattr(evolution_args, f"{constraint}_constraint")
        for constraint in constraints
    }
    optimize_val = evolution_args.optimize_val

    signs = {
        "latency": "<=",
        "flops": "<=",
        "accuracy": ">=",
        "efficiency": ">=",
    }
    newline = "\n"
    constraint_conditions = [
        f"{newline}  {metric} {signs[metric]} {thresholds[metric]:.6f}"
        for metric in constraints
    ]
    predict_scores = [
        f"{newline}  {score:.4f} predicted {metric}"
        for metric, score in net_info["predicted"].items()
    ]
    actual_scores = [
        f"{newline}  {score:.4f} {note_main_metric(metric, main_metric)}"
        for metric, score in net_info["metrics"].items()
    ]
    summary = (
        f"{newline}Found best architecture on {name} optimising {optimize_val} under"
        f"{''.join(constraint_conditions)}{newline}constraints."
        f"{newline}It achieves{''.join(predict_scores)}"
        f"{newline}and{''.join(actual_scores)}."
    )
    logger.info(summary)


def validate(
    learn_manager: LearnManager,
    supernet: CompositeSuperNet,
    params: Params,
    name: Literal["evolution", "accdict", "test"],
):
    is_test = name == "test"
    supernet.set_active_subnet(params[name]["net_config"])
    learn_manager.reset_running_statistics(supernet)
    results = learn_manager.validate(is_test=is_test, net=supernet, conf_matr=is_test)
    loss, metrics = results[0], results[1]
    conf_matr = results[2] if len(results) > 2 else None

    params[name]["metrics"].update(metrics)
    params[name] = get_predicted_values(
        params[name], params["predictor_methods"], verbose=learn_manager.is_root
    )
    params["confusion_matrix"] = conf_matr

    if learn_manager.is_root:
        print_summary(
            learn_manager.run_manager.cur_stage_config.evolution,
            params[name],
            name,
            learn_manager.strategy.main_metric,
        )
        metrics.update({"Loss": loss})
        metrics_postfix = {f"{k}/{name}": v for k, v in metrics.items()}
        learn_manager.tensorboard.write_metrics(metrics_postfix)
        if is_test:
            metrics_test = {f"{k}/val": v for k, v in metrics.items()}
            learn_manager.logger.reset_best_acc()
            learn_manager.logger.log(metrics_test)

    return params


def select_from_accdict(run_manager: RunManager, params: Params):
    args: EvolutionParams = run_manager.cur_stage_config.evolution
    acc_dict_path = os.path.join(run_manager.experiment_path, "pred_dataset/metric.csv")
    acc_dict = pd.read_csv(acc_dict_path)

    if args.constraint_type == "accuracy":
        mask = acc_dict["acc"] >= args.accdict_accuracy_constraint
    elif args.constraint_type == "latency":
        mask = acc_dict["latency"] <= args.latency_constraint
    elif args.constraint_type == "efficiency":
        mask = acc_dict["efficiency"] >= args.efficiency_constraint
    acc_dict = acc_dict[mask]

    # TODO: Подумать
    if args.optimize_val == "accuracy":
        argbest = acc_dict["acc"].argmax()
    elif args.optimize_val == "latency":
        if params["greed_heur_search"]:
            argbest = acc_dict["acc"].argmax()
        else:
            argbest = acc_dict["latency"].argmin()
    elif args.optimize_val == "efficiency":
        argbest = acc_dict["efficiency"].argmax()
    if params["choose_tiny"]:
        argbest = 0
    accdict = SubnetInfo(
        net_config=json.loads(acc_dict.iloc[argbest]["arch"]),
        metrics=dict(),
        predicted=dict(),
    )
    params["accdict"] = accdict
    return params


def choose_best(learn_manager: LearnManager, params: Params):
    args: EvolutionParams = learn_manager.run_manager.cur_stage_config.evolution
    metric = learn_manager.strategy.main_metric
    evolution = params["evolution"]
    accdict = params["accdict"]
    if args.optimize_val == "accuracy":
        choose_from_accdict = accdict["metrics"][metric] > evolution["metrics"][metric]
    elif args.optimize_val == "latency":
        choose_from_accdict = (
            accdict["predicted"]["latency"] < evolution["predicted"]["latency"]
        )
        if params["greed_heur_search"]:
            choose_from_accdict &= (
                accdict["metrics"][metric] > evolution["metrics"][metric]
            )
    elif args.optimize_val == "efficiency":
        choose_from_accdict = (
            accdict["predicted"]["efficiency"] > evolution["predicted"]["efficiency"]
        )

    if choose_from_accdict:
        params["test_from"] = "accdict"
        logger.info("pred_dataset net selected")
    else:
        params["test_from"] = "evolution"
        if learn_manager.is_root:
            logger.info("evolution net selected")
    params["test"] = copy.deepcopy(params[params["test_from"]])
    return params


def save_model(experiment_path, net_config, result_subnet, conf_matr):
    net_config_path = os.path.join(experiment_path, "subnet_config.json")
    with open(net_config_path, "w") as jout:
        json.dump(net_config, jout)

    result_path = os.path.join(experiment_path, "result_model.pt")
    torch.save(result_subnet.state_dict(), result_path)
    config_path = os.path.join(experiment_path, "result_model_config.json")
    with open(config_path, "w") as jout:
        json.dump(result_subnet.config, jout)

    if conf_matr is not None:
        df_cm = pd.DataFrame(conf_matr.detach().cpu())
        conf_matr_dump_path = os.path.join(experiment_path, "confusion_matrix.csv")
        df_cm.to_csv(conf_matr_dump_path)


def log_paradigma_fail(run_manager: RunManager):
    task = run_manager.task
    with open("./shared/results/log.csv", mode="a") as fout:
        metric_values = [-1, -1, -1, -1, "nas.sh", task]
        wrt = csv.writer(fout)
        wrt.writerow(metric_values)


def save_evolution_results(learn_manager: LearnManager, params: Params, success=True):
    """Подразумевается, что вызывается от рута! Сохраняет метрики и модель."""
    run_manager = learn_manager.run_manager
    args: EvolutionStageConfig = run_manager.cur_stage_config

    # metrics save
    interface_metrics = {}
    interface_metrics["optimize_val"] = args.evolution.optimize_val
    interface_metrics["acceptable_increase"] = params.pop("acceptable_increase", None)
    interface_metrics["main_metric"] = learn_manager.strategy.main_metric
    interface_metrics["val"] = {
        "evolution": copy.deepcopy(params["evolution"]),
        "accdict": copy.deepcopy(params["accdict"]),
    }
    interface_metrics["better_net"] = params.pop("test_from", None)
    interface_metrics["val"]["evolution"].pop("net_config", None)
    interface_metrics["val"]["accdict"].pop("net_config", None)
    interface_metrics["constraints"] = {
        metric: getattr(args.evolution, f"{metric}_constraint", None)
        for metric in [args.evolution.constraint_type]
    }
    if success:
        interface_metrics["test"] = copy.deepcopy(params["test"])
        interface_metrics["test"].pop("net_config")
    dump_metrics = {"evolution": interface_metrics}
    run_manager.update_run_metrics(dump_metrics)

    if not success:
        if run_manager.is_root:
            log_paradigma_fail(run_manager)
            logger.warning("Seach arch no success")
            save_interface(None, run_manager.args.common.supernet_config, stop=True)
        return

    stop = False
    stop_hpo = False
    test_main_metric = params["test"]["metrics"][learn_manager.strategy.main_metric]
    if (
        args.evolution.optimize_val == "latency"
        and args.evolution.constraint_type == "accuracy"
        and run_manager.task != "detection"
    ):
        if test_main_metric > args.evolution.accuracy_constraint:
            if run_manager.is_root:
                logger.info(
                    "The target metric has been reached. The HPO will not be applied"
                )
            stop_hpo = True
        elif (
            args.evolution.use_es
            and params.get("acceptable_increase", None) is not None
        ):
            if (
                test_main_metric
                < args.evolution.accuracy_constraint - params["acceptable_increase"]
            ):
                if run_manager.is_root:
                    logger.warning(
                        "The target metric is unreachable. The HPO will not be applied"
                    )
                    log_paradigma_fail()
                stop = True

    result_subnet = params["result_subnet"]
    conf_matr = params["confusion_matrix"]
    net_config = params["test"]["net_config"]
    save_model(run_manager.experiment_path, net_config, result_subnet, conf_matr)

    if args.evolution.preservation_of_interface != "not" and run_manager.is_root:
        save_interface(
            result_subnet,
            args.supernet_config,
            interface_metrics,
            preservation_of_interface=args.evolution.preservation_of_interface,
            experiment_path=run_manager.experiment_path,
            stop=stop,
            stop_hpo=stop_hpo,
        )


def create_supernet_and_learn_manager(run_manager: RunManager):
    supernet = build_composite_supernet_from_args(
        run_manager.args.common.supernet_config
    )
    supernet = supernet.to(run_manager.device).eval()
    device_ids = [run_manager._local_rank] if run_manager.device == "cuda" else None
    supernet: CompositeSuperNet = AttrPassDDP(
        supernet, device_ids=device_ids, find_unused_parameters=True
    )

    learn_manager = LearnManager(supernet, run_manager=run_manager)

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
    load_models(learn_manager, supernet, model_path=checkpoint_path)

    return learn_manager, supernet
