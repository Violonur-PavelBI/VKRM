import copy
import os

import torch
import torch.distributed as dist
from loguru import logger

from ofa.nas.evolution.utils import (
    SubnetInfo,
    evolution_preprocess,
    gather,
    create_supernet_and_learn_manager,
    validate,
    select_from_accdict,
    choose_best,
    save_evolution_results,
)
from ofa.nas.evolution.search import *
from ofa.run_manager import RunManager
from ofa.utils import EvolutionStageConfig


def worker(run_manager: RunManager):
    if run_manager.is_root:
        logger.add(run_manager.log_dump_path, diagnose=True)
    if run_manager.device == "cuda":
        torch.cuda.set_device(run_manager._local_rank)
    args: EvolutionStageConfig = run_manager.cur_stage_config
    do_evolution_search = (
        args.evolution.evolution_type == "distributed" or run_manager.is_root
    )

    params = evolution_preprocess(run_manager)
    finder = params["finder"]

    # start searching
    params["evolutuion_success"] = True
    net_info = SubnetInfo()
    try:
        if do_evolution_search:
            net_info = finder.run_evolution_search()
            params["finder"] = finder
    except SampleTimeOut as e:
        logger.info(e)
        params["evolutuion_success"] = False

    # TODO: distributed эволюцию нужно отрефакторить и протестить
    if args.evolution.evolution_type == "distributed":
        net_info = gather(params, net_info, run_manager.device)
    if run_manager.is_root:
        params["evolution"] = net_info

    if run_manager.is_root:
        logger.info("Run search in pred_dataset.")
        params["acc_dict_success"] = True
        try:
            params = select_from_accdict(run_manager, params)
        except Exception as e:
            params["acc_dict_success"] = False
            logger.warning(e)
            logger.warning("Could not load or process net from pred_dataset. ")
            params["accdict"] = SubnetInfo()

    if dist.is_initialized() and run_manager._world_size > 1:
        if run_manager.is_root:
            logger.info("Sync search result")
        del params["finder"]
        if args.evolution.evolution_type != "distributed":
            params_path = os.path.join(run_manager.experiment_path, "params.pt")
            if run_manager.is_root:
                torch.save(params, params_path)
            dist.barrier()
            if not run_manager.is_root:
                params = torch.load(params_path, map_location=run_manager.device)
            dist.barrier()
            if run_manager.is_root:
                os.remove(params_path)

    if run_manager.is_root:
        logger.info("Init supernet for validation")
    learn_manager, supernet = create_supernet_and_learn_manager(run_manager)
    success = params["acc_dict_success"] or params["evolutuion_success"]
    if success:
        if params["evolutuion_success"]:
            params = validate(learn_manager, supernet, params, name="evolution")
            if (
                args.evolution.evolution_accuracy_constraint
                > params["evolution"]["metrics"][args.strategy.main_metric]
            ):
                params["evolutuion_success"] = False
                if run_manager.is_root:
                    logger.warning(
                        "Real metrics for evolution network don't beat constraint "
                    )
        if params["acc_dict_success"]:
            params = validate(learn_manager, supernet, params, name="accdict")
            # Есть тонкий момент: при построении accdict используется не вся, а
            # часть валидации, поэтому мы можем получить метрику ниже тут порога
            # TODO: REFACTOR/FIX

        # TODO: REFACTOR: Это нужно вынести в choose_best.
        # Но есть риск запутаться и сломать всё. Поэтому только в феврале
        if params["acc_dict_success"] and params["evolutuion_success"]:
            if params["choose_tiny"]:
                params["test_from"] = "accdict"
                params["test"] = copy.deepcopy(params[params["test_from"]])
            else:
                params = choose_best(learn_manager, params)
        else:
            if not params["acc_dict_success"]:
                params["test_from"] = "evolution"
            elif not params["evolutuion_success"]:
                params["test_from"] = "accdict"
            params["test"] = copy.deepcopy(params[params["test_from"]])
        if run_manager.is_root:
            logger.info(f"Network chooset from {params['test_from']}")

        params = validate(learn_manager, supernet, params, name="test")
        if run_manager.is_root:
            params["result_subnet"] = supernet.get_active_subnet()
    if run_manager.is_root:
        save_evolution_results(learn_manager, params, success=success)
    dist.barrier()
    run_manager.stage_end()
