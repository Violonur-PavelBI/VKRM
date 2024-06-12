import os
from pathlib import Path
import torch

from ofa.networks import build_composite_supernet_from_args, init_supernet_pretrain
from ofa.training.progressive_shrinking import (
    prepare_for_elastic_task,
    train,
    validate,
)
from ofa.run_manager import LearnManager, RunManager
from ofa.utils import SupernetLearnStageConfig, AttrPassDDP
from loguru import logger


def worker(run_manager: RunManager):
    if run_manager.device == "cuda":
        torch.cuda.set_device(run_manager._local_rank)
    args: SupernetLearnStageConfig = run_manager.cur_stage_config

    # TODO: Refactor вынести в LearnConfig ???

    teacher_path = os.path.join(
        run_manager.experiment_path, "supernet_weights/teacher.pt"
    )
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
    args.exp.teacher_path = teacher_path if os.path.exists(teacher_path) else None
    args.exp.supernet_checkpoint_path = checkpoint_path

    net = build_composite_supernet_from_args(args.supernet_config).to(
        run_manager.device
    )

    flag_file = Path(run_manager.experiment_path) / "pretrain_load_flag"
    if (
        not flag_file.exists()
        and args.supernet_config.pretrain is not None
        and run_manager.task in ("keypoints", "detection")
    ):
        init_supernet_pretrain(net, args, verbose=run_manager.is_root)
        flag_file.touch()

    device_ids = [run_manager._local_rank] if run_manager.device == "cuda" else None
    net = AttrPassDDP(net, device_ids=device_ids, find_unused_parameters=True)
    learn_manager = LearnManager(net, run_manager=run_manager)
    learn_manager.broadcast()

    validate_func_dict = prepare_for_elastic_task(learn_manager)

    train(
        learn_manager,
        lambda _run_manager, epoch, is_test: validate(
            _run_manager, epoch, is_test, validate_func_dict
        ),
    )

    run_manager.stage_end()
