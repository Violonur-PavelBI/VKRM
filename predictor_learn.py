import os

import catboost as cb
import torch
import torch.distributed as dist
from loguru import logger
from typing import Literal
from ofa.run_manager import RunManager
from ofa.nas.arch_encoders import build_arch_encoder
from ofa.utils.configs_dataclasses import PredictorLearnStageConfig

from ofa.nas.predictors import ArchToNumberDataset, Predictor, PredictorCatboost


def worker(run_manager: RunManager):
    if not run_manager.is_root:
        dist.barrier()
        return

    args: PredictorLearnStageConfig = run_manager.cur_stage_config
    task = args.pred_learn.metric

    arch_encoder = build_arch_encoder(args)

    # TODO: REFACTOR change accuracy
    if args.pred_learn.data_path is None and task == "accuracy":
        args.pred_learn.data_path = os.path.join(
            run_manager.experiment_path, "pred_dataset/pred.dict"
        )
    dataset = ArchToNumberDataset(args.pred_learn.data_path)

    train_data, val_data, base_value = dataset.train_val_split(
        arch_encoder=arch_encoder
    )
    logger.info(f"Mean {task} in dataset: {base_value}")

    if args.pred_learn.model == "Perceptron":
        model = Predictor(arch_encoder=arch_encoder, device=run_manager.device)
    else:
        cb_model = cb.CatBoostRegressor(
            iterations=args.pred_learn.n_epochs * 10,
            train_dir=run_manager.current_stage_path,
        )
        model = PredictorCatboost(model=cb_model, arch_encoder=arch_encoder)

    os.makedirs(run_manager.current_stage_path, exist_ok=True)
    log_filename = os.path.join(
        run_manager.current_stage_path, f"{task}_predictor_learning.log"
    )
    with open(log_filename, "wt") as f_out:
        f_out.write(f"{task} predictor for {run_manager.experiment_path}\n")

    if isinstance(model, Predictor):
        train_loader, val_loader = dataset.build_data_loaders(train_data, val_data)
        best_loss = model.fit(
            train_loader,
            val_loader,
            run_manager.tensorboard,
            run_manager.device,
            args.pred_learn.init_lr,
            args.pred_learn.n_epochs,
            log_filename,
            verbose=True,
        )
    else:
        best_loss = model.fit(train_data, val_data, verbose=True)

    run_manager.update_run_metrics({f"{task}_predictor_learn": {"val_loss": best_loss}})

    predictor_path = getattr(args.predictors, f"{task}_predictor_path", None)
    if predictor_path is None:
        predictor_path = os.path.join(
            run_manager.experiment_path, f"{task}_predictor.pt"
        )
    logger.info(f"Saving {task} predictor to {predictor_path}")
    os.makedirs(os.path.dirname(predictor_path), exist_ok=True)
    torch.save(model, predictor_path)

    if dist.is_initialized():
        dist.barrier()

    run_manager.stage_end()
