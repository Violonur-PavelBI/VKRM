from __future__ import annotations

import copy
import os
from typing import Dict, List, Tuple, Union, TYPE_CHECKING

import catboost
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from loguru import logger
from tqdm import tqdm

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from ofa.utils import TensorBoardLogger

from ..arch_encoders import ArchEncoder

Array = Union[Tensor, np.ndarray]


class Predictor(nn.Module):
    def __init__(
        self,
        arch_encoder: ArchEncoder,
        hidden_size: int = 400,
        n_layers: int = 3,
        checkpoint_path: str = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.arch_encoder = arch_encoder
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        layers = [
            nn.Linear(self.arch_encoder.n_dim, self.hidden_size),
            nn.ReLU(inplace=True),
        ]
        for i in range(self.n_layers - 1):
            layers.extend(
                [
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Linear(self.hidden_size, 1, bias=False))
        self.layers = nn.Sequential(*layers).to(self.device)

        self.base_value = nn.Parameter(
            torch.zeros(1, device=self.device), requires_grad=False
        )

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            self.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x).squeeze()
        return x + self.base_value

    def predict(self, arch_dict_list: Union[Dict, List[Dict]]) -> Tensor:
        if isinstance(arch_dict_list, dict):
            arch_dict_list = [arch_dict_list]
        x = [self.arch_encoder.arch2feature(arch_dict) for arch_dict in arch_dict_list]
        x = torch.tensor(np.array(x)).float().to(self.device)
        return self.forward(x)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tb: TensorBoardLogger,
        device: torch.device,
        init_lr: float,
        n_epochs: int,
        log_filename: str,
        verbose: bool = True,
    ) -> float:
        criterion = torch.nn.MSELoss().to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=init_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, min_lr=0.00001
        )
        best_weights = copy.deepcopy(self.state_dict())
        best_loss = float("inf")

        for epoch in tqdm(range(n_epochs)):
            train_loss, val_loss = 0, 0
            self.train()
            for sample, label in train_loader:
                sample = sample.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                outputs = self.forward(sample)
                loss = criterion(outputs, label)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            train_loss /= len(train_loader)

            self.eval()
            for sample, label in val_loader:
                sample = sample.to(device)
                label = label.to(device)
                with torch.no_grad():
                    outputs = self.forward(sample)
                loss = criterion(outputs, label)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            lr = scheduler.optimizer.param_groups[0]["lr"]
            # scheduler.step()

            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(self.state_dict())

            if verbose:
                logstring = (
                    f"epoch {epoch: >3d}\t|\ttrain: {train_loss:.10f}\t|"
                    f"\tval: {val_loss:.10f} \t|\t lr: {lr}"
                )
                with open(log_filename, "a") as f_out:
                    f_out.write(logstring + "\n")
                tqdm.write(logstring)

                metrics = {"Loss/train": train_loss, "Loss/val": val_loss, "lr": lr}
                tb.write_metrics(metrics)

        if verbose:
            logger.info(f"Best validation loss = {best_loss}.")

        self.load_state_dict(best_weights)
        return best_loss


class PredictorCatboost(nn.Module):
    def __init__(
        self,
        model: catboost.CatBoostRegressor,
        arch_encoder: ArchEncoder,
        device: str = "cpu",
    ):
        super().__init__()
        self.model = model
        self.arch_encoder = arch_encoder
        self.device = device

    def _to_tensor(self, x: Array) -> Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float, device=self.device)
        return x

    def _to_ndarray(self, x: Array) -> np.ndarray:
        if isinstance(x, Tensor):
            x = x.cpu().numpy()
        return x

    def forward(self, x: Array) -> Tensor:
        x = self._to_ndarray(x)
        y = self.model.predict(x)
        y = self._to_tensor(y)
        return y

    def fit(
        self,
        train_data: Tuple[Array, Array],
        val_data: Union[Tuple[Array, Array], None] = None,
        verbose: bool = True,
    ) -> Union[float, None]:
        x_train, y_train = map(self._to_ndarray, train_data)
        self.model.fit(x_train, y_train)
        if val_data is None:
            return

        x_val, y_val = val_data
        z_train = self.forward(x_train)
        z_val = self.forward(x_val)
        x_val, y_val = map(self._to_tensor, val_data)

        ae_val = torch.abs(z_val - y_val)
        mae_val = ae_val.mean()
        if verbose:
            mae_train = torch.abs(z_train - y_train).mean()
            fitness = (ae_val / y_val > 0.02).float().mean()
            fitness_01 = (ae_val / y_val > 0.10).float().mean()
            logger.info(
                f"Best validation loss (MAE_val) = {mae_val:.4f}\n"
                f"fitness 0.02: {fitness:.4f}, fitness 0.10: {fitness_01:.4f}, "
                f"val/train MAE ratio: {mae_val / mae_train:.5f}"
            )
        return mae_val.item()

    def predict(self, arch_dict_list: Union[Dict, List[Dict]]):
        if isinstance(arch_dict_list, dict):
            arch_dict_list = [arch_dict_list]
        x = [self.arch_encoder.arch2feature(arch_dict) for arch_dict in arch_dict_list]
        x = np.array(x, dtype=np.float32)
        return self.forward(x)
