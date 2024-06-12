from typing import List, Union

import torch
from torch import Tensor

from dltools.data_providers import SegmentationProvider
from ofa.utils import (
    SegmentationStrategyConfig,
    SupernetLearnStageConfig,
    TensorBoardLogger,
)
from dltools.metrics import AverageIoU, PixelwiseAcc
from dltools.losses.segmentation import ComputeSegmentationLoss

from dltools.configs.dataset import SegmentationDatasetConfig

from .base_strategy import BaseContext, BaseStrategy, MetricDict
from .classification import ClassificationStrategy
from .utils.segmentation.visualise import visualise_segmentation


class Context(BaseContext):
    target: Tensor
    image_path: List[str]


class SegmentationStrategy(BaseStrategy):
    """Стратегия на сегментацию. Есть функциона для исследований в виде использования части классов.
    При его использовании получить адекватный таргет можно только через стратегию
    """

    compute_distil_output = ClassificationStrategy.compute_distil_output

    BACKGROUND_TYPES = ["add", "exist", "no"]

    def __init__(self, args: SupernetLearnStageConfig, device="cuda") -> None:
        """
        Args:
            num_classes: (int): количество классов
            classes_idx: (list|None): задаёт то, какие классы будут использоваться, если None, то берёт все
            classes_unused: (list|None): задаёт то, какие классы не будут использоваться если None, то берёт все
        """

        super().__init__(args, device)

        self.dataset: SegmentationDatasetConfig = self.args.dataset

        strategy: SegmentationStrategyConfig = self.args.strategy
        self.main_metric = strategy.main_metric
        self.loss_background = strategy.background_loss

        self.background_type = self.dataset.background
        if self.background_type not in self.BACKGROUND_TYPES:
            raise ValueError(
                f"{self.background_type = } not in {self.BACKGROUND_TYPES}"
            )
        if self.background_type in ["add", "exist"]:
            self.ignore_channels = [0]
        else:
            self.ignore_channels = None

        self.classes_idx = self.dataset.classes_idx
        self.classes_unused = self.dataset.classes_unused
        self.add_background = self.classes_idx or self.classes_unused
        self.num_classes = self.dataset.n_classes
        # Тут какой-то бардак с количеством классов
        if self.classes_idx is None and self.classes_unused is None:
            self.classes_idx = list(range(self.num_classes))
        else:
            if self.classes_unused is not None:
                if self.classes_idx is not None:
                    raise Exception(
                        "Only one of (classes_idx,classes_unused) must be != None"
                    )
                self.classes_idx = list(range(self.num_classes))
                self.classes_idx = list(
                    set(self.classes_idx) - set(self.classes_unused)
                )
            self.num_classes = len(self.classes_idx)
            # иначе оно было добавлено раньше
            if self.add_background:
                self.num_classes += 1

        self.loss_counter: Union[None, ComputeSegmentationLoss] = None

    def build_metrics_dict(self) -> MetricDict:
        # ! С метриками может быть беда на многоклассовой
        res = {
            "IoU": AverageIoU(threshold=0.5, ignore_channels=self.ignore_channels).to(
                self.device
            ),
            "PixelwiseAcc": PixelwiseAcc().to(self.device),
        }
        return res

    def prepare_batch(self, context: BaseContext) -> None:
        super().prepare_batch(context)
        # Нет кода, который отсеивает классы, на которых не обучаемся для
        # Маски, представленной индексами
        # Ещё это наверняка можно сделать быстрее
        # TODO: REFACTOR
        mask = context["target"]
        masks = [(mask == v) for v in self.classes_idx]
        mask = torch.stack(masks, axis=1)
        if self.add_background:
            background = (~mask.any(1))[:, None, :, :]
            mask = torch.concatenate([background, mask], 1)
        mask = mask.type(torch.float)
        context["target"] = mask

    def compute_loss(self, context: Context) -> None:
        if self.loss_counter is None:
            self.loss_counter = ComputeSegmentationLoss(
                strategy_cfg=self.args.strategy,
                kd_ratio=self.kd_ratio,
                background_type=self.background_type,
            )
        context["output"] = context["output"].sigmoid()
        loss = self.loss_counter(context)
        context["loss"] = loss

    def visualise(
        self,
        context: Context,
        data_provider: SegmentationProvider,
        indices: List[int],
        tensorboard: TensorBoardLogger,
        stage_name: str,
        epoch: int,
    ) -> None:
        background = self.background_type in ("add", "exist")
        fig = visualise_segmentation(context, self.n_classes, background, indices)
        tensorboard.tensorboard.add_figure(
            f"{stage_name}/image_target_pred", fig, global_step=epoch
        )
