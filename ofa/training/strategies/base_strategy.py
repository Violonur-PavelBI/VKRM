from abc import ABC, abstractmethod
from typing import Dict, List, TypedDict, Union

from torchmetrics import Metric
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss

from dltools.data_providers import DataProvider
from ofa.utils import SupernetLearnStageConfig, TensorBoardLogger

MetricDict = Dict[str, Metric]


class BaseContext(TypedDict):
    model: nn.Module
    teacher_model: nn.Module
    image: Tensor
    teacher_output: Union[Tensor, None]
    output: Tensor
    loss: Tensor
    target: Tensor


class BaseStrategy(ABC):
    """Класс, предоставляющий вспомогательные методы для обучения сети

    классы наследники должны определять методы, специализированные под конкретную задачу
    например классификация, сегментация, детекция и т.д"""

    def __init__(self, args: SupernetLearnStageConfig, device="cuda") -> None:
        self.args = args
        self.criterion: _Loss = None
        self.kd_ratio = args.learn_config.kd_ratio
        self.n_classes = args.dataset.n_classes
        self.device = device
        self.main_metric: str

    def prepare_batch(self, context: BaseContext) -> None:
        context["image"] = context["image"].to(self.device)
        context["target"] = context["target"].to(self.device)

    @abstractmethod
    def build_metrics_dict(self) -> MetricDict:
        """Создаёт словарь с метриками для заполнения"""
        pass

    @abstractmethod
    def compute_distil_output(self, context: BaseContext) -> None:
        """Получает выход с модели учителя"""
        pass

    def compute_output(self, context: BaseContext) -> None:
        context["output"] = context["model"](context["image"])

    @abstractmethod
    def compute_loss(self, context: BaseContext) -> None:
        """Вычисляет значение loss с учётом дистилляции"""

    @abstractmethod
    def visualise(
        self,
        context: BaseContext,
        data_provider: DataProvider,
        indices: List[int],
        tensorboard: TensorBoardLogger,
        stage_name: str,
        epoch: int,
    ) -> None:
        """Логирует картинку в тензорборд."""

    def update_metric(self, metrics: MetricDict, context: BaseContext) -> None:
        for _, metric in metrics.items():
            metric.update(output=[context["output"], context["target"]])

    def get_metric_vals(self, metrics: MetricDict) -> MetricDict:
        return {key: metrics[key].compute() for key in metrics}
