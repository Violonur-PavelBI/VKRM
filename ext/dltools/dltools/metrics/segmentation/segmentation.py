"""Common image segmentation metrics.
"""
import torch
from prettytable import PrettyTable
import pandas as pd
from torchmetrics import MeanMetric
from torch import distributed as dist

from segmentation_models_pytorch.utils import base as smp_base
from segmentation_models_pytorch.utils.metrics import IoU

from typing import Union, List, Tuple

# from utils.calculus.metrics import _fast_hist, jaccard_index, dice_coefficient, nanmean, \
#     overall_pixel_accuracy

metric_names = [
    "AverageIoU",
    "PixelwiseAcc",
    "overall_acc",
    "avg_per_class_acc",
    "avg_jacc",
    "avg_dice",
]
EPS = 1e-8


class SMPMetric(MeanMetric):
    def __init__(
        self, counter_cls: smp_base.Metric, threshold=0.5, ignore_channels=None
    ):
        super().__init__()
        self.counter = counter_cls(threshold=threshold, ignore_channels=ignore_channels)

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        counted = self.counter(output[0].detach(), output[1].detach()).item()
        return super().update(counted)

    def compute(self) -> float:
        return super().compute().cpu().item()


class AverageIoU(SMPMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(IoU, *args, **kwargs)


class PixelwiseAcc(MeanMetric):
    def __init__(self):
        super().__init__()

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        one = output[0].detach().argmax(1)
        two = output[1].detach().argmax(1)
        self.mean_value += (one == two).sum()
        self.weight += one.numel()

    def compute(self) -> float:
        return super().compute().cpu().item()


def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])


# TODO: Reimplement with faster hist, using sparse sum.
def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = (
        torch.bincount(
            num_classes * true[mask] + pred[mask], minlength=num_classes ** 2
        )
        .reshape(num_classes, num_classes)
        .float()
    )
    return hist


def overall_pixel_accuracy(hist):
    """Computes the total pixel accuracy.

    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.

    Args:
        hist: confusion matrix.

    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc


def per_class_pixel_accuracy(hist):
    """Computes the average per-class pixel accuracy.

    The per-class pixel accuracy is a more fine-grained
    version of the overall pixel accuracy. A model could
    score a relatively high overall pixel accuracy by
    correctly predicting the dominant labels or areas
    in the image whilst incorrectly predicting the
    possibly more important/rare labels. Such a model
    will score a low per-class pixel accuracy.

    Args:
        hist: confusion matrix.

    Returns:
        avg_per_class_acc: the average per-class pixel accuracy.
    """
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc


def jaccard_index(hist):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).

    Args:
        hist: confusion matrix.

    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    return avg_jacc


def dice_coefficient(hist):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.

    Args:
        hist: confusion matrix.

    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice


def eval_metrics(true, pred, num_classes, ignore_idc=[]):
    """Computes various segmentation metrics on 2D feature maps.

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        pred: a tensor of shape [B, H, W] or [B, 1, H, W].
        num_classes: the number of classes to segment. This number
            should be less than the ID of the ignored class.

    Returns:
        overall_acc: the overall pixel accuracy.
        avg_per_class_acc: the average per-class pixel accuracy.
        avg_jacc: the jaccard index.
        avg_dice: the dice coefficient.
    """
    hist = torch.zeros((num_classes, num_classes))
    for t, p in zip(true, pred):
        hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    avg_per_class_acc = per_class_pixel_accuracy(hist)
    avg_jacc = jaccard_index(hist)
    avg_dice = dice_coefficient(hist)
    return overall_acc, avg_per_class_acc, avg_jacc, avg_dice


def eval_metrics_by_batch(true, pred, num_classes):
    """Computes various segmentation metrics on 2D feature maps, by batch

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        pred: a tensor of shape [B, H, W] or [B, 1, H, W].
        num_classes: the number of classes to segment. This number
            should be less than the ID of the ignored class.

    Returns:
        per_class_acc: the average per-class pixel accuracy. [B, C]
        jacc: the jaccard index. [B, C]
        dice: the dice coefficient. [B, C]
        cfm : confusion unnormed matrix (hist) [B, C, C]
    """
    hist = torch.zeros((true.shape[0], num_classes, num_classes))
    per_class = torch.zeros((true.shape[0], num_classes))
    jaccard = torch.zeros((true.shape[0], num_classes))
    dice = torch.zeros((true.shape[0], num_classes))
    for i, (t, p) in enumerate(zip(true, pred)):
        hist[i] += _fast_hist(t.flatten(), p.flatten(), num_classes)
        per_class[i] = torch.diag(hist[i]) / (hist[i].sum(dim=1) + EPS)
        jaccard[i] = jaccard_index(hist[i])
        dice[i] = dice_coefficient(hist[i])
    return per_class, jaccard, dice, hist


class MetricEvaluator:
    """class which calculates, agregates, calls and prints metrics."""

    # metric names of averaged values
    avg_names = [
        "Overall Pixel Accuracy",
        "Per class PixAcc",
        "MIoU (aka. Jaccard)",
        "Dice Score (aka. F1)",
    ]
    # metric names per class
    per_names = ["Pixel Accuracy", "Jaccard (aka. IoU)", "Dice Metric(aka. F1-score)"]
    __last_instance = None

    def __init__(self, bd_class_dict: dict, dist=False, rank=0, iterative=True):
        self._defaults = (bd_class_dict, dist, rank, iterative)
        # adding
        names_indexes = []
        for key, value in bd_class_dict.items():
            if key.lower() not in ["__raster__", "ignored", "raster"] and value != 255:
                names_indexes += [[key, value]]
        ni_sorted = sorted(names_indexes, key=lambda x: x[1])
        self.ni = ni_sorted
        self.nc = len(ni_sorted)
        self._total_hist = torch.zeros((self.nc, self.nc)) if iterative else None
        self.is_dist = dist
        self.rank = rank
        self.running_loss = None
        # if criterion is not None here will be name of loss
        self._cumul = iterative
        self._agr_hist = None
        self._agr_loss = None
        self.counter = 0

        self.__class__.__last_instance = self

    """ total_hist - finction which unifies internal calculation of confusion matrix """

    @property
    def total_hist(self):
        if self._cumul:
            return self._total_hist
        else:
            return self._agr_hist.sum(0)

    @total_hist.setter
    def total_hist(self, tensor: torch.Tensor):
        assert (
            self._total_hist.shape == tensor.shape
        ), f"Get wrong shape! {tensor.shape}"
        self._total_hist = tensor

    @classmethod
    def register_criterion_name(cls, loss_class_def):
        cls.avg_names += [loss_class_def.__name__]
        cls.counter = 0
        cls.running_loss = None

    def __call__(self, true, pred, loss_scalar=None, num_classes_check=None):
        assert any(
            [
                # loss_scalar is None and len(self.avg_names) == 4,
                loss_scalar is not None
                and len(self.avg_names) == 5
            ]
        )
        if num_classes_check is not None:
            assert self.nc == num_classes_check, (
                f"Number of classes in subset didn't match,\n"
                f"In subset: {self.nc},\nmodel's : {num_classes_check}"
            )
        if self._cumul:
            self.cumulative(true, pred, loss_scalar)
        else:
            self.agr(true, pred, None)

    @staticmethod
    def eval_metrics_per_class(hist):
        """
        calculation metric per class,  hist - confusion matrix 2D - tensor []
        return: Pixel Accuracy, Jaccard (aka. IoU), Dice Metric(aka. F1-score)

        """

        # TODO: сделать удобнее
        with torch.no_grad():
            A_inter_B = torch.diag(hist)
            A = hist.sum(dim=1)
            B = hist.sum(dim=0)
            per_class_acc = A_inter_B / (A + EPS)
            jaccard = A_inter_B / (A + B - A_inter_B + EPS)
            dice = (2 * A_inter_B) / (A + B + EPS)
        return per_class_acc, jaccard, dice

    def eval_avg_metric(
        self, hist: torch.Tensor = None, running_loss: torch.FloatTensor = None
    ) -> Tuple[float, ...]:
        """
        calculation metrics of averaged values,  hist - confusion matrix ?D - tensor [], running_loss - current loss values ???
        return: average pixel accuracy, average jaccard (aka. IoU), average dice metric(aka. F1-score)

        """

        if hist is None:
            hist = self.total_hist
        if running_loss is None:
            running_loss = self.running_loss
        averages = [torch.diag(hist).sum() / (hist.sum() + EPS)]
        averages += [nanmean(x) for x in self.eval_metrics_per_class(hist)]
        if running_loss is not None:
            if running_loss.ndim == 0:
                averages += [running_loss]
            else:
                averages += [running_loss.mean()]
        return tuple([x.squeeze().item() for x in averages])

    def print_by_class(self, add_str="", flag_return=False) -> Union[None, str]:
        """
            make metrics info table with elements of class PrettyTable() if rank is 0
        +----------------------------------------------------------------------------------+
        |                              Epoch :  | Metrics |                               |
        +---------------+----------------+--------------------+----------------------------+
        |     Class     | Pixel Accuracy | Jaccard (aka. IoU) | Dice Metric(aka. F1-score) |
        +---------------+----------------+--------------------+----------------------------+

        """

        if self.rank == 0:
            table = PrettyTable()
            table.field_names = ["Class"] + self.per_names
            table.title = add_str + " | Metrics | "
            for (class_name, _), per_class_acc, iou, dice in zip(
                self.ni, *self.eval_metrics_per_class(self.total_hist)
            ):
                table.add_row(
                    [
                        class_name,
                        f"{per_class_acc.item():.4f}",
                        f"{iou.item():.4f}",
                        f"{dice.item():.4f}",
                    ]
                )

            if flag_return:
                print(table)
                return table
            else:
                print(table)
        else:
            pass

    def table_by_class(self):
        """return per class metrics info as DataFrame if rank is 0"""

        records = []
        if self.rank == 0:
            for (class_name, _), per_class_acc, iou, dice in zip(
                self.ni, *self.eval_metrics_per_class(self.total_hist)
            ):
                per_class_acc, iou, dice = [
                    x.item() for x in [per_class_acc, iou, dice]
                ]
                records += [
                    {
                        name: value
                        for name, value in zip(
                            ["Class name", *self.per_names],
                            [class_name, per_class_acc, iou, dice],
                        )
                    }
                ]
            df = pd.DataFrame(records)
            df.set_index("Class name")
            return df
        else:
            return None

    def print_means(self, purpose, epoch="", flag_return=False) -> Union[None, str]:

        """
        if cumul mood print info about average metrics:
                    train : Epoch number + Metrics: metric_name: metric_value
                    test : test + Metrics: metric_name: metric_value

        else calculate average metrics from self._agr_hist and self._agr_loss
                    train : Epoch number + Metrics: metric_name: metric_value
                    test : test + Metrics: metric_name: metric_value
        """

        if self._cumul:
            if self.rank == 0:
                info_string = "Metrics:"
                averages = self.eval_avg_metric(self.total_hist, self.running_loss)
                for metric_name, metric_value in zip(self.avg_names, averages):
                    if isinstance(metric_value, torch.Tensor):
                        if metric_value.ndim != 0:
                            metric_value.mean()
                        metric_value = metric_value.cpu().item()
                    info_string += f"  {metric_name}: {metric_value:.4f}  "
                msg = (
                    f"Epoch [{epoch}]: " + info_string
                    if purpose == "train"
                    else f"[{purpose}]:" + info_string
                )
                if flag_return:
                    print(msg)
                    return msg
                else:
                    print(msg)

        else:
            if self.rank == 0:
                info_string = "Metrics:"
                table = self.det_desc()
                _ = [
                    [metric_rec[i] for metric_rec in table]
                    for i in range(len(self.avg_names) - 1)
                ]
                averages = [sum(x) / len(x) for x in _]
                for metric_name, metric_value in zip(self.avg_names[:-1], averages):
                    if isinstance(metric_value, torch.Tensor):
                        if metric_value.ndim != 0:
                            metric_value.mean()
                        metric_value = metric_value.cpu().item()
                    info_string += f"  {metric_name}: {metric_value:.4f}  "
                msg = (
                    f"Epoch [{epoch}]: " + info_string
                    if purpose == "train"
                    else f"[{purpose}]:" + info_string
                )
                if flag_return:
                    print(msg)
                    return msg
                else:
                    print(msg)

    def cumulative(self, true, pred, loss):
        """
        method which provides combined accumulation, with storing only
        last accumulated calculation result. Calculation results are combined
        by weighted summation over the first dimension, which dimension of
        batch, i.e. it is recursive calculation of the average along batch.

        Arg: true – Tensor of integers, target markup.
             pred – Tensor of integers, predicted markup.
             loss - Tensor of None or calculated vector\scalar using FO.
        """

        with torch.no_grad():
            for t, p in zip(true, pred):
                self.total_hist += _fast_hist(t.flatten(), p.flatten(), self.nc)
                self._accumulate(
                    self.total_hist, _fast_hist(t.flatten(), p.flatten(), self.nc)
                )
            if self.is_dist:
                self.total_hist = self.total_hist.contiguous().cuda(
                    self.is_dist.get_rank()
                )
                self.is_dist.reduce(self.total_hist, 0, self.is_dist.ReduceOp.SUM)
                self.total_hist = self.total_hist.cpu()
            if loss is not None:
                self.running_loss, self.counter = self._accumulate_w_counter(
                    self.running_loss, loss, self.counter
                )

    def agr(
        self,
        true: torch.LongTensor,
        pred: torch.LongTensor,
        loss: Union[torch.FloatTensor, None] = None,
    ):
        """
        method that provides separate accumulation,
        with the storage of each final element of the calculation.
        The calculation results are placed by adding to the end of the container.

         Arg: true – Tensor of integers, target markup.
              pred – Tensor of integers, predicted markup.
              loss - Tensor of None or calculated vector\scalar using FO.
        """
        with torch.no_grad():
            # counting hist image by image
            for t, p in zip(true, pred):
                cur_hist = _fast_hist(t.flatten(), p.flatten(), self.nc)
                self._agr_hist = self._aggregate(self._agr_hist, cur_hist)

                # if self.dist:
                #     # sync gather and stack hist's from all replicas
                #    cur_hist = self._aggregate(cur_hist)
                # else:
                #     cur_hist = cur_hist.unsqueeze(0).cpu() # unsquuezing need to create 1 dim out, to stack by this dim, if not distributed

                # if self._agr_hist is None:
                #     self._agr_hist = cur_hist
                # else:
                #     self._agr_hist = torch.cat([self._agr_hist, cur_hist]) # so this is like append  if self._agr_hist already exists

                if loss is not None:

                    self._agr_loss = self._aggregate(self._agr_loss, loss)

                    # if self._agr_loss is None:
                    #     if self.dist:
                    #         loss_scalar = self._aggregate(loss_scalar)
                    #     else:
                    #         self._agr_loss = loss_scalar.unsqueeze(0)  # unsquuezing need to create 1 dim out, to stack by this dim, if not distributed
                    # else:
                    #     self._agr_loss = torch.stack([self._agr_loss, loss_scalar])

    def det_desc(self) -> List[Tuple[float]]:
        """calculate average metrics from self._agr_hist and self._agr_loss"""
        tozip = self._agr_hist
        out = []
        if self._agr_loss is not None:
            tozip = zip(self._agr_hist, self._agr_loss)

        for elem in tozip:
            out += [[*self.eval_avg_metric(elem)]]
        return out

    def eval_metrics_table(self):
        """method provides calculation of metrics in table. Table is expanded for each record in sample"""
        if self._cumul:
            # Bcz there not enough data to calculate this anyway
            raise NotImplementedError

        else:
            data = self.det_desc()
            named_records = [
                *map(
                    lambda metrics_rec: {
                        metric_name: metric_value
                        for metric_name, metric_value in zip(
                            self.avg_names, metrics_rec
                        )
                    },
                    data,
                )
            ]

            return pd.DataFrame.from_records(named_records).describe()

    @staticmethod
    def _transfer_gather(tensor: torch.Tensor, send_to_cpu=True):
        """Transfer tensor to cuda device by rank, gather on 0 rank, and transfer back to cpu if flag set"""

        tensor = tensor.contiguous().cuda(dist.get_rank())
        togather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        # gather hist's or losses from all replicas to all
        dist.all_gather(togather, tensor)
        # stacking by new dim by torch's default
        tensor = torch.stack(togather)
        if send_to_cpu:
            return tensor.cpu()
        return tensor

    @staticmethod
    def _transfer_reduce_sum(tensor: torch.Tensor, send_to_cpu=True) -> torch.Tensor:
        """reducing by sum"""
        tensor = tensor.contiguous().cuda(dist.get_rank())
        dist.reduce(tensor, 0, dist.ReduceOp.SUM)

        if send_to_cpu:
            return tensor.cpu()

        return tensor

    @staticmethod
    def _collect(
        agr_cont: Union[None, torch.Tensor], tensor_to_collect: torch.Tensor
    ) -> torch.Tensor:
        """if agr_cont is none, agr_cont would be equal to tensor_to_collect
        else it would be concatenation by first dim of both tensors"""
        is_dist = dist.is_initialized()
        if not is_dist:
            tensor_to_collect = tensor_to_collect.unsqueeze(0)

        if agr_cont is None:
            agr_cont = tensor_to_collect
        else:
            assert agr_cont.device == tensor_to_collect.device

            agr_cont = torch.cat([agr_cont, tensor_to_collect])

        return agr_cont

    def _aggregate(
        self, agr_container: Union[None, torch.Tensor], cur_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        method provides data collection and addition to the container.Type and device of
        current result and container are same, shape is same only with the exception of
        first dimension of container and  current result
        Arg: agr_container - Tensor that provides storage of calculation results.
             cur_tensor – Tensor, current result of calculations.
        """
        if self.is_dist:
            cur_tensor = self._transfer_gather(
                cur_tensor
            )  # this op add new dimension, size of world_size

        else:
            cur_tensor.unsqueeze(0).cpu()  # so there we also add this dim

        agr_container = self._collect(agr_container, cur_tensor)
        return agr_container

    def _accumulate(
        self, acc_container: Union[None, torch.Tensor], cur_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        method provides data collection, as recursive calculation of average\sum along batch dimension.
        Tensor shapes match except for missing acc_container dimension of  batch
        Arg:  acc_container– Tensor provides storage of calculation results. Initially can be empty
              cur_tensor – Tensor current result of calculations
        """
        if self.is_dist:
            cur_tensor = self._transfer_reduce_sum(cur_tensor)
        else:
            cur_tensor.cpu()
        if acc_container is None:
            acc_container = cur_tensor
        else:
            acc_container += cur_tensor

    def _accumulate_w_counter(
        self,
        acc_container: Union[None, torch.Tensor],
        cur_tensor: torch.Tensor,
        counter: Union[float, int],
    ) -> Tuple[torch.Tensor, int]:

        """
        method provides data collection as recursive calculation of average along dimension of batch
        with counter.Shape of tensors match everywhere, except for missing batch dimension of accumulating container
        Arg:  acc_container– Tensor provides storage of calculation results. Initially can be empty
              cur_tensor – Tensor current result of calculations
              counter – integer, counter for calculating average recursively

        """
        if self.is_dist:
            cur_tensor = self._transfer_reduce_sum(cur_tensor)
            cur_tensor /= self.is_dist.get_world_size()
        else:
            cur_tensor.cpu()

        if acc_container is None:
            acc_container = cur_tensor
            assert counter == 0
            counter += 1
        else:
            acc_container = (acc_container * counter + cur_tensor) / (counter + 1)
            counter += 1

        return acc_container, counter

    def reset(self):
        self.__init__(*self._defaults)
