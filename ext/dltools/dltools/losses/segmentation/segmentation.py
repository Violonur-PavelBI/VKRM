from typing import Optional

import numpy as np
import torch
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss
from segmentation_models_pytorch.losses.constants import MULTICLASS_MODE
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, MSELoss, KLDivLoss

__all__ = ["CombinedLoss", "CustomBCELoss", "DiceBCELoss", "ComputeSegmentationLoss"]


class CustomBCELoss(nn.Module):
    """
    input, target: [batch, channels, height, width] 
    """

    def __init__(self, ignore_index: Optional[int] = None) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = 1e-7

    def forward(self, input: Tensor, target: Tensor):
        if self.ignore_index is not None:
            mask = torch.arange(input.size(1)) != self.ignore_index
            input = input[:, mask]
            target = target[:, mask]
        loss = -(
            target * torch.log(input + self.eps)
            + (1 - target) * torch.log(1 - input + self.eps)
        )
        return loss.mean()


class DiceBCELoss(nn.Module):
    def __init__(
        self, ignore_index: Optional[int] = None, dice_weight: Optional[float] = 0.5
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(mode=MULTICLASS_MODE, ignore_index=ignore_index)
        self.bce = CustomBCELoss(ignore_index=ignore_index)
        self.dice_weight = np.clip(dice_weight, 0, 1)

    def forward(self, input: Tensor, target: Tensor):
        target_argmax = target.argmax(1)
        dice_loss = self.dice(input, target_argmax)
        bce_loss = self.bce(input, target)
        combined_loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * bce_loss
        return combined_loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        ignore_index: Optional[int] = None,
        dice_weight: Optional[float] = 0,
        lovasz_weight: Optional[float] = 0,
        ce_weight: Optional[float] = 0,
        bce_weight: Optional[float] = 0,
        mse_weight: Optional[float] = 0,
    ) -> None:
        weights_sum = dice_weight + lovasz_weight + ce_weight + bce_weight + mse_weight
        assert weights_sum > 0
        super().__init__()
        self.dice = DiceLoss(mode=MULTICLASS_MODE, ignore_index=ignore_index)
        self.lovasz = LovaszLoss(mode=MULTICLASS_MODE, ignore_index=ignore_index)
        self.ce = CrossEntropyLoss(
            ignore_index=ignore_index if ignore_index is not None else -100
        )
        self.bce = CustomBCELoss(ignore_index=ignore_index)
        self.mse = MSELoss()
        self.dice_weight = dice_weight / weights_sum
        self.lovasz_weight = lovasz_weight / weights_sum
        self.ce_weight = ce_weight / weights_sum
        self.bce_weight = bce_weight / weights_sum
        self.mse_weight = mse_weight / weights_sum

    def forward(self, input: Tensor, target: Tensor):
        target_argmax = target.argmax(1)
        dice_loss = self.dice(input, target_argmax)
        lovasz_loss = self.lovasz(input, target_argmax)
        ce_loss = self.ce(input, target_argmax)
        bce_loss = self.bce(input, target)
        mse_loss = self.mse(input, target)
        combined_loss = (
            self.dice_weight * dice_loss
            + self.lovasz_weight * lovasz_loss
            + self.ce_weight * ce_loss
            + self.bce_weight * bce_loss
            + self.mse_weight * mse_loss
        )
        return combined_loss


class ComputeSegmentationLoss:
    CRITS = {
        "LovaszLoss": LovaszLoss,
        "DiceLoss": DiceLoss,
        "CrossEntropyLoss": CrossEntropyLoss,
        "BCELoss": CustomBCELoss,
        "DiceBCELoss": DiceBCELoss,
        "CombinedLoss": CombinedLoss,
    }

    def __init__(
        self, strategy_cfg=None, kd_ratio: float = 0, background_type: str = "exist"
    ) -> None:
        super().__init__()

        self.kd_ratio = kd_ratio
        self.background_type = background_type

        CritCls = self.CRITS[strategy_cfg.loss]
        crit_args = {}
        if (
            self.background_type in ["add", "exist"]
            and not strategy_cfg.background_loss
        ):
            crit_args = {"ignore_index": 0}

        if CritCls in (DiceLoss, LovaszLoss):
            crit_args["mode"] = MULTICLASS_MODE
        elif CritCls is DiceBCELoss:
            crit_args["dice_weight"] = strategy_cfg.dice_weight
        elif CritCls is CombinedLoss:
            crit_args["dice_weight"] = strategy_cfg.dice_weight
            crit_args["lovasz_weight"] = strategy_cfg.lovasz_weight
            crit_args["ce_weight"] = strategy_cfg.ce_weight
            crit_args["bce_weight"] = strategy_cfg.bce_weight
            crit_args["mse_weight"] = strategy_cfg.mse_weight
        self.criterion = CritCls(**crit_args)

        DISTIL_CRIT_POLICIES = {"strategy", "mse", "none", "kl"}
        distil_crit_policy = strategy_cfg.distil_loss
        if distil_crit_policy == "strategy":
            self.distil_criterion = CritCls(**crit_args)
        elif distil_crit_policy == "mse":
            self.distil_criterion = MSELoss()
        elif distil_crit_policy == "kl":
            self.distil_criterion = KLDivLoss(reduction="batchmean")
        elif distil_crit_policy == "none":
            self.distil_criterion = None
        else:
            raise ValueError(
                f"Invalid distillation policy: {distil_crit_policy} not in {DISTIL_CRIT_POLICIES}"
            )

    def __call__(self, context):
        if isinstance(self.criterion, (DiceLoss, LovaszLoss, CrossEntropyLoss)):
            target = context["target"].argmax(1)
        else:
            target = context["target"]

        loss = self.criterion(context["output"], target)
        if (
            self.distil_criterion is not None
            and self.kd_ratio > 0
            and "teacher_output" in context
            and context["teacher_output"] is not None
        ):
            if isinstance(
                self.distil_criterion, (DiceLoss, LovaszLoss, CrossEntropyLoss)
            ):
                distil_target = context["teacher_output"].argmax(1)
            else:
                distil_target = context["teacher_output"]
            loss_to_teacher = self.distil_criterion(context["output"], distil_target)
            loss = self.kd_ratio * loss_to_teacher + (1 - self.kd_ratio) * loss
        return loss
