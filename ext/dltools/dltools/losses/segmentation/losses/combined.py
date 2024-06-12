import torch
from torch.nn import Module
from ..losses import _BaseLoss, CELoss, SegmFocalLoss

# __all__ = []
from ..losses import DiceLoss, DiceLossSq


class Cmb_CE_DiceLoss(Module):
    """
    CrossEntropy + DiceLoss,
        Note: reducef 'c' and generalized weights with 'c'

    """

    def __init__(self, alpha=0.1, weights=None, weights_dice=True):
        super(Cmb_CE_DiceLoss, self).__init__()
        self.DiceLoss = DiceLoss(
            reduction="mb", weights=weights if weights_dice else None
        )
        self.CELoss = CELoss(reduction="n", weights=weights)
        self.alpha = alpha

    def __call__(self, gt, logits):
        loss_dice = self.DiceLoss(gt, logits)
        loss_ce = self.CELoss(gt, logits)
        loss_ce = loss_ce.mean([1, 2])
        # loss_dice = loss_dice.mean([1])
        self.dice = loss_dice
        self.ce = loss_ce

        loss_batch = loss_ce * self.alpha + loss_dice

        return loss_batch


class Cmb_FL_DiceLoss(torch.nn.Module):
    """
    FocalLoss + DiceLoss,
        Note: reducef 'c' and generalized weights with 'c'

    """

    def __init__(
        self,
        alpha=1.3,
        weights=None,
        weights_dice=True,
        ignore_idc=[],
        use_softmax=False,
    ):
        # super(Cmb_FL_DiceLoss, self).__init__(ignore_idc = ignore_idc)
        super(Cmb_FL_DiceLoss, self).__init__()
        self.DiceLoss = DiceLoss(
            reduction="mb",
            weights=weights if weights_dice else None,
            use_softmax=use_softmax,
        )
        self.FLLoss = SegmFocalLoss(
            reduction="mb", weights=weights, use_softmax=use_softmax
        )
        self.alpha = alpha

    def __call__(self, gt, logits):
        loss_dice = self.DiceLoss(gt, logits)
        loss_fl = self.FLLoss(gt, logits)
        loss_fl = loss_fl.view(gt.shape[0], -1).mean(1)
        loss_dice = loss_dice.mean([1])
        self.pdice = loss_dice
        self.fl = loss_fl

        loss_batch = (loss_fl * self.alpha + loss_dice) / (1 + self.alpha)

        return loss_batch


class Cmb_BCE_DiceLossSq(torch.nn.Module):
    """
    BCE + DiceLossSq,
        Note: reducef 'c' and generalized weights with 'c'

    """

    def __init__(
        self, alpha=4, weights=None, weights_dice=True, ignore_idc=[], use_softmax=False
    ):
        super().__init__()
        # super(Cmb_FL_DiceLoss, self).__init__(ignore_idc = ignore_idc)
        self.DiceLoss = DiceLossSq(
            reduction="mb",
            weights=weights if weights_dice else None,
            use_softmax=use_softmax,
        )
        # self.BCE = torch.nn.BCEWithLogitsLoss(
        #     weight=torch.Tensor(weights).reshape(len(weights), 1, 1), reduction="none"
        # )
        # self.BCE = torch.nn.BCEWithLogitsLoss(
        #     weight=torch.Tensor(weights).reshape(len(weights), 1, 1) if weights is not None else None, reduction="none"
        # )
        self.BCE = torch.nn.BCELoss(
            weight=torch.Tensor(weights).reshape(len(weights), 1, 1)
            if weights is not None
            else None,
            reduction="none",
        )
        self.alpha = alpha
        self.use_softmax = use_softmax

    def __call__(self, gt, logits):
        loss_dice = self.DiceLoss(gt, logits)
        gt_hot, probas = self.DiceLoss.filter_and_2probas(gt, logits)
        if self.BCE.weight is not None:
            if self.BCE.weight.device != gt_hot.device:
                self.BCE.to(gt_hot.device)
        loss_bce = self.BCE(probas, gt_hot)
        loss_bce = loss_bce.contiguous().view(gt.shape[0], -1).mean(1)
        loss_batch = (loss_bce * self.alpha + loss_dice) / (1 + self.alpha)

        return loss_batch


class Cmb_FL_DiceLoss(torch.nn.Module):
    """
    FocalLoss + DiceLoss,
        Note: reducef 'c' and generalized weights with 'c'

    """

    def __init__(
        self,
        alpha=0.1,
        weights=None,
        weights_dice=True,
        ignore_idc=[],
        use_softmax=False,
    ):
        # super(Cmb_FL_DiceLoss, self).__init__(ignore_idc = ignore_idc)
        super(Cmb_FL_DiceLoss, self).__init__()
        self.DiceLoss = DiceLoss(
            reduction="mb",
            weights=weights if weights_dice else None,
            use_softmax=use_softmax,
        )
        self.FLLoss = SegmFocalLoss(
            reduction="mb", weights=weights, use_softmax=use_softmax
        )
        self.alpha = alpha

    def __call__(self, gt, logits):
        loss_dice = self.DiceLoss(gt, logits)
        loss_fl = self.FLLoss(gt, logits)
        loss_fl = loss_fl.view(gt.shape[0], -1).mean(1)
        loss_dice = loss_dice.mean([1])
        self.pdice = loss_dice
        self.fl = loss_fl

        loss_batch = (loss_fl * self.alpha + loss_dice) / (1 + self.alpha)

        return loss_batch


class Cmb_FL_DiceLossSq(torch.nn.Module):
    """
    FocalLoss + DiceLoss,
        Note: reducef 'c' and generalized weights with 'c'

    """

    def __init__(
        self,
        alpha=0.1,
        weights=None,
        weights_dice=True,
        ignore_idc=[],
        use_softmax=False,
    ):
        # super(Cmb_FL_DiceLoss, self).__init__(ignore_idc = ignore_idc)
        super(Cmb_FL_DiceLossSq, self).__init__()
        self.DiceLoss = DiceLossSq(
            reduction="sbc",
            weights=[x**2 for x in weights] if weights_dice else None,
            use_softmax=use_softmax,
        )
        self.FLLoss = SegmFocalLoss(
            reduction="mb", weights=weights, use_softmax=use_softmax
        )
        self.alpha = alpha

    def __call__(self, gt, logits):
        loss_dice = self.DiceLoss(gt, logits)
        loss_fl = self.FLLoss(gt, logits)
        loss_fl = loss_fl.view(gt.shape[0], -1).mean(1)
        loss_dice = loss_dice.mean([1])
        self.pdice = loss_dice
        self.fl = loss_fl

        loss_batch = (loss_fl * self.alpha + loss_dice) / (1 + self.alpha)

        return loss_batch


# class HardCoded_SegmFL_DiceLoss(_BaseLoss):
# """
# SegmFocalLoss + DiceLoss,
#     Note: reducef 'c' and generalized weights with 'c'
#
# """
#
# def __init__(self, alpha=1, weights=None, ignore_idc = []):
#     super(HardCoded_SegmFL_DiceLoss, self).__init__(ignore_idc = ignore_idc)
#     self.DiceLoss = DiceLoss(ignore_idc = ignore_idc, reduction='mb', weights=weights)
#     self.FLLoss = SegmFocalLoss(ignore_idc = ignore_idc, reduction='mb', weights=weights)
#     self.alpha = alpha
#
# def __call__(self, gt, logits):
#     loss_dice = self.DiceLoss(gt, logits)
#     loss_fl = self.FLLoss(gt, logits)
#     self.dice = loss_dice
#     self.segfl = loss_fl
#
#     loss_batch = (loss_fl * self.alpha + loss_dice)/(1 + self.alpha)
#
#     return loss_batch


class CombineLosses(Module):
    """
    Combine 2 losses result will be: :math:`crit = crit1 + \alpha * crit2`

    """

    def __init__(self, crit1, crit2, alpha=0.1):
        super().__init__()
        self.crit1 = crit1
        self.crit2 = crit2
        self.alpha = alpha

    def __call__(self, gt, logits):
        loss = (
            self.crit1(gt, logits).mean() + self.alpha * self.crit2(gt, logits).mean()
        )
        return loss
