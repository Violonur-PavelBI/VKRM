from ..base import _BaseLoss
import torch
from typing import Union, List


class DRCE(_BaseLoss):
    EPS = 1e-6
    workdims = "bwh"

    def __init__(
        self,
        weights=None,
        alphas: Union[torch.FloatTensor, float, None, List[float]] = None,
        betas: Union[torch.FloatTensor, float, List[float]] = 3.0,
        reduction="m",  # main params
        use_softmax=False,
        ignore_idc=[],
        pseudo_ignore_idc=[],  # subclass params
    ):

        super(_BaseLoss, self).__init__(
            reduction=reduction,
            weights=weights,
            ignore_idc=ignore_idc,
            pseudo_ignore_idc=pseudo_ignore_idc,
            use_softmax=use_softmax,
        )

        self.minreducef = lambda x: x.sum(
            1
        )  # sum by class dimension. If we now want sum we need to use FocalBinaryLoss for each class
        self.alphas = alphas
        self.betas = betas

    def forward(
        self, gt: Union[torch.LongTensor, torch.ByteTensor], pred: torch.FloatTensor
    ) -> torch.FloatTensor:
        gt_hot, probas = self.filter_and_2probas(gt, pred)
        probas = probas.softmax(1) if self.use_softmax else pred
        if self.alphas is None:
            # 1 x C x 1 x 1
            self.alphas = torch.FloatTensor(
                [1 / probas.shape[1] for _ in range(probas.shape[1])],
                device=probas.device,
            ).reshape(1, probas.shape[1], 1, 1)

        if not isinstance(self.betas, torch.FloatTensor):
            # 1 x C x 1 x 1
            self.betas = torch.FloatTensor(
                [self.betas] * probas.shape[1], device=probas.device
            ).reshape(1, probas.shape[1], 1, 1)

        # TODO: fix where only trues
        lesser_mask = (probas <= self.alphas).float()
        left_probs = probas * lesser_mask
        right_probs = probas * (1 - lesser_mask).float()
        log_pt = (
            left_probs + (right_probs + self.betas) / (self.betas + 1) + (1 - gt_hot)
        ).log()
        loss = -1 * log_pt
        loss = self.minreducef(loss)
        loss = self.reducef(loss)
        return loss
