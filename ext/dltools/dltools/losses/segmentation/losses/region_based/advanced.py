from ..base import _BaseLoss
from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import torch.distributed as dist


class TverskiyLoss(_BaseLoss):
    f"""Computes the Tverskiy loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the Tverskiy criteria so we
    return the negated Tverskiy criteria.

    Args:
        @param alpha:coefficent to balance FP, can be a tensor of size [C], which can used to different
                    balancing FP for each class
        @param beta: coefficent to balance FN, can be a tensor of size [C], which can used to different
                    balancing FN for each class
        @param weights: a tensor,List or np.array of shape [ C ].
        @param generalized_weights: parameter which controls how to compute weights dynamically,
            typically it will reduce one-hotted labels to [B] ,[C],[B x C] dimension, and if passed True,
            static weights didn't be used.
        @param reduction: one of ['s','m','mc','mb','mbc','sb','sc','sbc']
            in abbreviations:
                            'm': mean
                            's': sum
                            'b': will not reduce batch dimension (must be first dim)
                            'c': will not reduce class dimension (must be second)
                            'n': will not reduce at all (can't be used with this loss)
        @param ignore_idc: class idc, or list of idcs to ignore, use fuzzy indexing.
            Only 0 or positive indexes can be used. Now cant'be used with pseudo ignore_idc

        @param pseudo_ignore_idc: class idc, or list of idcs to ignore, only 0 or positive indexes can be used.
            Weight in loss will be 0, but it's plane will be used in normalization (in softmax)
            Note: for this class idc also must have plane in logits C dimensions.
                If this option used automatically set's use_softmax to True
        @param eps: added to the denominator for numerical stability. default eps = 1e-9
        @param smooth: a scalar, which added to numerator and denominator to handle 1 pix case
    Returns:
        tverskiy_loss: the Tverskiy loss.
    """
    workdims = "bc"

    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        smooth=1,
        weights=None,
        # generalized_weights=False,
        # reduction='mbc',
        ignore_idc=[],
        eps=1e-2,
        use_softmax=False,
        pseudo_ignore_idc=[],
    ):
        self.min_reduction = "sbc"
        self.eps = eps
        super(TverskiyLoss, self).__init__(
            # reduction=reduction,
            reduction="mbc",
            weights=weights,
            # weights=None,
            ignore_idc=ignore_idc,
            pseudo_ignore_idc=pseudo_ignore_idc,
            use_softmax=use_softmax,
        )

        self.minreduce = self.create_reduction(
            redefine_rstr=self.min_reduction, workdims="bcwh"
        )
        # self._genwts = generalized_weights

        self.reducef = self.create_reduction(redefine_rstr="n")
        # assert not all([hasattr(self, 'weights'), self._genwts])
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def __call__(self, gt, logits):
        """
        Args:
            @param gt: a tensor of shape [B, 1, H, W].
            @param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
        Returns:
            tverskiy_loss: the Tverskiy loss.
        """

        gt_hot, probas = self.filter_and_2probas(gt, logits)

        TP = probas * gt_hot.float()
        FP = (1 - gt_hot.float()) * probas
        FN = gt_hot.float() * (1 - probas)

        FP = self.minreduce(FP)  # reducing by sum on HW
        FN = self.minreduce(FN)  # reducing by sum on HW
        TP = self.minreduce(TP)  # reducing by sum on HW

        tverskiy_loss = 1 - (2.0 * TP + self.smooth) / (
            self.smooth
            + self.eps
            + self.alpha * FP
            + self.beta * FN
            + (2.0 * TP + self.smooth)
        )

        tverskiy_loss = tverskiy_loss.view(*tverskiy_loss.shape, 1, 1)

        # if hasattr(self, 'weights') and not self._genwts:
        #     tverskiy_loss = self.apply_weights(tverskiy_loss)
        # elif self._genwts:
        #     tverskiy_loss = self.apply_weights(tverskiy_loss, weights=self.generalized_weights(gt_hot, self._genwts))

        tverskiy_loss = self.reducef(tverskiy_loss.squeeze())

        return tverskiy_loss


class FocalTverskiyLoss(TverskiyLoss):
    f"""Computes the Focal Tverskiy loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the Tverskiy criteria so we
    return the negated Tverskiy criteria as loss.

    Args:
        @param alpha:coefficent to balance FP, can be a tensor of size [C], which can used to different
                    balancing FP for each class
        @param beta: coefficent to balance FN, can be a tensor of size [C], which can used to different
                    balancing FN for each class
        @param gamma: power of loss coeeficient
        @param weights: a tensor,List or np.array of shape [ C ].
        @param generalized_weights: parameter which controls how to compute weights dynamically,
            typically it will reduce one-hotted labels to [B] ,[C],[B x C] dimension, and if passed True,
            static weights didn't be used.
        @param reduction: one of ['s','m','mc','mb','mbc','sb','sc','sbc']
            in abbreviations:
                            'm': mean
                            's': sum
                            'b': will not reduce batch dimension (must be first dim)
                            'c': will not reduce class dimension (must be second)
                            'n': will not reduce at all (can't be used with this loss)
        @param ignore_idc: class idc, or list of idcs to ignore, use fuzzy indexing.
            Only 0 or positive indexes can be used. Now cant'be used with pseudo ignore_idc

        @param pseudo_ignore_idc: class idc, or list of idcs to ignore, only 0 or positive indexes can be used.
            Weight in loss will be 0, but it's plane will be used in normalization (in softmax)
            Note: for this class idc also must have plane in logits C dimensions.
                If this option used automatically set's use_softmax to True
        @param eps: added to the denominator for numerical stability. default eps = 1e-9
        @param smooth: a scalar, which added to numerator and denominator to handle 1 pix case
    Returns:
        focal_tverskiy_loss: the Focal Tverskiy loss.
    """

    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        gamma=2,
        smooth=1,
        weights=None,
        # generalized_weights=False,
        # reduction='mbc',
        ignore_idc=[],
        eps=1e-2,
        use_softmax=False,
        pseudo_ignore_idc=[],
    ):
        super(FocalTverskiyLoss, self).__init__(
            alpha=alpha,
            beta=beta,
            smooth=smooth,
            weights=weights,
            # reduction=reduction,
            # weights=None,
            ignore_idc=ignore_idc,
            eps=eps,
            use_softmax=use_softmax,
            pseudo_ignore_idc=pseudo_ignore_idc,
        )

        self.gamma = gamma

    def __call__(self, gt, logits):
        """
        Args:
            @param gt: a tensor of shape [B, 1, H, W].
            @param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
        Returns:
            focal_tverskiy_loss: the Tverskiy loss in power of gamma.
        """

        tverskiy_loss = super(FocalTverskiyLoss, self).__call__(gt, logits)
        focal_tverskiy_loss = tverskiy_loss.pow(1 / self.gamma)

        return focal_tverskiy_loss


def ifscalar2tensor(x):
    if not isinstance(x, torch.Tensor):
        if isinstance(x, (float, int)):
            return torch.Tensor([x])
        elif isinstance(x, list):
            return torch.Tensor(x)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            raise NotImplemented
    else:
        return x


class ACFTverskiy(nn.Module):
    def __init__(
        self,
        num_classes=None,
        alpha=0.5,
        beta=0.5,
        gamma=1,
        delta=0.5,
        buffer_size=200,
        momentum=0.9,
        smooth=0.1,
        weights=None,
    ):
        assert num_classes is not None

        self.defaults = dict(
            alpha=ifscalar2tensor(alpha),
            beta=ifscalar2tensor(beta),
            gamma=ifscalar2tensor(gamma),
            delta=ifscalar2tensor(delta),
            buffer_size=buffer_size,
            num_classes=num_classes,
            momentum=0.9,
        )
        self.smooth = smooth
        self.buffer_offset = 0
        super().__init__()
        """
            @param alpha: float or tensor of shape [1xC], FP starting multiplier;
            @param beta: float or tensor of shape [1xC], FN starting multiplier;
            @param gamma: float or tensor of shape [1xC],TP starting multiplier;
            @param delta: float or tensor of shape [1xC], Focal coefficient (power) of Tverskiy loss;
            @param buffer_size: size of buffer of keeped FP,FN,TP tensors;
            @param num_classes: number of classes;
            @param weights: Unused, for similarity purposes
            @momentum: momentum to change sigmas
        """

        self.register_buffer("alpha", ifscalar2tensor(alpha).unsqueeze(0))
        self.register_buffer("beta", ifscalar2tensor(beta).unsqueeze(0))
        self.register_buffer("gamma", ifscalar2tensor(gamma).unsqueeze(0))
        self.register_buffer("delta", ifscalar2tensor(delta))
        self.register_buffer("buffer", torch.ones(buffer_size, num_classes, 3))
        self.momentum = momentum
        self.register_buffer("sigmas", torch.ones(num_classes, 3).float())
        if dist.is_initialized():
            self.ws = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.ws = 0
            self.rank = 0

    def reset(self):
        self.__init__(**self.defaults)

    def __call__(self, gt, probas):
        if self.alpha.device != gt.device:
            self.to(gt.device)
        bs = gt.shape[0]
        gt_hot = F.one_hot(gt + 1, probas.shape[1] + 1)[..., 1:].permute(0, -1, 1, 2)

        TP = probas * gt_hot.float()
        FP = (1 - gt_hot.float()) * probas
        FN = gt_hot.float() * (1 - probas)

        FP = FP.sum([-1, -2])  # reducing by sum on HW
        FN = FN.sum([-1, -2])  # reducing by sum on HW
        TP = TP.sum([-1, -2])  # reducing by sum on HW
        if self.training:
            memories = torch.stack([FP.detach(), FN.detach(), TP.detach()], dim=-1)
            if self.ws > 1:
                memories_list = [
                    torch.zeros_like(memories) for _ in range(dist.get_world_size())
                ]
                bs *= dist.get_world_size()
                dist.all_gather(memories_list, memories)
                memories = torch.cat(memories_list, dim=0)

            if self.buffer_offset + bs < self.buffer.shape[0]:
                self.buffer[
                    self.buffer_offset : self.buffer_offset + bs, ...
                ] = memories
                self.buffer_offset += bs

                self.sigmas = (
                    self.buffer[: self.buffer_offset].std(0) * (1 - self.momentum)
                    + self.sigmas * self.momentum
                )
            else:
                self.buffer_offset = self.buffer.shape[0]
                self.sigmas = (
                    self.buffer.std(0) * (1 - self.momentum)
                    + self.sigmas * self.momentum
                )
                self.buffer[:-bs] = self.buffer[bs:].clone()
                self.buffer[
                    self.buffer_offset - bs : self.buffer_offset, ...
                ] = memories

            alpha = (self.sigmas[..., 1].pow(2) + self.sigmas[..., 2].pow(2) + 1).pow(
                0.5
            ) / (self.sigmas[..., 0].pow(2) + 1).pow(0.5)
            beta = (self.sigmas[..., 0].pow(2) + self.sigmas[..., 2].pow(2) + 1).pow(
                0.5
            ) / (self.sigmas[..., 1].pow(2) + 1).pow(0.5)
            gamma = (self.sigmas[..., 0].pow(2) + self.sigmas[..., 1].pow(2) + 1).pow(
                0.5
            ) / (self.sigmas[..., 2].pow(2) + 1).pow(0.5)

            self.alpha = alpha.unsqueeze(0) * self.momentum + self.alpha * (
                1 - self.momentum
            )
            self.beta = beta.unsqueeze(0) * self.momentum + self.beta * (
                1 - self.momentum
            )
            self.gamma = gamma.unsqueeze(0) * self.momentum + self.gamma * (
                1 - self.momentum
            )

        tverskiy_loss = (self.alpha * FP + self.beta * FN) / (
            self.alpha * FP + self.beta * FN + 2 * self.gamma * TP + self.smooth
        )
        focal_tverskiy = tverskiy_loss.pow(self.delta)

        return focal_tverskiy


class RocVectorLoss(_BaseLoss):
    f"""Experimental RocVectorloss.
    Computes vector length to maximize TP, and minimize FP,FN as they was on sphere

    Args:
        @param alpha:coefficent to balance FP, can be a tensor of size [C], which can used to different
                    balancing FP for each class
        @param beta: coefficent to balance FN, can be a tensor of size [C], which can used to different
                    balancing FN for each class
        @param weights: a tensor,List or np.array of shape [ C ].
        @param generalized_weights: parameter which controls how to compute weights dynamically,
            typically it will reduce one-hotted labels to [B] ,[C],[B x C] dimension, and if passed True,
            static weights didn't be used.
        @param reduction: one of ['s','m','mc','mb','mbc','sb','sc','sbc']
            in abbreviations:
                            'm': mean
                            's': sum
                            'b': will not reduce batch dimension (must be first dim)
                            'c': will not reduce class dimension (must be second)
                            'n': will not reduce at all (can't be used with this loss)
        @param ignore_idc: class idc, or list of idcs to ignore, use fuzzy indexing.
            Only 0 or positive indexes can be used. Now cant'be used with pseudo ignore_idc

        @param pseudo_ignore_idc: class idc, or list of idcs to ignore, only 0 or positive indexes can be used.
            Weight in loss will be 0, but it's plane will be used in normalization (in softmax)
            Note: for this class idc also must have plane in logits C dimensions.
                If this option used automatically set's use_softmax to True
        @param eps: added to the denominator for numerical stability. default eps = 1e-9
        @param smooth: a scalar, which added to numerator and denominator to handle 1 pix case
    Returns:
        roc_vector_loss: roc_vector_loss
    """
    workdims = "bc"

    def __init__(
        self,
        reduction="mbc",
        weights=None,
        ignore_idc=[],
        eps=1e-2,
        use_softmax=False,
        pseudo_ignore_idc=[],
    ):
        self.min_reduction = "sbc"
        self.eps = eps
        super(RocVectorLoss, self).__init__(
            # reduction=reduction,
            reduction="mbc",
            # weights=weights,
            weights=None,
            ignore_idc=ignore_idc,
            pseudo_ignore_idc=pseudo_ignore_idc,
            use_softmax=use_softmax,
        )

        self.minreduce = self.create_reduction(
            redefine_rstr=self.min_reduction, workdims="bcwh"
        )

        self.reducef = self.create_reduction(redefine_rstr="n")

    def __call__(self, gt, logits):
        """
        Args:
            @param gt: a tensor of shape [B, 1, H, W].
            @param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
        Returns:
            tverskiy_loss: the Tverskiy loss.
        """

        gt_hot, probas = self.filter_and_2probas(gt, logits)

        TP = probas * gt_hot.float()
        FP = (1 - gt_hot.float()) * probas
        FN = gt_hot.float() * (1 - probas)

        FP = self.minreduce(FP)  # reducing by sum on HW
        FN = self.minreduce(FN)  # reducing by sum on HW
        TP = self.minreduce(TP)  # reducing by sum on HW
        GTK = self.minreduce(gt_hot.float())  # B x C

        FPstar = FP * GTK / GTK.sum(1, keepdim=True)
        NRK = (self.eps + 1) / (GTK + self.eps + 1) ** 2
        LK = ((GTK - TP) ** 2 + (FPstar**2 + FN**2)) * NRK
        vector_loss = self.reducef(LK)
        return vector_loss
