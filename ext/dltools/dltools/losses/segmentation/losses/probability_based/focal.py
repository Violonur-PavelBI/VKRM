import numpy as np
import torch
import torch.nn as nn


class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*weights*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when weights is float
    """

    def __init__(self, alpha=[1.0, 1.0], gamma=2, ignore_index=None, reduction="mean"):
        super(BinaryFocalLoss, self).__init__()
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ["none", "mean", "sum"]

        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = np.asarray(self.alpha)
            self.alpha = np.reshape(self.alpha, 2)
            assert (
                self.alpha.shape[0] == 2
            ), "the `weights` shape is not match the number of class"
        elif isinstance(self.alpha, (float, int)):
            self.alpha = np.asarray(
                [self.alpha, 1.0 - self.alpha], dtype=np.float
            ).view(2)

        else:
            raise TypeError("{} not supported".format(type(self.alpha)))

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_loss = (
            -self.alpha[0]
            * torch.pow(torch.sub(1.0, prob), self.gamma)
            * torch.log(prob)
            * pos_mask
        )
        neg_loss = (
            -self.alpha[1]
            * torch.pow(prob, self.gamma)
            * torch.log(torch.sub(1.0, prob))
            * neg_mask
        )

        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

        if num_pos == 0:
            loss = neg_loss
        else:
            loss = pos_loss / num_pos + neg_loss / num_neg
        return loss


class FocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*weights*(1-pt)*log(pt)
    :param weights: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example.
    """

    def __init__(
        self,
        weights=[0.25, 0.75],
        gamma=2,
        balance_index=-1,
        reduction="mean",
        use_softmax=False,
    ):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = len(weights)
        self.alpha = weights
        self.gamma = gamma
        self.eps = 1e-6
        self.reduction = reduction
        self.use_softmax = use_softmax

        if isinstance(self.alpha, (list, tuple)):
            self.alpha = torch.Tensor(list(self.alpha))
            self.alpha = (
                self.alpha if self.alpha.sum() == 1 else self.alpha / self.alpha.sum()
            )
        elif isinstance(self.alpha, (float, int)):
            assert 0 < self.alpha < 1.0, "weights should be in `(0,1)`)"
            assert balance_index > -1
            weights = torch.ones(self.num_class)
            weights *= 1 - self.alpha
            weights[balance_index] = self.alpha
            self.alpha = weights

        elif isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha

        else:
            raise TypeError(
                "Not support weights type, expect `int|float|list|tuple|torch.Tensor`"
            )

    def forward(self, target, logit):
        logit = logit.softmax(1) if self.use_softmax else logit

        if logit.shape[1] != self.num_class:
            self.num_class = logit.shape[1]
            self.alpha = torch.Tensor(
                [1 / self.num_class for i in range(self.num_class)]
            )
            print(
                f"FocalLossOri: alpha changed to array of "
                f"len {self.num_class} of values equal to {1 / self.num_class:.4f}"
            )
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]

        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        pt = logit.gather(1, target).view(-1) + self.eps  # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            self.alpha = self.alpha.to(logpt.device)
        alpha_class = self.alpha.gather(0, target.view(-1))
        logpt = alpha_class * logpt
        loss = -1 * torch.pow(1.0 - pt, self.gamma) * logpt

        if self.reduction == "none":
            loss = loss
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss.mean()
        return loss
