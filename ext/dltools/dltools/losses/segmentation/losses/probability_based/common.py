import torch

from .. import _BaseLoss

math = "/math:"

# class CELoss(_BaseLoss):
#     workdims = 'bwh'
#     def __init__(self,
#                  weights=None,
#                  generalized_weights=False,
#                  reduction='m',
#                  ignore_idc=[],
#                  eps=1e-3,
#                  use_softmax=False,
#                  pseudo_ignore_idc=[],
#                  ):
#         assert 'c' not in reduction, "reducef 'c' cant be used with CE type Losses"
#         super(CELoss, self).__init__(reduction=reduction,
#                                      weights=weights,
#                                      ignore_idc=ignore_idc,
#                                      pseudo_ignore_idc=pseudo_ignore_idc,
#                                      use_softmax=use_softmax)
#         self.min_reduction = 'sb'
#
#         self.eps = eps
#         # assert generalized_weights in [False, 'c', 'bc', 'b']
#         assert generalized_weights in [False, 'c', '']
#         self._genwts = generalized_weights
#         assert not all([hasattr(self, 'weights'), self._genwts])
#
#     def __call__(self, gt, logits):
#         gt_hot, probas = self.filter_and_2probas(gt, logits)
#
#         if hasattr(self, 'weights') and not (self._genwts):
#             if self.weights.device != gt_hot.device:
#                 self.weights = self.weights.to(gt_hot.device)
#             if self.weights.ndim == 2 and gt_hot.shape[:2] == self.weights.shape:
#                 mult_mask = (self.weights.reshape(self.weights.shape, 1, 1) * gt_hot).sum(1)
#             elif self.weights.ndim == 4 and self.weights.shape[1] == probas.shape[1]:
#                 mult_mask = (self.weights * gt_hot).sum(1)
#             elif self.weights.ndim == 3 and self.weights.shape == probas.shape[1:]:
#                 mult_mask = (self.weights.unsqueeze(1) * gt_hot).sum(1)
#             elif self.weights.ndim == 1 and gt_hot.shape[1] == self.weights.shape[0]:
#                 mult_mask = (self.weights.reshape(1, self.weights.shape[0], 1, 1) * gt_hot).sum(1)
#             else:
#                 raise TypeError(
#                     f"self.weights didn't have 1,3,4 dims, or shapes of gt and weights didn't match: " \
#                     f"weights shape :{self.weights.shape}, gt_hot shape: {gt_hot.shape}")
#             loss = cross_entropy(probas, gt, reduction = 'none')
#             loss = loss * mult_mask
#         elif self._genwts:
#             if self._genwts == 'c':
#                 weights = 1 / (torch.pow(gt_hot.sum([0, 2, 3], 2)) + self.eps)
#                 loss = cross_entropy(probas, gt, weights = weights)
#             else:
#                 raise NotImplementedError(
#                     f'Generalization {self._genwts} for this loss not implemented, use WEWeighter classes instead')
#         else:
#             loss = cross_entropy(probas, gt, reduction='none', ignore_index = -1)
#         loss = self.reducef(loss.unsqueeze(1))
#         return loss


class SegmFocalLoss(_BaseLoss):
    workdims = "bwh"
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*weights*(1-pt)*log(pt)

    :param weights: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
        Note: ignored class must be in weights with weight 0, :: Maybe we need to adjust dim in pt?
    """

    def __init__(
        self,
        weights=None,
        gamma=2,
        reduction="m",  # main params
        use_softmax=False,
        ignore_idc=[],
        pseudo_ignore_idc=[],  # subclass params
    ):
        super(SegmFocalLoss, self).__init__(
            reduction=reduction,
            weights=weights,
            ignore_idc=ignore_idc,
            pseudo_ignore_idc=pseudo_ignore_idc,
            use_softmax=use_softmax,
        )
        if weights is not None:
            self.num_classes = len(weights)

        self.gamma = gamma
        self.eps = 1e-6
        self.reduction = reduction
        self.minreducef = lambda x: x.sum(
            1
        )  # sum by class dimension. If we now want sum we need to use FocalBinaryLoss for each class

    def forward(self, gt, logit):
        gt_hot, probas = self.filter_and_2probas(gt, logit)

        probas = probas.softmax(1) if self.use_softmax else logit
        # HotFix
        # if probas.shape[1] != self.num_classes:
        #     raise RuntimeError(f"dim 1 of logits: [{probas.shape[1]}] didn't match num classes: {len(self.alpha)}")

        probas = (probas + self.eps) * gt_hot
        logpt = (probas + (1 - gt_hot)).log()

        logpt = self.apply_weights(logpt)
        loss = -1 * torch.pow(1.0 - probas, self.gamma) * logpt
        loss = self.minreducef(loss)
        loss = self.reducef(loss)
        return loss


class CELoss(_BaseLoss):
    workdims = "bwh"
    """
    This is a implementation of cross entropy which supports ignoring class

    :param weights: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example

    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
        Note: ignored class must be in weights with weight 0, :: Maybe we need to adjust dim in pt?
    """

    def __init__(
        self,
        weights=None,
        reduction="m",  # main params
        use_softmax=False,
        ignore_idc=[],
        pseudo_ignore_idc=[],  # subclass params
    ):
        super(CELoss, self).__init__(
            reduction=reduction,
            weights=weights,
            ignore_idc=ignore_idc,
            pseudo_ignore_idc=pseudo_ignore_idc,
            use_softmax=use_softmax,
        )
        if weights is not None:
            self.num_classes = len(weights)

        self.eps = 1e-6
        self.reduction = reduction
        self.minreducef = lambda x: x.sum(
            1
        )  # sum by class dimension. If we now want sum we need to use BCELoss for each class

    def forward(self, gt, logit):
        gt_hot, probas = self.filter_and_2probas(gt, logit)

        probas = probas.softmax(1) if self.use_softmax else logit
        # HotFix
        # if probas.shape[1] != self.num_classes:
        #     raise RuntimeError(f"dim 1 of logits: [{probas.shape[1]}] didn't match num classes: {len(self.alpha)}")
        # print(gt.unique())
        probas = (probas + self.eps) * gt_hot
        logpt = (probas + (1 - gt_hot)).log()

        logpt = self.apply_weights(logpt)
        loss = -1 * logpt
        loss = self.minreducef(loss)
        loss = self.reducef(loss)
        return loss
