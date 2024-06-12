from .. import _BaseLoss
import torch


class DiceLoss(_BaseLoss):
    f"""Computes the Sørensen–Dice loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice score so we
    return the negated dice score.

    Args:

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
        dice_loss: the Sørensen–Dice loss.
    """
    workdims = "bc"

    def __init__(
        self,
        weights=None,
        generalized_weights=False,
        reduction="mbc",
        ignore_idc=[],
        eps=1e-2,
        use_softmax=False,
        pseudo_ignore_idc=[],
        smooth=0,
    ):
        self.min_reduction = "sbc"
        self.eps = eps
        super(DiceLoss, self).__init__(
            reduction=reduction,
            weights=weights,
            ignore_idc=ignore_idc,
            pseudo_ignore_idc=pseudo_ignore_idc,
            use_softmax=use_softmax,
        )
        self.minreduce = self.create_reduction(
            redefine_rstr=self.min_reduction, workdims="bcwh"
        )
        self._genwts = generalized_weights
        if reduction[1:] == self.workdims:
            self.reducef = self.create_reduction(redefine_rstr="n")
        assert not all([hasattr(self, "weights"), self._genwts])
        self.smooth = smooth

    def __call__(self, gt, logits):
        """
        Args:
            @param gt: a tensor of shape [B, 1, H, W].
            @param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """

        gt_hot, probas = self.filter_and_2probas(gt, logits)
        intersection = probas * gt_hot.float()
        cardinality = probas + gt_hot.float()
        intersection = self.minreduce(intersection)  # [B x C]
        cardinality = self.minreduce(cardinality)
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (
            self.smooth + cardinality + self.eps
        )
        dice_loss = dice_loss.view(*dice_loss.shape, 1, 1)  # [B x C x 1 x 1 ]

        if hasattr(self, "weights") and not self._genwts:
            dice_loss = self.apply_weights(dice_loss)
        elif self._genwts:
            dice_loss = self.apply_weights(
                dice_loss, weights=self.generalized_weights(gt_hot, self._genwts)
            )

        dice_loss = self.reducef(dice_loss)
        # dice_loss = self.reducef((2. * intersection + self.eps)/ (cardinality + self.eps))
        return dice_loss


class JaccardLoss(_BaseLoss):
    f"""Computes the Jaccard loss, a.k.a the IoU loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard criteria so we
    return the negated jaccard criteria..

    Args:

        @param weights: a tensor,List or np.array of shape [ C ].
        @param generalized_weights: one of 'b','c','bc'or False. parameter which controls how to compute weights
            dynamically, typically it will reduce one-hotted labels by sum to one of [B] ,[C] or [B x C] dimension,
            and if passed True, static weights didn't be used.
            Look _BaseLoss.generalized_weights method for more information.
        @param reduction: one of ['s','m','mc','mb','mbc','sb','sc','sbc']
            in abbreviations:
                            'm': mean
                            's': sum
                            'b': will not reduce batch dimension (must be first dim)
                            'c': will not reduce class dimension (must be second)
                            'n': will not reduce at all (can't be used with this loss)
        @param ignore_idc: class idc, or list of idcs to ignore, use fuzzy indexing. Only 0 or positive indexes can
                be used. Now cant'be used with pseudo ignore_idc
        @param pseudo_ignore_idc: class idc, or list of idcs to ignore, only 0 or positive indexes can be used.
                Weight in loss will be 0, but it's plane will be used in normalization (in softmax)
            Note: for this class idc also must have plane in logits C dimensions.
                If this option used automatically set's use_softmax to True
        @param eps: added to the denominator for numerical stability. default eps = 1e-9
        @param smooth: a scalar, which added to numerator and denominator to handle 1 pix case
    Returns:
        jacc_loss: the Jaccard loss.."""

    workdims = "bc"

    def __init__(
        self,
        weights=None,
        generalized_weights=False,
        reduction="mbc",
        ignore_idc=[],
        eps=1e-2,
        use_softmax=False,
        pseudo_ignore_idc=[],
        smooth=0,
    ):

        self.min_reduction = "sbc"
        self.eps = eps
        super(JaccardLoss, self).__init__(
            reduction=reduction,
            weights=weights,
            ignore_idc=ignore_idc,
            pseudo_ignore_idc=pseudo_ignore_idc,
            use_softmax=use_softmax,
        )

        self.minreduce = self.create_reduction(
            workdims="bcwh", redefine_rstr=self.min_reduction
        )
        self._genwts = generalized_weights
        assert not all([hasattr(self, "weights"), self._genwts])
        if reduction[1:] == self.workdims:
            self.reducef = self.create_reduction(redefine_rstr="n")
        self.smooth = smooth

    def __call__(self, gt, logits):
        """
            Args:
            gt: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or probas of the model output.
            eps: added to the denominator for numerical stability.

        Returns:
            jacc_loss: the Jaccard loss.
        """

        gt_hot, probas = self.filter_and_2probas(gt, logits)
        intersection = probas * gt_hot.float()
        cardinality = probas + gt_hot.float()
        union = cardinality - intersection
        intersection = self.minreduce(intersection)
        union = self.minreduce(union)
        # jacc_loss = self.reducef(((intersection +self.eps)/ (union + self.eps)))
        jacc_loss = 1 - (intersection + self.smooth) / (self.smooth + union + self.eps)
        jacc_loss = jacc_loss.view(*jacc_loss.shape, 1, 1)
        if hasattr(self, "weights") and not self._genwts:
            jacc_loss = self.apply_weights(jacc_loss)
        elif self._genwts:
            jacc_loss = self.apply_weights(
                jacc_loss, weights=self.generalized_weights(gt_hot, self._genwts)
            )
        jacc_loss = self.reducef(jacc_loss)

        return jacc_loss


class DiceLossSq(DiceLoss):
    def __call__(self, gt, logits):
        """
        Args:
            @param gt: a tensor of shape [B, 1, H, W].
            @param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """

        gt_hot, probas = self.filter_and_2probas(gt, logits)
        intersection = probas * gt_hot.float()  # new_changes
        cardinality = probas**2 + gt_hot.float() ** 2  # changes
        intersection = self.minreduce(intersection)
        cardinality = self.minreduce(cardinality)
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (
            self.smooth + cardinality + self.eps
        )
        dice_loss = dice_loss.view(*dice_loss.shape, 1, 1)

        if hasattr(self, "weights") and not self._genwts:
            dice_loss = self.apply_weights(dice_loss)
        elif self._genwts:
            dice_loss = self.apply_weights(
                dice_loss, weights=self.generalized_weights(gt_hot, self._genwts)
            )

        dice_loss = self.reducef(dice_loss.squeeze())
        # dice_loss = self.reducef((2. * intersection + self.eps)/ (cardinality + self.eps))
        return dice_loss


class DiceLossFS(_BaseLoss):
    f"""Computes the Sørensen–Dice loss, with shift by `epsilon` * `weights` in denominator.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.

    Args:
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
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    workdims = "bc"

    def __init__(
        self,
        weights=None,
        generalized_weights=False,
        reduction="mbc",
        ignore_idc=[],
        eps=1e-2,
        use_softmax=False,
        pseudo_ignore_idc=[],
    ):
        self.min_reduction = "sbc"
        self.eps = eps
        super().__init__(
            reduction=reduction,
            weights=weights,
            ignore_idc=ignore_idc,
            pseudo_ignore_idc=pseudo_ignore_idc,
            use_softmax=use_softmax,
        )
        self.minreduce = self.create_reduction(
            redefine_rstr=self.min_reduction, workdims="bcwh"
        )
        self._genwts = generalized_weights
        assert not all([hasattr(self, "weights"), self._genwts])
        self.freq_shift = None

    def __call__(self, gt, logits):
        """
        Args:
            @param gt: a tensor of shape [B, 1, H, W].
            @param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """

        gt_hot, probas = self.filter_and_2probas(gt, logits)
        intersection = probas * gt_hot.float()
        cardinality = probas + gt_hot.float()

        if hasattr(self, "weights") and not self._genwts:
            if self.freq_shift is None:
                num_cls = probas.shape[1]
                self.freq_shift = (
                    torch.Tensor([self.eps])
                    .repeat(num_cls)
                    .reshape(1, num_cls, 1, 1)
                    .to(probas.device)
                )
                self.freq_shift = self.apply_weights(self.freq_shift)
        elif self._genwts:
            if self.freq_shift is None:
                num_cls = probas.shape[1]
                self.freq_shift = (
                    torch.Tensor([self.eps])
                    .repeat(num_cls)
                    .reshape(1, num_cls, 1, 1)
                    .to(probas.device)
                )
                self.freq_shift = self.apply_weights(
                    self.freq_shift,
                    weights=self.generalized_weights(gt_hot, self._genwts),
                )
        else:
            self.freq_shift = self.eps
        intersection = self.minreduce(intersection)
        cardinality = self.minreduce(cardinality)
        if intersection.ndim != self.freq_shift.ndim:
            self.freq_shift = self.freq_shift.reshape(1, probas.shape[1])
        dice_loss = 2.0 * intersection / (cardinality + self.freq_shift)
        dice_loss = self.reducef(dice_loss)
        # dice_loss = self.reducef((2. * intersection + self.eps)/ (cardinality + self.eps))
        return 1 - dice_loss


class JaccardLossFS(_BaseLoss):
    f"""Computes the Jaccard loss with shift by `epsilon` * `weights` in denominator, a.k.a the IoU loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated jaccard loss..

    Args:
        @param weights: a tensor,List or np.array of shape [ C ].
        @param generalized_weights: one of 'b','c','bc'or False. parameter which controls how to compute weights
            dynamically, typically it will reduce one-hotted labels by sum to one of [B] ,[C] or [B x C] dimension,
            and if passed True, static weights didn't be used.
            Look _BaseLoss.generalized_weights method for more information.
        @param reduction: one of ['s','m','mc','mb','mbc','sb','sc','sbc']
            in abbreviations:
                            'm': mean
                            's': sum
                            'b': will not reduce batch dimension (must be first dim)
                            'c': will not reduce class dimension (must be second)
                            'n': will not reduce at all (can't be used with this loss)
        @param ignore_idc: class idc, or list of idcs to ignore, use fuzzy indexing. Only 0 or positive indexes can
                be used. Now cant'be used with pseudo ignore_idc
        @param pseudo_ignore_idc: class idc, or list of idcs to ignore, only 0 or positive indexes can be used.
                Weight in loss will be 0, but it's plane will be used in normalization (in softmax)
            Note: for this class idc also must have plane in logits C dimensions.
                If this option used automatically set's use_softmax to True
        @param eps: added to the denominator for numerical stability. default eps = 1e-9
    Returns:
        jacc_loss: the Jaccard loss.."""

    workdims = "bc"

    def __init__(
        self,
        weights=None,
        generalized_weights=False,
        reduction="mbc",
        ignore_idc=[],
        eps=1e-2,
        use_softmax=False,
        pseudo_ignore_idc=[],
    ):

        self.min_reduction = "sbc"
        self.eps = eps
        super(JaccardLossFS, self).__init__(
            reduction=reduction,
            weights=weights,
            ignore_idc=ignore_idc,
            pseudo_ignore_idc=pseudo_ignore_idc,
            use_softmax=use_softmax,
        )

        self.minreduce = self.create_reduction(
            workdims="bcwh", redefine_rstr=self.min_reduction
        )
        self._genwts = generalized_weights
        self.freq_shift = None
        assert not all([hasattr(self, "weights"), self._genwts])

    def __call__(self, gt, logits):
        """
            Args:
            gt: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or probas of the model output.
            eps: added to the denominator for numerical stability.

        Returns:
            jacc_loss: the Jaccard loss.
        """

        gt_hot, probas = self.filter_and_2probas(gt, logits)
        intersection = probas * gt_hot.float()
        cardinality = probas + gt_hot.float()
        union = cardinality - intersection

        if hasattr(self, "weights") and not self._genwts:
            if self.freq_shift is None:
                num_cls = probas.shape[1]
                self.freq_shift = (
                    torch.Tensor([self.eps])
                    .repeat(num_cls)
                    .reshape(1, num_cls, 1, 1)
                    .to(probas.device)
                )
                self.freq_shift = self.apply_weights(self.freq_shift)
        elif self._genwts:
            if self.freq_shift is None:
                num_cls = probas.shape[1]
                self.freq_shift = (
                    torch.Tensor([self.eps])
                    .repeat(num_cls)
                    .reshape(1, num_cls, 1, 1)
                    .to(probas.device)
                )
                self.freq_shift = self.apply_weights(
                    self.freq_shift,
                    weights=self.generalized_weights(gt_hot, self._genwts),
                )
        else:
            self.freq_shift = self.eps

        intersection = self.minreduce(intersection)
        union = self.minreduce(union)
        if intersection.ndim != self.freq_shift.ndim:
            self.freq_shift = self.freq_shift.reshape(1, probas.shape[1])
        # jacc_loss = self.reducef(((intersection +self.eps)/ (union + self.eps)))
        jacc_loss = intersection / (union + self.freq_shift)
        jacc_loss = self.reducef(jacc_loss)
        jacc_loss = 1 - jacc_loss
        return jacc_loss
