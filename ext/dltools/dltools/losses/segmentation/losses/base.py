from warnings import warn

from typing import Union, Tuple
import torch
from torch import einsum
from torch.nn import functional as F


class _BaseLoss(torch.nn.Module):
    """
    Abstract class which impement different reductions, weighting, generalizing, and filtering
        Args:
            @param reduction: one of ['s','m','n','mc','mb','mbc','sb','sc','sbc']
                in abbreviations:
                    'm': mean
                    's': sum
                    'b': will not reduce batch dimension (must be first dim)
                    'c': will not reduce class dimension (must be second)
                    'n': will not reduce at all
                Note: for Dice loss and Jaccard loss 'n' reducef can't be used. Generally reducef 'n' can be used
                only for pointwise losses. For  CE, or losses mixed with CE reducef 'c' can't be applied,
                because it turns to BCE.
            @param weights: list or torch.Tensor of classes weights, used to recalibrate loss.
            @param ignore_idc: class idc, or list of idcs to ignore, use fuzzy indexing. Only 0 or positive indexes
                can be used. For now cant'be used with pseudo ignore_idc
            @param pseudo_ignore_idc: class idc, or list of idcs to ignore, only 0 or positive indexes can be used.
                Weight in loss will be 0, but it's plane will be used in normalization (in softmax)
                ____________________________________________________________________________________________________
                Note: for this class idc also must have plane in logits C dimensions. If this option used
                    automatically set's use_softmax to True
            @param use_softmax: flag to use softmax for raw logits, or not to use if softmax was used already

    """

    workdims = "bcwh"

    def create_reduction(self, workdims=None, redefine_rstr=None):

        rstr = self.rstr if redefine_rstr is None else redefine_rstr
        workdims = self.workdims if workdims is None else workdims
        assert (
            rstr[0] in ["s", "m", "n"]
            if len(rstr) == 1
            else all(
                [
                    rstr[0] in ["s", "m", "n"],
                    rstr[1:] in ["c", "b", "bc"],
                    *[char in workdims for char in rstr[1:]],
                ]
            )
        ), (
            f"{rstr} is wrong reducef mode or some"
            f" of dims not in workdims = {workdims}!"
        )
        if rstr == "n":
            return lambda x: x
        else:
            if rstr[0] == "m":
                reduce_f = lambda x, args: x.mean(args) if len(args) > 0 else x.mean()
            elif rstr[0] == "s":
                reduce_f = lambda x, args: x.sum(args) if len(args) > 0 else x.sum()
            reduce_dims = [i for i, char in enumerate(workdims) if char not in rstr]
            reduce_function = lambda x: reduce_f(x, reduce_dims)

        return reduce_function

    def __init__(
        self,
        reduction="n",
        weights=None,
        ignore_idc=[],
        pseudo_ignore_idc=[],
        use_softmax=True,
    ):
        super(_BaseLoss, self).__init__()
        # self.reducef = (self,)
        self.rstr = reduction
        self.reducef = self.create_reduction()
        if weights is not None:
            if isinstance(weights, list):
                self.weights = torch.Tensor(weights).reshape(1, len(weights), 1, 1)
            elif isinstance(weights, torch.Tensor):
                assert weights.ndimension() in [1, 2, 4], (
                    "If weights passed like torch.Tensor, "
                    "that tensor must have 1,2 or 4 dimensions"
                )
                if weights.ndimension() == 2:
                    assert weights.shape[0] == 1
                    weights = weights.reshape(weights.shape + (1, 1))
                if weights.ndimension() == 1:
                    weights = weights.reshape(1, len(weights), 1, 1)
                if weights.ndimension() == 4:
                    assert all([weights.shape[i] == 1 for i in [0, 2, 3]])
                self.register_buffer("weights", weights, persistent=False)

        assert (
            isinstance(ignore_idc, (list, int)) and all([x >= 0 for x in ignore_idc])
            if isinstance(ignore_idc, list)
            else ignore_idc >= 0
        )
        ignore_idc = (
            [ignore_idc] if not isinstance(ignore_idc, list) else ignore_idc
        )  # for better slicing

        self.ignore_idc = ignore_idc

        self.use_softmax = use_softmax
        assert (
            isinstance(pseudo_ignore_idc, (list, int))
            and all([x > 0 for x in pseudo_ignore_idc])
            if isinstance(pseudo_ignore_idc, list)
            else pseudo_ignore_idc >= 0
        )
        pseudo_ignore_idc = (
            [pseudo_ignore_idc]
            if not isinstance(pseudo_ignore_idc, list)
            else pseudo_ignore_idc
        )
        self.pseudo_ignore_idc = pseudo_ignore_idc
        if len(pseudo_ignore_idc) > 0:
            warn("Use softmax changed to True, as result of using preudo_ignore_idc!")
            self.use_softmax = True
        else:
            self.use_softmax = use_softmax
        if len(pseudo_ignore_idc) > 0 and len(ignore_idc) > 0:
            raise NotImplementedError

    #     @staticmethod
    def apply_weights(
        self,
        *args: Tuple[Union[torch.FloatTensor, torch.LongTensor]],
        weights: torch.FloatTensor = None,
    ):
        """
        Method to statically or dynamically apply weights to tensor or tuple of tensors  if weights exist
        Args:
            tensor or tuple of tensors. Shape [BxCxWxH]
            weights: optional, for dynamic weights calculations.
        Note:
            if weights passed, weights shape can be either [1xCx1x1] or [BxCx1x1].
        """
        if weights is None and hasattr(self, "weights"):
            assert self.weights is not None
            weights = self.weights
            if weights.device != args[0].device:
                weights = weights.to(args[0].device)
            for ii, tensor in enumerate(args):
                assert all(
                    [
                        any(
                            [
                                weights.ndimension() == tensor.ndimension(),
                                weights.ndimension() + 1 == tensor.ndimension(),
                            ]
                        ),
                        weights.shape[1] == tensor.shape[1],
                    ]
                ), f"""
                Weights ndim not as input {ii} ndim or 1 dimenstion didnt match:
                w ndim:{weights.ndimension()}, input ndim: {tensor.ndimension()},
                weights shape: {weights.shape}, input shape: {tensor.shape}
                """
            return (
                tuple([tensor * weights for tensor in args])
                if len(args) != 1
                else tensor * weights
            )
        elif weights is not None:
            for ii, tensor in enumerate(args):
                assert all(
                    [
                        weights.ndimension() == tensor.ndimension(),
                        #                             weights.shape[1] == tensor.shape[1],
                        #                             weights.shape[0] == tenspr.shape[0],
                    ]
                ), f"""
                Weights ndim not as input {ii} ndim or 1 dimenstion didnt match:
                w ndim:{weights.ndimension()}, input ndim: {tensor.ndimension()},
                weights shape: {weights.shape}, input shape: {tensor.shape}
                """
            return (
                tuple([tensor * weights for tensor in args])
                if len(args) != 1
                else tensor * weights
            )
        else:
            return args if len(args) != 1 else args[0]

    def forward(
        self, gt: Union[torch.LongTensor, torch.ByteTensor], pred: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Must be implemented by child class of Losses"""

        raise NotImplementedError

    @staticmethod
    def generalized_weights(
        onehot: Union[torch.ByteTensor, torch.LongTensor, torch.FloatTensor],
        by: str = "bc",
        gamma: float = -2,
        eps: float = 1e-10,
    ):
        """Function to compute weights dynamically
        Args:
            onehot: one hot encoded ground true
            by: one of 'b','c', 'bc'. By which dimension must be generalized.
            eps: added to denominator for numerical stability
            gamma: power of weights, default - 2
        weights = 1 / (reducef(onehot,'s' + by) + eps)^2
        """
        warn(
            f"Note: After generalization by {by} dimeshions sum by this dimension or dimensions allways be equal to 1"
        )
        assert len(onehot.shape) > 2
        assert by in ["c", "bc", "b"]
        if by == "b":
            warn(
                "Generalizing only by batch dimension is Bad practice for segmentation,"
                " but can be good for some gan applications"
            )
        #         weights =  1 / (( onehot.sum([i for i in range(onehot.ndimension()) if i not in [0,1]]) + 1e-10) ** 2)
        # weights = (onehot.sum()) / ((einsum(f"bcwh->{by}", onehot).type(torch.float32) + eps) ** 2)
        weights = (einsum(f"bcwh->{by}", onehot).type(torch.float32) + eps) ** gamma
        weights = weights.reshape(
            [onehot.shape[i] if char in by else 1 for i, char in enumerate("bcwh")]
        )

        return weights

    def filter_and_2probas(
        self,
        gt: Union[torch.LongTensor, torch.ByteTensor],
        logits: torch.FloatTensor,
        redefine_use_softmax: bool = False,
        positive_filter: bool = True,
    ):
        """
        Method to filter out from raw logits and gt's ingored idcs or pseudo ignored idcs, logits will normalize in
        C dimension by softmax if use_softmax flag passed. GT represnted in 'one hotted' format of shape [BxCxWxH]
         with sparse dimension C (there only one nonzero element that equal 1), where all values can be just 0 or 1.
        Args:
            @param gt: ground True of shape [BxHxW]
            @param logits: raw output of model with shape [BxCxHxW]
            @param positive_filter: filter out all classes with negative id
        Returns:
            gt_hot, probas
        """
        use_softmax = (
            self.use_softmax if not redefine_use_softmax else redefine_use_softmax
        )
        assert all([logits.ndimension() == 4, gt.ndimension() == 3])

        assert any([hasattr(self, "num_classes"), logits is not None])
        num_classes = (
            self.num_classes
            if hasattr(self, "num_classes")
            else logits.shape[1] + len(self.ignore_idc)
        )
        # Filter Logits
        if len(self.ignore_idc) > 0:
            # Questionable filtration. Ignore idc need to filter out only from onehotted gt's,
            # but can work if cls_idc on top of gt
            # Edit: Needed to apply ignored mask anyway, that's why lesser computation needed
            logits = logits[
                :,
                [
                    cl_idc
                    for cl_idc in range(num_classes)
                    if cl_idc not in self.ignore_idc
                ],
                ...,
            ]
            probas = F.softmax(logits, dim=1) if use_softmax else logits
        if not positive_filter:
            gt_hot = torch.eye(num_classes + len(self.ignore_idc), device=gt.device)[
                gt.squeeze(1)
            ]
            gt_hot = (
                gt_hot.permute(0, 3, 1, 2).float().type(logits.type()).to(logits.device)
            )

            if len(self.ignore_idc) > 0:
                gt_hot = gt_hot[
                    :,
                    [
                        cl_idc
                        for cl_idc in range(num_classes)
                        if cl_idc not in self.ignore_idc
                    ],
                    ...,
                ]
                # ignored_mask = (gt_hot[:, [self.ignore_idc], ...]).sum(1) == 0
                # gt_hot = gt_hot[:, [cl_idc for cl_idc in range(num_classes) if cl_idc not in self.ignore_idc], ...]
                # probas = F.softmax(logits * ignored_mask.float(), dim=1) if use_softmax else logits * ignored_mask.float()
            elif len(self.pseudo_ignore_idc) > 0:
                gt_hot = gt_hot[
                    :,
                    [
                        cl_idc
                        for cl_idc in range(num_classes)
                        if cl_idc not in self.pseudo_ignore_idc
                    ],
                    ...,
                ]
                probas = F.softmax(logits, dim=1)
                probas = probas[
                    :,
                    [
                        cl_idc
                        for cl_idc in range(num_classes)
                        if cl_idc not in self.pseudo_ignore_idc
                    ],
                    ...,
                ]
            else:
                probas = F.softmax(logits, dim=1) if use_softmax else logits
        else:
            # shifting all to + 1, and filtering out first dim
            gt_hot = torch.eye(num_classes + 1, device=gt.device)[
                gt.squeeze(1) + 1
            ].permute(0, 3, 1, 2)

            # Applying ignored mask
            if not use_softmax:
                probas = logits * (1 - gt_hot[:, 0:1, :, :])
            else:
                probas = F.softmax(logits, dim=1) * (1 - gt_hot[:, 0:1, :, :])
            # filtering shifted dim:
            gt_hot = gt_hot[:, 1:, :, :]
        return gt_hot, probas
