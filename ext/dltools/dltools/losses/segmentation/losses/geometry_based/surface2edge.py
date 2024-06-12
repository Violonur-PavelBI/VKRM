from functools import reduce

import torch
from torch.nn import functional as F
from typing import Union, Tuple, List
from .. import _BaseLoss
from torch import jit

@jit.script
def simple_line(dist:torch.FloatTensor, local_metric: torch.FloatTensor):
    return dist * (1-local_metric)

@jit.script
def interval_line(dist, local_metric, max_dist):
    return (1 - local_metric) * torch.clamp(dist, -1 * max_dist, max_dist)

lines = {
    "simple" :simple_line,
    "interval": interval_line
    }


class Surf2EdgeLoss(_BaseLoss):
    def __init__(
        self,
        window_size,
        reduction="m",
        weights=None,
        generalized_weights=False,
        ignore_idc=[],
        pseudo_ignore_idc=[],
        window_padding_mode="reflect",
        num_classes=None,
        border_height=None,
        border_width=None,
        max_dist=None,
        device=None,
        window_mode="simple",
        use_softmax=False,
    ):
        """
        Computes Surf2Edge Loss.
        Args:
            @param window_size : int,list or tuple of odd ints. Size of running mean window of pixel accuracy.
                Pixel Accuracy calculates for each plane of raw logits, by using functional conv2d
                with filter all values equals 1/S(window_size),where S is surface of window,if 'simple' window mode used.
            @param reduction : one of ['s','m','mc','mb','mbc','sb','sc','sbc'].
                For more information look BaseLoss reducef.
            @param weights:  a tensor,List or np.array of shape [ C ].
            @param generalized_weights: one of ['b','c','bc'] or False, None. Parameter which controls how to compute
                weights dynamically, typically it will reduce one-hotted labels to [B] ,[C],[B x C] shape tensor, and if
                passed True, static weights ignored.
            @param ignore_idc: class idc, or list of idcs to ignore, use fuzzy indexing. Only 0 or positive indexes can
                be used. Now cant'be used with pseudo ignore_idc
            @param pseudo_ignore_idc: class idc, or list of idcs to ignore, only nonegative integers can be used.
                Weight in loss will be 0, but it's plane will be used in normalization (in softmax)
                Note: for this class idc also must have plane in logits C dimensions. If this option used then flag
                    use_softmax set's to True
            @param window_padding_mode: how to pad. one of torch.nn.Conv2d padding modes
            @param num_classes: number of classes, if passed, then loss called filter will be allready set to
                (num_classes,1, window_size[0],window_size[1]) shape, else filter will be repeated in logits C dimension
                (second dimension,logits.shape[1]), then method __call__ called.
            @param border_height: python scalar. Describes height of loss amplification on edge if running acc-> 1.
            @param border_width: python scalar. Describes width of edge in distance units.
            @param max_dist: python scalar, describes max loss amplification by distance in distance units.
            @param device: torch.device. If passed filter weights will be placed on this device, else default torch
                behavior will be used.
            @param window_mode: For now other running means modes not implemented yet. Controls filter values for
                different types of running means.
            @param use_softmax : flag, controls raw logits comed, or probas normalized in C dimension.

        """

        super(Surf2EdgeLoss, self).__init__(
            reduction=reduction,
            weights=weights,
            ignore_idc=ignore_idc,
            pseudo_ignore_idc=pseudo_ignore_idc,
            use_softmax=use_softmax,
        )

        if window_mode != "simple":
            raise NotImplementedError
        self.w_pad_mode = window_padding_mode
        self.bh = border_height if border_height else window_size
        self.bw = border_width if border_width else window_size / 2

        assert all([window_size % 2 == 1, self.bw <= window_size / 2])

        self._genwts = generalized_weights
        assert not all([hasattr(self, "weights"), self._genwts])

        if isinstance(window_size, int):
            window_size = (window_size, window_size)
            window_size: Tuple[int,int]
        elif isinstance(window_size, (list, tuple)):
            assert len(window_size) == 2
        else:
            raise TypeError("Window size must be int, or one of list or tuple")
        self.window_size = window_size

        _filter = torch.ones(
            (1, 1, *window_size),
            dtype=torch.float32,
            requires_grad = False
        ) / (
            (self.window_size[0] * self.window_size[1])
        )
        self.register_buffer("filter", _filter, False)
        self.filter: torch.Tensor
        self.filter = self.filter.to(device) if device else self.filter
        self.max_dist = max_dist
        self.by_dist = lines["interval"] if self.max_dist else lines["simple"]
        self.paddings = self._get_paddings(window_size)


    @torch.no_grad()
    def local_accuracy(self, gt_hot:torch.IntTensor, probas:torch.FloatTensor)-> torch.FloatTensor:
        """Calculated window mean of accuracy

        Args:
            gt_hot (torch.IntTensor): ground truth
            probas (torch.FloatTensor): predicted probabilities


        Returns:
            torch.FloatTensor: accuracy calculated in windows
        """

        _, top_1_class = probas.topk(1, dim=1)
        top_1_class = top_1_class.squeeze(1)
        pred_hot, _ = self.filter_and_2probas(top_1_class, probas)
        equals = (gt_hot == pred_hot).float().detach()
        if self.filter.shape[0] != probas.shape[1]:
            self.filter = self.filter.repeat(probas.shape[1], 1, 1, 1)
        #         with torch.no_grad():
        equals = F.pad(
            equals,
            self._paddings,
            mode=self.w_pad_mode,
            )

        wm_pixacc = F.conv2d(
            equals,
            self.filter,
        )
        return wm_pixacc

    @staticmethod
    def _get_paddings(_windows: List[int]):
        return [
            x // 2
            for x in reduce(
                lambda x, y: (x if isinstance(x, (tuple, list)) else [x] * 2)
                + [y] * 2,
                _windows,
                )
            ],

    def _loss_localized_mask(self, dist, wm_pixacc):
        loss_mask = torch.sign(dist) * wm_pixacc / (
            1 + 1 / self.bh + torch.tanh(torch.abs(dist) - self.bw)
            ) + (
                self.by_dist(dist, wm_pixacc, self.max_dist)
                if self.max_dist
                else self.by_dist(dist, wm_pixacc)
                )
        return loss_mask


    def __call__(self, gt, logits, dist, return_loss_mask=False):

        self.filter = (
            self.filter.to(gt.device)
            if self.filter.device != gt.device
            else self.filter
        )
        assert (
            self.filter.device == logits.device == dist.device
        ), f"some of gt, logits, dist not on device {self.filter.device}"

        gt_hot, probas = self.filter_and_2probas(gt, logits)
        wm_pixacc = self.local_accuracy(gt_hot, probas)
        loss_mask = self._loss_localized_mask(dist, wm_pixacc)

        if return_loss_mask:
            return loss_mask

        loss = torch.einsum("bcwh,bcwh->bcwh", probas, loss_mask)
        #         if self.weights is not None:

        if self._genwts:
            loss = self.apply_weights(
                loss,
                weights=self.generalized_weights(gt_hot, by=self._genwts, eps=1e-2),
            )
        else:
            loss = self.apply_weights(loss)
        loss = self.reducef(loss)
        return loss
