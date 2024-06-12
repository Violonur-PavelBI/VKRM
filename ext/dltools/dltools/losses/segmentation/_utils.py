from functools import reduce
from typing import Set, Iterable

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as distance
from torch import Tensor


def smul(shape):
    size = 1
    for dim in shape:
        size *= dim
    return size


smul = lambda x: (reduce((lambda y, z: y * z), x))


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def simplex(t: Tensor, axis=1) -> bool:
    """Function to check,that on axis sum of elements is close to 1. Checks onehots and probs after Softmax.
    If it return False, this mean that this tensor is not onehot encoded or (for probs) didn't used Softmax.
    """
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def normalized_tanh(x, dim=1, eps=1e-8):
    x = torch.tanh(x)
    x = x / (torch.abs(x).sum(dim) + eps)
    return x


def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)
    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res
