import torch
import torch.nn as nn
from torch import einsum
from typing import List


class SegSimpleCrossEntropy(nn.Module):
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"] if "idc" in kwargs.keys() else None
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, target, probs):

        log_p = (probs[:, self.idc, ...] + 1e-10).log()
        mask = target[:, self.idc, ...].type(torch.float32)

        loss = -einsum("bcwh,bcwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


# class TruncatedLoss(nn.Module):
#
#     def __init__(self, q=0.7, k=0.5, trainset_size=500000000):
#         super(TruncatedLoss, self).__init__()
#         self.q = q
#         self.k = k
#         self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
#
#     def forward(self, probas, targets, indexes):
#
#         Yg = torch.gather(probas, 1, torch.unsqueeze(targets, 1))
#
#         loss = ((1 - (Yg ** self.q)) / self.q) * self.weight[indexes] - ((1 - (self.k ** self.q)) / self.q) * \
#                self.weight[indexes]
#         loss = torch.mean(loss)
#
#         return loss
#
#     def update_weight(self, probas, targets, indexes):
#
#         Yg = torch.gather(probas, 1, torch.unsqueeze(targets, 1))
#         Lq = ((1 - (Yg ** self.q)) / self.q)
#         Lqk = np.repeat(((1 - (self.k ** self.q)) / self.q), targets.size(0))
#         Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
#         Lqk = torch.unsqueeze(Lqk, 1)
#
#         condition = torch.gt(Lqk, Lq)
#         self.weight[indexes] = condition.type(torch.cuda.FloatTensor)
