import torch
import torch.nn as nn


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction="mean", use_softmax=False, weights=[]):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction
        self.use_softmax = use_softmax
        self.weights = weights

    @staticmethod
    def prob_flatten(inputs, labels):
        assert inputs.dim() in [4, 5]
        num_class = inputs.size(1)
        if inputs.dim() == 4:
            inputs = inputs.permute(0, 2, 3, 1).contiguous()
            inputs_flatten = inputs.view(-1, num_class)
        elif inputs.dim() == 5:
            inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
            inputs_flatten = inputs.view(-1, num_class)
        labels_flatten = labels.reshape(-1)
        return inputs_flatten, labels_flatten

    def lovasz_softmax_flat(self, inputs, labels):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            weight = self.weights[c] if self.weights else 1.0
            target_c = (labels == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(
                weight
                * torch.dot(
                    loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))
                )
            )
        losses = torch.stack(losses)

        if self.reduction == "none":
            loss = losses
        elif self.reduction == "sum":
            loss = losses.sum()
        elif self.reduction == "mean":
            loss = losses.mean()
        else:
            raise NotImplementedError
        return loss

    def forward(self, labels, inputs):
        if self.use_softmax:
            inputs = inputs.softmax(1)
        inputs, labels = self.prob_flatten(inputs, labels)
        losses = self.lovasz_softmax_flat(inputs, labels)
        return losses
