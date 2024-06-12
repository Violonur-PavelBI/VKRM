import torch


class WEWeighterByLog:
    # TODO: need fix; ema averaging labels instead of ema averaging of loss weights coefficients
    def __init__(self, num_classes, by="c", exp_alpha=0.9, exp_beta=0.2, use_std=None):
        self.alpha = exp_alpha
        self.nc = num_classes
        self.beta = exp_beta
        self.counter = 0
        self.ustd = use_std
        self.wm = None
        #         self.lblcont = None
        self.by = by
        self.iter = 1

    def __call__(self, labels):
        with torch.no_grad():
            self.iter += 1
            # new_weights = (torch.eye(self.nc + 1)[labels+1]).permute([0, -1] + [i for i in range(1, labels.ndim)])
            # shifting, to properly use ignored (-1) class
            new_weights = (torch.eye(self.nc + 1)[labels + 1]).permute([0, 3, 1, 2])
            # filter out ignored dim
            new_weights = new_weights[:, 1:, ...]
            new_weights = torch.einsum(f"bcwh->{self.by}", new_weights)
            #             self.lblcont = new_weights if self.lblcont is None else (self.lblcont*(self.iter-1)+new_weights)/(self.iter)
            new_weights = torch.log10(
                (new_weights.sum() + 1) / (new_weights + 1 + self.beta)
            )
            if self.wm is None:
                self.wm = new_weights
            else:
                self.wm = self.wm * self.alpha + new_weights * (1 - self.alpha)
            return self.wm


class WEWeighterBySquare:
    # TODO: need fix; ema averaging labels instead of ema averaging of loss weights coefficients
    def __init__(self, num_classes, alpha=0.9, beta=2e-2, use_std=None):
        self.alpha = alpha
        self.nc = num_classes
        self.beta = beta
        self.counter = 0
        self.ustd = use_std
        self.wm = None

    def __call__(self, labels):
        # shifting, to properly use ignored (-1) class
        new_weights = torch.eye(self.nc + 1)[labels.flatten() + 1]
        # filter out ignored dim
        new_weights = new_weights[:, 1:]
        # calc weights
        new_weights = new_weights.sum(0)
        # new_weights = new_weights.sum() ** 2 / torch.pow(new_weights + self.beta, 2)
        new_weights = 1 / torch.pow(new_weights + self.beta, 2)
        if self.wm is None:
            self.wm = new_weights
        else:
            self.wm = self.wm * self.alpha + new_weights * (1 - self.alpha)
        return self.wm


class TMWeighter:
    def __init__(self, num_classes, beta=2e-2, use_std=None):
        self.nc = num_classes
        self.beta = beta
        self.counter = 0
        self.stat = None

    def __call__(self, labels):
        # shifting, to properly use ignored (-1) class
        new_weights = torch.eye(self.nc + 1)[labels.flatten() + 1]
        # filter out ignored dim
        new_weights = new_weights[:, 1:]
        self.stat = (
            new_weights.sum(0) if self.stat is None else self.stat + new_weights.sum(0)
        )
        self.counter += 1
        # calc weights
        self.wm = self.counter / (self.stat + self.beta)
        return self.wm
