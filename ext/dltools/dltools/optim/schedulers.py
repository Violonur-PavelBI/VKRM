# from  ssh://git@10.24.65.46:997/platform/kernels/seg_lib.git

from torch.optim.lr_scheduler import (
    CyclicLR,
    StepLR,
    MultiStepLR,
    MultiplicativeLR,
    OneCycleLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    _LRScheduler,
)
from math import log, exp


class TrianglePeriodicSteps(StepLR):
    def __init__(
        self,
        optimizer,
        step_size=1,
        gamma=1.41,
        beta=0.95,
        dzyta=0.99,
        last_epoch=-1,
        turn_p=15,
        start=0,
        adjust_groups_lr_by={},
    ):
        self.start = start
        self.beta = beta
        self.dzyta = dzyta
        self.turn_p = turn_p
        self.adjust_groups_lr_by = (
            adjust_groups_lr_by  # This need to change some group lr agter milestone,
        )
        # this work as {milestone_epoch:{'group_id':'adjust_by'}}

        super(TrianglePeriodicSteps, self).__init__(
            optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch
        )

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch in self.adjust_groups_lr_by.keys():
            for milestone, adjust_dict in self.adjust_groups_lr_by.items():
                if milestone == epoch:
                    for group_id, lr_modifier in adjust_dict.items():
                        self.optimizer.param_groups[group_id]["lr"] = (
                            self.optimizer.param_groups[group_id]["lr"] * lr_modifier
                        )

        if epoch % self.turn_p == 0 and epoch > self.start:
            self.gamma = self.beta / self.gamma
            self.beta = self.beta * self.dzyta
        if epoch > self.start:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group["lr"] = lr


class MockScheduler(_LRScheduler):
    def __init__(self, optimizer):
        super(MockScheduler, self).__init__(optimizer)
        self.optim = optimizer

    def step(self, *args, **kwargs):
        pass

    @staticmethod
    def default_get_lr(optim):
        lrs = []
        for param_group in optim.param_groups():
            lrs += [param_group["lr"]]

    def get_lr(self):
        return self.default_get_lr(self.optim)


class WarmUpSch:
    @staticmethod
    def calc_gamma_for_group(start_lr, end_lr, warmup_ep):
        return exp(log(end_lr / start_lr) / warmup_ep)

    def __init__(self, opt, start_lr, warmup_ep):
        self.opt = opt
        self.gammas = []
        self._warmup_ep = warmup_ep
        self._counter = 0
        for param_group in self.opt.param_groups:
            self.gammas += [
                self.calc_gamma_for_group(start_lr, param_group["lr"], warmup_ep)
            ]
            param_group["lr"] = start_lr

    def step(self):
        if self._counter < self._warmup_ep:
            for param_group, gamma in zip(self.opt.param_groups, self.gammas):
                param_group["lr"] *= gamma
                self._counter += 1


__all__ = [ x.__name__ for x in
    [
    CyclicLR,
    StepLR,
    MultiStepLR,
    MultiplicativeLR,
    OneCycleLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    TrianglePeriodicSteps,
    WarmUpSch,
    ]
]
