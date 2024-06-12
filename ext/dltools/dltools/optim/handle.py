# from  ssh://git@10.24.65.46:997/platform/kernels/seg_lib.git

from typing import Any, Union, Dict
import torch
from torch.optim.optimizer import Optimizer
from torch.nn import Module

# from torch.nn import Module as Loss #FIXME: write apropriate abstract class for losses
from torch.cuda.amp.grad_scaler import GradScaler
from ..utils.base_utils import ScalerMock, CfgContext  # TODO NOTE: CfgContext in future can be childs of pydantic.BaseModel


BBS = 4  # Basic Batch size to recalculate effective lr


class MockOptimizer(Optimizer):
    def init(self, params) -> None:
        pass

    def step(self):
        pass

    def zero_grad(self):
        pass


class OptimizerHandler:
    _optim_module = "seglib.optim"  # Module from which optimizer would be imported, and initilized by parameters from cfg

    def __init__(
        self,
        optimizer: Union[Optimizer, None],
        delayed_batch_size: int,
        scaler: Union[GradScaler, ScalerMock, None] = None,
    ):
        self._opt = optimizer
        self._dlbs = delayed_batch_size
        self._counter = 1
        self.scaler = scaler

    def step(self):
        if self._opt:
            if self._counter < self._dlbs:
                self._counter += 1
            else:
                if self.scaler:
                    self.scaler.step(self._opt)
                    self._counter = 1
                    self.scaler.update()
                    self._opt.zero_grad()
                else:
                    self._opt.step()
                    self._opt.zero_grad()
                    self._counter = 1
        else:
            pass
    # NOTE: TODO: FIX this after creating pydantic reference class. Now Deprecated
    @classmethod
    def from_cfg(
        cls,
        STAGE_CONTEXT: CfgContext,
        cfg: Dict[str, Any],
        model: Module,
        scaler: Union[None, GradScaler] = None,
        effective_lr=True,
    ):
        raise NotImplementedError()
        # TODO: implement with pydantic.BaseModel
        # delayed_batch_size = cfg["delayed_batch_size"]

        # if STAGE_CONTEXT.is_train:
        #     optim_class_def, optim_params = kwrgs_by_signature(
        #         cfg, cls._optim_module, STAGE_CONTEXT.is_not_silent
        #     )
        #     if effective_lr:
        #         # effective lr dependency: default batch used is aprox to 4, delayed = 1, world_size = 1.
        #         # effective lr func: lr_2 = lr_1 * ((minibatch_size * delayed_batch_size * world_size - 1) / (default_batch_size - 1))^ 0.5
        #         optim_params["lr"] = (
        #             optim_params["lr"]
        #             * (
        #                 (STAGE_CONTEXT.batch_size * delayed_batch_size * STAGE_CONTEXT.world_size - 1)
        #                 / (BBS - 1)
        #             )
        #             ** 0.5
        #         )

        #     if "optimizer_eval_dict" not in cfg.keys():

        #         optim = optim_class_def(model.parameters(), **optim_params)

        #     else:
        #         # This is option for different lrs for different parts of model, use wisely!
        #         # NOTE: remember to use module or _modules atribute, for keys when using DDP
        #         param_groups = eval(cfg["optimizer_eval_dict"])
        #         optim = optim_class_def(*param_groups, **optim_params)

        #     # optim_class_def - class object of optimizer, optim_class_params - apropriate params parsed from cfg
        # else:
        #     optim = None
        #     delayed_batch_size = 1

        # return cls(optim, delayed_batch_size, scaler)


class BaseLossCalcHandler:
    _crit_module = "seglib.calculus.losses"
    # Module from which criterion would be imported, class must be defined in cfg, and initilized by parameters from cfg
    # Temporary Deprecated feature
    def __init__(
        self,
        criterion: Module,
        delayed_batch_size: int,
        STAGE_CONTEXT: CfgContext,
        scaler: Union[GradScaler, ScalerMock, None] = None,
    ):
        self.crit = criterion
        self._dlbs = delayed_batch_size
        self.__totloss = None
        self.__count = 0
        self.__train = STAGE_CONTEXT.is_train
        self.scaler = scaler

    def __call__(self, gt, pred):
        """Calculate loss, and gradients with respect to delayed_batch_size"""
        if self.crit:
            loss_current = self.crit(gt, pred)  # can be non scalar
            # tensor can be of shapes: [B] , [C] , [B x C], [B x C x H x W] - segm, [B x H x W] - also segm

            scalar_current_loss = (
                loss_current.mean()
                if loss_current.shape != torch.Size([1])
                else loss_current
            )
            if self.__train:
                # multiplying by 1 / self._dlbs bcz we didn't want to rescale betas in Adam-like optims
                # and momentum in SGD-like optims
                # Lr shift already 
                if self.scaler is None:

                    (scalar_current_loss / self._dlbs).backward() 

                else:
                    self.scaler.scale(scalar_current_loss / self._dlbs).backward()

            if self.__count < self._dlbs:
                if self.__totloss is None:
                    # there can be non scalar loss, if we need to log loss
                    #  by image or by batch bach, or both.
                    self.__totloss = loss_current.detach() / self._dlbs
                else:
                    self.__totloss += loss_current.detach() / self._dlbs

                self.__count += 1

            else:

                self.__totloss = None
                self.__count = 0

            return loss_current.detach()
        else:
            return None

    @classmethod
    def from_cfg(
        cls,
        STAGE_CONTEXT: CfgContext,
        cfg: dict,
        scaler: Union[GradScaler, ScalerMock, None] = None,
    ):
        raise NotImplementedError()
        # TODO: implement with pydantic.BaseModel
        # Temporary Deprecated feature
        if cls.__name.lower() in cfg.keys():
            cls._crit_module = cfg[cls.name.lower()]

        criterion_def, criterion_params = recursive_kwargs_by_signature(
            cfg, cls._crit_module, STAGE_CONTEXT.is_not_silent
        )
        if criterion_def:
            criterion = criterion_def(**criterion_params)
            # NOTE: if preprocessing used, "weights" param or anything else, which was calculated
            # must already be placed in cfg.'
            delayed_batch_size = cfg["delayed_batch_size"]

        else:
            criterion = None
            delayed_batch_size = 1

        return cls(criterion, delayed_batch_size, STAGE_CONTEXT, scaler)
