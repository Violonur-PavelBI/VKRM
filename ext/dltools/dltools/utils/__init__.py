import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from .base_utils import *
from .train_utils import *
from ..data import *
from .utils import DistributedSubsetSampler
from .base_utils import make_divisible
from . import max_batch_finder

def get_optimizer(config, model):
    '''returns optimizer by given OptimizerConfig and model parameters'''
    if config.name in optim.__dict__.keys():
        return getattr(optim, config.name)(params=model, **config.params)


def get_scheduler(config, opt):
    '''returns scheduler by given SchedulerConfig and optimizer'''
    if config.name in lr_sched.__dict__.keys():
        return getattr(lr_sched, config.name)(optimizer=opt, **config.params)


def get_loss(config):
    '''returns loss by given LossConfig'''
    if config.name in nn.__dict__.keys():
        return getattr(nn, config.name)(**config.params)


def get_dataset(config):
    raise NotImplementedError
