import torch
import json
from itertools import product
from tqdm import tqdm
from torch.utils.data import DataLoader
from copy import deepcopy
import os

# from  ssh://git@10.24.65.46:997/platform/kernels/seg_lib.git


# from coolname import generate_slug # Needed to create cool names of experiments like "amazing-yellow-nindza-turtle"

class CfgContext:
    """Class which stores task context params"""
    # TODO: change to use frozen dataclass with `__post_init__`

    def __init__(
        self,
        debug=False,
        verbose=False,
        target_device="cuda",
        uuid=None,
        rank=0,
        purpose="train",
        epochs=0,
        prep_epochs=0,
        warmup_ep=0,
        world_size=1,
        jobs=4,
        rasterize=None,
        profile=False,
        mixed=False,
        exp_name=None,
        hparams=None,
        task_dir=None,
        findlr=None,
        delayed_batch_size=1,
        division=1,
    ):

        self.debug = debug
        self.verbose = verbose
        self.rank = rank
        os.environ["RANK"] = str(rank) 
        self.epochs = epochs
        self.prep_epochs = prep_epochs
        self.warmup_ep = warmup_ep
        self.world_size = world_size
        self.uuid = uuid
        self.jobs = jobs
        self.rasterize = rasterize
        self.purpose = purpose
        self.is_profiled = profile
        self.is_mixed = mixed
        self.exp_name = exp_name
        self.task_dir = task_dir
        self.hparams = hparams
        self.findlr = findlr
        self.division = division
        self.delayed_batch_size = delayed_batch_size
        # NOTE: after this is __post__init__
        self.is_train = purpose == "train" or purpose == "search"
        self.is_first = self.rank == 0
        self.is_dist = self.world_size > 1
        self.is_not_silent = (self.verbose and self.is_first) or self.debug
        self.use_cuda = target_device == "cuda"


class ContextManagerMocker:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        return False

    def step(self, *args, **kwargs):
        pass


def autocast_if_mix(is_mix):
    if is_mix:
        return torch.cuda.amp.autocast()
    else:
        return ContextManagerMocker()


def no_grad_by_purpose(is_train):
    if not is_train:
        return torch.no_grad()
    else:
        return ContextManagerMocker()


def ScalerMock():
    def __init__(self):
        pass

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


def profiler_if_needed(is_profiled, exp_name="prof"):
    if is_profiled:
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(
            "/app/task_info/profilerlog/{}".format(exp_name)
        )
        return torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=3, warmup=4, active=7, repeat=1),
            on_trace_ready=tensorboard_trace_handler,
            with_stack=True,
        )
    else:
        return ContextManagerMocker()


def printlog_metric_means(metric_names, runs, purpose, epoch=""):
    info_string = "Metrics:"
    for metric_name, metric_value in zip(metric_names, runs):
        info_string += f"\t{metric_name}: {metric_value.item():.4f}"
    print(
        (f"Epoch [{epoch}]:" if purpose == "train" else f"[{purpose}]:") + info_string
    )


def tqdmw(iterator: DataLoader, is_silent: bool, rank=0, **kwargs):
    """
    :param iterator: torch.utils.data.Dataloader object
    :param verbose: bool, flag which controls returning of tqdm class object
    :param local_rank: rank of process in process group
    :param kwargs: kwargs to pass to tqdm class constructor
    :return: tqdm instance
    """
    if "desc" in kwargs.keys():
        subdesc = kwargs.pop("desc")
    else:
        subdesc = ""
    if torch.is_distributed:
        desc = "{}: ".format(rank) + subdesc
    else:
        desc = subdesc

    return tqdm(iterator, desc=desc, disable=is_silent, **kwargs)


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
