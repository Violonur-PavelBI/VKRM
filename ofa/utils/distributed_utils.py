import math

import torch
import torch.distributed as dist
from loguru import logger
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

__all__ = [
    "AttrPassDDP",
    "DistributedSubsetSampler",
    "DistributedMetric",
    "DistributedTensor",
]


class AttrPassDDP(DistributedDataParallel):
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class DistributedSubsetSampler(DistributedSampler):
    """Allow Subset Sampler in Distributed Training"""

    def __init__(
        self, dataset, num_replicas=None, rank=None, shuffle=True, sub_index_list=None
    ):
        super(DistributedSubsetSampler, self).__init__(
            dataset, num_replicas, rank, shuffle
        )
        self.sub_index_list = sub_index_list  # numpy

        self.num_samples = int(
            math.ceil(len(self.sub_index_list) * 1.0 / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        if dist.get_rank() == 0:
            logger.info(
                "Use DistributedSubsetSampler: %d, %d"
                % (self.num_samples, self.total_size)
            )

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.sub_index_list), generator=g).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        indices = self.sub_index_list[indices].tolist()
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


# TODO: delete?
class DistributedMetric(object):
    """
    Horovod: average metrics from distributed training.
    лучше это не юзать TODO: delete
    """

    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.0, dtype=torch.float32)
        self.count = torch.tensor(0.0, dtype=torch.float32)

    def update(self, val, delta_n=1):
        """Accumulate sum with avg reduce between workers

        delta_n тут нахер не нужен, возможно.
        Чтобы делить не на количество батчей, а на количество семплов
        TODO: Попробовать заменить all_gather на all_reduce

        В torch 1.12 завезли AVG операцию в all_reduce"""
        val_for_update = delta_n * val
        world_size = dist.get_world_size()
        tensor_list = [torch.zeros_like(val_for_update) for _ in range(world_size)]
        dist.all_gather(tensor_list, val_for_update)
        avg_reduce = sum(tensor_list).detach().cpu() / world_size
        self.sum += avg_reduce
        self.count += delta_n

    @property
    def avg(self):
        return self.sum / self.count


class DistributedTensor(object):
    """Используется для распределённого подсчёта BN статистик"""

    def __init__(self, name):
        self.name = name
        self.sum = None
        self.count = torch.tensor(0.0, dtype=torch.float32)
        self.synced = False

    def update(self, val, delta_n=1):
        self.synced = False
        val_for_update = val * delta_n
        if self.sum is None:
            self.sum = val_for_update.detach()
        else:
            self.sum += val_for_update.detach()
        self.count += delta_n

    @property
    def avg(self):
        if not self.synced:
            world_size = dist.get_world_size()
            tensor_list = [torch.zeros_like(self.sum) for _ in range(world_size)]
            torch.distributed.all_gather(tensor_list, self.sum)
            self.sum = sum(tensor_list).detach().cpu() / world_size
            self.synced = True
        return self.sum / self.count
