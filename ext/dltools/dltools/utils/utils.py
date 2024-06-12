import math
import torch
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from loguru import logger
from typing import Tuple, Union
from functools import partial
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


_DATASET_MAX_PARTITION_COEFF=0.025 # minimum 40 batches (opt-steps) per epoch. For my opinion it need to be 0.01 or lesser;
_CLAS_MIN_REASONABLE_BATCH=64
_SEGM_MIN_REASONABLE_BATCH=16
_SEGM_BASE_IMG_SURF=384**2
_DET_MIN_REASONABLE_BATCH=64
_KEYP_MIN_REASONABLE_BATCH=64
_PER_GPU_REASONING_COEF = 4

def _calculate_accumulation_steps_needed(batch_wanted:int, batch_possible_per_gpu:int, num_gpus:int):
    assert batch_wanted > batch_possible_per_gpu * num_gpus
    batch_per_gpu_accumulated = batch_wanted // num_gpus
    accumulation_steps = round(batch_per_gpu_accumulated / batch_possible_per_gpu)
    if accumulation_steps == 1:
        accumulation_steps +=1
    return accumulation_steps

def _balance_between_gpus_for_max_part(max_partition_batch, min_per_gpu, max_per_gpu, available_ngpus):

    if max_partition_batch // min_per_gpu > available_ngpus:
        batch_wanted = max_partition_batch
        if batch_wanted // available_ngpus > max_per_gpu:
            # so probably we need to accumulate gradients
            acc_grad_steps = _calculate_accumulation_steps_needed(batch_wanted,max_per_gpu, available_ngpus)
            # finetune batch between max_per_gpu and min_per_gpu:
            # this takes only few steps.
            for batch_per_gpu in range(max_per_gpu, min_per_gpu-1, -1):
                if batch_per_gpu * available_ngpus * acc_grad_steps > max_partition_batch:
                    continue
                else:
                    return batch_per_gpu, available_ngpus, acc_grad_steps
            else:
                for n_gpus in range(available_ngpus, 0, -1):
                    if batch_per_gpu * n_gpus * acc_grad_steps > max_partition_batch:
                        continue
                    else:
                        return batch_per_gpu, n_gpus, acc_grad_steps
                # return batch_per_gpu, available_ngpus -1 , acc_grad_steps
        else:
            batch_per_gpu = batch_wanted // available_ngpus
            return batch_per_gpu, available_ngpus, 1

    else:
        # this regime then not need all gpus; cz overall batch would be bigger than max partition
        # Here train task will use bigger that _DATASET_MAX_PARTITION_COEFF part of data if we use min_task_batch;
        # which is bad. So we can lower gpu usage to reasonable numbers
        n_gpus = max_partition_batch // min_per_gpu
        return min_per_gpu, n_gpus, 1


def _batch_gpu_usage_balancer(
    min_task_batch: int,
    max_per_gpu: int,
    available_ngpus: int,
    dataset_size: int,
    img_size: Union[None, int] = None # this for now is not used
)-> Tuple[int,int,int]:
    """Function which tries to balance batch for several gpus and estimates how better divide between gpus.
    Some gpus can be not used at all. Also estimates aggregated batch size (for gradient accumulating)

    Args:
        min_task_batch (int): Batch minimum for good task convergence
        current_max_possible_batch (int): Max possible batch for this task. Can be calculated using max batch finder
        available_ngpus (int): How many there are gpus for this task
        dataset_size (int): Self described
    Returns:
        Tuple[int,int,int]: batch_per_gpu, n_gpu, agregated_batch_size
    """
    max_partition_batch = round(dataset_size * _DATASET_MAX_PARTITION_COEFF)
    min_per_gpu = min(max_per_gpu, _PER_GPU_REASONING_COEF)
    if max_partition_batch <= _PER_GPU_REASONING_COEF: # maybe condition needed max_partition_batch < min_per_gpu; not sure yet
        min_dataset_adequate = round(_PER_GPU_REASONING_COEF / _DATASET_MAX_PARTITION_COEFF)
        raise UserWarning(f"Dataset to smol, add atleast {min_dataset_adequate - dataset_size} training examples")
    elif max_partition_batch < min_task_batch:
        # This is low data regime; we want max batch between this too, so batch wanted is `max_partition_batch`
        return _balance_between_gpus_for_max_part(max_partition_batch, min_per_gpu, max_per_gpu, available_ngpus)
    else:
        batch_atleast_wanted = min_task_batch
        # this is normal regime; there we can use atleast `min_task_batch`` and lesser than maximum `max_partition_batch``
        if batch_atleast_wanted > available_ngpus * max_per_gpu:
            # There we need accumulating gradients
            acc_steps = _calculate_accumulation_steps_needed(batch_atleast_wanted, max_per_gpu, available_ngpus)
            if max_per_gpu * available_ngpus * acc_steps > max_partition_batch:
                acc_steps-=1
            return max_per_gpu, available_ngpus, acc_steps
        else:
            acc_steps = 1
            if max_per_gpu * available_ngpus > max_partition_batch:
                if max_partition_batch //available_ngpus > 2:
                    return max_partition_batch // available_ngpus, available_ngpus, acc_steps
                else:
                    acc_steps = 2
                    return 2, available_ngpus, acc_steps
            else:
                return max_per_gpu, available_ngpus, acc_steps

def optim_param_adjuster_on_batch_change(
    batch_or: int,
    batch_new: int,
    lr_or:int,
    wd_or:int,
    beta1_or:float, # for  sgd maybe equals to momentum
    beta2_or:float = None
    ):
    new_lr = lr_or * (batch_new / batch_or) ** 0.5

    # this cz lr already in (b1/b0)^0.5 times bigger and epoch is lesser by b0/b1 times, so (b1/b0) / (b1/b0)^0.5 = (b1/b0)^0.5
    new_wd = wd_or * (batch_new / batch_or) ** 0.5

    # for betas bcz of ema effective window is firest_el/ (1-q) (limit of geometric proggression)
    # we need to lower this if we want same results
    # also gradients is (b1/b0)^0.5 times more accurate (std of gradients lesser by this term times) and it already handled in lr
    # b0 / (1-beta1_0) = b1 / (1-beta1_1) # ema effective window_size0 = window_size1
    # beta1_1 = 1 - b1/b0 (1-beta)
    new_beta1 = 1- (batch_new/batch_or) * (1-beta1_or)
    new_beta2 = beta2_or # bcz terms of reduction of windows size and std of gradients squares more accurate calculations cancels each other
    assert new_beta1 < new_beta2 # bcz always ema of squares must be bigger than ema for adam-like optims to work
    return new_lr, new_wd, new_beta1, new_beta2

clas_balancer = partial(_batch_gpu_usage_balancer, _CLAS_MIN_REASONABLE_BATCH)
segm_balancer = partial(_batch_gpu_usage_balancer, _SEGM_MIN_REASONABLE_BATCH)
det_balancer = partial(_batch_gpu_usage_balancer, _DET_MIN_REASONABLE_BATCH)
keyp_balancer = partial(_batch_gpu_usage_balancer, _KEYP_MIN_REASONABLE_BATCH)
