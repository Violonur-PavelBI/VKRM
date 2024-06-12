from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch

def calculate_stats(dataset: Dataset, input_idx = 0, num_jobs=4, batch_size = 8):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_jobs = num_jobs,  prefetch_factor=3)
    example_input = dataset[0][input_idx]
    _calculate_stats(dataloader, example_input, input_idx)

def _calculate_stats(dataloader:DataLoader, example_input:torch.Tensor, input_idx):
    # assume that image is [C x H x W]
    means = torch.zeros_like(example_input.mean(1,2))
    sqrmeans =  torch.zeros_like(means) # [C]
    counts = 0
    for batch in dataloader:
        counts +=1
        inputs = batch[input_idx]
        inputs: torch.Tensor # Assume B x C x H x W
        means  = (inputs.mean(0,2,3) + means * (counts-1)) / counts
        sqrmeans = (inputs.square().mean(0,2,3) + sqrmeans * (counts - 1)) / counts
    std = (sqrmeans - means.square()).sqrt()
    return means, std
