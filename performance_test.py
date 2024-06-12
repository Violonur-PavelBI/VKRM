# Скрипт для прогона подсети ofa на наборе данных с минимальными аугментациями
# Запуск через указание пути до рабочей директории
# python performance_test.py --workdir exp/ofa
# Из-за пробрасывания атрибутов в WrapperModule и CompositeSubNet не работает thop.profile.

import argparse
import json
import os
import tqdm

from typing import List

import random
import numpy as np
import torch

from models.ofa.networks import CompositeSubNet
from loguru import logger


def flush_cache(device):
    x = torch.empty(int(40 * (1024**2)), dtype=torch.int8, device=device)
    x.zero_()


def perf(
    model: CompositeSubNet,
    image_size: List[int],
    num_runs: int,
    warmup_runs: int,
    num_tensors: int,
    device: torch.device,
    deterministic: bool,
):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = model.to(device).eval()

    dummy_input = torch.randn(1, 3, *image_size).to(device)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    reps = num_runs * num_tensors
    timings = torch.zeros(reps)

    for _ in range(warmup_runs):
        _ = model(dummy_input)

    # flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    # logger.info(f"Flops:  {flops}")
    # logger.info(f"Params: {params}")

    with torch.no_grad():
        for i in tqdm.tqdm(range(reps)):
            if i > 0 and i % num_runs == 0:
                dummy_input = torch.randn(1, 3, *image_size).to(device)

            flush_cache(device)
            torch.cuda._sleep(1_000_000)

            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time

    mean_latency = torch.mean(timings)
    std_latency = torch.std(timings)
    logger.info(f"latency: {mean_latency:.4f} ± {std_latency:.4f}")

    fps = 1000.0 / mean_latency
    logger.info(f"fps: {fps:.2f}")

    return mean_latency


def build_net(args: argparse.Namespace):
    with open(args.model_config) as j_in:
        config = json.load(j_in)
    model = CompositeSubNet.build_from_config(config)
    state_dict = torch.load(args.model_weight, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--num-tensors", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--det", action="store_true")
    args = parser.parse_args()

    args.model_config = os.path.join(args.workdir, "result_model_config.json")
    args.model_weight = os.path.join(args.workdir, "result_model.pt")

    return args


def main():
    args = get_args()
    net = build_net(args)

    perf(
        net,
        args.image_size,
        args.num_runs,
        args.warmup_runs,
        args.num_tensors,
        torch.device(args.device),
        args.det,
    )


if __name__ == "__main__":
    main()
