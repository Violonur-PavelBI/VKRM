import argparse
import json
import random
from pathlib import Path
import time
from typing import List, Tuple

import numpy as np
import torch
import tqdm
from loguru import logger

from models.ofa.networks import CompositeSubNet, CompositeSuperNet

from ofa.networks import build_composite_supernet_from_args
from ofa.utils import build_config_from_file


def freeze(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Путь до yaml конфига. Аргумент игнорируется, если была указана workdir.",
    )
    parser.add_argument(
        "--workdir", type=Path, help="Путь до папки с обученной суперсетью."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Размер датасета для предиктора.",
    )
    parser.add_argument(
        "--det", action="store_true", help="Выбирать первые сети детерминированно."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Количество замеров для каждой подсети.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Номер GPU.",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Использовать CPU вместо GPU."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default="output/latency",
        help="Путь до папки для сохранения результатов.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Случайный сид.")
    args = parser.parse_args()

    cp: Path = args.config_path
    wd: Path = args.workdir
    assert cp or wd

    return args


def create_supernet(config_path: Path, workdir: Path, device: torch.device):
    if workdir:
        config_path = workdir / "config.yaml"

    config = build_config_from_file(config_path)
    supernet = build_composite_supernet_from_args(config.common.supernet_config)

    if workdir:
        checkpoint_path = workdir / "supernet_weights/model_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        supernet.load_state_dict(state_dict)

    supernet.to(device)
    supernet.eval()

    return supernet, config


def sample_arch_configs(
    supernet: CompositeSuperNet,
    size: int,
    det: bool,
    image_size: Tuple[int, int],
    out: Path,
):
    subnet_ids = []

    if det:
        subnet_ids.extend(supernet.get_subnets_grid())
        if len(subnet_ids) > size:
            subnet_ids = random.sample(subnet_ids, k=size)

    for i in range(len(subnet_ids), size):
        arch_config = supernet.sample_active_subnet()
        subnet_ids.append(arch_config)

    for i in range(len(subnet_ids)):
        arch_config = subnet_ids[i]
        arch_config["r"] = image_size
        net_id = json.dumps(arch_config)
        subnet_ids[i] = net_id

    subnet_ids = sorted(subnet_ids)

    with open(out / "subnet_ids.dict", "w") as f:
        json.dump(subnet_ids, f, indent=4)

    return subnet_ids


def measure_latency(
    subnet: CompositeSubNet,
    runs: int,
    image_size: Tuple[int, int],
    device: torch.device,
    seed: int,
    warmup_cut=5,
):
    cuda = device.type == "cuda"
    if cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    timings = torch.zeros(runs)
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    with torch.no_grad():
        for i in range(runs):
            x = torch.randn(1, 3, *image_size, generator=g, device=device)

            if cuda:
                starter.record()
                _ = subnet(x)
                ender.record()
                torch.cuda.synchronize()

                curr_time = starter.elapsed_time(ender)
                timings[i] = curr_time
            else:
                start = time.time()
                _ = subnet(x)
                end = time.time()

                timings[i] = 1000 * (end - start)

    # Я пробовал разные вещи, но первые 2-3 прогона были с сильным отклонением
    timings = timings[warmup_cut:] if runs > 15 else timings
    mean_latency = torch.mean(timings).item()
    std_latency = torch.std(timings).item()

    return mean_latency, std_latency


def build_dataset(
    supernet: CompositeSuperNet,
    subnet_ids: List[str],
    out: Path,
    runs: int,
    image_size: Tuple[int, int],
    device: torch.device,
    seed: int,
    warmup_runs: int = 25,
):
    latency_dict = {}

    subnet = supernet.get_active_subnet()
    warmup_tensor = torch.randn(1, 3, *image_size).to(device)
    with torch.no_grad():
        for i in range(warmup_runs):
            _ = subnet(warmup_tensor)
    time.sleep(1)

    for arch_s in tqdm.tqdm(subnet_ids):
        arch_config = json.loads(arch_s)
        supernet.set_active_subnet(arch_config)
        subnet = supernet.get_active_subnet()
        latency = measure_latency(subnet, runs, image_size, device, seed)
        latency_dict[arch_s] = latency

    with open(out / "latency_stats.dict", "w") as f:
        json.dump(latency_dict, f, indent=4)

    latency_dict_without_std = {k: v[0] for k, v in latency_dict.items()}
    with open(out / "latency_set.json", "w") as f:
        json.dump(latency_dict_without_std, f, indent=4)

    return latency_dict


def main():
    args = get_args()
    freeze(args.seed)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    supernet, config = create_supernet(args.config_path, args.workdir, device)
    image_size = config.common.dataset.image_size

    subnet_ids = sample_arch_configs(supernet, args.size, args.det, image_size, out)
    latency_dict = build_dataset(
        supernet, subnet_ids, out, args.runs, image_size, device, args.seed
    )

    latency_values = torch.tensor(list(latency_dict.values()))

    latency_mean, latency_std_mean = latency_values.mean(dim=0)
    latency_std_min = latency_values[:, 1].min().item()
    latency_std_median = latency_values[:, 1].median().item()
    latency_std_max = latency_values[:, 1].max().item()
    latency_std = {
        "min": latency_std_min,
        "median": latency_std_median,
        "mean": latency_std_mean.item(),
        "max": latency_std_max,
    }
    logger.info(f"mean latency in dataset: {latency_mean}")
    logger.info(f"{latency_std = }")


if __name__ == "__main__":
    main()
