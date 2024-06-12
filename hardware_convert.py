import argparse
import json
import random
import tqdm
from pathlib import Path
from typing import Tuple

import torch
from models.ofa.heads.detection.yolo_v4 import PostprocessMode
from models.ofa.heads.segmentation.zero import ZeroSegmentationHead
from models.ofa.heads.segmentation.segmentation import SegmentationHead
from models.ofa.primitives.static import Argmax
from ofa.utils import build_config_from_file
from ofa.networks import build_composite_supernet_from_args, CompositeSuperNet
from ofa.utils.conversion import convert_single_subnet, HPM_SCHEDULER_PATH


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ins",
        type=Path,
        nargs="+",
        required=True,
        help="Список путей до yaml конфигов для построения суперсетей.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Путь до папки для сохранения результата.",
    )
    parser.add_argument(
        "--hpm",
        type=str,
        default=HPM_SCHEDULER_PATH,
        help=f"Путь до папки с HPM_SCHEDULER. По умолчанию {HPM_SCHEDULER_PATH}.",
    )
    parser.add_argument(
        "--image-size", type=int, nargs=2, required=True, help="Размер изображения."
    )
    parser.add_argument(
        "--classes", type=int, required=True, help="Количество классов."
    )
    parser.add_argument(
        "--det", action="store_true", help="Выбирать часть подсетей детерминированно."
    )
    parser.add_argument(
        "--nets", type=int, default=1000, help="Количество подсетей. По умолчанию 1000."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Случайный сид. По умолчанию 42."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Печатать выход hpm_scheduler-а."
    )
    parser.add_argument(
        "--saveplat",
        action="store_true",
        help="Сохранять промежуточные файлы подсетей после семплирования.",
    )
    parser.add_argument("--onnx", action="store_true", help="Конвертировать в ONNX.")
    parser.add_argument(
        "--systems", type=str, nargs="+", required=True, help="module или elbrus."
    )
    parser.add_argument("--mode", type=str, default=None, help="min or max")
    args = parser.parse_args()
    return args


def create_supernet_and_examples(
    config_path: Path, system: str, n_classes: int, image_size: Tuple[int, int]
):
    config = build_config_from_file(config_path).common.supernet_config
    if system == "module" and n_classes == 3:
        config.head.n_classes = 4
    else:
        config.head.n_classes = n_classes
    if hasattr(config.head, "image_size"):
        config.head.image_size = image_size

    supernet = build_composite_supernet_from_args(config)
    supernet.eval()

    input = torch.randn(1, 3, *image_size)
    with torch.no_grad():
        output = supernet(input)

    return supernet, input, output


def make_subnet_configs(
    supernet: CompositeSuperNet, det_grid: bool, num_subnets: int, seed: int, mode: str
):
    net_id_list = set()
    if det_grid:
        subnets_grid = supernet.get_subnets_grid()
        for net_setting in subnets_grid:
            net_id = json.dumps(net_setting)
            net_id_list.add(net_id)
        if len(net_id_list) > num_subnets:
            net_id_list = random.sample(net_id_list, k=num_subnets)
            net_id_list = set(net_id_list)

    random.seed(seed)
    while len(net_id_list) < num_subnets:
        if mode == "min":
            supernet.set_min_net()
            subnet_arch_desc = supernet.get_active_arch_desc()
        elif mode == "max":
            supernet.set_max_net()
            subnet_arch_desc = supernet.get_active_arch_desc()
        else:
            subnet_arch_desc = supernet.sample_active_subnet()
        net_id = json.dumps(subnet_arch_desc)
        net_id_list.add(net_id)
    net_id_list = list(net_id_list)
    net_id_list.sort()

    return net_id_list


def create_and_convert_subnets(
    config_path: Path,
    system: str,
    det_grid: bool,
    num_subnets: int,
    seed: int,
    out: Path,
    verbose: bool,
    saveplat: bool,
    convert_to_onnx: bool,
    hpm_path: str,
    image_size: Tuple[int, int],
    n_classes: int,
    mode: str,
):
    supernet, input, output = create_supernet_and_examples(
        config_path, system, n_classes, image_size
    )
    net_id_list = make_subnet_configs(supernet, det_grid, num_subnets, seed, mode)

    for i, net_id in tqdm.tqdm(
        enumerate(net_id_list), desc=f"{config_path.stem} {system}"
    ):
        subnet_arch_desc = json.loads(net_id)
        supernet.set_active_subnet(subnet_arch_desc)
        subnet = supernet.get_active_subnet().eval()
        subnet_arch_desc["r"] = image_size
        if isinstance(subnet.head, SegmentationHead) or isinstance(
            subnet.head, ZeroSegmentationHead
        ):
            subnet.head.act = Argmax(1)
            subnet.head.act.keepdim = True
        if hasattr(subnet.head, "postprocess") and isinstance(
            subnet.head.postprocess, PostprocessMode
        ):
            subnet.head.postprocess = PostprocessMode.PLATFORM
        try:
            with torch.no_grad():
                output = subnet(input)
        except Exception as e:
            print(i, e)
            print(subnet_arch_desc)

        convert_single_subnet(
            subnet,
            subnet_arch_desc,
            str(i),
            input,
            output,
            out / system / config_path.stem,
            system,
            verbose=verbose,
            saveplat=saveplat,
            convert_to_onnx=convert_to_onnx,
            hpm_path=hpm_path,
        )


if __name__ == "__main__":
    args = get_args()

    for config_path in args.ins:
        for system in args.systems:
            create_and_convert_subnets(
                config_path=config_path,
                system=system,
                det_grid=args.det,
                num_subnets=args.nets,
                seed=args.seed,
                out=args.out,
                verbose=args.verbose,
                saveplat=args.saveplat,
                convert_to_onnx=args.onnx,
                hpm_path=args.hpm,
                image_size=tuple(args.image_size),
                n_classes=args.classes,
                mode=args.mode,
            )
