# TODO: обновить или удалить.


import argparse
from ofa.utils.common_tools import build_config_from_file
import os
import torch
import torch.distributed as dist
from ofa.run_manager import DefaultLearnConfig, LearnManager
from ofa.training.progressive_shrinking import load_models, validate
from ofa.utils import AttrPassDDP
import yaml
from loguru import logger
from train_ofa_net import build_supernet


def worker(gpu=0, args=None):
    os.environ["LOCAL_RANK"] = str(gpu)

    worker_global_rank = int(os.environ["RANK"]) * args._global.ngpus_per_node + gpu
    os.environ["RANK"] = worker_global_rank

    # Initialize DDP
    logger.info("init process group")
    if args._global.distributed:
        dist.init_process_group(backend="nccl", rank=worker_global_rank)
    else:
        dist.init_process_group(backend="nccl", rank=worker_global_rank, world_size=1)
    logger.info("process group inited!")
    torch.cuda.set_device(gpu)
    args.image_size = [int(img_size) for img_size in args.image_size]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    teacher_path = os.path.join(args._global.workdir, "supernet_weights/teacher.pt")
    checkpoint_path = os.path.join(
        args._global.workdir, "supernet_weights/checkpoint.pt"
    )
    args.exp.teacher_path = teacher_path if os.path.exists(teacher_path) else None
    args.exp.supernet_checkpoint_path = (
        checkpoint_path if os.path.exists(checkpoint_path) else None
    )

    run_config = DefaultLearnConfig(dataset_name=args.dataset_type)
    run_config.init_from_dict(args.__dict__)

    args.ks_list = [int(ks) for ks in args.ks_list]
    args.expand_list = [float(e) for e in args.expand_list]
    args.depth_list = [int(d) for d in args.depth_list]

    args.width_mult_list = [float(width_mult) for width_mult in args.width_mult_list]

    net = build_supernet(args).to(args.common.exp.device)
    net = AttrPassDDP(net, device_ids=[gpu], find_unused_parameters=True)

    compression = False
    stage_dir = os.path.join(args._global.workdir, args.stage_name)
    distributed_run_manager = LearnManager(
        stage_dir,
        net,
        run_config,
        compression,
        is_root=(worker_global_rank == 0),
        args=args,
    )
    if args._global._path_to_model:
        path_to_model = args.path_to_model
    else:
        path_to_model = args._global.supernet_checkpoint_path

    distributed_run_manager.load_model(path_to_model)

    if isinstance(args.image_size, tuple):
        validate_image_sizes = {args.image_size}
    else:
        validate_image_sizes = [min(max(args.image_size)), max(max(args.image_size))]
    validate_func_dict = {
        "image_size_list": validate_image_sizes,
        "ks_list": sorted({min(args.ks_list), max(args.ks_list)}),
        "expand_ratio_list": sorted({min(args.expand_list), max(args.expand_list)}),
        "depth_list": sorted({min(net.depth_list), max(net.depth_list)}),
    }
    validate_func = lambda _run_manager, epoch, is_test: validate(
        _run_manager, epoch, is_test, **validate_func_dict
    )
    metrics, _val_log = validate_func(distributed_run_manager, epoch=0, is_test=False)
    metrics, _val_log = validate_func(distributed_run_manager, epoch=0, is_test=True)
    if worker_global_rank == 0:
        logger.info(f"result on val: loss:{metrics['Loss']}\ttop1:{metrics['top1']}")
        logger.info(f"result on test: loss:{metrics['Loss']}\ttop1:{metrics['top1']}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--master-addr", help="ip:port", default="localhost:28500")
    parser.add_argument("--path-to-model", help="path/to/file", default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--distributed", action="store_true", default=False)

    args = parser.parse_args()
    parsed = build_config_from_file(args.config_file)

    parsed._global.config_path = args.config_file
    parsed._global.debug = args.debug
    parsed._global.world_size = args.world_size
    parsed._global.rank = args.rank
    parsed._global.master_addr_full = args.master_addr
    parsed._global._path_to_model = args.path_to_model
    parsed._global.stage = args.stage
    parsed._global.distributed = args.distributed
    return parsed


def main():
    args = get_args()
    if args._global.dont_use_env:
        os.environ["RANK"] = str(args._global.rank)
        # считается, что каждый экземпляр скрипта видит одинаковое количество карт
        args._global.ngpus_per_node = torch.cuda.device_count()
        args._global.world_size = args._global.ngpus_per_node * args._global.world_size

        os.environ["WORLD_SIZE"] = str(args._global.world_size)
        master_arrd = args._global.master_addr_full.split(":")
        os.environ["MASTER_ADDR"] = master_arrd[0]
        os.environ["MASTER_PORT"] = master_arrd[1]

    stage = args._global.stage if args._global.stage else args.stages[-1]
    stage_config = args.stages_configs[stage]
    stage_config._global = args._global
    if args._global.distributed:
        import torch.multiprocessing as mp

        mp.spawn(worker, (stage_config,), nprocs=torch.cuda.device_count())
    else:
        worker(0, stage_config)


if __name__ == "__main__":
    main()
