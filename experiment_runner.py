import argparse
import fcntl
import yaml

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from loguru import logger
from shutil import rmtree, copy2
from typing import Union

from main import main
from ofa.utils import build_index, Config


def set_field(config: dict, key: str, value):
    """
    ключ задаётся в виде строки разделённой точкой
    каждая точка означает вложенность

    key1.key2.key3 -> config['key1']['key2']['key3']
    """
    qwe = key.split(".")
    d = config
    for k in qwe[:-1]:
        d = d.get(k, None)
        if d is None:
            raise KeyError(f"Trying to set `{key}`: error in `{k}` key.")
    d[qwe[-1]] = value


def get_args_function(loaded_config, config_path, rank, port_shift, run_from_scratch):
    raw_config = deepcopy(loaded_config)
    config = Config(**loaded_config)
    config.raw_config = raw_config
    parsed = config
    parsed.common.exp.resume = not run_from_scratch
    parsed.common.config_path = config_path
    parsed.common.world_size = 1
    parsed.common.rank = 0
    parsed.common.master_addr_full = f"localhost:{28500+rank+port_shift}"
    return parsed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config-path",
        type=Path,
        required=True,
        help="Путь к файлу базовой конфигурации эксперимента",
    )
    parser.add_argument(
        "--exp-config-path",
        type=Path,
        required=True,
        help="Путь к конфигу с параметрами для варьирования и их значениями",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Путь к папке для сохранения результатов.",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        help="Путь до папки с датасетами.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Перезапустить только неудачные эксперименты"
        " (по умолчанию с последней успешно завершившейся эпохи).",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Игнорировать успешно завершившиеся стадии и эпохи"
        " при перезапуске неудачных экспериментов.",
    )
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--port-shift", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    base_config_path: Path = args.base_config_path
    datasets_dir: Union[Path, None] = args.datasets_dir
    if args.datasets_dir is not None:
        datasets_dir = args.datasets_dir.absolute()
    else:
        datasets_dir = None
    output_dir: Path = args.output_dir
    index_path = output_dir / "index.yaml"
    resume_failed_exps: bool = args.resume
    run_from_scratch: bool = args.from_scratch
    world_size: int = args.world_size
    rank: int = args.rank
    port_shift: int = args.port_shift

    if not index_path.exists():
        build_index(args.exp_config_path, index_path, 4)
    with open(index_path) as fin:
        index = yaml.load(fin, yaml.Loader)

    output_stat = output_dir / "experiments.log"
    logger.info("Reading config.")

    with open(index_path) as fin_index:
        index_all = yaml.load(fin_index, yaml.Loader)
    if resume_failed_exps:
        index = []
        index_base = list(index_all.keys())
        index_base.sort()
        with open(output_stat, "r+") as fin_log:
            fcntl.flock(fin_log, fcntl.LOCK_EX)
            log_old_exp = yaml.load(fin_log, yaml.Loader)
            for i in log_old_exp:
                if log_old_exp[i]["returncode"]:
                    time = datetime.now().replace(microsecond=0)
                    name_for_old = f"{output_stat.stem}_old_{time}.log"
                    copy2(output_stat, output_stat.parent / name_for_old)
                    index_true = log_old_exp.copy()
                    for i in log_old_exp:
                        if log_old_exp[i]["returncode"]:
                            index_true.pop(i)
                    fin_log.seek(0)
                    if index_true != {}:
                        yaml.dump(index_true, fin_log)
                    fin_log.truncate()
                    break
            fcntl.flock(fin_log, fcntl.LOCK_UN)
        for i in log_old_exp:
            index_base.remove(
                int(log_old_exp[i]["dir"].split(str(output_dir) + "/")[1])
            )
            if log_old_exp[i]["returncode"]:
                index.append(int(log_old_exp[i]["dir"].split(str(output_dir) + "/")[1]))
        index.extend(index_base)
    else:
        index = list(index_all.keys())
        open(output_stat, "w").close()
        run_from_scratch = True
    index.sort()
    index = list(map(int, index))

    with open(base_config_path) as fin:
        base_config = yaml.load(fin, yaml.Loader)
    logger.info("Config read.")

    for idx in range(rank, len(index), world_size):
        cur_config = deepcopy(base_config)
        for key, value in index_all[index[idx]].items():
            set_field(cur_config, key, value)

        output_dir_i = (output_dir / str(index[idx])).absolute()
        if output_dir_i.is_dir() and run_from_scratch:
            rmtree(output_dir_i)
        output_dir_i = str(output_dir_i)

        if datasets_dir is not None:
            dataset_path = cur_config["common"]["dataset"]["dataset_path"]
            cur_config["common"]["dataset"]["dataset_path"] = str(
                datasets_dir / dataset_path
            )
        cur_config["common"]["exp"]["exp_root_path"] = output_dir_i
        cur_config["common"]["exp"]["workdir"] = output_dir_i

        args = get_args_function(
            cur_config,
            config_path=base_config_path,
            rank=rank,
            port_shift=port_shift,
            run_from_scratch=run_from_scratch,
        )

        try:
            main(args)
            returncode = 0
        except:
            returncode = 1

        metadata = {
            index[idx]: {
                "dir": output_dir_i,
                "returncode": returncode,
            }
        }
        with open(output_stat, "a") as f_out:
            fcntl.flock(f_out, fcntl.LOCK_EX)
            yaml.dump(metadata, f_out)
            fcntl.flock(f_out, fcntl.LOCK_UN)

        logger.info(f"task {idx} has completed with {returncode = }")
