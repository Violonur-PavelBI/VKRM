import argparse
import json
import numpy as np
import pandas as pd
import tqdm

from pathlib import Path
from typing import Dict, List, Union


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="Путь до папки с результатами измерений характеристик "
        "сетей на вычислителе в виде нескольких json файлов.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Папка для сохранения итоговых датасетов, "
        "полученных в результате усреднения нескольких измерений на вычислителе.",
    )
    parser.add_argument("--systems", type=str, required=True, help="module или elbrus.")
    args = parser.parse_args()
    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir if args.out_dir is not None else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict: Dict[str, Dict[str, List[float]]] = {}
    leftout_archs: List[dict] = []
    for file_path in tqdm.tqdm(list(in_dir.iterdir())):
        if file_path.suffix != ".json":
            continue
        try:
            with open(file_path, "r") as f:
                if args.systems == "module":
                    data = json.load(f)
                else:
                    [data] = json.load(f)
        except Exception as e:
            leftout_archs.append({file_path.stem: "error"})
            with open(out_dir / "leftout_archs.json", "w") as f:
                json.dump(leftout_archs, f, indent=4)
            continue
        for subnet in data:  # tqdm.tqdm(data, desc=file_path.stem):
            if args.systems == "module":
                arch_desc = subnet["ofa_meta"]["supernet_dict"]
            else:
                arch_desc = data[subnet]["meta_data"]["supernet_dict"]
                subnet = data[subnet]
            net_id = arch_desc
            if "FPS" not in subnet or "arithmetic_efficiency_mac" not in subnet:
                leftout_archs.append({file_path.stem: net_id})
                with open(out_dir / "leftout_archs.json", "w") as f:
                    json.dump(leftout_archs, f, indent=4)
                continue
            if net_id not in dataset_dict:
                dataset_dict[net_id] = {"latency": [], "efficiency": []}
            lat = 1000 / subnet["FPS"]
            eff = subnet["arithmetic_efficiency_mac"]
            dataset_dict[net_id]["latency"].append(lat)
            dataset_dict[net_id]["efficiency"].append(eff)

    dataset_list: List[List[Union[str, float]]] = [
        [subnet_id, np.mean(stats["latency"]), np.mean(stats["efficiency"])]
        for subnet_id, stats in dataset_dict.items()
    ]
    dataset_list = sorted(dataset_list, key=lambda x: x[0])

    df = pd.DataFrame(dataset_list, columns=["arch", "latency", "efficiency"])
    df.to_csv(out_dir / "lat_eff_dataset.csv", index=False)

    lat_dict = {subnet_id: lat for subnet_id, lat, _ in dataset_list}
    with open(out_dir / "latency.json", "w") as f:
        json.dump(lat_dict, f, indent=0)

    eff_dict = {subnet_id: eff for subnet_id, _, eff in dataset_list}
    with open(out_dir / "efficiency.json", "w") as f:
        json.dump(eff_dict, f, indent=0)
