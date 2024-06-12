import json
import os
from pathlib import Path
import subprocess


import onnx
import torch
from models.ofa.networks import CompositeSubNet
from tempfile import TemporaryDirectory


HPM_SCHEDULER_PATH = "/workspace/proj/hpm"


colours = (
    "\033[0m",  # End of colour
    "\033[36m",  # Cyan
    "\033[91m",  # Red
    "\033[35m",  # Magenta
)


err_colouring = colours[2] + "{}" + colours[0]


def execute_bash(command: str, verbose: bool = False):
    if verbose:
        print(f"running command: {command}")

    p = subprocess.Popen(
        command,
        shell=True,
        executable="/bin/bash",
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    p.wait()

    if verbose:
        stdout = p.stdout
        stdout.flush()
        stdout_cont = stdout.read()
        stderr = p.stderr
        stderr.flush()
        stderr_cont = stderr.read()
        print(
            stdout_cont.decode("utf-8"),
            err_colouring.format(stderr_cont.decode("utf-8")),
        )


def convert_single_subnet(
    subnet: CompositeSubNet,
    subnet_arch_desc: dict,
    name: str,
    input_example: torch.Tensor,
    output_example: torch.Tensor,
    save_dir: Path,
    system: str = "module",
    verbose: bool = False,
    saveplat: bool = False,
    convert_to_onnx: bool = False,
    hpm_path: str = HPM_SCHEDULER_PATH,
):
    model_dir: Path = save_dir / name
    model_dir.mkdir(exist_ok=True, parents=True)
    if saveplat:
        plat_dir_path = model_dir
        model_desk_fp = os.path.join(model_dir, "model.json")
        model_weight_fp = os.path.join(model_dir, "model.bin")
    else:
        plat_dir = TemporaryDirectory()
        plat_dir_path = Path(plat_dir.name)
        model_desk_fp = os.path.join(plat_dir_path, "model.json")
        model_weight_fp = os.path.join(plat_dir_path, "model.bin")

    metadata = {}
    metadata["supernet_dict"] = subnet_arch_desc
    with open(model_dir / "ofa_meta.json", "w") as js_out:
        json.dump(metadata, js_out)

    if convert_to_onnx:
        try:
            onnx_path = str(model_dir / "model.onnx")
            torch.onnx.export(
                subnet,
                input_example,
                model_dir / "model.onnx",
                export_params=True,
                opset_version=12,
                do_constant_folding=False,
                input_names=["input"],
                output_names=["output"],
            )
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            print(e)
            print(name)

    subnet.toPlatform(
        model_path=plat_dir_path,
        input_example=input_example,
        output_example=output_example,
    )

    if system == "elbrus":
        system_model = "8sv"
        data_format = "channel"
    elif system == "module":
        system_model = "nm6408"
        data_format = "pixel"
    else:
        raise NotImplementedError(f"{system = } is not supported")

    result_file = model_dir / f"{name}_{system}"
    command = (
        f"{hpm_path}/hpm_scheduler "
        f"-system_type={system} -system_model={system_model} "
        f"-in_desc={model_desk_fp} -in_weights={model_weight_fp} "
        f'-out_weights="{result_file}.weights" -out_desc="{result_file}.descr" '
        f"-data_format={data_format} -gen_weights=false -stage=0 "
        f"> {model_dir}/run_time_data.dat"
    )
    execute_bash(command, verbose)
    if not saveplat:
        plat_dir.cleanup()
