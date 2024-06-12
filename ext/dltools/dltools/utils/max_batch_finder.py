import subprocess,shlex
import time
import argparse
import time, os,sys, shutil
from math import log2
from typing import Callable, List, Any, Union, AnyStr, Dict
import signal
import yaml
from pydantic import BaseModel, FilePath, Field
import importlib
import tempfile
from copy import deepcopy
from functools import wraps,partial
from operator import itemgetter
from loguru import logger

def _create_parser()-> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_cmdargs = subparsers.add_parser("cmd", help="usage from comand line args")
    parser_cmdargs.add_argument("-s","--script", type=str, help="Path to python script")
    parser_cmdargs.add_argument("-b", "--batch", type=int, help="Start batch", default=4)
    parser_cmdargs.add_argument("-n", "--name", type=str, help="Batch argument name in script, including \"--\"", default="--batch")
    parser_cmdargs.add_argument("--args", type=str, help = "Arguments to script in \"\", splitted by space (as like as you pass them in terminal)", default="")
    parser_cmdargs.add_argument("-t", "--timeout", type=int, help="Timeout in seconds for process", default=120)
    # cfg-yaml subparser
    parser_cfgyaml = subparsers.add_parser("cfg", help='usage from yaml config file')
    parser_cfgyaml.add_argument("-f", '--file', type=str, help="path to config file in yaml format")
    parser_cmdargs.set_defaults(func = find_batch_from_cmdargs)
    parser_cfgyaml.set_defaults(func = find_batch_from_yamlcfg)

    # cfg-yaml module option
    parser_module = subparsers.add_parser("mod", help="usage to call function from specific module; defined in yaml-cfg config")
    parser_module.add_argument("-f", '--file', type=str, help="path to config file in yaml format, look on FindBatchCfgFromModule")
    parser_module.set_defaults(func= find_batch_from_module)

    # hidden parser for this script reusage in subprocess
    _parser_function = subparsers.add_parser("_callf", help = argparse.SUPPRESS)
    _parser_function.add_argument("--try-batch", required=True, type = int)
    _parser_function.add_argument("-f", '--file', type=str, help="path to config file in yaml format, look on FindBatchCfgFromModule")
    _parser_function.set_defaults(func = _call_function)

    # cfg-yaml with script target cfg override
    _parser_override = subparsers.add_parser(
        "cfg-override",
        help="""usage: call with cfg argument, look on example yaml;
        script target yaml would be temporary overriden then searcing occurs""")
    _parser_override.add_argument("-f", "--file", type=str, help = "usage to call with target script cfg override; look on  FindBatchCfgWithCfgOverride")
    _parser_override.set_defaults(func = _find_with_target_cfg_override)
    return parser

class StartBatchAlreadyBroken(Exception):
    """Exception which indicated broken script, call of this script or already broken batch"""

class FindBatchCfg(BaseModel):
    script: FilePath
    script_args: Union[AnyStr, List[AnyStr], None]
    start_batch:int
    batch_arg_name:str
    # start_batch: int= Field(alias="batch")
    # batch_arg_name: str = Field(alias="name")
    timeout: Union[int,None] = 120

class FindBatchCfgFromModule(BaseModel):
    module: str
    function: str
    function_args: Dict[str, Any]
    start_batch: int
    batch_arg_name: str
    timeout: Union[int,None] = 120

class FindBatchCfgWithCfgOverride(BaseModel):
    script: FilePath
    start_batch: int
    batch_arg_name: str
    timeout: Union[int,None] = 120
    script_args: Union[str, List[str], None]
    script_target_cfg_fp: FilePath
    script_cfg_arg_name:str



def _get_cfg_from_yaml_wrapper(func: Callable = None, param_validator: BaseModel=None):
    if func is None and not param_validator is None:
        return partial(_get_cfg_from_yaml_wrapper, param_validator = param_validator)
    else:
        @wraps(func)
        def _wrapper(argparse_args):
            fp = argparse_args.file
            with open(fp,"r") as fp:
                cfg = yaml.safe_load(fp)
            # validate args by provided pudantic model
            cfg = param_validator(**cfg)
            return func(cfg)
        return _wrapper


def half_step_search(max_ok_batch:int, min_bad_batch:int, test_f: Callable, min_diff = 2):
    if min_bad_batch - max_ok_batch > min_diff:
        step = (min_bad_batch - max_ok_batch) //2
        batch = max_ok_batch + step
        is_ok = test_f(batch)
        if is_ok:
            return half_step_search(batch, min_bad_batch, test_f)
        else:
            return half_step_search(max_ok_batch, batch, test_f)
    else:
        return max_ok_batch


def find_batch_from_cmdargs(args):
    cfg = FindBatchCfg(
        script=args.script,
        script_args=args.args,
        start_batch=args.batch,
        batch_arg_name=args.name,
        timeout = args.timeout
    )
    batch = _find_batch(
        cfg.script,
        cfg.batch_arg_name,
        cfg.start_batch,
        cfg.script_args,
        cfg.timeout,
    )
    return batch

@_get_cfg_from_yaml_wrapper(param_validator=FindBatchCfg)
def find_batch_from_yamlcfg(cfg:FindBatchCfg):
    batch = _find_batch(
        cfg.script,
        cfg.batch_arg_name,
        cfg.start_batch,
        cfg.script_args,
        cfg.timeout
    )
    return batch

def find_batch_from_module(args: argparse.Namespace):
    # didn't wrapper here cz also need fp.
    cfg_file = args.file
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
    cfg = FindBatchCfgFromModule(**cfg)
    kwargs = _create_this_script_call(cfg_file)
    kwargs.update(
        dict(start_batch = cfg.start_batch,
             timeout = cfg.timeout
             ))
    return _find_batch(**kwargs, script_arg_strf = _script_arg_combine_insert)

def _script_arg_combine_append(script_args, batch_arg_name, batch)->List[str]:
    return [batch_arg_name, str(batch), *script_args]

def _script_arg_combine_insert(script_args, batch_arg_name, batch)->List[str]:
    return [*script_args, batch_arg_name, str(batch)]

def nested_setitem(path):
    path = path.split(".")
    def setitem(xs, newval):
        origin =xs
        for subpath in path[:-1]:
            xs = xs[subpath]
        assert xs.get(path[-1]) is not None
        xs [path[-1]] = newval
        return origin
    return setitem

def _partial_script_arg_combine_and_override_cfg( target_cfg_fp, target_cfg_argparse_name)->Callable:
    with open(target_cfg_fp, "r") as f:
        target_cfg = yaml.safe_load(f)
    def temp_yaml_gen_provider():
        while True:
            with tempfile.TemporaryDirectory() as dir:
                yield os.path.join(dir, "batch-finder-temp.yaml")
    tempfilegen = temp_yaml_gen_provider()

    def _script_arg_combine_and_override_cfg(script_args, batch_arg_name, batch)->List[str]:
        modified_cfg = deepcopy(target_cfg)
        settitem = nested_setitem(batch_arg_name)
        modified_cfg = settitem(modified_cfg, batch)
        tmpcfgfp = next(tempfilegen)
        with open(tmpcfgfp, 'w', encoding="utf-8") as f:
            yaml.safe_dump(modified_cfg, f)
        script_args = [*script_args] # to not modify old list
        script_args+=[target_cfg_argparse_name, tmpcfgfp]
        return [ str(x) for x in script_args]

    return _script_arg_combine_and_override_cfg


def _find_batch(script,
                batch_arg_name,
                start_batch,
                script_args,
                timeout,
                script_arg_strf = _script_arg_combine_append):
    curbatch_ok = True
    batch = start_batch
    mult = 1
    if isinstance(script_args, str):
        script_args = script_args.replace("\"", "").split(" ")
        script_args = [x for x in script_args if x] # filtering empties
    batch_arg_name = batch_arg_name.replace("\"", "")
    try:
        while curbatch_ok:
            # Finding multiplier of 2
            new_script_args = script_arg_strf(script_args, batch_arg_name, batch)
            curbatch_ok, err_msg = _boolify_process_result(script, new_script_args,timeout=timeout)
            if curbatch_ok:
                last_ok_batch = start_batch * mult
                mult *= 2
                batch = start_batch * mult

        else:
            test_f = lambda batch:  _boolify_process_result(
                script,
                script_arg_strf(script_args, batch_arg_name, batch),
                timeout=timeout
                )[0]
            ok_batch = half_step_search(last_ok_batch, batch, test_f)
    except UnboundLocalError as e:
        err_msg = r"""Start batch already can't be used, or wrong parameters for script passed, or script broken"
                 Last procces executed error msg:
                 {}""".format(err_msg)
        logger.error(err_msg)
        raise StartBatchAlreadyBroken(err_msg)

    logger.info("Max possible batch: {}".format(ok_batch))
    return ok_batch


def _cast_process(script, script_args, wait=True, timeout=120):
    if not isinstance(script, str):
        script = str(script)
    assert os.path.exists(script)
    script_args = [str(arg) for arg in script_args]
    cmd = [script, *script_args]
    executable = None
    cmd.insert(0, sys.executable)
    cmdargs = cmd
    # This always returns Completed process instace, so process alwyays terminates.
    # Maybe somehow we need add timewait argument, to add some time to wait before starting
    # new process
    p = subprocess.run(
        args=cmdargs,
        executable =executable,
        stdout= subprocess.PIPE,
        stderr= subprocess.PIPE,
        timeout = timeout,
        encoding="utf-8",
        text=True,
        preexec_fn=os.setpgrp
    )
    return  p.returncode, p.stderr, p.stdout


def _boolify_process_result(script, script_args, wait=True, timeout=120):
    if not wait:
        raise NotImplementedError()
    _rcode, err,out = _cast_process(script, script_args, wait, timeout)
    if _rcode !=0:
        if _rcode ==2:
            logger.error("ChildProcessError\n" + err)
            raise ChildProcessError(err)
        return False, err
    else:
        return True, err



@_get_cfg_from_yaml_wrapper(param_validator=FindBatchCfgWithCfgOverride)
def _find_with_target_cfg_override(cfg: FindBatchCfgWithCfgOverride):
    # t_cfgfp - filepath for cfg of target script
    # tcfg - target script arg name for tcfg
    t_cfgfp = cfg.script_target_cfg_fp
    tcfg_name = cfg.script_cfg_arg_name

    _partial_script_arg_comb = _partial_script_arg_combine_and_override_cfg(t_cfgfp, tcfg_name)
    return _find_batch(
        cfg.script,
        cfg.batch_arg_name,
        cfg.start_batch,
        cfg.script_args,
        cfg.timeout,
        script_arg_strf=_partial_script_arg_comb,
    )

@_get_cfg_from_yaml_wrapper(param_validator=FindBatchCfgFromModule)
def _call_function(cfg: FindBatchCfgFromModule):
    cwd = os.getcwd()
    module_path = cfg.module if os.path.exists(cfg.module) else os.path.join(cwd, cfg.module)
    top_module_path = module_path.split(".")[:1][0]
    module_name = ".".join(module_path.split(".")[1:])
    if not module_name:
        module_name = top_module_path.split(os.path.sep)[-1]
        top_module_path = os.path.sep.join(top_module_path.split(os.path.sep)[:-1])
    assert os.path.exists(top_module_path)
    assert any([os.path.exists(os.path.join(top_module_path, module_name)),os.path.exists(os.path.join(top_module_path, module_name+'.py'))])

    sys.path.append(top_module_path)
    mod = importlib.import_module(module_name)
    sys.path.pop(-1)
    function = getattr(mod, cfg.function)
    batch_arg_name = cfg.batch_arg_name
    batch = args.try_batch
    fargs = cfg.function_args
    fargs.update({batch_arg_name: batch})
    function(**fargs)



def _create_this_script_call(call_cfg_path):
    this_script = __file__
    function_cmd = "_callf"
    script_args = [
        function_cmd,
        "-f", call_cfg_path,
    ]

    return dict(script = this_script,
               batch_arg_name = "--try-batch",
               script_args = script_args
               )


def parse_args(args):
    parser = _create_parser()
    return parser.parse_args(args)

def main():
    args = parse_args(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()