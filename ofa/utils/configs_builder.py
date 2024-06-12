import fcntl
import time
from collections import defaultdict
from multiprocessing import JoinableQueue, Process
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Union
from loguru import logger

import yaml
from tqdm import tqdm
from yaml import SafeLoader, load


class Rule(TypedDict):
    type: str
    values: str


class Param:
    """Класс обозначающий один варьируемый параметр. На данный момент тут больше параметров чем надо.
    Каждый параметр имеет ряд состояний на основе объединения которых устанавливаются параметры для эксперимента
    """

    def __init__(self, key, values=None, rules=None, set_by_rule=False):
        self.cur_idx = 0
        self.parent: Param = None
        self.childs: List[Param] = None
        self.rules: Union[None, Dict[str, Rule]] = rules
        self.values: Union[None, List] = values
        self.key = key
        self.set_by_rule = set_by_rule

    @property
    def complete(self) -> bool:
        return self.cur_idx >= len(self.values)

    def __bool__(self) -> bool:
        return bool(self.values) and not self.complete

    def __nonzero__(self) -> bool:
        return self.__bool__()

    def __repr__(self) -> str:
        return f"Param: key = {self.key}, idx = {self.cur_idx}"

    @property
    def cur_val(self) -> Any:
        if not bool(self):
            return None
        return self.values[self.cur_idx]

    @property
    def cur_rul(self) -> Union[Dict, None]:
        if self.rules is None or not bool(self):
            return None
        return self.rules.get(self.cur_val, None)


def get_rules(p: Param):
    """Парсит правила из исходного словаря в единый формат"""
    rules = defaultdict(list)
    if not isinstance(p.rules, dict):
        return
    val_rules = p.rules.get(p.cur_val, None)
    if val_rules is None:
        return
    for key, rule in val_rules.items():
        if not isinstance(rule, dict):
            rule: Rule = {"values": rule, "type": "set"}
        rules[key].append(rule)
    return rules


def eval_rules(values, rules: List[Rule]) -> list:
    """Для одного набора значений из конфига производит фильтрацию
    набором правил"""
    set_flag = False
    for rule in rules:
        rule: Rule
        if rule["type"] == "set":
            if set_flag and values != rule["values"]:
                return []
            values = rule["values"]
        elif rule["type"] == "exclude":
            values = list(filter(lambda x: x in rule["values"], values))
        else:
            raise ValueError(f"rule type must be in [set, exclude], got {rule['type']}")
    return values


def update_graph_rules(graph_rules: Dict[str, List], param: Param):
    """Для обновления словаря по значениям которого лежат листы"""
    cur_rules = get_rules(param)
    if cur_rules:
        for k, v in cur_rules.items():
            graph_rules[k].extend(v)


def build_params(exp_config, start=0, count=None, source_params=None) -> List[Param]:
    """Делает список параметров, или пересоздаёт часть списка"""
    graph_rules = defaultdict(list)

    count = count or len(exp_config.keys())
    params = source_params or [None] * count
    keys = list(exp_config.keys())
    for param in params[:start]:
        update_graph_rules(graph_rules, param)

    for i in range(start, count):
        key = keys[i]
        data = exp_config[key]
        if isinstance(data, dict):
            values = data.get("values", None)
            rules = data.get("rules", None)
        else:
            values = data
            rules = None
        if key in graph_rules:
            values = eval_rules(values, graph_rules[key])
        values = values if isinstance(values, list) else [values]
        param = Param(key, values, rules)
        update_graph_rules(graph_rules, param)

        params[i] = param
    return params


def modify_config(params: List[Param]) -> dict:
    """Возвращает словарь с изменёнными параметрами"""
    keys = set(map(lambda x: x.key, params))
    changed_params = {}
    for p in params:
        changed_params[p.key] = p.cur_val
        param_rules = p.cur_rul
        if param_rules is None:
            continue
        for rule_key, rule_val in p.cur_rul.items():
            if rule_key not in keys:
                changed_params[rule_key] = rule_val

    return changed_params


def dump_task(queue: JoinableQueue, index_path: Path):
    """Дописывает в файл уникальные для данного эксперимента параметры"""
    while True:
        idx, modified_params = queue.get()
        with open(index_path, "a") as f_out:
            fcntl.flock(f_out, fcntl.LOCK_EX)
            yaml.dump({idx: modified_params}, f_out, sort_keys=False)
            fcntl.flock(f_out, fcntl.LOCK_UN)

        queue.task_done()


def build_index(exp_config_path: Path, index_path: Path, dump_process_count: int = 4):
    with open(exp_config_path) as fin:
        exp_config = load(fin, SafeLoader)

    params = build_params(exp_config=exp_config)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    idx = 0
    p = tqdm()
    try_counter = 0
    max_graph_depth = len(params) - 1
    configs_queue = JoinableQueue()

    dumpers = [
        Process(target=dump_task, args=(configs_queue, index_path), daemon=True)
        for _ in range(dump_process_count)
    ]

    for d in dumpers:
        d.start()

    while not params[0].complete:
        try_counter += 1
        if all(params):
            configs_for_dump = modify_config(params)
            configs_queue.put((idx, configs_for_dump))
            p.set_postfix(size_of_queue=configs_queue.qsize())
            p.update(1)
            idx += 1
        cur_depth = len(params) - 1
        params[cur_depth].cur_idx += 1
        while not all(params):
            if params[0].complete:
                break
            if params[cur_depth].complete:
                cur_depth -= 1
                params[cur_depth].cur_idx += 1
                continue
            if cur_depth < max_graph_depth:
                params = build_params(
                    exp_config, start=cur_depth + 1, source_params=params
                )
    while True:
        if configs_queue.qsize() == 0:
            break
        p.set_postfix(remaining_configs=configs_queue.qsize())
        p.update(0)
        time.sleep(0.5)
    configs_queue.join()
    for d in dumpers:
        d.terminate()
    p.update(0)
    logger.info(
        f"Потенциальных конфигураций: {try_counter}. Итоговое количество: {idx}"
    )
