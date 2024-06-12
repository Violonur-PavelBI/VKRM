import tqdm

from typing import Any, Dict, List, Tuple, Union

import torch
from torch.utils.data import DataLoader

from torchmetrics import Metric

from models.ofa.networks import CompositeSubNet

from .utils import (
    forward_with_threshold,
    filter_preds_by_conf,
    compute_metric,
    check_requirements,
)


def joint_iou_conf_search(
    model: CompositeSubNet,
    dataloader: DataLoader,
    metric: Metric,
    metrics_dict: Dict[str, Tuple[float, bool]],
    conf_thresholds_list: List[float],
    iou_thresholds_list: List[float],
    device=Union[str, torch.device],
    verbose: bool = True,
    precision: str = "single",
) -> Tuple[float, float, List[Dict[str, Any]]]:
    conf_idx, conf_n = 0, len(conf_thresholds_list)
    iou_idx, iou_n = 0, len(iou_thresholds_list)
    conf_cur = conf_thresholds_list[conf_idx]
    iou_cur = iou_thresholds_list[iou_idx]

    preds_list, target_list = forward_with_threshold(
        model,
        dataloader,
        conf_thresholds=conf_cur,
        nms_iou_threshold=iou_cur,
        device=device,
        precision=precision,
    )

    idx = 0
    res_list = []
    t = tqdm.tqdm(
        total=iou_n + conf_n - 1, desc="iou & conf search", disable=not verbose
    )
    while conf_idx < conf_n and iou_idx < iou_n:
        res = {"idx": idx, "conf": conf_cur, "iou": iou_cur}
        metric_res = compute_metric(metric, preds_list, target_list)
        sat = check_requirements(metric_res, metrics_dict, res)
        res_list.append(res)
        t.set_postfix(res)

        if sat["all"]:
            break
        elif "recall" in sat and sat["recall"]:
            conf_idx += 1
            if conf_idx == conf_n:
                break
            conf_cur = conf_thresholds_list[conf_idx]
            preds_list = filter_preds_by_conf(preds_list, conf_cur)
        elif "imprecision" in sat and sat["imprecision"]:
            iou_idx += 1
            if iou_idx == iou_n:
                break
            iou_cur = iou_thresholds_list[iou_idx]
            preds_list, target_list = forward_with_threshold(
                model,
                dataloader,
                conf_thresholds=conf_cur,
                nms_iou_threshold=iou_cur,
                device=device,
                precision=precision,
            )
        else:
            conf_idx += 1
            iou_idx += 1
            if conf_idx == conf_n or iou_idx == iou_n:
                break
            conf_cur = conf_thresholds_list[conf_idx]
            iou_cur = iou_thresholds_list[iou_idx]
            preds_list, target_list = forward_with_threshold(
                model,
                dataloader,
                conf_thresholds=conf_cur,
                nms_iou_threshold=iou_cur,
                device=device,
                precision=precision,
            )

        idx += 1
        t.update(1)

    res_list = sorted(
        res_list, key=lambda x: (x["sat"]["all"], x["margin"]), reverse=True
    )
    best_res = res_list[0]
    best_conf_threshold = best_res["conf"]
    best_nms_iou_threshold = best_res["iou"]

    return best_conf_threshold, best_nms_iou_threshold, res_list
