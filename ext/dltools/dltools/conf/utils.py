import copy

from typing import Dict, List, Tuple, Union, Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torchmetrics import Metric

from models.ofa.networks import CompositeSubNet
from models.ofa.heads.detection.yolo_v4 import YoloV4DetectionHead, PostprocessMode
from models.ofa.heads.keypoints.yolo_v7.kps_head import YoloV7KeypointsHead

from ..metrics.detection.detection import detection_yolov4_preprocess_for_map
from ..metrics.yolo_kps.yolo_kps import yolo_kps_preprocess_for_map


def forward_with_threshold(
    model: CompositeSubNet,
    dataloader: DataLoader,
    conf_thresholds: Union[List[float], float, None] = None,
    nms_iou_threshold: Union[float, None] = None,
    device: Union[str, torch.device] = "cpu",
    precision: str = "single",
) -> Tuple[List[Dict[str, Tensor]], List[Dict[str, Tensor]]]:
    head = model.head
    preprocess_for_map: Callable
    if isinstance(head, YoloV4DetectionHead):
        preprocess_for_map = detection_yolov4_preprocess_for_map
    elif isinstance(head, YoloV7KeypointsHead):
        preprocess_for_map = yolo_kps_preprocess_for_map
    else:
        raise NotImplementedError(head.__class__.__name__)
    head.postprocess = PostprocessMode.PLATFORM
    if conf_thresholds is not None:
        original_conf_thresholds = head.change_conf_thresholds(conf_thresholds)
    if nms_iou_threshold is not None:
        original_nms_iou_threshold = head.change_nms_iou_threshold(nms_iou_threshold)

    preds_list = []
    target_list = []

    device = torch.device(device) if isinstance(device, str) else device
    with torch.no_grad():
        for context in dataloader:
            context["model"] = model

            context["image"] = context["image"].to(device)
            context["target"] = context["target"].to(device)
            if precision == "half":
                context["image"] = context["image"].half()

            dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=precision == "mixed"
            ):
                output_processed = model(context["image"])
            if precision != "single":
                output_processed = output_processed.float()
            context["target_pred"] = output_processed

            preds, target = preprocess_for_map(context)

            preds_list.extend(preds)
            target_list.extend(target)

    head.postprocess = PostprocessMode.NMS
    if conf_thresholds is not None:
        head.change_conf_thresholds(original_conf_thresholds)
    if nms_iou_threshold is not None:
        head.change_nms_iou_threshold(original_nms_iou_threshold)

    return preds_list, target_list


def select_class(
    preds_list: List[Dict[str, Tensor]],
    target_list: List[Dict[str, Tensor]],
    class_idx: int,
):
    preds_mask = [d["labels"] == class_idx for d in preds_list]
    target_mask = [d["labels"] == class_idx for d in target_list]

    preds_list_cls = [
        {k: v[m] for k, v in d.items()} for d, m in zip(preds_list, preds_mask)
    ]
    target_list_cls = [
        {k: v[m] for k, v in d.items()} for d, m in zip(target_list, target_mask)
    ]

    return preds_list_cls, target_list_cls


def filter_preds_by_conf(
    preds_list: List[Dict[str, Tensor]], conf_thresholds: Union[Tensor, float]
) -> List[Dict[str, Tensor]]:
    filtered_preds_list = []
    for preds in preds_list:
        labels, scores = preds["labels"], preds["scores"]
        if isinstance(conf_thresholds, Tensor):
            thresholds_expanded = conf_thresholds[labels.long()]
            mask = scores >= thresholds_expanded
        else:
            mask = scores >= conf_thresholds
        filtered_preds = {}
        for key in preds.keys():
            filtered_preds[key] = preds[key][mask]
        filtered_preds_list.append(filtered_preds)
    return filtered_preds_list


def compute_metric(
    metric: Metric,
    preds_list: List[Dict[str, Tensor]],
    target_list: List[Dict[str, Tensor]],
) -> Dict[str, Tensor]:
    for p, t in zip(preds_list, target_list):
        metric.update([copy.deepcopy(p)], [copy.deepcopy(t)])
    metric_results = metric.compute()
    metric.reset()
    return metric_results


def check_requirements(
    metric_res: Dict[str, Tensor],
    metrics_dict: Dict[str, Tuple[float, bool]],
    res: Dict[str, Union[float, bool]],
):
    margin = 1000000.0
    sat = {}
    for name, (requirement, greater) in metrics_dict.items():
        value = metric_res[name].cpu().item()
        res[name] = value
        if greater:
            margin_name = value - requirement
        else:
            margin_name = requirement - value
        sat[name] = margin_name >= 0
        margin = min(margin_name, margin)
    sat["all"] = margin >= 0.0
    res["sat"] = sat
    res["margin"] = margin
    return sat
