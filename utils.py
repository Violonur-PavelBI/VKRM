# Временный файл
from collections import OrderedDict
from typing import Dict
import os

import numpy as np
from models import core as c
import torch
from tqdm import tqdm
from loguru import logger
from models.ofa.heads.segmentation.segmentation import SegmentationHead
from models.ofa.heads.segmentation.zero import ZeroSegmentationHead

from models.ofa.primitives.static import Argmax
from ofa.utils.configs_dataclasses import SupernetConfig
from models.ofa.heads.detection.yolo_v4 import PostprocessMode, YoloV4DetectionHead
from models.ofa.heads.keypoints.yolo_v7.kps_head import YoloV7KeypointsHead


def save_interface(
    model=None,
    args: SupernetConfig = None,
    metrics=None,
    stop=False,
    stop_hpo=False,
    preservation_of_interface="default",
    experiment_path=None,
):
    """TODO: refactor ???"""
    stop_hpo |= stop
    try:
        from interfaces.tools import load_interface, save_interface

        try:
            old_interface: OrderedDict = load_interface()
        except FileNotFoundError as e:
            logger.warning(f"{e} not found paradigma interface will create!")
            old_interface = OrderedDict()
            old_interface["nas"] = OrderedDict()
            old_interface["nas"]["algname"] = "ofa"
        if model == None:
            stop = True
            save_interface(params=None, interface=old_interface, name_component="nas")
            return
        summary = OrderedDict()
        summary["ofa_subnet_config"] = model.config
        # TODO: remove `supernet`/`subnet` mismatch
        summary["ofa_subnet_cls"] = args.type
        metrics = metrics or {}
        summary["metrics"] = OrderedDict(metrics)
        old_interface["stop"] = stop
        old_interface["stop_hpo"] = stop_hpo
        old_interface["pretrained"] = True
        if preservation_of_interface == "default":
            save_interface(
                params=model.state_dict(),
                interface=old_interface,
                name_component="nas",
                summary=summary,
            )
        elif preservation_of_interface == "local":
            save_interface(
                params=model.state_dict(),
                interface=old_interface,
                name_component="nas",
                summary=summary,
                path_to_interface=experiment_path,
            )
    except ModuleNotFoundError as e:
        if e.name == "interfaces":
            logger.warning(f"{e.name} not found paradigma stuff don`t use!")
        else:
            raise e


def _save(tensor: torch.Tensor, tensor_name: str, model_path: str):
    """Добавляет shape к имени и сохраняет тензор"""
    shape = "_".join(map(str, tensor.shape))
    _tensor_name = f"{tensor_name}_{shape}.bin"
    _tensor_name = os.path.join(model_path, _tensor_name)
    _tensor: np.ndarray = tensor.cpu().detach().numpy()
    _tensor.tofile(_tensor_name)


def convert_with_all_tensors(
    model: c.Module,
    input: torch.Tensor,
    model_path=".",
    cuda=False,
    fp16=False,
    check=False,
    preservation_of_intermediate_tensor=True,
):
    """Конвертирует и сохраняет выходы всех слоёв через функционал конвертера
    моделька ставится в eval.

    check -- сверка выходов восстановленой модели с исходной

    Прокручивает форвард для yolo head
    """

    model.eval()

    if isinstance(model.head, (SegmentationHead, ZeroSegmentationHead)):
        model.head.act = Argmax(1)
        model.head.act.keepdim = True

    src_model_out: torch.Tensor = model(input)
    logger.info("convert model")
    if isinstance(model.head, (YoloV4DetectionHead, YoloV7KeypointsHead)):
        model.head.postprocess = PostprocessMode.NONE
        src_model_out = model(input)
        if isinstance(model.head, YoloV7KeypointsHead):
            src_model_out = src_model_out[0] + src_model_out[1]

    model.cpu().toPlatform(
        model_path, input_example=input, output_example=src_model_out
    )
    l_model = c.Module.fromPlatform(model_path)
    l_model.eval()

    if cuda:
        l_model.cuda()
    if fp16:
        l_model.half()
    if preservation_of_intermediate_tensor:
        os.makedirs(model_path + "/tensors/", exist_ok=True)

    with torch.no_grad():
        out = l_model(input)

    if check:  # TODO: сравнение для коллекций
        logger.info(src_model_out)
        logger.info(torch.allclose(src_model_out, out[0]))

    out_dict = out[1]  # dict with all outputs
    out_dict: Dict[str, torch.Tensor]
    logger.info("save tensors")
    bar_len = len(out_dict.keys())
    p_bar = tqdm(total=bar_len)
    for key, value in out_dict.items():
        shape = "_".join(map(str, value.shape))
        tensor_name = f"{key}_({shape}).bin"
        p_bar.set_description(tensor_name)
        p_bar.refresh()

        tensor_name = os.path.join(model_path, "tensors", tensor_name)
        tensor: np.ndarray = value.cpu().detach().numpy()
        if preservation_of_intermediate_tensor:
            tensor.tofile(tensor_name)
        if key.split(".")[1] == "input":
            _tensor_name = f"{key.split('.')[1]}.bin"
            _tensor_name = os.path.join(model_path, _tensor_name)
            tensor.tofile(_tensor_name)

        p_bar.update(1)

    # Если YOLO -- перезаписываем конвертацию.
    if isinstance(model.head, YoloV4DetectionHead):
        model.head.postprocess = PostprocessMode.PLATFORM
        src_model_out = model(input)
        model.cpu().toPlatform(
            model_path, input_example=input, output_example=src_model_out
        )

        model.head.postprocess = PostprocessMode.NONE
        with torch.no_grad():
            out_to_plat = model(input)

        levels_out = []
        for i in range(model.head.levels):
            sigmoid = model.head.sigmoids[i]
            x = out_to_plat[i]
            x = sigmoid(x)
            levels_out.append(x)

        value = model.head.yolo_layer(levels_out)
        key = f"yolo_decoder"
        _save(value, key, model_path)

        # Функция идёт по картинкам (она одна)
        value = model.head.thresholding(value)
        key = f"thresholding"
        _save(value[0], key, model_path)

        # Функция идёт по картинкам (она одна)
        value = model.head.nms(value)
        key = f"nms"
        _save(value[0], key, model_path)
        tensor = value.cpu().detach().numpy()

    # сохраняем последний выходной слой как output
    _tensor_name = "output.bin"
    _tensor_name = os.path.join(model_path, _tensor_name)

    tensor.tofile(_tensor_name)

    # Если YOLO -- перезаписываем конвертацию.
    if isinstance(model.head, YoloV7KeypointsHead):
        model.head.postprocess = PostprocessMode.PLATFORM
        src_model_out = model(input)
        model.cpu().toPlatform(
            model_path, input_example=input, output_example=src_model_out
        )

        model.head.postprocess = PostprocessMode.NONE
        with torch.no_grad():
            out_to_plat = model(input)

        det_features = []
        for i in range(model.head.levels):
            det_feature = out_to_plat[0][i]
            sigmoid = model.head.sigmoids[i]
            out_to_plat[0][i] = sigmoid(det_feature)

        value = model.head.kps_decode(*out_to_plat)
        key = f"yolo_decoder"
        _save(value, key, model_path)

        # Функция идёт по картинкам (она одна)
        value = model.head.thresholding(value)
        key = f"thresholding"
        _save(value[0], key, model_path)

        # Функция идёт по картинкам (она одна)
        value = model.head.nms(value)
        key = f"nms"
        _save(value[0], key, model_path)
        tensor = value.cpu().detach().numpy()

    # сохраняем последний выходной слой как output
    _tensor_name = "output.bin"
    _tensor_name = os.path.join(model_path, _tensor_name)
    tensor.tofile(_tensor_name)
