from typing import List, Tuple, Union

from loguru import logger
import torch

from ofa.utils.common_tools import build_config_from_file
from ofa.utils.configs_dataclasses import (
    Config,
    SupernetConfig,
    BackboneConfig,
    NeckConfig,
    HeadConfig,
    ClassificationHeadConfig,
    SupernetLearnStageConfig,
)

from models.ofa.networks import CompositeSuperNet
from models.ofa.backbones import backbone_name2class
from models.ofa.hooks import DynamicHookCatcherLst
from models.ofa.necks import neck_name2class
from models.ofa.heads import head_name2class

from dltools.anchors import build_auto_anchors


def build_backbone(args: BackboneConfig):
    SUPERNET_CLS = backbone_name2class.get(args.type, None)
    if SUPERNET_CLS is None:
        raise NotImplementedError(args.type)
    net = SUPERNET_CLS(**args.dict())
    return net


def build_neck(args: NeckConfig):
    NECK_CLS = neck_name2class.get(args.type, None)
    if NECK_CLS is None:
        raise NotImplementedError(args.type)
    net = NECK_CLS(**args.dict())
    return net


def build_head(args: HeadConfig):
    HEAD_CLS = head_name2class.get(args.type, None)
    if HEAD_CLS is None:
        raise NotImplementedError(args.type)
    net = HEAD_CLS(**args.dict())
    return net


def build_composite_supernet_and_args_from_config(
    config_path: str,
) -> Tuple[CompositeSuperNet, Config]:
    args = build_config_from_file(config_path)

    head_config = args.common.supernet_config.head
    if (
        "anchors" in head_config.dict()
        and "auto_anchors" in head_config.dict()
        and head_config.auto_anchors
    ):
        head_config.anchors = build_auto_anchors(
            args.common.dataset, len(head_config.strides)
        )

    composite_net = build_composite_supernet_from_args(args.common.supernet_config)
    return composite_net, args


def build_composite_supernet_from_args(args: SupernetConfig) -> CompositeSuperNet:
    """Собирает по частям композитную сеть с FPN данная функция отвечает
    за то, чтобы по параметрам из конфига получить параметры классов, чтобы всё
    нормально соединилось между собой

    args: `SupernetConfig`"""

    backbone = build_backbone(args.backbone)
    max_hook_count = len(backbone.layers_to_hook)
    catch_counter = args.backbone.catch_counter or max_hook_count
    backbone_hooks = DynamicHookCatcherLst(backbone, catch_counter)

    # Не совсем хороший парсинг, потому что он привязан к названием поля
    # У статических модулей там int, у динамических list
    channels_info: List[Union[List[int], int]] = []
    for info in backbone_hooks.hooks_info:
        prefix = "in" if info["hook_type"] == "pre" else "out"
        module_conf = info["module_conf"]
        if f"{prefix}_channel_list" in module_conf:
            key = f"{prefix}_channel_list"
        elif f"{prefix}_channels" in module_conf:
            key = f"{prefix}_channels"
        info = module_conf[key]
        channels_info.append(info)

    if args.neck is not None:
        args.neck.in_channels = channels_info
        neck = build_neck(args.neck)
    else:
        neck = None

    if isinstance(args.head, ClassificationHeadConfig):
        args.head.in_channels = channels_info[-1]
    head = build_head(args.head)

    SuperNetCls = CompositeSuperNet
    composite_net = SuperNetCls(backbone_hooks, neck, head)
    return composite_net


def init_supernet_pretrain(
    supernet: CompositeSuperNet, args: SupernetLearnStageConfig, verbose=False
):
    """Загружает веса ранее обученной суперсети без головы"""
    pretrain = args.supernet_config.pretrain
    pretrain_path = f"pretrain/{pretrain}/model_best.pt"
    if verbose:
        logger.info(f"Use pretrain from {pretrain_path}")
    pretrain_weights = torch.load(pretrain_path, map_location="cpu")
    # TODO: Это должно делаться по условию
    net_state = {k[7:]: v for k, v in pretrain_weights["state_dict"].items()}
    # delete head keys
    for head_key in supernet.head.state_dict().keys():
        full_key = f"head.{head_key}"
        if full_key in net_state:
            del net_state[full_key]

    supernet.load_state_dict(net_state, strict=False)
    if verbose:
        logger.info(f"Pretrain loaded! {pretrain_path}")
