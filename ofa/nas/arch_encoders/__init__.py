from typing import Union

from ofa.utils.configs_dataclasses import (
    EvolutionStageConfig,
    PredictorLearnStageConfig,
)

from .abstract import ArchEncoder
from .backbones import build_arch_encoder_backbone
from .heads import build_arch_encoder_head
from .necks import build_arch_encoder_neck
from .composite import CompositeEncoder, CompositeEncoderCatboost, ArchDesc


def build_arch_encoder(args: Union[PredictorLearnStageConfig, EvolutionStageConfig]):
    if isinstance(args, PredictorLearnStageConfig):
        if args.pred_learn.image_size_list:
            image_size_list = args.pred_learn.image_size_list
        elif args.pred_learn.height_list and args.pred_learn.width_list:
            image_size_list = []
            for h in args.pred_learn.height_list:
                for w in args.pred_learn.width_list:
                    image_size_list.append((h, w))
        else:
            image_size_list = [args.dataset.image_size]
    else:
        image_size_list = [args.dataset.image_size]
    backbone_encoder = build_arch_encoder_backbone(args)
    neck_encoder = build_arch_encoder_neck(args)
    head_encoder = build_arch_encoder_head(args)

    if (
        isinstance(args, PredictorLearnStageConfig)
        and args.pred_learn.model == "Catboost"
    ):
        composite_encoder_cls = CompositeEncoderCatboost
    else:
        composite_encoder_cls = CompositeEncoder
    arch_encoder = composite_encoder_cls(
        backbone_encoder=backbone_encoder,
        neck_encoder=neck_encoder,
        head_encoder=head_encoder,
        image_size_list=image_size_list,
    )
    return arch_encoder
