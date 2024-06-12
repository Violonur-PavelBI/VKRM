from typing import Union

from ofa.utils.configs_dataclasses import (
    EvolutionStageConfig,
    PredictorLearnStageConfig,
)
from .mbnet import MobileNetArchEncoder, MobileNetArchEncoderCatboost
from .resnet import ResNetArchEncoder

from models.ofa.backbones import OFAMobileNet, OFAResNet


def build_arch_encoder_backbone(
    args: Union[PredictorLearnStageConfig, EvolutionStageConfig],
) -> Union[MobileNetArchEncoder, ResNetArchEncoder]:
    backbone_config = args.supernet_config.backbone
    if backbone_config.type in [
        OFAMobileNet.__name__,
    ]:
        if (
            isinstance(args, PredictorLearnStageConfig)
            and args.pred_learn.model == "Catboost"
        ):
            arch_encoder = MobileNetArchEncoderCatboost(backbone_config)
        else:
            arch_encoder = MobileNetArchEncoder(backbone_config)
    elif backbone_config.type in [
        OFAResNet.__name__,
    ]:
        arch_encoder = ResNetArchEncoder(backbone_config=backbone_config)
    else:
        raise NotImplementedError
    return arch_encoder
