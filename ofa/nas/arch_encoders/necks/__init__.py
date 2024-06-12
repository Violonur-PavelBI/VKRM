from typing import Union

from ofa.utils.configs_dataclasses import (
    EvolutionStageConfig,
    PredictorLearnStageConfig,
)

from .fpn import FPNEncoder


def build_arch_encoder_neck(
    args: Union[PredictorLearnStageConfig, EvolutionStageConfig]
):
    neck_config = args.supernet_config.neck
    if neck_config is None or neck_config.type[:3] != "NAS":
        return None

    arch_encoder = FPNEncoder(neck_config=neck_config)
    return arch_encoder
