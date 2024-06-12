# TODO: подумать.

from typing import Union

from ofa.utils.configs_dataclasses import (
    EvolutionStageConfig,
    PredictorLearnStageConfig,
)

from .segmentation import SegmentationHeadEncoder


def build_arch_encoder_head(
    args: Union[PredictorLearnStageConfig, EvolutionStageConfig]
):
    head_config = args.supernet_config.head
    if head_config.type != "NASSegmentationHead":
        return None
    arch_encoder = SegmentationHeadEncoder(head_config=head_config)
    return arch_encoder
