from .base_strategy import BaseStrategy

from .segmentation import SegmentationStrategy


from ofa.utils.configs_dataclasses import StrategyConfig


def get_strategy_class(args: StrategyConfig) -> BaseStrategy:
    """Return strategy for task
    Args
        args: stage config"""
    strategy_name2class = {
        ClassificationStrategy.__name__: ClassificationStrategy,
        SegmentationStrategy.__name__: SegmentationStrategy,
        KeypointsStrategy.__name__: KeypointsStrategy,
        YoloV4DetectionStrategy.__name__: YoloV4DetectionStrategy,
        YoloV7KeypointsStrategy.__name__: YoloV7KeypointsStrategy,
    }
    return strategy_name2class[args.type]
