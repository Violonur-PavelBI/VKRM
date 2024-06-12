from typing import Callable


class _BaseAugSet():
    """Базовый класс для сетов аугментаций
    _heuristic_hook_for_stat_calc -  лист из функций, который будут вызываться во время работы 
        компонетны эвристик (если в рамках технологии) для вычисления статистик (среднего и стд) для нормализации каналов и других штучек.
    """
    _heuristic_hook_for_stat_calc = List[Callable]
    _custom_colate_fn в них аугментации сразу всего батча для CutMix, MixUP
    
    @staticmethod
    def _custom_collate_fn(...):
        Пример
        default_torch_collate_fn
        for f in _custom_batch_aug:
            batch = f(batch)
        return batch

class ClassificationAugSet(AbsAug)
    """Примерный состав Compose, OneOf, RandomResizedCrop|RandomSizedCrop, RandomCrop"""
    pass

class SegmentationAugSet(AbsAug)
    pass

class DetectionAugSet(AbsAug)
    pass

class KeyPointsAugSet(AbsAug)
    pass
