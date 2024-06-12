# Про ауги
## Ауги обучения
Преподсчёт 
Примерный состав аугментаций для экспериментов (обучение)
Общий примерный состав
- Compose
  - RandAugment (Platform.kernels.augs) только для классификации?
  - Normalize 
  - RandomResizedCrop(224, scale=(0.4, 1)) как в торчвижоне, нужен ли BBoxSafe для детекции?
  - HorizontalFlip
  - * CutMix, MixUp (только для классификации)
> CutMix, MixUp (только для классификации) и порядок безразличен, так как они обрабатывают батч. Вероятно будут встроенны лосс, лоссы.  
> для Normalize параметры посчитываются во время работы компонетны эвристик (если в рамках технологии).  
> RandAugment из Platform.kernels.augs требует починки.

 для вычисления статистик (среднего и стд) для нормализации каналов и других штучек.

 - Классификация:
   - RandomResizedCrop(224,224, scale=(0.4, 1), ratio=(0.75, 1.33))  | Resize(224,224) | RandomCrop(224,224)  
   - RandAugment (num_ops = 2, magnitude = 15) (Platform.kernels.augs? или из торчвижона)   
   - CutMix, MixUp встроенные в ошибку  
   - HorizontalFlip  
   - Normalize

 - Сегментация:
   - RandomResizedCrop(416, scale=(0.4, 1), ratio=(0.75, 1.33))  
   - RandAugment (num_ops = 2, magnitude = 15) (возможен из Platform.kernels.augs)?
   - либо RandomBrigntessContrast   
   - HorizontalFlip, Флипы.      
   - Normalize

 - Детекция:
   - RandomResizedCrop(416, scale=(0.4, 1), ratio=(0.75, 1.33)) 
   - RandAugment (num_ops = 2, magnitude = 15) (возможен из Platform.kernels.augs)?   
   - HorizontalFlip
   - Normalize

 - Keypoints:
   - RandomResizedCrop(416, scale=(0.4, 1), ratio=(0.75, 1.33)) 
   - RandAugment (num_ops = 2, magnitude = 15) (возможен из Platform.kernels.augs)? 
   - HorizontalFlip, Флипы.  Либо с флипами но прописав пары левая\правая.
   - RandomResizedCrop(416, scale=(0.4, 1))  
   - Normalize  


На подумать:
- MaskDropout Нужно посчитать сколько в среднем контуров на изображении в случае сегментации, либо в случае детекции, и взять 25% квантиль от гистограммы как число закрашиваемых областей\боксов.
- Mozaic? что-то там. Паша Батырь покажет
> Проверить:
> Как влияет RandomResizedCrop -> RandAugment; RandAugment -> RandomResizedCrop.
> Неплохо бы пообучать с разными аугментациями на трейне и посмотреть какие варианты аугментаций на вале лучше отражают то, что наблюдалось на трейте, т.е. у каких пар наборов агументаций трейн-тест меньше условное discrepancy

## Ауги валидации тестирования

- классификация:
    - PadIfNeeded + CenterCrop(224) | Resize(224,224) + CenterCrop(224) | NoOP (оригинальное)
    - Normalize

- Сегментация:
  - NearTestValResize(64|32) | NearTestValPad(64|32) ( NearTestValResize - Ресайзит до ближайшего делящегося на 64|32,  NearTestValPad - паддит до  ближайшего делящегося на 64|32)
  - Normalize
- Детекция:
  - 
  - Normalize
- Keypoints:


## Пример кода для
```python
class _BaseTrainAugSet(): 
    """Базовый класс для сетов аугментаций
    _heuristic_hook_for_stat_calc -  лист из функций, который будут вызываться во время работы 
        компонетны эвристик (если в рамках технологии) для вычисления статистик (среднего и стд) для нормализации каналов и других штучек.
    """
    _heuristic_hook_for_stat_calc = List[Callable]
    _custom_colate_fn в них аугментации сразу всего батча для CutMix, MixUP (Так как это еще должен лосс учитывать, то мб внутрь лосса идёт)
    def __init__(self, size)
    
    @staticmethod
    def _custom_collate_fn(...):
        Пример
        default_torch_collate_fn
        for f in _custom_batch_aug:
            batch = f(batch)
        return batch

class ClassificationAugSet(_BaseTrainAugSet):
    """Примерный состав Compose, OneOf, RandomResizedCrop|RandomSizedCrop, RandomCrop"""
    
    def __init__(self, size = (224,224)):
        pass
class SegmentationAugSet(_BaseTrainAugSet):
    def __init__(self, size = (416,416)):
        pass

class DetectionAugSet(_BaseTrainAugSet):
    def __init__(self, size = (416,416)):
        pass

class KeyPointsAugSet(_BaseTrainAugSet):
    def __init__(self, size = (416,416)):
        pass

```