# Репозиторий c примитивами для написания моделей
Данный репозиторий содержит:
 - Примитивы для создания моделей для репозитория models.
 - Механизм конвертации и методы для каждого вида слоя.
 - TODO: Механизм переключение между фрейворками для создания и обучения нейронных сетей.
 - TODO: Тестирование примитивов на прогон forward и конвертацию.

## Шаблоны архитектурных модулей (ROADMAP CONCEPT)
Атрибуты шаблонов показывают имена и тип атрибутов, которые должны быть выставлены у наследников, после `__init__`.
Про страйды для подключении голов детекции, FPN, и других неков:
```python
LAYER_NAME=str # Имя слоя который имеет stride!=1
LAYER_INPUT_CHANNELS=int # Число каналов слоя на входе
LAYER_OUTPUT_CHANNELS=int # Число каналов на выходе
GD_FACTORS=int # Глобальный downsample_factor. (Во сколько раз меньше карта признаков по H-W по сравнению с самым первым input'ом, учитывается в конструкторе сети) Глобальный страйд для слоев, соотвествуют значениям downsample_factor. TODO: Мб стоит прояснить про local_downsample_factor и global_dowbsample factor (первый относительно модуля-класса, относительно слоя картинки)

class Backbone(Module):
    r"""Класс, определяющий необходимые атрибуты, для описания бэкбонов

    Аргументы:
    - `output_channels` - глубина выходной карты признаков.
    - `downsample_factor` - показатель отношения $ \frac{HW_{in}}{HW_{out}} $ , для ряда декодеров,
    желательно чтобы был целой степенью двойки.
    - `input_channels` -  глубина входной карты признаков.
    """
    output_channels: int
    downsample_factor: int
    input_channels: int
    gd_factors: Dict[GD_FACTORS, List[Tuple(LAYER_NAME,LAYER_INPUT_CHANNELS,LAYER_OUTPUT_CHANNELS)]]
    _trained_additional_meta: dict(
        channel_format = Literal["RGB", "IR", "RGBA", "BGR"]
    )
    @abstractmethod
    def __init__(self) -> None:
        """Must be implemented in child class"""
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Must be implemented in child class"""
        pass
```


## Структура репозитория (NEED UPDATE):
- `__init__.py`  - содержит базовый класс `core.Module`, классы контейнеры `core.Sequential`, `core.ModuleList` и абстрактные классы, которые добавляют нужный функционал для первых двух типов классов. А так же `__init__.py` содержит все примитивы, которые импортируются при создании моделей.
- `functional/` - содержит функции, которые выполняются без инициализации в forward.
- `converter.py` - содержит классы и функции, которые отвечают за управлением механизма конвертации в формат `platform`.
- `activation.py, conv.py, dropout.py, linear.py, ...` - содержат примитивы, из который будут собираться модели. Для каждого примитива описываются методы конвертации `toPlatform` и `fromPlatform`
## Пример использования:
Создание моделей с помощью примитивов с репозитория `core`.
```python
import core


class MyModel(core.Module):
    def __init__(self):
        self.conv_block1 = core.Sequential(core.Conv2d(3, 16, 3, 1, 1),
                                           core.BatchNorm2d(16),
                                           core.ReLU())
        self.conv_block2 = core.Sequential(core.Conv2d(32, 64, 3, 1, 1),
                                           core.BatchNorm2d(16),
                                           core.ReLU())
        self.conv_block3 = core.Sequential(core.Conv2d(64, 128, 3, 1, 1),
                                           core.BatchNorm2d(16),
                                           core.ReLU())

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        return out
```

Инициализированную модель, созданную с помощью данного репозитория можно сохранить в формате `platform`, который представляет с собой граф архитектуры сети, описанный в файле `model.json`, а так же веса сети в бинарном файле `model.bin`.
```python
model = MyModel()
model.toPlatform(model_path="path/to/model")

# выходные файлы при сохранении модели в формате platform
output_file_1 >> "path/to/model/model.json"
output_file_2 >> "path/to/model/model.bin"
```

Для обратной конвертации необходимо воспользоваться класс методом: `core.Module.fromPlatform`

```python
model = core.Module.fromPlatform(model_path="path/to/model")
```

## Шаблоны архитектурных модулей (ROADMAP CONCEPT)
Атрибуты шаблонов показывают имена и тип атрибутов, которые должны быть выставлены у наследников, после `__init__`.
Про страйды для подключении голов детекции, FPN, и других неков:
```python
LAYER_NAME=str # Имя слоя который имеет stride!=1
LAYER_INPUT_CHANNELS=int # Число каналов слоя на входе
LAYER_OUTPUT_CHANNELS=int # Число каналов на выходе
GLOBAL_STRIDE_LEVEL=int # Глобальный уровень страйда для слоев, соотвествуют значениям при которых переключается глобальный downsample_factor. TODO: Мб стоит прояснить про local_downsample_factor и global_dowbsample factor (первый относительно модуля-класса, относительно слоя картинки)
# TODO: maybe rename to GLOBAL_DOWNSAMPLE_LEVEL
# TODO: Добавить концепт перехвата выхода финального слоя
class Backbone(Module):
    r"""Класс, определяющий необходимые атрибуты, для описания бэкбонов

    Аргументы:
    - `output_channels` - глубина выходной карты признаков.
    - `downsample_factor` - показатель отношения $ \frac{HW_{in}}{HW_{out}} $ , для ряда декодеров,
    желательно чтобы был целой степенью двойки.
    - `input_channels` -  глубина входной карты признаков.
    """
    output_channels: int
    downsample_factor: int
    input_channels: int
    strides: Dict[GLOBAL_STRIDE_LEVEL, List[Tuple(STRIDED_LAYER, LAYER_INPUT_CHANNELS, LAYER_OUTPUT_CHANNELS)]]
    _trained_additional_meta: dict(
        channel_format = Literal["RGB", "IR", "RGBA", "BGR"]
    )
    @abstractmethod
    def __init__(self) -> None:
        """Must be implemented in child class"""
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Must be implemented in child class"""
        pass
```

## RoadMap:
- [ ] Остыковка от репозитория models.
  - [ ] Оформить README.md описанием и дальнейшими планами.
  - [ ] Создание репозитория `core` или  `m_core`.
- [ ] Смена между фреймворками `torch/plat`.
- [ ] Работы с конвертором:
    - [x] Добавить входные и выходные тензоры для обработки с помощью планировщика.
    - [ ] Добавить тесты конвертации и востановления для сетей\блоков с несколькими входами и выходами.
    - [x] Переработать методы конвертации для слоев `core.functional.interpolate`, `core.upsample`.
    - [x] Переделать работу метода is_leaf. Нужно убрать костыль с `__module__ == "torch.nn.*"` для всех примитивов
    - [ ] Для классов `core.functional` добавить методы `__call__`, которые будут вызывать функции из `torch.nn.functional` или `plat.nn.functional`.
    - [ ] Поменять способ выбора метода конвертации у примитивов core.functional, а так же при обратной конвертации.
    - [ ] ~~Добавить state_dict для ициниализированной модели из формата `platform`~~. Будет реализовано в `kernel_api/utils/loaders`
    - [ ] Переработать механизм обратной конвертации.
    - [x] Отрефакторить классы примитивов.
