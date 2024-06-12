# How to use

## Создание окружения
В репозитории предусмотрена возможность использования базового окружения и окружения для разработки.
Они отличаются зависимостями и docker файлом.
В репозитории есть файл для docker-compose

## Weird trick for better life

```bash
python script.py -h
```

## Train
```bash
python main.py --config-path config.yaml 
```

## Evaluate

TODO: 
- Добавить пути к тетрадкам для просмотра результатов обучения на различных задачах
- Добавить пример кода для загрузки модели из json
- Добавить ссылку на interfaces и пример работы с interface.pt

## Launch experiments

### Run
```bash
python experiment_runner.py --base-config-path config.yaml --exp-config-path ./configs/old/exp_segmentation_config_example.yaml --output-dir ./output/experiment --datasets-dir /workspace/labs_db/paradigma/
```
### Resume
```bash
python experiment_runner.py --base-config-path config.yaml --exp-config-path ./configs/old/exp_segmentation_config_example.yaml --output-dir ./output/experiment --datasets-dir /workspace/labs_db/paradigma/ --resume
```
#### Resume with a passage through all stages
```bash
python experiment_runner.py --base-config-path config.yaml --exp-config-path ./configs/old/exp_segmentation_config_example.yaml --output-dir ./output/experiment --datasets-dir /workspace/labs_db/paradigma/ --resume --from-scratch
```

## Получение датасета для предиктора характеристик на железках

### Создание и конвертация сетей
1. Необходим hpm_scheduler со всеми разрешениями (`chmod 777`), при отсутствии:
```bash
cd HPM_SCHEDULER_PATH 
git clone http://10.24.65.46:999/high-performance-computing-module/software/hpm_scheduler
cd hpm_scheduler
make .
```
2. Генерация сетей:
```bash
python hardware_convert.py --ins CONFIG_PATH_1 CONFIG_PATH_2 --out FOLDER --image-size HEIGHT WIDTH --classes N_CLASSES --systems module elbrus [--hpm HPM_SCHEDULER_PATH --det --nets NUM_NETS]
```
### Работа на вычислителе
1. Для корректной работы необходимо совпадение версий hpm_scheduler на железке с локальной версией при конвертации в пункте выше.
2. Запуск скриптов:
- Эльбрус `root@10.24.65.222` (пароль ***) 
  1. Переместить папку с сконвертированными сетями в `/home/nm6408/module_test/demo/ofa_supernet_test/FOLDER`
  2. Получение характеристик
  ```bash
  cd /home/nm6408/module_test/validator
  ./net_stats home/nm6408/module_test/demo/ofa_supernet_test/FOLDER path/to/result.json input.json
  ```
- Модуль `nm6408@10.24.65.123` (пароль ***)
  1. Переместить папку с сконвертированными сетями в `/home/user/demo_paradigma/demo/ofa_supernet_test/FOLDER`
  2. Получение характеристик
  ```bash
  cd /home/user/demo_paradigma/validator/VERSION_FOLDER
  ./test_elbrus -net=/home/user/demo_paradigma/demo/ofa_supernet_test/FOLDER -out=path/to/result.json -inp=input.json [-lam=5]
  ```
### Преобразование результатов в датасет для обучения предиктора
1.  Переместить json(-ы) с железок в папку `FOLDER`.
2.  Спарсить:
```bash
python hardware_stats_to_dataset.py --in-dir FOLDER [--out-dir OUT_DIR] 
``` 
3. В `OUT_DIR` будут файлы, пути к которым можно указывать в конфигах для обучения предикторов.


# Разработчикам
- В репозитории используется репозиторий модель из Платформы. В продакшене оно стоит уже в базовом образе. Для разработки же надо скачать submodule и поставить в режиме разработчика.
- Разработка идёт по принципу: новая фича -> новая ветка.
- для форматирования используется:
  - `black` для кода;
  - `isort` для сортировки импортов.
- В коде должны максимально использоваться аннотация типов.
- При использовании argparse следует добавлять описание аргументов (help=) для облегчения понимания.
- В репу не заливаются файлы для IDE.
- docstring функций/классов должны быть актуальными:
  - После работы с классом/функцией нужно убедиться, что её docstring актуален.
  - Во время работы с классом/функцией следует написать/дописать docstring, если там ничего нет.
- Jupyter тетрадки должны заливаться без изображений (С очищенными выходами ячеек). Только если эти картинки в качестве документации.
- TODO: возможно стоит написать про рекомендуемые плагины.
