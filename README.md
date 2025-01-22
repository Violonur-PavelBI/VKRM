# How to use

## Создание окружения
В репозитории есть файл для docker-compose

## Weird trick for better life

```bash
python script.py -h
```

## Train
```bash
python main.py --config-path config.yaml 
```

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
### Преобразование результатов в датасет для обучения предиктора
1.  Переместить json(-ы) с железок в папку `FOLDER`.
2.  Спарсить:
```bash
python hardware_stats_to_dataset.py --in-dir FOLDER [--out-dir OUT_DIR] 
``` 
3. В `OUT_DIR` будут файлы, пути к которым можно указывать в конфигах для обучения предикторов.
