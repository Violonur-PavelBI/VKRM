from typing import Dict, List, Literal, Tuple, Union

from pydantic import BaseModel

from dltools.configs.dataset import DatasetConfigsUnion


class ExperimentConfig(BaseModel):
    use_amp: bool
    debug: bool
    dont_use_env: bool
    exp_prefix: str = ""
    exp_root_path: str
    resume: bool = False
    seed: Union[int, None] = None
    deterministic: bool = False
    validation_frequency: int
    workdir: Union[str, None] = None
    teacher_path: Union[str, None] = None
    supernet_checkpoint_path: Union[str, None] = None
    clearml_project: Union[str, None] = None
    clearml_task: Union[str, None] = None
    device: Literal["cuda", "cpu"] = "cuda"
    local_world_size_on_cpu: int = 1


class BackboneConfig(BaseModel):
    """
    Параметры MobileNet: `use_se`, `act_func`, `n_stages`.

    Параметры ResNet: `base_depth_list`, `downsample_mode`.
    """

    type: str
    ks_list: List[int]
    depth_list: List[int]
    expand_ratio_list: List[float]
    width_mult_list: List[float]
    use_se: bool = False
    act_func: str = "relu"
    n_stages: int = 5
    base_stage_width: List[int] = [16, 16, 24, 40, 80, 112, 160]
    base_depth_list: List[int] = [2, 2, 4, 2]
    downsample_mode: str = "avgpool_conv"
    bn_eps: float
    bn_momentum: float
    bn_param: Union[Tuple[float, float], None] = None
    catch_counter: Union[int, None] = None

    def __init__(self, **data) -> None:
        data["bn_param"] = (data["bn_momentum"], data["bn_eps"])
        super().__init__(**data)


class NeckConfig(BaseModel):
    """
    type: str -- Семейство сети.
    levels: int -- Количество уровней.
    in_channels: list[list[int] | int] -- Список списков входных каналов.
        Внешний список по уровням, внутренний по возможному числу каналов.
        Аргумент не задаётся в конфиге, но забирается от бэкбона.
    mid_channels: list[int] | int -- Список возможных промежуточных каналов.
        Значение одинаково для всех уровней.
        Активное число подбирается NAS-ом из указанных в конфиге вариантов.
    out_channels: list[list[int]] | list[int] -- Список списков выходных каналов.
        Внешний список по уровням, внутренний по возможному числу каналов.
        Активное число подбирается NAS-ом из указанных в конфиге вариантов.
    kernel: int -- Ядро свёрток после слияния карт.
        Не используется, если `convs` != `None`.
    convs: list[list[tuple[int, int]]] -- Описание свёрток после слияния карт.
        Внешний список по уровням, внутренний по порядку применения.
        Каждая свёртка задаётся количеством каналов и размером ядра.
        Количество каналов последних свёрток уровней должно совпадать с `out_channels`.
    merge_policy: str -- `'sum'` | `'cat'`
    act_func: str -- Активация.
    use_bias: bool -- Использовать bias в свёртках или нет.
    upsample_mode: str -- `'deconv'` | `'nearest'` | `'bilinear'` | `'bicubic'`
        В NAS FPN нет варианта `'deconv'`.
    """

    type: str
    levels: int
    in_channels: Union[List[Union[List[int], int]], None] = None
    mid_channels: int

    # dynamic
    convs: Union[List[List[Tuple[int, int]]], None] = None

    # nas
    out_channel_list: Union[List[int], None] = None
    depth_list: Union[List[int], None] = None
    kernel_list: Union[List[int], None] = None
    width_list: Union[List[float], None] = None

    # zero segmentation head
    single_out: bool = False

    merge_policy: Literal["sum", "cat"] = "sum"
    act_func: str = "relu"
    use_bias: bool
    upsample_mode: str


class HeadConfig(BaseModel):
    type: str
    n_classes: int
    use_bias: bool


class ClassificationHeadConfig(HeadConfig):
    type: Literal["DynamicClassificationHead"]
    in_channels: Union[List[int], int, None] = None
    dropout_rate: float
    act_func: Union[str, None] = None


class SegmentationHeadConfig(HeadConfig):
    type: Literal[
        "SegmentationHead",
        "DynamicSegmentationHead",
        "ZeroSegmentationHead",
        "DynamicZeroSegmentationHead",
    ]
    levels: Union[int, None] = None
    in_channels: List[int]
    width_list: Union[List[float], None] = None
    mid_channels: Union[int, None] = None
    upsample_mode: str
    upsample_factor: int
    final_conv_kernel: int
    merge_policy: Union[str, None] = None


class HeatmapKeypointsHeadConfig(HeadConfig):
    type: Literal["HeatmapKeypointsHead"]
    levels: int
    in_channels: int
    kernel: int
    upsample_mode: str
    upsample_factor: int


class YoloV4DetectionHeadConfig(HeadConfig):
    type: Literal["YoloV4DetectionHead", "DynamicYoloV4DetectionHead"]
    levels: int
    in_channels: List[int]
    width_list: Union[List[float], None] = None
    conf_thresholds: Union[float, List[float]]
    strides: List[int]
    anchors: List[List[List[int]]]
    kernel: int
    nms_iou_threshold: float
    nms_top_k: int
    image_size: Tuple[int, int]
    auto_anchors: bool = True


class YoloV7KeypointsHeadConfig(HeadConfig):
    type: Literal["YoloV7KeypointsHead", "DynamicYoloV7KeypointsHead"]
    in_channels: List[Union[int, List[int]]]
    width_list: Union[List[float], None] = None
    conf_threshold: float
    nms_iou_threshold: float
    nms_top_k: int
    strides: List[int]
    anchors: List[List[List[int]]]
    inplace: bool = True
    n_kpt: Union[int, None]
    dw_conv_kpt: bool = False
    auto_anchors: bool = False
    kpt_factor: int = 1
    v7mode: bool = False
    image_size: Tuple[int, int]


HeadConfigsUnion = Union[
    ClassificationHeadConfig,
    SegmentationHeadConfig,
    HeatmapKeypointsHeadConfig,
    YoloV4DetectionHeadConfig,
    YoloV7KeypointsHeadConfig,
]


class SupernetConfig(BaseModel):
    type: str
    backbone: BackboneConfig
    neck: Union[NeckConfig, None]
    head: HeadConfigsUnion
    pretrain: Union[None, Literal["coco", "imagenet"]] = None


class StrategyConfig(BaseModel):
    type: str
    main_metric: str


class ClassificationStrategyConfig(StrategyConfig):
    type: Literal["ClassificationStrategy"]
    main_metric: str = "top1"
    soft_target: bool = False
    label_smoothing: float = 0.1
    weight_for_loss: Union[None, str] = None


class SegmentationStrategyConfig(StrategyConfig):
    type: Literal["SegmentationStrategy"]
    main_metric: str = "IoU"
    loss: str = "BCELoss"
    distil_loss: str = "none"
    background_loss: bool = True
    dice_weight: float = 0.0
    lovasz_weight: float = 0.0
    ce_weight: float = 0.0
    bce_weight: float = 0.0
    mse_weight: float = 0.0


class HeatmapKeypointsStrategyConfig(StrategyConfig):
    type: Literal["HeatmapKeypointsStrategy"]
    main_metric: str = "negME"
    loss: str


class DetectionYoloV4StrategyConfig(StrategyConfig):
    """
    cls_pw: float -- `pos_weight` классов в `BCEWithLogitsLoss` (TODO: убрать или заменить на вектор?)
    obj_pw: float -- `pos_weight` объектности в `BCEWithLogitsLoss`
    iou_loss_ratio: float -- доля IoU в лоссе объектности
    iou_type: str -- `'GIoU'` | `'DIoU'` | `'CIoU'` | `'EIoU'` | `'ECIoU'`
        При другом входе используется обычный IoU.
    obj_loss_balance: list[float] -- Коэффициенты лосса объектности для разных уровней
        в порядке увеличения страйда
    box_loss_scale: float -- Коэффициент лосса боксов
    obj_loss_scale: float -- Коэффициент лосса объектности
    cls_loss_scale: float -- Коэффициент лосса классификации
    label_smoothing: float -- Коэффициент label smoothing для классов и объектности
        [0, 1] -> [label_smoothing / 2, 1 - label_smoothing / 2]
    focal_loss_gamma: float -- Коэффициент гамма в FocalLoss(BCEWithLogitsLoss)
        для классов и объектности
    """

    type: Literal["YoloV4DetectionStrategy"]
    main_metric: str = "map"
    cls_pw: float = 1.0
    obj_pw: float = 1.0
    iou_loss_ratio: float = 1.0
    iou_type: str = "CIoU"
    obj_loss_balance: List[float] = [4.0, 1.0, 0.4]
    box_loss_scale: float = 0.05
    obj_loss_scale: float = 0.6
    cls_loss_scale: float = 0.015
    label_smoothing: float = 0
    focal_loss_gamma: float = 0


class YoloV7KeypointsStrategyConfig(StrategyConfig):
    """Фактически повторяет конфиг YoloV4, но добавляет пару полей для точек"""

    type: Literal["YoloV7KeypointsStrategy"]
    main_metric: str = "kp_acc"
    n_kpt: int
    cls_pw: float = 1.0
    obj_pw: float = 1.0
    iou_loss_ratio: float = 1.0
    iou_type: str = "CIoU"
    obj_loss_balance: List[float] = [4.0, 1.0, 0.4]
    box_loss_scale: float = 0.05
    obj_loss_scale: float = 0.6
    cls_loss_scale: float = 0.015
    kpt_loss_scale: float = 0.1
    label_smoothing: float = 0
    focal_loss_gamma: float = 0
    kpt_label: bool = True
    anchor_t: float = 4
    weigh_kps_in_box: bool = True
    weigh_boxes_in_image: bool = False
    kpt_loss_type: Literal["oks", "lp", "huber", "wing"] = "oks"
    oks_loss_type: Literal["default", "yolov8-formula", "yolov8-cocoeval"] = "default"
    oks_loss_denom: Literal["area", "hypotenuse", "hypotenuse_squared"] = "area"
    oks_loss_sigmas: Union[float, List[float], None] = None
    lp_loss_power: float = 1
    huber_loss_delta: float = 10
    wing_loss_w: float = 10
    wing_loss_e: float = 2


StrategyConfigsUnion = Union[
    ClassificationStrategyConfig,
    SegmentationStrategyConfig,
    HeatmapKeypointsStrategyConfig,
    DetectionYoloV4StrategyConfig,
    YoloV7KeypointsStrategyConfig,
]


class LearnConfig(BaseModel):
    task: Union[str, None] = None
    opt_type: str = "adam"
    momentum: float = 0.9
    nesterov: bool = True
    init_lr: float = 0.05
    weight_decay: float = 4e-5
    no_decay_keys: str = "bn#bias"
    betas: Tuple[float, float] = (0.9, 0.999)
    dynamic_batch_size: int = 1
    n_epochs: int = 100
    warmup_epochs: int = 10
    warmup_lr: float = 0
    lr_schedule_type: str = "cosine"
    kd_ratio: float = 0  # TODO move to strategy ?
    kd_type: str = "ce"  # TODO move to strategy
    teach_choice_strategy: Literal["old", "big"] = "old"
    subnet_sample_strategy: Literal["old", "sandwich", "big_and_other"] = "old"
    temperature: Union[float, None] = None  # TODO move to strategy
    grad_clip_value: Union[float, None] = None
    grad_clip_norm: Union[float, None] = None
    early_stopping: bool = False
    pre_stop_epochs: int = 3
    early_stopping_lyambda: float = 0.5
    pipe_width: float = 0.001
    weight_loss: bool = True
    iters_to_accumulate: Union[int, None] = None
    use_sam: bool = False


class PredictorsConfig(BaseModel):
    accuracy_predictor_path: Union[str, None] = None
    latency_predictor_path: Union[str, None] = None
    efficiency_predictor_path: Union[str, None] = None


class ThresholdSearchParams(BaseModel):
    """
    Args:
        - goals: {metric_name: (threshold, greater_is_better)}
        E. g.:
            - detection:
                goals = {
                    "recall": (0.8, True),
                    "imprecision": (0.1, False)
                }
            - keypoints:
                goals = {
                    "kp_acc": (0.85, True)
                }
    """

    metric: Literal["ImprecisionRecall", "KeypointAccuracy"]
    goals: Dict[str, Tuple[float, bool]]
    min_conf: float = 0.01
    max_conf: float = 0.99
    num_thresholds: int = 99
    nms_iou_search: bool = True
    double_check: bool = False
    search_on_subnet: bool = True


class CommonConfig(BaseModel):
    exp: ExperimentConfig
    supernet_config: SupernetConfig
    predictors: Union[PredictorsConfig, None] = None
    dataset: Union[DatasetConfigsUnion, None] = None
    strategy: Union[StrategyConfigsUnion, None] = None
    learn_config: Union[LearnConfig, None] = None
    threshold: Union[ThresholdSearchParams, None] = None
    config_path: Union[str, None] = None
    world_size: int = 1
    rank: int = 0
    master_addr_full: str = "localhost:28500"
    ngpus_per_node: int = 1


class SupernetLearnStageConfig(CommonConfig):
    stage_type: Literal["SupernetLearn"]


class BuildAccDatasetParams(BaseModel):
    n_subnets: int = 4000
    det_grid: bool = False
    n_data_samples: int = 2000
    save_nets: bool = False
    threshold_calibrate: bool = False


class PredictorDatasetStageConfig(CommonConfig):
    stage_type: Literal["ArchAccDatasetBuild"]
    build_acc_dataset: BuildAccDatasetParams


class PredictorLearnParams(BaseModel):
    metric: Literal["accuracy", "latency", "efficiency"] = "accuracy"
    model: Literal["Perceptron", "Catboost"] = "Perceptron"
    n_epochs: int = 100
    init_lr: float = 0.003
    data_path: Union[str, None] = None
    image_size_list: Union[List[Tuple[int, int]], None] = None
    height_list: Union[List[int], None] = None
    width_list: Union[List[int], None] = None


class PredictorLearnStageConfig(CommonConfig):
    stage_type: Literal["PredictorLearn"]
    pred_learn: PredictorLearnParams


class EvolutionParams(BaseModel):
    evolution_type: str = "simple_with_latency_threshold"
    generations: int = 30
    population_size: int = 200
    parent_ratio: float = 0.25
    mutation_ratio: float = 0.5
    mutate_probability: float = 0.1
    optimize_val: Literal["accuracy", "latency", "efficiency"] = "accuracy"
    constraint_type: Literal["accuracy", "latency", "efficiency"] = "accuracy"
    accuracy_constraint: Union[float, List[float]] = 0.01
    evolution_accuracy_constraint: Union[float, None] = None
    accdict_accuracy_constraint: Union[float, None] = None
    efficiency_constraint: float = 0.01
    flops_constraint: float = 1000000
    latency_constraint: float = 1000
    preservation_of_interface: Literal["default", "not", "local"] = "default"
    use_es: bool = True
    predictor_err_compensation: float = 1.02


class EvolutionStageConfig(CommonConfig):
    stage_type: Literal["EvolutionOnSupernet"]
    evolution: EvolutionParams


class FinetuneStageConfig(CommonConfig):
    stage_type: Literal["Finetune"]
    choose_best: bool = True
    is_test: bool = True


class ThresholdSearchStageConfig(CommonConfig):
    stage_type: Literal["ThresholdSearch"]


StageConfigsUnion = Union[
    SupernetLearnStageConfig,
    PredictorDatasetStageConfig,
    PredictorLearnStageConfig,
    EvolutionStageConfig,
    FinetuneStageConfig,
    ThresholdSearchStageConfig,
]


def update_stage_config(stage_config, global_config):
    for k, v in global_config.items():
        if k not in stage_config or stage_config[k] is None:
            stage_config[k] = v
        elif isinstance(v, Dict):
            update_stage_config(stage_config[k], global_config[k])


class Config(BaseModel):
    common: CommonConfig
    stages: Dict[str, StageConfigsUnion]
    execute: List[str]
    raw_config: Union[Dict, None] = None

    def __init__(self, **data) -> None:
        for stage_config in data["stages"].values():
            update_stage_config(stage_config, data["common"])
        super().__init__(**data)
