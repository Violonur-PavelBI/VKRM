from pydantic import BaseModel
from pydantic import confloat
from typing import Union, Literal, Tuple, List

__all__ = ["DatasetConfig"]


class DatasetConfig(BaseModel):
    type: str
    name: str = ""
    n_classes: int
    dataset_path: str
    valid_size: Union[float, None] = None
    train_batch_size: int
    test_batch_size: int
    image_size: Tuple[int, int]
    n_worker: int
    use_randalbu: bool = False
    randalbu_num_ops: int = 2
    randalbu_magnitude: int = 15
    pin_memory: bool = True
    use_norm: bool = True
    mean: List = [0.5, 0.5, 0.5]
    std: List = [0.5, 0.5, 0.5]


class CSVAnnotatedDatasetConfig(DatasetConfig):
    type: Literal["CSVAnnotated"]
    ann_file: str


class SegmentationDatasetConfig(DatasetConfig):
    type: Literal["Segmentation"]
    ann_file: str
    background: Literal["add", "exist", "no"]
    pipeline_crop_train: confloat(gt=0, le=1)
    classes_idx: Union[List[int], None] = None
    classes_unused: Union[List[int], None] = None


class HeatmapKeypointsDatasetConfig(DatasetConfig):
    type: Literal["HeatmapKeypoints"]
    ann_file_train: str
    ann_file_test: str
    sigma: int
    pad: bool
    classes_idx: Union[List[int], None] = None


class DetectionYoloV4DatasetConfig(DatasetConfig):
    type: Literal["DetectionYoloV4"]
    img_train_prefix: str = ""
    img_test_prefix: str = ""
    ann_file_train: str = "train.json"
    ann_file_test: str = "test.json"
    augs: str = "min"
    mixup_prob: confloat(ge=0, le=1) = 0
    mosaic_prob: confloat(ge=0, le=1) = 0
    mosaic_offset: confloat(ge=0.1, le=0.5) = 0.2


class KeypointsYoloV7DatasetConfig(DatasetConfig):
    type: Literal["KeypointsYoloV7"]
    img_train_prefix: str = ""
    img_test_prefix: str = ""
    ann_file_train: str = "train.json"
    ann_file_test: str = "test.json"
    n_kpt: int
    kpt_label: bool = True
    mixup_prob: confloat(ge=0, le=1) = 0
    mosaic_prob: confloat(ge=0, le=1) = 0
    mosaic_offset: confloat(ge=0.1, le=0.5) = 0.2


DatasetConfigsUnion = Union[
    CSVAnnotatedDatasetConfig,
    SegmentationDatasetConfig,
    HeatmapKeypointsDatasetConfig,
    DetectionYoloV4DatasetConfig,
    KeypointsYoloV7DatasetConfig,
]
