import os.path as osp
import os
from pathlib import Path

import albumentations as albu
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor
from PIL import Image
from torch.utils.data import Dataset
import cv2
from ...data_providers.randalbu import RandAlbument

from ...configs.dataset import SegmentationDatasetConfig
from ...utils import make_divisible

from .. import DataProvider, DataProvidersRegistry
from ..classification.csv_annotated import read_csv_ann

cv2.setNumThreads(0)
os.environ["OMP_NUM_THREADS"] = "1"

__all__ = ["SegmentationProvider"]


class SegmentationDataset(Dataset):
    def __init__(
        self,
        annotation: list,
        img_root,
        transform,
    ) -> None:
        """Класс для работы с данными для сегментации, раметка который в виде пар
        `путь к картинке`:`путь к маске`. Маски хранятся в png, где в каждом канале индекс класса для пикселя
        например (1,1.1) либо к PNGp

        Args:
          annotation: (list[tuple]): список пар с путями к картинке и маске относительно img_root
          img_root: (str): путь к папке относительно которой формируются пути к изображениям
          transform: (albumentations.Compouse): собранный пайплайн аугментаций из albumentations
        """

        super().__init__()
        self.ann = annotation
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.ann)

    def _pull_image(self, img_path: Path):
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        return image

    def _pull_mask(self, mask_path: Path):
        mask = Image.open(mask_path)
        mask = np.array(mask)
        return mask

    def __getitem__(self, index):
        record = self.ann[index]
        img_filename = record[0]
        mask_filename = record[1]
        img_path = Path(osp.join(self.img_root, img_filename))
        mask_path = Path(osp.join(self.img_root, mask_filename))

        image = self._pull_image(img_path)
        mask = self._pull_mask(mask_path)

        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return {
            "image": image,
            "target": mask,
            "image_path": str(img_path),
        }


@DataProvidersRegistry.registry
class SegmentationProvider(DataProvider):
    """Провайдер для сегментации,

    работает на фиксированном размере изображения"""

    NAME = "Segmentation"

    def __init__(
        self,
        init_config: SegmentationDatasetConfig,
        world_size: int = None,
        rank: int = None,
    ):
        self.ann_file = osp.join(init_config.dataset_path, init_config.ann_file)
        self.train_crop_ratio = init_config.pipeline_crop_train

        self.train_ann, self.test_ann = read_csv_ann(self.ann_file)

        super().__init__(init_config, world_size, rank)

    def build_train_transform(self, image_size=None):
        if image_size is None:
            image_size = self.image_size
        divisor = 32
        crop_height = make_divisible(self.train_crop_ratio * image_size[0], divisor)
        crop_width = make_divisible(self.train_crop_ratio * image_size[1], divisor)

        train_transform = [
            albu.Resize(*image_size),
            albu.HorizontalFlip(p=0.5),
        ]
        if self.use_randalbu:
            ra = RandAlbument(
                image_size,
                num_ops=self.randalbu_num_ops,
                magnitude=self.randalbu_magnitude,
            )
            train_transform.append(ra)
        train_transform += [
            albu.RandomCrop(height=crop_height, width=crop_width, always_apply=True),
            self.normalize,
            ToTensor(transpose_mask=True),
        ]
        return albu.Compose(train_transform)

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.image_size
        test_transform = [
            #! TODO REFACTOR
            # Resize -- временное решение
            albu.Resize(*image_size),
            self.normalize,
            ToTensor(transpose_mask=True),
        ]
        return albu.Compose(test_transform)

    def train_dataset_builder(
        self, train_transforms, is_train=True
    ) -> SegmentationDataset:
        return SegmentationDataset(
            self.train_ann,
            self.dataset_path,
            train_transforms,
        )

    def test_dataset_builder(self, test_transforms) -> SegmentationDataset:
        # использование val_path в test_dataset тянется с исходной репы
        return SegmentationDataset(
            self.test_ann,
            self.dataset_path,
            test_transforms,
        )
