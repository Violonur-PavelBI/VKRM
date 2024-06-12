import os
from abc import ABC, abstractmethod
from typing import Callable, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler

from ..configs import DatasetConfig
from ..utils import DistributedSubsetSampler
from .randalbu import RandAlbument

__all__ = ["DataProvider"]


from albumentations import ImageOnlyTransform


class Div255(ImageOnlyTransform):
    """Div image by 255"""

    def __init__(self, always_apply: bool = False, p: float = 1):
        super().__init__(always_apply, p)
        self.max_pixel_value = 255

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img.astype(np.float32) / self.max_pixel_value

    def get_transform_init_args_names(self):
        return ("max_pixel_value",)


class DataProvider(ABC):
    """Базовый провайдер. Фактически является провайдером для классификацией
    и часть функционала нужно вынести отсюда, и сделать методы абстрактными"""

    collate_fn: Union[Callable, None] = None

    def __init__(
        self, init_config: DatasetConfig, world_size: int = None, rank: int = None
    ):
        self.n_classes = init_config.n_classes
        self.dataset_path = init_config.dataset_path
        self.valid_size = init_config.valid_size
        self.train_batch_size = init_config.train_batch_size
        self.test_batch_size = init_config.test_batch_size
        self.image_size = init_config.image_size
        self.n_worker = init_config.n_worker
        self.pin_memory = init_config.pin_memory
        self.world_size = world_size
        self.rank = rank
        self.use_randalbu = init_config.use_randalbu
        self.randalbu_num_ops = init_config.randalbu_num_ops
        self.randalbu_magnitude = init_config.randalbu_magnitude
        self.use_norm = init_config.use_norm
        self.mean = init_config.mean
        self.std = init_config.std

        self._train_path = os.path.join(self.dataset_path, "train")
        self._val_path = os.path.join(self.dataset_path, "val")

        train_transform = self.build_train_transform()
        valid_transforms = self.build_valid_transform()
        self.train_dataset = self.train_dataset_builder(train_transform)
        self.test_dataset = self.test_dataset_builder(valid_transforms)

        # build samplers
        if self.valid_size is not None:
            # train test split from train dataset
            if not isinstance(self.valid_size, int):
                assert isinstance(self.valid_size, float) and 0 < self.valid_size < 1
                self.valid_size = int(len(self.train_dataset) * self.valid_size)

            self.valid_dataset = self.train_dataset_builder(
                valid_transforms, is_train=False
            )
            train_indexes, valid_indexes = self.random_sample_valid_set(
                len(self.train_dataset), self.valid_size
            )

            if self.world_size is not None:
                self.train_sampler = DistributedSubsetSampler(
                    self.train_dataset,
                    self.world_size,
                    self.rank,
                    True,
                    np.array(train_indexes),
                )
                print(1)
                self.valid_sampler = DistributedSubsetSampler(
                    self.valid_dataset,
                    self.world_size,
                    self.rank,
                    True,
                    np.array(valid_indexes),
                )
                self.test_sampler = DistributedSampler(
                    self.test_dataset, self.world_size, self.rank
                )
            else:
                print(2)
                self.train_sampler = SubsetRandomSampler(train_indexes)
                self.valid_sampler = SubsetRandomSampler(valid_indexes)
                self.test_sampler = None
        else:
            self.valid_dataset = self.test_dataset
            if self.world_size is not None:
                print(3.1)
                self.train_sampler = DistributedSampler(
                    self.train_dataset, self.world_size, self.rank
                )
                self.test_sampler = DistributedSampler(
                    self.test_dataset, self.world_size, self.rank
                )
            else:
                print(3.2)
                self.train_sampler = None
                self.test_sampler = None
            print(3)
            self.valid_sampler = self.test_sampler

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.train_loader = self.train_loader_builder()
        self.valid_loader = self.valid_loader_builder()
        self.test_loader = self.test_loader_builder()

    # TODO!!! REFACTOR Вернул кеширование лоадера, пока таким вот костыльным способом
    # Не сказать, что это костыль, это просто ленивая инициализация лоадера
    def train_loader_builder(self) -> DataLoader:
        if self.train_loader is None:
            print(self.train_sampler,self.train_sampler.rank ,self.train_sampler.shuffle ,self.world_size,(False if self.train_sampler else True))
            self.train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.train_batch_size,
                sampler=self.train_sampler,
                shuffle=(False if self.train_sampler else True),
                num_workers=self.n_worker,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory ,
                persistent_workers=self.n_worker > 0,
            )
        return self.train_loader

    def valid_loader_builder(self) -> DataLoader:
        if self.valid_loader is None:
            print(self.valid_sampler,self.valid_sampler.rank ,self.valid_sampler.dataset,(False if self.valid_sampler else True))
            self.valid_loader = DataLoader(
                dataset=self.valid_dataset,
                batch_size=self.test_batch_size,
                sampler=self.valid_sampler,
                shuffle=False,
                num_workers=self.n_worker,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
                persistent_workers=self.n_worker > 0,
            )
        return self.valid_loader

    def test_loader_builder(self) -> DataLoader:
        if self.test_loader is None:
            self.test_loader = DataLoader(
                dataset=self.test_dataset,
                batch_size=self.test_batch_size,
                sampler=self.test_sampler,
                shuffle=False ,
                num_workers=self.n_worker,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
                persistent_workers=self.n_worker > 0,
            )
        return self.test_loader

    @abstractmethod
    def train_dataset_builder(self, train_transforms, is_train=True) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def test_dataset_builder(self, test_transforms) -> Dataset:
        raise NotImplementedError

    @property
    def normalize(self):
        if self.use_norm:
            return A.Normalize(mean=self.mean, std=self.std)
        return Div255()

    def build_train_transform(self, image_size=None):
        if image_size is None:
            image_size = self.image_size

        train_transforms = [
            A.Resize(self.image_size[0], self.image_size[1]),
        ]

        if self.use_randalbu:
            ra = RandAlbument(
                image_size,
                num_ops=self.randalbu_num_ops,
                magnitude=self.randalbu_magnitude,
            )
            train_transforms.append(ra)

        train_transforms += [
            self.normalize,
            ToTensorV2(),
        ]

        train_transforms = A.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.image_size
        return A.Compose(
            [
                A.Resize(self.image_size[0], self.image_size[1]),
                self.normalize,
                ToTensorV2(),
            ]
        )

    def build_sub_train_loader(
        self,
        n_images,
        batch_size,
        num_worker=None,
        world_size=None,
        rank=None,
        cache_loader=True,
    ):
        """Эта функция используется когда нам нужно обновить статистики батч норма на валидации.
        семплы кешируются в оперативную память в лист"""
        subtrain_key = f"sub_train_{self.image_size}"
        if self.__dict__.get(subtrain_key, None) is None:
            if num_worker is None:
                num_worker = self.n_worker

            n_samples = len(self.train_dataset)
            g = torch.Generator()
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()

            new_train_dataset = self.train_dataset_builder(
                self.build_train_transform(image_size=self.image_size)
            )
            chosen_indexes = rand_indexes[:n_images]
            # TODO Refactor -- выпилить DistributedSubsetSampler
            # Использовать torch.utils.data.random_split для создания подмножества
            if world_size is not None:
                sub_sampler = DistributedSubsetSampler(
                    new_train_dataset,
                    world_size,
                    rank,
                    True,
                    np.array(chosen_indexes),
                )

            else:
                sub_sampler = None
            sub_data_loader = DataLoader(
                new_train_dataset,
                batch_size=batch_size,
                sampler=sub_sampler,
                num_workers=num_worker,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
            )
            # TODO refactor  это кешинрование не факт, что хорошо
            if cache_loader:
                self.__dict__[subtrain_key] = []
                for data in sub_data_loader:
                    self.__dict__[subtrain_key].append(data)
            else:
                self.__dict__[subtrain_key] = sub_data_loader
        return self.__dict__[subtrain_key]

    @staticmethod
    def random_sample_valid_set(train_size, valid_size):
        """делает два тензора и индексами для обучения и валидации"""
        assert train_size > valid_size

        g = torch.Generator()
        rand_indexes = torch.randperm(train_size, generator=g).tolist()

        valid_indexes = rand_indexes[:valid_size]
        train_indexes = rand_indexes[valid_size:]
        return train_indexes, valid_indexes
