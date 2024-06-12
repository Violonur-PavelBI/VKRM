import random
from typing import Dict, List, Tuple, TypedDict, Union

import numpy as np

from .abstract import ArchEncoder
from .backbones import MobileNetArchEncoder, ResNetArchEncoder
from .necks import FPNEncoder
from .heads import SegmentationHeadEncoder


class ArchDesc(TypedDict):
    backbone: Dict[str, List[Union[int, float]]]
    neck: Union[Dict[str, Union[int, List[int], str]], None]
    head: None
    r: Tuple[int, int]


class CompositeEncoder(ArchEncoder):
    def __init__(
        self,
        backbone_encoder: Union[MobileNetArchEncoder, ResNetArchEncoder],
        neck_encoder: Union[FPNEncoder, None],
        head_encoder: Union[SegmentationHeadEncoder, None],
        image_size_list: List[Tuple[int, int]],
    ):
        self.backbone_encoder = backbone_encoder
        self.neck_encoder = neck_encoder
        self.head_encoder = head_encoder
        self.hs_list = [size[0] for size in image_size_list]
        self.ws_list = [size[1] for size in image_size_list]

        self.backbone_dim = self.backbone_encoder.n_dim
        self.neck_dim = self.neck_encoder.n_dim if self.neck_encoder is not None else 0
        self.head_dim = self.head_encoder.n_dim if self.head_encoder is not None else 0
        self.n_dim = self.backbone_dim + self.neck_dim + self.head_dim

        self.h_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="h")
        self.w_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="w")

    def _build_info_dict(self, target):
        if target == "h":
            target_dict = self.h_info
            choices = self.hs_list
        elif target == "w":
            target_dict = self.w_info
            choices = self.ws_list
        else:
            raise NotImplementedError

        target_dict["L"].append(self.n_dim)
        for val in choices:
            target_dict["val2id"][val] = self.n_dim
            target_dict["id2val"][self.n_dim] = val
            self.n_dim += 1
        target_dict["R"].append(self.n_dim)

    def arch2feature(self, arch_dict: ArchDesc) -> np.ndarray:
        arch_backbone = arch_dict["backbone"]
        arch_neck = arch_dict["neck"]
        arch_head = arch_dict["head"]
        h, w = arch_dict["r"]

        feature = np.zeros(self.n_dim)

        feature_backbone = self.backbone_encoder.arch2feature(arch_backbone)
        feature[: self.backbone_dim] = feature_backbone

        if self.neck_encoder is not None:
            feature_neck = self.neck_encoder.arch2feature(arch_neck)
            feature[self.backbone_dim : self.backbone_dim + self.neck_dim] = (
                feature_neck
            )

        if self.head_encoder is not None:
            feature_head = self.head_encoder.arch2feature(arch_head)
            feature[
                self.backbone_dim
                + self.neck_dim : self.backbone_dim
                + self.neck_dim
                + self.head_dim
            ] = feature_head

        feature[self.h_info["val2id"][h]] = 1
        feature[self.w_info["val2id"][w]] = 1

        return feature

    def feature2arch(self, feature: np.ndarray) -> ArchDesc:
        feature_backbone = feature[: self.backbone_dim]
        arch_backbone = self.backbone_encoder.feature2arch(feature_backbone)

        if self.neck_encoder is not None:
            feature_neck = feature[
                self.backbone_dim : self.backbone_dim + self.neck_dim
            ]
            arch_neck = self.neck_encoder.feature2arch(feature_neck)
        else:
            arch_neck = None

        if self.head_encoder is not None:
            feature_head = feature[
                self.backbone_dim
                + self.neck_dim : self.backbone_dim
                + self.neck_dim
                + self.head_dim
            ]
            arch_head = self.head_encoder.feature2arch(feature_head)
        else:
            arch_head = None

        for j in range(self.h_info["L"][0], self.h_info["R"][0]):
            if feature[j] == 1:
                h = self.h_info["id2val"][j]
        for j in range(self.w_info["L"][0], self.w_info["R"][0]):
            if feature[j] == 1:
                w = self.w_info["id2val"][j]
        img_sz = h, w

        arch_dict: ArchDesc = {
            "backbone": arch_backbone,
            "neck": arch_neck,
            "head": arch_head,
            "r": img_sz,
        }
        return arch_dict

    def random_sample_arch(self) -> ArchDesc:
        arch_backbone = self.backbone_encoder.random_sample_arch()

        if self.neck_encoder is not None:
            arch_neck = self.neck_encoder.random_sample_arch()
        else:
            arch_neck = None

        if self.head_encoder is not None:
            arch_head = self.head_encoder.random_sample_arch()
        else:
            arch_head = None

        h = random.choice(self.hs_list)
        w = random.choice(self.ws_list)
        img_sz = (h, w)

        arch_dict: ArchDesc = {
            "backbone": arch_backbone,
            "neck": arch_neck,
            "head": arch_head,
            "r": img_sz,
        }
        return arch_dict

    def random_resample(self, arch_dict: ArchDesc, mutate_prob: int) -> None:
        self.backbone_encoder.random_resample(arch_dict["backbone"], mutate_prob)
        if self.neck_encoder is not None:
            self.neck_encoder.random_resample(arch_dict["neck"], mutate_prob)
        if self.head_encoder is not None:
            self.head_encoder.random_resample(arch_dict["head"], mutate_prob)
        if random.random() < mutate_prob:
            h = random.choice(self.hs_list)
            w = random.choice(self.ws_list)
            img_sz = (h, w)
            arch_dict["r"] = img_sz


class CompositeEncoderCatboost(CompositeEncoder):
    def __init__(
        self,
        backbone_encoder: Union[MobileNetArchEncoder, ResNetArchEncoder],
        neck_encoder: Union[FPNEncoder, None],
        head_encoder: Union[SegmentationHeadEncoder, None],
        image_size_list: List[Tuple[int, int]],
    ):
        self.backbone_encoder = backbone_encoder
        self.neck_encoder = neck_encoder
        self.head_encoder = head_encoder
        self.hs_list = [size[0] for size in image_size_list]
        self.ws_list = [size[1] for size in image_size_list]

        self.backbone_dim = self.backbone_encoder.n_dim
        self.neck_dim = self.neck_encoder.n_dim if self.neck_encoder is not None else 0
        self.head_dim = self.head_encoder.n_dim if self.head_encoder is not None else 0
        self.n_dim = self.backbone_dim + self.neck_dim + self.head_dim + 2

        self.h_info = None
        self.w_info = None

    def _build_info_dict(self, target):
        pass

    def arch2feature(self, arch_dict: ArchDesc) -> np.ndarray:
        arch_backbone = arch_dict["backbone"]
        arch_neck = arch_dict["neck"]
        arch_head = arch_dict["head"]
        h, w = arch_dict["r"]

        feature = np.zeros(self.n_dim)

        feature_backbone = self.backbone_encoder.arch2feature(arch_backbone)
        feature[: self.backbone_dim] = feature_backbone

        if self.neck_encoder is not None:
            feature_neck = self.neck_encoder.arch2feature(arch_neck)
            feature[self.backbone_dim : self.backbone_dim + self.neck_dim] = (
                feature_neck
            )

        if self.head_encoder is not None:
            feature_head = self.head_encoder.arch2feature(arch_head)
            feature[
                self.backbone_dim
                + self.neck_dim : self.backbone_dim
                + self.neck_dim
                + self.head_dim
            ] = feature_head

        feature[-2] = h / max(self.hs_list)
        feature[-1] = w / max(self.ws_list)
        return feature

    def feature2arch(self, feature: np.ndarray) -> ArchDesc:
        # TODO: implement for distributed evolution
        raise NotImplementedError
