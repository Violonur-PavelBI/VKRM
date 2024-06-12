import math
from typing import Any, Dict, Optional, Tuple, Union

from albumentations import (
    Affine,
    Compose,
    Equalize,
    OneOf,
    Posterize,
    Sharpen,
    Solarize,
)
from albumentations.core.bbox_utils import BboxParams
from albumentations.core.keypoints_utils import KeypointParams
from torch import Tensor, arange, linspace

from .functional import AutoContrast, Brightness, Color, Contrast, Identity

__all__ = ["RandAlbument"]


class RandAlbument(Compose):
    def __init__(
        self,
        img_size: Tuple = (640, 480),
        num_ops: int = 2,
        magnitude: int = 5,
        num_magnitude_bins: int = 31,
        bbox_params: Optional[Union[dict, "BboxParams"]] = None,
        keypoint_params: Optional[Union[dict, "KeypointParams"]] = None,
    ) -> None:
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.img_size = img_size

        super(RandAlbument, self).__init__(
            transforms=self._tranforms(),
            bbox_params=bbox_params,
            keypoint_params=keypoint_params,
        )

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "__class_fullname__": Compose.get_class_fullname(),
            "p": self.p,
            "transforms": [t._to_dict() for t in self.transforms],
        }

    def add_targets(self, additional_targets: Optional[Dict[str, str]]) -> None:
        return super().add_targets(additional_targets)

    def _augmentation_space(self) -> Dict[str, Tensor]:
        return {
            "ShearX": linspace(0.0, 0.3, self.num_magnitude_bins),
            "ShearY": linspace(0.0, 0.3, self.num_magnitude_bins),
            "TranslateX": linspace(
                0.0, 150.0 / 331.0 * self.img_size[1], self.num_magnitude_bins
            ),
            "TranslateY": linspace(
                0.0, 150.0 / 331.0 * self.img_size[0], self.num_magnitude_bins
            ),
            "Rotate": linspace(0.0, 30.0, self.num_magnitude_bins),
            "Brightness": linspace(0.0, 0.9, self.num_magnitude_bins),
            "Color": linspace(0.0, 0.9, self.num_magnitude_bins),
            "Contrast": linspace(0.0, 0.9, self.num_magnitude_bins),
            "Sharpness": linspace(0.0, 0.9, self.num_magnitude_bins),
            "Posterize": 8
            - (arange(self.num_magnitude_bins) / ((self.num_magnitude_bins - 1) / 4))
            .round()
            .int(),
            "Solarize": linspace(255.0, 0.0, self.num_magnitude_bins),
        }

    def _tranforms(self):
        meta_augs = self._augmentation_space()
        _magnitude = {}
        for _name_aug in meta_augs.keys():
            _magnitude[_name_aug] = (
                float(meta_augs[_name_aug][self.magnitude].item())
                if meta_augs[_name_aug].ndim > 0
                else 0.0
            )

        transforms = [
            OneOf(
                [
                    OneOf(
                        [
                            Affine(
                                scale=1.0,
                                rotate=0.0,
                                shear={
                                    "x": (-1)
                                    * math.degrees(math.atan(_magnitude["ShearX"]))
                                },
                            ),
                            Affine(
                                scale=1.0,
                                rotate=0.0,
                                shear={
                                    "x": math.degrees(math.atan(_magnitude["ShearX"]))
                                },
                            ),
                        ]
                    ),
                    OneOf(
                        [
                            Affine(
                                scale=1.0,
                                rotate=0.0,
                                shear={
                                    "y": (-1)
                                    * math.degrees(math.atan(_magnitude["ShearY"]))
                                },
                            ),
                            Affine(
                                scale=1.0,
                                rotate=0.0,
                                shear={
                                    "y": math.degrees(math.atan(_magnitude["ShearY"]))
                                },
                            ),
                        ]
                    ),
                    OneOf(
                        [
                            Affine(
                                scale=1.0,
                                rotate=0.0,
                                shear=[0.0, 0.0],
                                translate_px={
                                    "x": (-1) * int(_magnitude["TranslateX"])
                                },
                            ),
                            Affine(
                                scale=1.0,
                                rotate=0.0,
                                shear=[0.0, 0.0],
                                translate_px={"x": int(_magnitude["TranslateX"])},
                            ),
                        ]
                    ),
                    OneOf(
                        [
                            Affine(
                                scale=1.0,
                                rotate=0.0,
                                shear=[0.0, 0.0],
                                translate_px={
                                    "x": [0, 0],
                                    "y": (-1) * int(_magnitude["TranslateY"]),
                                },
                            ),
                            Affine(
                                scale=1.0,
                                rotate=0.0,
                                shear=[0.0, 0.0],
                                translate_px={
                                    "x": [0, 0],
                                    "y": int(_magnitude["TranslateY"]),
                                },
                            ),
                        ]
                    ),
                    OneOf(
                        [
                            Affine(rotate=(-1) * _magnitude["Rotate"]),
                            Affine(rotate=_magnitude["Rotate"]),
                        ]
                    ),
                    OneOf(
                        [
                            Contrast(contrast=1.0 - _magnitude["Contrast"]),
                            Contrast(contrast=1.0 + _magnitude["Contrast"]),
                        ]
                    ),
                    OneOf(
                        [
                            Brightness(brightness=1.0 - _magnitude["Brightness"]),
                            Brightness(brightness=1.0 + _magnitude["Brightness"]),
                        ]
                    ),
                    AutoContrast(p=0.5),
                    Identity(p=0.5),
                    OneOf(
                        [
                            Color(saturation_factor=1.0 + _magnitude["Color"]),
                            Color(saturation_factor=1.0 - _magnitude["Color"]),
                        ]
                    ),
                    Sharpen(
                        alpha=[0, 0.05],
                        lightness=[
                            1.0 + _magnitude["Sharpness"],
                            1.0 + _magnitude["Sharpness"],
                        ],
                    ),
                    Posterize(num_bits=_magnitude["Posterize"]),
                    Solarize(threshold=_magnitude["Solarize"]),
                    Equalize(mode="pil"),
                ],
                p=1,
            )
        ] * self.num_ops

        return transforms
