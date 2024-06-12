import torch
import torchvision.transforms.functional as F
import numpy as np
import copy

from typing import  Tuple, Dict
from albumentations import ImageOnlyTransform


class AutoContrast(ImageOnlyTransform):
    """
        Maximize (normalize) image contrast. This function calculates a
        histogram of the input image (or mask region), removes ``cutoff`` percent of the
        lightest and darkest pixels from the histogram, and remaps the image
        so that the darkest pixel becomes black (0), and the lightest
        becomes white (255).
        :param image: The image to process.
        :return: An image.
    """

    def __init__(
                self,
                always_apply: bool =False,
                p:float = 1
                ) -> None:
        super(AutoContrast, self).__init__(always_apply, p)

    def apply(self, image, **params):

        img = copy.copy(image)
        _tensor = torch.from_numpy(img)
        _tensor = _tensor.permute((2, 0, 1))

        return np.array(F.autocontrast(_tensor).permute(1, 2, 0))

    def get_transform_init_args_names(self):
        return list()


class Identity(ImageOnlyTransform):

    def __init__(self, 
                always_apply: bool = False,
                p: float = 1) -> None:

        super(Identity, self).__init__(p = p, always_apply = always_apply)

    def apply(self, image, **params):
        return image

    def get_transform_init_args_names(self):
        return list()


class Color(ImageOnlyTransform):

    def __init__(self,
                 saturation_factor: float,
                 always_apply: bool =False,
                 p: float = 0.5
                ) -> None:

        super(ImageOnlyTransform, self).__init__(always_apply, p)
        self.saturation_factor = saturation_factor

    def apply(self, image, saturation_factor, **params):

        img = copy.copy(image)
        _tensor = torch.from_numpy(img)
        _tensor = _tensor.permute((2, 0, 1))

        return np.array(F.adjust_saturation(_tensor, saturation_factor).permute(1, 2, 0))

    def get_transform_init_args_names(self) -> Tuple:
        return ("saturation_factor",)

    def get_params(self) -> Dict[str, float]:
        return {"saturation_factor": self.saturation_factor}



class Contrast(ImageOnlyTransform):

    def __init__(self,
                 contrast: float,
                 always_apply: bool = False,
                 p: float = 0.5
                ) -> None:

        super(ImageOnlyTransform, self).__init__(always_apply, p)
        self.contrast = contrast

    def apply(self, image, contrast, **params):

        img = copy.copy(image)
        _tensor = torch.from_numpy(img)
        _tensor = _tensor.permute((2, 0, 1))

        return np.array(F.adjust_contrast(_tensor, contrast).permute(1, 2, 0))

    def get_transform_init_args_names(self) -> Tuple:
        return ("contrast",)

    def get_params(self) -> Dict[str, float]:
        return {"contrast": self.contrast}

class Brightness(ImageOnlyTransform):

    def __init__(self,
                 brightness: float,
                 always_apply: bool = False,
                 p: float = 0.5
                ) -> None:

        super(ImageOnlyTransform, self).__init__(always_apply, p)
        self.brightness = brightness

    def apply(self, image, brightness, **params):

        img = copy.copy(image)
        _tensor = torch.from_numpy(img)
        _tensor = _tensor.permute((2, 0, 1))

        return np.array(F.adjust_brightness(_tensor, brightness).permute(1, 2, 0))

    def get_transform_init_args_names(self) -> Tuple:
        return ("brightness",)

    def get_params(self) -> Dict[str, float]:
        return {"brightness": self.brightness}