from models.backbones import Backbone
from models.core import AdaptiveAvgPool2d, Dropout, Linear, LinearOpt, Tensor
from models.core.abstract import AbsClass2DModel
from models.core.container import Sequential
from models.heads.classifier import ClsAdpHead

from collections import OrderedDict
from typing import Optional


class ClassificationNetwork(Sequential, AbsClass2DModel, register=False):
    def __init__(
        self,
        backbone: Backbone,
        num_classes: int = 1000,
        dropout_p=0,
    ) -> None:
        super().__init__(
            OrderedDict(
                [
                    ("backbone", backbone),
                    (
                        "head",
                        ClsAdpHead(
                            backbone.output_channels,
                            num_classes=num_classes,
                            dropout_p=dropout_p,
                        ),
                    ),
                ]
            )
        )
        # super(ClassificationNetwork,AbsClass2DModel).__init__()
        self.num_classes = num_classes
