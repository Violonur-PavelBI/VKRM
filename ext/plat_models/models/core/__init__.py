import sys
import json
from abc import abstractclassmethod, ABC

from .functional import ops
from .activation import ReLU, Sigmoid, LeakyReLU, ReLU6
from .linear import Linear
from .norm import BatchNorm1d, BatchNorm2d, InstanceNorm2d, GroupNorm
from .pooling import MaxPool1d, AvgPool1d, AdaptiveAvgPool1d, AdaptiveMaxPool1d
from .pooling import MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d
from .upsampling import Upsample
from .padding import ReplicationPad2d
from .dropout import Dropout
from .custom import LinearOpt
from . import functional
from .module import Module
from .abstract import (
    Node,
    Backbone,
    AbsDetect2DModel,
    AtomicNetwork,
    AbsClass2DModel,
    AbsSegm2DModel,
    CLASMODELS,
    SEGMMODELS,
    DETMODELS,
    IMGMODELS,
    BACKBONES,
    NODES,
    HEADS,
    NETWORKS,
)
from .tensor import Tensor
from .conv import Conv1d, Conv2d, ConvTranspose2d
from .container import Sequential, ModuleList, ModuleDict

# TODO При восстановления сети plat2torch загрузка весов производиться с использованием torch
# torch.from_numpy(np.frombuffer())
# подумaть как это будет работать с переключением torch -> platlib

Norm2d = BatchNorm2d
NLF = ReLU
# nlf = functional.relu
flatten = functional.flatten
cat = functional.cat
Flatten = functional.ops.Flatten
# nlf = functional.relu # TODO: fix

assert sys.version_info >= (3, 7), "python version must be >= 3,7"
