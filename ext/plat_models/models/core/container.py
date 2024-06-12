import torch.nn as nn
from .meta import _CoreInterface, ABCEnforcedProperties


class ModuleList(nn.ModuleList, _CoreInterface, metaclass=ABCEnforcedProperties):
    """Redefined ModuleList, for docstring look `torch.nn.ModuleList`"""


class Sequential(nn.Sequential, _CoreInterface, metaclass=ABCEnforcedProperties):
    """Redefined Sequential, for docstring look `torch.nn.Sequential`"""


class ModuleDict(nn.ModuleDict, _CoreInterface, metaclass=ABCEnforcedProperties):
    """Redefined ModuleDict, for docstring look `torch.nn.ModuleDict"""
