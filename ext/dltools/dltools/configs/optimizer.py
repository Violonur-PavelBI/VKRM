from pydantic import BaseModel
from pydantic import StrictStr
from typing import Dict, Any


__all__ = ['OptimizerConfig']


class OptimizerConfig(BaseModel):
    '''Base optimizer dataclass'''
    name: StrictStr
    params: Dict[StrictStr, Any]
