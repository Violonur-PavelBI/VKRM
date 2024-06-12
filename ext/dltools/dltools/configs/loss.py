from pydantic import BaseModel
from pydantic import StrictStr
from typing import Dict, Any


__all__ = ['LossConfig']


class LossConfig(BaseModel):
    '''Base loss dataclass'''
    name: StrictStr
    params: Dict[StrictStr, Any]
