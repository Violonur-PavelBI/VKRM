from pydantic import BaseModel
from pydantic import StrictStr, StrictInt
from typing import Dict, Any


__all__ = ['SchedulerConfig']


class SchedulerConfig(BaseModel):
    '''Base scheduler dataclass'''
    name: StrictStr
    params: Dict[str, Any]
