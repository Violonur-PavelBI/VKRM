from typing import Literal, TypedDict

HookType = Literal["pre", "forward"]


class HookInfo(TypedDict):
    module: str
    hook_type: HookType
    module_conf: dict
