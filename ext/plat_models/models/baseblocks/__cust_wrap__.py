from platform_typing_api import ClassPacker, PlatformIntParam, PlatformListParam
from .resnet_basic import BasicBlock

#   Example of wrapping BasicBlock


# BasicBlock = ClassPacker(
#     BasicBlock,
#     platform_definition = dict(
#         inplanes = PlatformIntParam(...),
#         planes = PlatformIntParam(...),
#         stride = PlatformIntParam(...),
#         downsample: Optional[Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[
#     Callable[..., Module]] = None
#     )
# )
