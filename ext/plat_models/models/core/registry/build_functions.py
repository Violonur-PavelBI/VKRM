# from open-mmlab/mmengine/build_functions
import inspect
import logging
from typing import TYPE_CHECKING, Any, Optional, Union
from .cfgdict import CfgDict
from .registry import Registry


def build_from_cfg(
    cfg: Union[dict, CfgDict],
    registry: Registry,
    default_args: Optional[Union[dict, CfgDict]] = None,
) -> Any:
    """Build a module from  dict when it is a class uration, or
    call a function from  dict when it is a function uration.
    If the global variable default scope (:obj:`DefaultScope`) exists,
    :meth:`build` will firstly get the responding registry and then call
    its own :meth:`build`.
    At least one of the ``cfg`` and ``default_args`` contains the key "type",
    which should be either str or class. If they all contain it, the key
    in ``cfg`` will be used because ``cfg`` has a high priority than
    ``default_args`` that means if a key exists in both of them, the value of
    the key will be ``cfg[key]``. They will be merged first and the key "type"
    will be popped up and the remaining keys will be used as initialization
    arguments.
    Examples:
        >>> from mmengine import Registry, build_from_cfg
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     def __init__(self, depth, stages=4):
        >>>         self.depth = depth
        >>>         self.stages = stages
        >>> cfg = dict(type='ResNet', depth=50)
        >>> model = build_from_cfg(cfg, MODELS)
        >>> # Returns an instantiated object
        >>> @MODELS.register_module()
        >>> def resnet50():
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='resnet50'), MODELS)
        >>> # Return a result of the calling function
    Args:
        cfg (dict or CfgDict or ):  dict. It should at least
            contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict or CfgDict or , optional): Default
            initialization arguments. Defaults to None.
    Returns:
        object: The constructed object.
    """
    # Avoid circular import
    from logging import getLogger

    Logger = getLogger()

    if not isinstance(cfg, (dict, CfgDict)):
        raise TypeError(f"cfg should be a dict, CfgDict or , but got {type(cfg)}")

    if "type" not in cfg:
        if default_args is None or "type" not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f"but got {cfg}\n{default_args}"
            )

    if not isinstance(registry, Registry):
        raise TypeError(
            "registry must be a mmengine.Registry object, " f"but got {type(registry)}"
        )

    if not (isinstance(default_args, (dict, CfgDict)) or default_args is None):
        raise TypeError(
            "default_args should be a dict, CfgDict,  or None, "
            f"but got {type(default_args)}"
        )

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    # Instance should be built under target scope, if `_scope_` is defined
    # in cfg, current default scope should switch to specified scope
    # temporarily.
    scope = args.pop("_scope_", None)
    with registry.switch_scope_and_registry(scope) as registry:
        obj_type = args.pop("type")
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    f"{obj_type} is not in the {registry.name} registry. "
                    f"Please check whether the value of `{obj_type}` is "
                    "correct or it was registered as expected. More details "
                    "can be found at "
                    "https://mmengine.readthedocs.io/en/latest/advanced_tutorials/.html#import-the-custom-module"  # noqa: E501
                )
        # this will include classes, functions, partial functions and more
        elif callable(obj_type):
            obj_cls = obj_type

        else:
            raise TypeError(
                f"type must be a str or valid type, but got {type(obj_type)}"
            )

        try:

            obj = obj_cls(**args)  # type: ignore

            if (
                inspect.isclass(obj_cls)
                or inspect.isfunction(obj_cls)
                or inspect.ismethod(obj_cls)
            ):
                Logger.debug(
                    f"An `{obj_cls.__name__}` instance is built from "  # type: ignore # noqa: E501
                    "registry, and its implementation can be found in "
                    f"{obj_cls.__module__}"
                )

            else:
                Logger.debug(
                    "An instance is built from registry, and its constructor is {}".format(
                        obj_cls
                    )
                )
            return obj

        except Exception as e:
            # Normal TypeError does not print class name.
            cls_location = "/".join(obj_cls.__module__.split("."))  # type: ignore
            raise type(e)(
                f"class `{obj_cls.__name__}` in "  # type: ignore
                f"{cls_location}.py: {e}"
            )
