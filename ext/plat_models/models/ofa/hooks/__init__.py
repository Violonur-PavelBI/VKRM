from .catcher_lst import HookCatcherLst, DynamicHookCatcherLst

backbone_hooks_name2class = {
    HookCatcherLst.__name__: HookCatcherLst,
    DynamicHookCatcherLst.__name__: DynamicHookCatcherLst,
}
