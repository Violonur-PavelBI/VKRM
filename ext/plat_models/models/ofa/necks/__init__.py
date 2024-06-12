from .fpn import FPN, DynamicFPN, NASFPN

neck_name2class = {
    FPN.__name__: FPN,
    DynamicFPN.__name__: DynamicFPN,
    NASFPN.__name__: NASFPN,
}
