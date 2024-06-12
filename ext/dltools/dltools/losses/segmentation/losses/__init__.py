"""
There's only losses,which had been checked

"""
from .base import _BaseLoss
from .probability_based import FocalLoss_Ori, CELoss, SegmFocalLoss, LovaszSoftmax
from .region_based import (
    DiceLoss,
    JaccardLoss,
    TverskiyLoss,
    DiceLossSq,
    ACFTverskiy,
    DiceLossFS,
    JaccardLossFS,
    FocalTverskiyLoss,
    RocVectorLoss,
)
from .combined import (
    Cmb_CE_DiceLoss,
    Cmb_FL_DiceLoss,
    Cmb_BCE_DiceLossSq,
    Cmb_FL_DiceLossSq,
)

# from utils.calculus.probability_based.lovasz import  lovasz_softmax as LovaszSoftmaxOri
# from utils.calculus.probability_based.lovasz_toolbox import LovaszSoftmax as LovaszSoftmaxTol
FocalLossOri = FocalLoss_Ori
# FocalLoss = SegmFocalLoss
__all__ = [
    x.__name__
    for x in [
        SegmFocalLoss,
        # FocalLoss,
        DiceLoss,
        DiceLossSq,
        DiceLossFS,
        JaccardLoss,
        JaccardLossFS,
        # Surf2EdgeLoss,
        # LovaszSoftmaxOri,
        # LovaszSoftmaxTol,
        LovaszSoftmax,
        # BinaryFocalLoss,
        CELoss,
        Cmb_CE_DiceLoss,
        Cmb_FL_DiceLoss,
        Cmb_BCE_DiceLossSq,
        TverskiyLoss,
        ACFTverskiy,
        RocVectorLoss,
        FocalTverskiyLoss,
    ]
]
