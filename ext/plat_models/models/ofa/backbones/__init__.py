from .ofa_mobilenetv3 import MobileNet
from .ofa_supernet_mbv3 import OFAMobileNet
from .ofa_resnet import ResNet
from .ofa_supernet_resnet import OFAResNet

backbone_name2class = {
    MobileNet.__name__: MobileNet,
    OFAMobileNet.__name__: OFAMobileNet,
    ResNet.__name__: ResNet,
    OFAResNet.__name__: OFAResNet,
}
