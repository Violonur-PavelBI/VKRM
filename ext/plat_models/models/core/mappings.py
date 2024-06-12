from .functional import *
from .activation import ReLU, LeakyReLU, ReLU6, Sigmoid
from .conv import Conv1d, Conv2d, ConvTranspose2d
from .norm import BatchNorm2d, InstanceNorm2d, GroupNorm
from .pooling import MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d
from .upsampling import Upsample
from .dropout import Dropout
from .linear import Linear
from .base_primitive import REVERSED_REGISTRY, PRIMITIVE_REGISTRY

Torch2Plat_fromFX = PRIMITIVE_REGISTRY
Plat2Torch_fromJSON = REVERSED_REGISTRY
# Torch2Plat_fromFX = {
#     "model_input":                  Input.toPlatform,
#     "model_output":                 Output.toPlatform,
#     "getattr":                      GetAttr.toPlatform,
#     "getitem":                      GetItem.toPlatform,
#     "add":                          Add.toPlatform,
#     "max":                          models.core.functional.ops.Max.toPlatform,
#     "mul":                          models.core.functional.ops.Mul.toPlatform,
#     "_VariableFunctionsClass.cat":  models.core.functional.ops.Concat.toPlatform,
#     #################### modules as finctions ########################
#     "interpolate":                  models.core.functional.ops.Interpolate.toPlatform,
#     "relu":                         ReLU.toPlatform,
#     "relu6":                        ReLU.toPlatform,
# }

# Plat2Torch_fromJSON = {
#     "module": {
#         "dense":        Linear.fromPlatform,
#         # "dense_opt":    LinearOpt.fromPlatform,
#         "conv1d":       Conv1d.fromPlatform,
#         "conv2d":       Conv2d.fromPlatform,
#         # "lstm":       convert_LSTM,
#         "batchnorm2d":  BatchNorm2d.fromPlatform,
#         "deconv2d":     ConvTranspose2d.fromPlatform,
#         "relu":         ReLU.fromPlatform,
#         "relu6":        ReLU6.fromPlatform,
#         "leakyrelu":    LeakyReLU.fromPlatform,
#         # "prelu":      convert_PReLU,
#         "pool2d":       MaxPool2d.fromPlatform,
#         "sigmoid":      Sigmoid.fromPlatform,
#         # "softmax":    convert_Softmax,
#         # "tanh":       convert_Tanh,
#         # "mish":       convert_Mish,
#         # "softplus":   convert_Softplus,
#         "upsample":     Upsample.fromPlatform,
#         "dropout":      Dropout.fromPlatform,
#     },
#     "functional": {
#         "concat":       models.core.functional.ops.Concat.fromPlatform,
#         "interpolate":  models.core.functional.ops.Interpolate.fromPlatform,
#         "max":          models.core.functional.ops.Max.fromPlatform,
#         "add":          models.core.functional.ops.Add.fromPlatform,
#         "mul":          models.core.functional.ops.Mul.fromPlatform,
#     }
# }
