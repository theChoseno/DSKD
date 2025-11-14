import os
from .resnet import (
    resnet8,
    resnet14,
    resnet20,
    resnet32,
    resnet44,
    resnet56,
    resnet110,
    resnet8x4,
    resnet32x4,
)
from .resnet_adapt import (
    resnet8_adapt,
    resnet14_adapt,
    resnet20_adapt,
    resnet32_adapt,
    resnet44_adapt,
    resnet56_adapt,
    resnet110_adapt,
    resnet8x4_adapt,
    resnet32x4_adapt,
)
from .resnetv2 import ResNet50, ResNet34, ResNet18, ResNet101
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half
from .mobilenetv1 import MobileNetV1
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .resnet_hier import hresnet8, hresnet8x4


cifar100_model_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "../../../download_ckpts/cifar_teachers/"
)
cifar_model_dict = {
#     teachers
    "resnet56": (
        resnet56,
        cifar100_model_prefix + "resnet56_vanilla/ckpt_epoch_240.pth",
    ),
#     "resnet56": (
#         resnet56,
#         cifar100_model_prefix + "resnet56_v2-162-best.pth",
#     ),
    "resnet110": (
        resnet110,
        cifar100_model_prefix + "resnet110_vanilla/ckpt_epoch_240.pth",
    ),
#     "resnet32x4": (
#         resnet32x4,
#         cifar100_model_prefix + "resnet32x4_v2-179-best.pth",
#     ),
    "resnet32x4": (
        resnet32x4,
        cifar100_model_prefix + "resnet32x4_vanilla/ckpt_epoch_240.pth",
    ),
    "ResNet50": (
        ResNet50,
        cifar100_model_prefix + "ResNet50_vanilla/ckpt_epoch_240.pth",
    ),
#     "ResNet34": (
#         ResNet34, 
#         cifar100_model_prefix + "resnet34_v2-10-regular.pth",
#     ),
    "ResNet34": (
        ResNet34, 
        cifar100_model_prefix + "resnet34_v2-198-best.pth",
    ),
    "ResNet101": (
        ResNet101,
        cifar100_model_prefix + "ResNet101_vanilla/resnet101_v2-200-best.pth",
    ),
    "wrn_40_2": (
        wrn_40_2,
        cifar100_model_prefix + "wrn_40_2_vanilla/ckpt_epoch_240.pth",
    ),
    "vgg13": (vgg13_bn, cifar100_model_prefix + "vgg13_vanilla/ckpt_epoch_240.pth"),
    # students
    "resnet8_adapt": (resnet8_adapt, None),
    "resnet14_adapt": (resnet14_adapt, None),
    "resnet20_adapt": (resnet20_adapt, None),
    "resnet32_adapt": (resnet32_adapt, None),
    "resnet44_adapt": (resnet44_adapt, None),
    "resnet8x4_adapt": (resnet8x4_adapt, None),
    "hresnet8x4": (hresnet8x4, None),
    "resnet8": (resnet8, None),
    "resnet14": (resnet14, None),
    "resnet20": (resnet20, None),
    "resnet32": (resnet32, None),
    "resnet44": (resnet44, None),
    "resnet8x4": (resnet8x4, None),
    "ResNet18": (ResNet18, None),
    "wrn_16_1": (wrn_16_1, None),
    "wrn_16_2": (wrn_16_2, None),
    "wrn_40_1": (wrn_40_1, None),
    "vgg8": (vgg8_bn, None),
    "vgg11": (vgg11_bn, None),
    "vgg16": (vgg16_bn, None),
    "vgg19": (vgg19_bn, None),
    "MobileNetV2": (mobile_half, None),
    "MobileNetV1": (MobileNetV1, None),
    "ShuffleV1": (ShuffleV1, cifar100_model_prefix + "shufflenetv1_vanilla/student_best"),
#     "ShuffleV1": (ShuffleV1, cifar100_model_prefix + "shufflenetv1_vanilla/shufflenetv1-165-best.pth"),
    "ShuffleV2": (ShuffleV2, None),
}
