from ..core.container import ModuleList, Sequential
from ..baseblocks.hrnet_basic import (
    HrNetBasicBlock,
    HrNetBottleneck,
    HighResolutionModule,
)
from ..core.abstract import Backbone, BACKBONES

from ..core import Conv2d, ReLU, BatchNorm2d, Upsample
from ..core import functional as F
from ..core.functional.ops import Concat as cat
import torch

BN_MOMENTUM = 0.1

blocks_dict = {"BASIC": HrNetBasicBlock, "BOTTLENECK": HrNetBottleneck}


class HRNetV2(Backbone):
    def __init__(self, n_class, **kwargs):
        super(HRNetV2, self).__init__()
        extra = {
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": (4, 4),
                "NUM_CHANNELS": (48, 96),
                "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 4,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": (4, 4, 4),
                "NUM_CHANNELS": (48, 96, 192),
                "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 3,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": (4, 4, 4, 4),
                "NUM_CHANNELS": (48, 96, 192, 384),
                "FUSE_METHOD": "SUM",
            },
            "FINAL_CONV_KERNEL": 1,
        }

        # stem net
        self.input_channels = 3
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = ReLU(inplace=True)
        self.downsample_factor = 4

        self.layer1 = self._make_layer(HrNetBottleneck, 64, 64, 4)

        self.stage2_cfg = extra["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels
        )

        self.stage3_cfg = extra["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels
        )

        self.stage4_cfg = extra["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True
        )

        self.output_channels = sum(self.stage4_cfg["NUM_CHANNELS"])
        ## Convertation fix
        self.br1_final_up = Upsample(scale_factor=2, mode="bilinear")
        self.br2_final_up = Upsample(scale_factor=4, mode="bilinear")
        self.br3_final_up = Upsample(scale_factor=8, mode="bilinear")

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        Sequential(
                            Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            BatchNorm2d(
                                num_channels_cur_layer[i], momentum=BN_MOMENTUM
                            ),
                            ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        Sequential(
                            Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            ReLU(inplace=True),
                        )
                    )
                transition_layers.append(Sequential(*conv3x3s))

        return ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        self.downsample_factor *= stride

        return Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return Sequential(*modules), num_inchannels

    def forward(self, x, return_feature_maps=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        # x0_h, x0_w = x[0].size(2), x[0].size(3)
        # x1 = F.interpolate(
        #     x[1], size=(x0_h, x0_w), mode="bilinear", align_corners=False
        # )
        # x2 = F.interpolate(
        #     x[2], size=(x0_h, x0_w), mode="bilinear", align_corners=False
        # )
        # x3 = F.interpolate(
        #     x[3], size=(x0_h, x0_w), mode="bilinear", align_corners=False
        # )
        x1 = self.br1_final_up(x[1])
        x2 = self.br2_final_up(x[2])
        x3 = self.br3_final_up(x[3])

        x = torch.cat([x[0], x1, x2, x3], 1)

        # x = self.last_layer(x)
        return x


def hrnetv2(num_classes=1000, pretrained=False):
    model = HRNetV2(n_class=num_classes)
    if pretrained:
        # FIXME
        model.load_state_dict(
            torch.load("./subprj/pretrained/hrnetv2_w48-imagenet.pth"), strict=False
        )

    return model


BACKBONES.register_module("hrnetv2", module=hrnetv2)
