from typing import List, Literal

from torch import Tensor

from ... import core as c

from ..primitives.static import MergeLayer
from .conv_blocks import ConvBlock


class MergeBlock(c.Module):
    """
    Собирает фичи в одну с постепенным увеличением масштаба.
    Возможны две политики агрегирования: сложение и конкатенация.
    """

    merge_policies_list = ["cat", "sum"]

    def __init__(
        self,
        levels: int,
        channels: int,
        use_bias: bool = False,
        upsample_mode: str = "nearest",
        merge_policy: Literal["sum", "cat"] = "sum",
    ) -> None:
        super().__init__()
        self.levels = levels

        self.upsample_mode = upsample_mode

        self.merge_policy = merge_policy
        if self.merge_policy not in self.merge_policies_list:
            raise ValueError(f"{merge_policy = } not in {self.merge_policies_list}")

        self.channels = channels
        self.expanded_channels = channels
        if self.merge_policy == "cat":
            self.expanded_channels = channels * 2
        self.use_bias = use_bias

        self.ups = c.ModuleList(
            [
                c.ConvTranspose2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=self.use_bias,
                )
                if self.upsample_mode == "deconv"
                else c.Upsample(mode=self.upsample_mode, scale_factor=2)
                for _ in range(self.levels - 1)
            ]
        )
        self.merge = MergeLayer(self.merge_policy)
        self.convs = c.ModuleList(
            [
                ConvBlock(
                    in_channels=self.expanded_channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    bias=self.use_bias,
                    use_bn=True,
                    act_func="relu",
                )
                for _ in range(self.levels - 1)
            ]
        )

    def forward(self, input_features: List[Tensor]) -> Tensor:
        output = input_features[-1]

        for i in range(self.levels - 2, -1, -1):
            upsample = self.ups[i]
            skip_input = input_features[i]
            conv = self.convs[i]

            output = upsample(output)
            output = self.merge(output, skip_input)
            output = conv(output)

        return output
