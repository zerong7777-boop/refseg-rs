from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class PortableConvNormAct(nn.Module):
    """Checkpoint-compatible subset of MMDet ConvModule used by ChannelMapper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        num_groups: int = 32,
        act: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.activate = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        return self.activate(x)


class PortableChannelMapper(nn.Module):
    """Pure torch ChannelMapper with checkpoint-compatible parameter names."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        kernel_size: int = 3,
        num_outs: Optional[int] = None,
        num_groups: int = 32,
        act: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, (list, tuple)) and len(in_channels) > 0
        if num_outs is None:
            num_outs = len(in_channels)
        self.out_channels = out_channels
        self.convs = nn.ModuleList([
            PortableConvNormAct(
                in_channel,
                out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                num_groups=num_groups,
                act=act,
                bias=bias,
            )
            for in_channel in in_channels
        ])
        self.extra_convs = nn.ModuleList()
        for idx in range(len(in_channels), num_outs):
            extra_in_channels = in_channels[-1] if idx == len(in_channels) else out_channels
            self.extra_convs.append(
                PortableConvNormAct(
                    extra_in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    num_groups=num_groups,
                    act=act,
                    bias=bias,
                )
            )

    def forward(self, inputs: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        inputs = list(inputs)
        assert len(inputs) == len(self.convs)
        outs = [conv(feature) for conv, feature in zip(self.convs, inputs)]
        for idx, extra_conv in enumerate(self.extra_convs):
            if idx == 0:
                outs.append(extra_conv(inputs[-1]))
            else:
                outs.append(extra_conv(outs[-1]))
        return tuple(outs)
