from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PortableSinePositionalEncoding(nn.Module):
    """Portable variant of MMDet's sine positional encoding."""

    def __init__(
        self,
        num_feats: int,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: Optional[torch.Tensor], input: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert not (mask is None and input is None)

        if mask is not None:
            batch_size, height, width = mask.size()
            device = mask.device
            mask = mask.to(torch.int)
            not_mask = 1 - mask
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
        else:
            batch_size, _, height, width = input.shape
            device = input.device
            x_embed = torch.arange(1, width + 1, dtype=torch.float32, device=device)
            x_embed = x_embed.view(1, 1, -1).repeat(batch_size, height, 1)
            y_embed = torch.arange(1, height + 1, dtype=torch.float32, device=device)
            y_embed = y_embed.view(1, -1, 1).repeat(batch_size, 1, width)

        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).view(
            batch_size, height, width, -1
        )
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).view(
            batch_size, height, width, -1
        )
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
