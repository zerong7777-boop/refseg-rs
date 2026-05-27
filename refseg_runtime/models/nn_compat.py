from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn


def build_norm_layer(cfg: Optional[dict], num_features: int):
    cfg = dict(cfg or {"type": "LN"})
    norm_type = str(cfg.get("type", "LN")).upper()
    eps = cfg.get("eps", 1e-5)
    requires_grad = cfg.get("requires_grad", True)
    if norm_type == "LN":
        layer = nn.LayerNorm(num_features, eps=eps)
    elif norm_type == "BN":
        layer = nn.BatchNorm1d(num_features, eps=eps)
    elif norm_type == "BN2D":
        layer = nn.BatchNorm2d(num_features, eps=eps)
    else:
        raise ValueError(f"Unsupported norm layer type: {cfg.get('type')}")
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return norm_type, layer


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor.div(keep_prob)


def build_dropout(cfg: Optional[dict]) -> nn.Module:
    cfg = dict(cfg or {})
    drop_type = cfg.get("type", "Dropout")
    if drop_type == "DropPath":
        return DropPath(cfg.get("drop_prob", 0.0))
    if drop_type == "Dropout":
        return nn.Dropout(cfg.get("drop_prob", cfg.get("p", 0.0)))
    raise ValueError(f"Unsupported dropout layer type: {drop_type}")


def _build_activation(cfg: Optional[dict]) -> nn.Module:
    cfg = dict(cfg or {"type": "ReLU", "inplace": True})
    act_type = cfg.get("type", "ReLU").upper()
    if act_type == "RELU":
        return nn.ReLU(inplace=cfg.get("inplace", True))
    if act_type == "GELU":
        return nn.GELU()
    raise ValueError(f"Unsupported activation type: {cfg.get('type')}")


class FFN(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        num_fcs: int = 2,
        act_cfg: Optional[dict] = None,
        ffn_drop: float = 0.0,
        dropout_layer: Optional[dict] = None,
        add_identity: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        if num_fcs != 2:
            raise ValueError(f"Only num_fcs=2 is supported, got {num_fcs}")
        act = _build_activation(act_cfg)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            ("0", nn.Linear(embed_dims, feedforward_channels)),
                            ("1", act),
                            ("2", nn.Dropout(ffn_drop)),
                        ]
                    )
                ),
                nn.Linear(feedforward_channels, embed_dims),
            ]
        )
        self.dropout = nn.Dropout(ffn_drop)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()
        self.add_identity = add_identity

    def forward(self, x: torch.Tensor, identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.layers[0](x)
        out = self.layers[1](out)
        out = self.dropout(out)
        out = self.dropout_layer(out)
        if self.add_identity:
            if identity is None:
                identity = x
            out = identity + out
        return out


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.proj_drop = nn.Dropout(dropout)
        self.dropout_layer = nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **_: object,
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        out, _ = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = self.proj_drop(out)
        out = self.dropout_layer(out)
        return identity + out


def _constant_init(module: nn.Module, weight: float = 0.0, bias: float = 0.0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, weight)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _xavier_init(module: nn.Module, bias: float = 0.0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.xavier_uniform_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([h_ * w_ for h_, w_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (height, width) in enumerate(value_spatial_shapes):
        value_l = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, height, width)
        sampling_grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l = nn.functional.grid_sample(
            value_l,
            sampling_grid_l,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads,
        1,
        num_queries,
        num_levels * num_points,
    )
    output = (
        torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
    ).sum(-1).view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        im2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = False,
        norm_cfg: Optional[dict] = None,
        value_proj_ratio: float = 1.0,
        **_: object,
    ) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, got {embed_dims} and {num_heads}")
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.im2col_step = im2col_step
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.norm_cfg = norm_cfg
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        _constant_init(self.sampling_offsets, weight=0.0, bias=0.0)
        device = next(self.parameters()).device
        thetas = torch.arange(self.num_heads, dtype=torch.float32, device=device) * (2.0 * torch.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads,
            1,
            1,
            2,
        ).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        _constant_init(self.attention_weights, weight=0.0, bias=0.0)
        _xavier_init(self.value_proj, bias=0.0)
        _xavier_init(self.output_proj, bias=0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **_: object,
    ) -> torch.Tensor:
        del key, level_start_index
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert spatial_shapes is not None
        assert reference_points is not None
        assert int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum()) == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / (
                offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}")

        output = multi_scale_deformable_attn_pytorch(
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
        )
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity
