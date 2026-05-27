from __future__ import annotations

from copy import deepcopy
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_compat import FFN, build_dropout, build_norm_layer


def to_2tuple(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


class AdaptivePadding(nn.Module):
    def __init__(self, kernel_size=1, stride=1, dilation=1, padding: str = "corner") -> None:
        super().__init__()
        assert padding in ("same", "corner")
        self.padding = padding
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)

    def get_pad_shape(self, input_shape: Sequence[int]) -> Tuple[int, int]:
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = (input_h + stride_h - 1) // stride_h
        output_w = (input_w + stride_w - 1) // stride_w
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = F.pad(x, [0, pad_w, 0, pad_h])
            else:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x


class PortablePatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 96,
        kernel_size: int = 4,
        stride: int = 4,
        padding: str | int | tuple = "corner",
        dilation: int = 1,
        bias: bool = True,
        patch_norm: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)
        self.projection = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.norm = build_norm_layer(dict(type="LN"), embed_dims)[1] if patch_norm else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if self.adap_padding is not None:
            x = self.adap_padding(x)
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class PortablePatchMerging(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int | None = None,
        padding: str | int | tuple = "corner",
        dilation: int = 1,
    ) -> None:
        super().__init__()
        stride = stride or kernel_size
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        sample_dim = kernel_size[0] * kernel_size[1] * in_channels
        self.norm = build_norm_layer(dict(type="LN"), sample_dim)[1]
        self.reduction = nn.Linear(sample_dim, out_channels, bias=False)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, input_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        batch_size, seq_len, channels = x.shape
        height, width = input_size
        assert seq_len == height * width
        x = x.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        if self.adap_padding is not None:
            x = self.adap_padding(x)
            height, width = x.shape[-2:]
        x = self.sampler(x).transpose(1, 2)
        out_h = (height + 2 * self.sampler.padding[0] - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1) - 1) // self.sampler.stride[0] + 1
        out_w = (width + 2 * self.sampler.padding[1] - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1) - 1) // self.sampler.stride[1] + 1
        x = self.norm(x)
        x = self.reduction(x)
        return x, (out_h, out_w)


class WindowMSA(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: Tuple[int, int],
        qkv_bias: bool = True,
        qk_scale=None,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        wh, ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * ww - 1, wh, 1, ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer("relative_position_index", rel_position_index)
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(batch_size // num_windows, num_windows, self.num_heads, num_tokens, num_tokens)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, num_tokens, num_tokens)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class ShiftWindowMSA(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        qkv_bias: bool = True,
        qk_scale=None,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
        )
        self.drop = build_dropout(dict(type="DropPath", drop_prob=drop_path_rate))

    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = x.shape
        x = x.view(
            batch_size,
            height // self.window_size,
            self.window_size,
            width // self.window_size,
            self.window_size,
            channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(-1, self.window_size, self.window_size, channels)

    def window_reverse(self, windows: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size = int(windows.shape[0] / (height * width / self.window_size / self.window_size))
        x = windows.view(
            batch_size,
            height // self.window_size,
            width // self.window_size,
            self.window_size,
            self.window_size,
            -1,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(batch_size, height, width, -1)

    def forward(self, query: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        batch_size, seq_len, channels = query.shape
        height, width = hw_shape
        assert seq_len == height * width
        query = query.view(batch_size, height, width, channels)
        pad_r = (self.window_size - width % self.window_size) % self.window_size
        pad_b = (self.window_size - height % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        height_pad, width_pad = query.shape[1], query.shape[2]
        if self.shift_size > 0:
            shifted_query = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            img_mask = torch.zeros((1, height_pad, width_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = self.window_partition(img_mask).view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None
        query_windows = self.window_partition(shifted_query).view(-1, self.window_size ** 2, channels)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)
        shifted_x = self.window_reverse(attn_windows, height_pad, width_pad)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :height, :width, :].contiguous()
        x = x.view(batch_size, height * width, channels)
        return self.drop(x)


class PortableSwinBlock(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        window_size: int = 7,
        shift: bool = False,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = build_norm_layer(dict(type="LN"), embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.norm2 = build_norm_layer(dict(type="LN"), embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type="DropPath", drop_prob=drop_path_rate),
            act_cfg=dict(type="GELU"),
            add_identity=True,
        )

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)
        x = x + identity
        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)
        return x


class PortableSwinBlockSequence(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        depth: int,
        window_size: int,
        qkv_bias: bool,
        qk_scale,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate,
        downsample: nn.Module | None,
    ) -> None:
        super().__init__()
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        self.blocks = nn.ModuleList([
            PortableSwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=bool(i % 2),
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
            )
            for i in range(depth)
        ])
        self.downsample = downsample

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]):
        for block in self.blocks:
            x = block(x, hw_shape)
        if self.downsample is not None:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        return x, hw_shape, x, hw_shape


class PortableSwinTransformer(nn.Module):
    def __init__(
        self,
        pretrain_img_size=384,
        in_channels: int = 3,
        embed_dims: int = 128,
        patch_size: int = 4,
        window_size: int = 12,
        mlp_ratio: int = 4,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        strides=(4, 2, 2, 2),
        out_indices=(1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.3,
    ) -> None:
        super().__init__()
        del pretrain_img_size
        self.out_indices = out_indices
        self.patch_embed = PortablePatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            kernel_size=patch_size,
            stride=strides[0],
            patch_norm=patch_norm,
        )
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        self.stages = nn.ModuleList()
        in_ch = embed_dims
        for i in range(len(depths)):
            downsample = None
            if i < len(depths) - 1:
                downsample = PortablePatchMerging(
                    in_channels=in_ch,
                    out_channels=2 * in_ch,
                    stride=strides[i + 1],
                )
            stage = PortableSwinBlockSequence(
                embed_dims=in_ch,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_ch,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[: i + 1])],
                downsample=downsample,
            )
            self.stages.append(stage)
            if downsample is not None:
                in_ch = downsample.out_channels
        self.num_features = [int(embed_dims * 2 ** i) for i in range(len(depths))]
        for i in out_indices:
            self.add_module(f"norm{i}", build_norm_layer(dict(type="LN"), self.num_features[i])[1])

    def forward(self, x: torch.Tensor):
        x, hw_shape = self.patch_embed(x)
        x = self.drop_after_pos(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs
