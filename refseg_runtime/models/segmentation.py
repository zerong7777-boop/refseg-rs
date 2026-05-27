from __future__ import annotations

import copy
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_to_two_channel_logits(pred: torch.Tensor) -> torch.Tensor:
    foreground_logits = pred
    background_logits = -pred
    return torch.stack([background_logits, foreground_logits], dim=1)


class DiceLoss:
    def __init__(
        self,
        axis: int = 1,
        smooth: float = 1e-6,
        reduction: str = "sum",
        square_in_union: bool = False,
    ) -> None:
        self.axis = axis
        self.smooth = smooth
        self.reduction = reduction
        self.square_in_union = square_in_union

    @staticmethod
    def _one_hot(x: torch.Tensor, classes: int, axis: int = 1) -> torch.Tensor:
        return torch.stack([torch.where(x == c, 1, 0) for c in range(classes)], axis=axis)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.axis)

    def __call__(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        targ = self._one_hot(targ, pred.shape[self.axis])
        pred = self.activation(pred)
        sum_dims = list(range(2, len(pred.shape)))
        inter = torch.sum(pred * targ, dim=sum_dims)
        union = (
            torch.sum(pred ** 2 + targ, dim=sum_dims)
            if self.square_in_union
            else torch.sum(pred + targ, dim=sum_dims)
        )
        dice_score = (2.0 * inter + self.smooth) / (union + self.smooth)
        loss = 1 - dice_score
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class RefSegLoss:
    def __init__(self, dice_weight: float = 0.1) -> None:
        self.dice_loss = DiceLoss()
        self.ce_weight = torch.FloatTensor([0.9, 1.1])
        self.dice_weight = dice_weight

    def __call__(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice_loss(pred, targ)
        ce_weight = self.ce_weight.to(device=pred.device, dtype=pred.dtype)
        ce_loss = F.cross_entropy(pred, targ, weight=ce_weight)
        return (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss


class QueryConditionedLocalGate(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        query_dim: int,
        hidden_dim: Optional[int] = None,
        gate_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or query_dim
        gate_channels = gate_channels or feature_channels
        self.query_norm = nn.LayerNorm(query_dim)
        self.query_proj = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_channels * 2),
        )
        self.hint_proj = nn.Conv2d(1, feature_channels, kernel_size=1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(feature_channels, gate_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels, 1, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor, hint: torch.Tensor, query_vec: torch.Tensor) -> torch.Tensor:
        query_params = self.query_proj(self.query_norm(query_vec))
        scale, bias = torch.chunk(query_params, 2, dim=1)
        scale = torch.sigmoid(scale).unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        feat_cond = feat * scale + bias
        hint_cond = self.hint_proj(hint)
        gate = torch.sigmoid(self.fuse(feat_cond + hint_cond))
        return feat * gate


class UNetSegHead(nn.Module):
    def __init__(
        self,
        input_channels_list: Sequence[int],
        output_channels: int,
        query_gate_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList([
            self._build_adjust_block(input_channels, output_channels)
            for input_channels in input_channels_list
        ])
        self.upconv1 = nn.Sequential(
            self._build_adjust_block(output_channels, output_channels),
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=2, stride=2),
        )
        self.upconv2 = nn.Sequential(
            self._build_adjust_block(output_channels, output_channels),
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=2, stride=2),
        )
        self.upconv3 = nn.Sequential(
            self._build_adjust_block(output_channels, output_channels),
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=2, stride=2),
        )
        self.upconv4 = nn.Sequential(
            self._build_adjust_block(output_channels, output_channels),
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=2, stride=2),
        )
        self.changechannel1 = nn.Sequential(
            nn.Conv2d(2 * output_channels, output_channels, kernel_size=1, stride=1),
            self._build_adjust_block(output_channels, output_channels),
        )
        self.changechannel2 = nn.Sequential(
            nn.Conv2d(2 * output_channels, output_channels, kernel_size=1, stride=1),
            self._build_adjust_block(output_channels, output_channels),
        )
        self.changechannel3 = nn.Sequential(
            nn.Conv2d(2 * output_channels, output_channels, kernel_size=1, stride=1),
            self._build_adjust_block(output_channels, output_channels),
        )
        query_gate_cfg = copy.deepcopy(query_gate_cfg) if query_gate_cfg is not None else {}
        self.enable_query_gate = bool(query_gate_cfg.pop("enable", False))
        self.query_gate_mid = None
        self.query_gate_high = None
        self.mid_hint_conv = None
        self.high_hint_conv = None
        if self.enable_query_gate:
            hidden_dim = int(query_gate_cfg.pop("hidden_dim", output_channels))
            gate_channels = int(query_gate_cfg.pop("gate_channels", output_channels))
            if query_gate_cfg:
                raise ValueError(f"Unexpected query_gate_cfg keys: {sorted(query_gate_cfg.keys())}")
            self.mid_hint_conv = nn.Conv2d(output_channels, 1, kernel_size=1)
            self.high_hint_conv = nn.Conv2d(output_channels, 1, kernel_size=1)
            self.query_gate_mid = QueryConditionedLocalGate(
                feature_channels=output_channels,
                query_dim=output_channels,
                hidden_dim=hidden_dim,
                gate_channels=gate_channels,
            )
            self.query_gate_high = QueryConditionedLocalGate(
                feature_channels=output_channels,
                query_dim=output_channels,
                hidden_dim=hidden_dim,
                gate_channels=gate_channels,
            )
        self.final_conv = nn.Conv2d(output_channels, 1, kernel_size=1)

    @staticmethod
    def _build_adjust_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(
        self,
        input_features: Iterable[torch.Tensor],
        query_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.enable_query_gate and query_vec is None:
            raise ValueError("query_vec must be provided when query gating is enabled")
        conv_results = [conv_layer(feature) for conv_layer, feature in zip(self.conv_layers, input_features)]
        x = conv_results[-1]

        x = self.upconv1(x)
        x = F.interpolate(x, size=conv_results[-2].shape[2:], mode="bilinear", align_corners=False)
        x = self.changechannel1(torch.cat([x, conv_results[-2]], dim=1))

        x = self.upconv2(x)
        x = F.interpolate(x, size=conv_results[-3].shape[2:], mode="bilinear", align_corners=False)
        x = self.changechannel2(torch.cat([x, conv_results[-3]], dim=1))
        if self.enable_query_gate and self.query_gate_mid is not None and self.mid_hint_conv is not None:
            mid_hint = torch.sigmoid(self.mid_hint_conv(x))
            x = self.query_gate_mid(x, mid_hint, query_vec)

        x = self.upconv3(x)
        x = F.interpolate(x, size=conv_results[-4].shape[2:], mode="bilinear", align_corners=False)
        x = self.changechannel3(torch.cat([x, conv_results[-4]], dim=1))
        if self.enable_query_gate and self.query_gate_high is not None and self.high_hint_conv is not None:
            high_hint = torch.sigmoid(self.high_hint_conv(x))
            x = self.query_gate_high(x, high_hint, query_vec)

        x = self.upconv4(x)
        return self.final_conv(x)
