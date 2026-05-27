from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .fusion import SparseTransformerFusion
from .nn_compat import DropPath, FFN, MultiScaleDeformableAttention, MultiheadAttention, build_norm_layer


def get_text_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
) -> torch.Tensor:
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor) -> torch.Tensor:
        sin_x = x * scale / dim_t
        return torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3).flatten(2)

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy and len(pos_res) >= 2:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    return torch.cat(pos_res, dim=-1)


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim: int, l_dim: int, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(v_dim, embed_dim)
        self.l_proj = nn.Linear(l_dim, embed_dim)
        self.values_v_proj = nn.Linear(v_dim, embed_dim)
        self.values_l_proj = nn.Linear(l_dim, embed_dim)
        self.out_v_proj = nn.Linear(embed_dim, v_dim)
        self.out_l_proj = nn.Linear(embed_dim, l_dim)
        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self) -> None:
        for layer in (
            self.v_proj,
            self.l_proj,
            self.values_v_proj,
            self.values_l_proj,
            self.out_v_proj,
            self.out_l_proj,
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(
        self,
        vision: torch.Tensor,
        lang: torch.Tensor,
        attention_mask_v: Optional[torch.Tensor] = None,
        attention_mask_l: Optional[torch.Tensor] = None,
    ):
        batch_size, tgt_len, _ = vision.size()
        query_states = self.v_proj(vision) * self.scale
        key_states = self._shape(self.l_proj(lang), -1, batch_size)
        value_v_states = self._shape(self.values_v_proj(vision), -1, batch_size)
        value_l_states = self._shape(self.values_l_proj(lang), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        attn_weights_t = attn_weights.transpose(1, 2)
        attn_weights_l = attn_weights_t - torch.max(attn_weights_t, dim=-1, keepdim=True)[0]
        if attention_mask_v is not None:
            attention_mask_v = attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_weights_l.masked_fill_(attention_mask_v, float("-inf"))
        attn_weights_l = attn_weights_l.softmax(dim=-1)

        if attention_mask_l is not None:
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(batch_size, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)

        attn_weights_v = torch.softmax(attn_weights, dim=-1)
        attn_probs_v = torch.nn.functional.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = torch.nn.functional.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        attn_output_v = attn_output_v.view(batch_size, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2).reshape(batch_size, tgt_len, self.embed_dim)
        attn_output_l = attn_output_l.view(batch_size, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2).reshape(batch_size, src_len, self.embed_dim)
        return self.out_v_proj(attn_output_v), self.out_l_proj(attn_output_l)


class SingleScaleBiAttentionBlock(nn.Module):
    def __init__(
        self,
        v_dim: int,
        l_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        init_values: float = 1e-4,
    ) -> None:
        super().__init__()
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones(v_dim), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones(l_dim), requires_grad=True)

    def forward(
        self,
        visual_feature: torch.Tensor,
        lang_feature: torch.Tensor,
        attention_mask_v: Optional[torch.Tensor] = None,
        attention_mask_l: Optional[torch.Tensor] = None,
    ):
        visual = self.layer_norm_v(visual_feature)
        lang = self.layer_norm_l(lang_feature)
        delta_v, delta_l = self.attn(visual, lang, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l)
        visual = visual + self.drop_path(self.gamma_v * delta_v)
        lang = lang + self.drop_path(self.gamma_l * delta_l)
        return visual, lang


class PortableDetrTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        self_attn_cfg: Optional[dict] = None,
        ffn_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.self_attn_cfg = dict(self_attn_cfg or dict(embed_dims=256, num_heads=8, dropout=0.0))
        self.self_attn_cfg.setdefault("batch_first", True)
        self.ffn_cfg = dict(ffn_cfg or dict(
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ))
        self.norm_cfg = dict(norm_cfg or dict(type="LN"))
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        self.norms = nn.ModuleList([build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(2)])

    def forward(self, query: torch.Tensor, query_pos: torch.Tensor, key_padding_mask: Optional[torch.Tensor], **kwargs):
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)
        return query


class PortableDeformableDetrTransformerEncoderLayer(PortableDetrTransformerEncoderLayer):
    def __init__(
        self,
        self_attn_cfg: Optional[dict] = None,
        ffn_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.self_attn_cfg = dict(self_attn_cfg or dict(
            embed_dims=256,
            num_heads=8,
            dropout=0.0,
            batch_first=True,
            num_levels=4,
        ))
        self.self_attn_cfg.setdefault("batch_first", True)
        self.ffn_cfg = dict(ffn_cfg or dict(
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ))
        self.norm_cfg = dict(norm_cfg or dict(type="LN"))
        self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        self.norms = nn.ModuleList([build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(2)])


class PortableSparseGateTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        layer_cfg: Optional[dict] = None,
        text_layer_cfg: Optional[dict] = None,
        fusion_layer_cfg: Optional[dict] = None,
        num_cp: int = 0,
        num_feature_levels: int = 4,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layer_cfg = dict(layer_cfg or {})
        self.text_layer_cfg = dict(text_layer_cfg or {})
        self.fusion_layer_cfg = dict(fusion_layer_cfg or {})
        self.num_cp = num_cp
        self.num_feature_levels = num_feature_levels

        self.layers = nn.ModuleList([
            PortableDeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = nn.ModuleList([
            PortableDetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = nn.ModuleList([
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        self.sparse_transformer_fusion = SparseTransformerFusion(
            embed_dims=self.embed_dims, num_heads=8, num_levels=num_feature_levels, num_points=4, dropout=0.1
        )
        self.gate_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1),
                nn.Tanh(),
            )
            for _ in range(num_feature_levels)
        ])

    @staticmethod
    def get_encoder_reference_points(
        spatial_shapes: torch.Tensor,
        valid_ratios: torch.Tensor,
        device,
    ) -> torch.Tensor:
        reference_points_list = []
        for lvl, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        key_padding_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        memory_text: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        pos_text: Optional[torch.Tensor] = None,
        text_self_attention_masks: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        output = query
        reference_points = self.get_encoder_reference_points(spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers and memory_text is not None:
            batch_size, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = torch.arange(n_text, device=memory_text.device).float().unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
                pos_text = get_text_sine_pos_embed(pos_text, num_pos_feats=self.embed_dims, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(position_ids[..., None], num_pos_feats=self.embed_dims, exchange_xy=False)

        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers and memory_text is not None:
                output, memory_text = self.fusion_layers[layer_id](
                    visual_feature=output,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )
            if self.text_layers and memory_text is not None and text_self_attention_masks is not None:
                num_heads = self.text_layers[layer_id].self_attn_cfg.get("num_heads", 8)
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=pos_text if pos_text is not None else None,
                    attn_mask=~text_self_attention_masks.repeat(num_heads, 1, 1),
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
            )

        vis_text_fusion = self.sparse_transformer_fusion(
            query=output,
            query_pos=query_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points=reference_points,
            text_feats=memory_text,
        )

        gated_fusions = []
        for lvl, (height, width) in enumerate(spatial_shapes):
            start_idx = level_start_index[lvl]
            end_idx = level_start_index[lvl + 1] if (lvl + 1) < len(level_start_index) else output.shape[1]
            output_lvl = output[:, start_idx:end_idx, :].permute(0, 2, 1).reshape(-1, self.embed_dims, height, width)
            fusion_lvl = vis_text_fusion[:, start_idx:end_idx, :].permute(0, 2, 1).reshape(-1, self.embed_dims, height, width)
            gate = self.gate_modules[lvl](fusion_lvl)
            gated_feature = output_lvl + gate * fusion_lvl
            gated_fusions.append(gated_feature.flatten(2).permute(0, 2, 1))

        return torch.cat(gated_fusions, dim=1), memory_text
