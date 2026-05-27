from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseTransformerFusion(nn.Module):
    """Portable fusion block extracted from the MMDetection implementation."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        use_ffn: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.use_layernorm = use_layernorm
        self.use_ffn = use_ffn

        self.visual_q_proj = nn.Linear(embed_dims, embed_dims)
        self.visual_k_proj = nn.Linear(embed_dims, embed_dims)
        self.visual_v_proj = nn.Linear(embed_dims, embed_dims)
        self.text_k_proj = nn.Linear(embed_dims, embed_dims)
        self.text_v_proj = nn.Linear(embed_dims, embed_dims)
        self.sampling_offsets = nn.Linear(embed_dims * 2, num_heads * num_levels * num_points * 2)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

        if self.use_layernorm:
            self.norm = nn.LayerNorm(embed_dims)
        if self.use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 4),
                nn.ReLU(),
                nn.Linear(embed_dims * 4, embed_dims),
                nn.Dropout(dropout),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in (
            self.visual_q_proj,
            self.visual_k_proj,
            self.visual_v_proj,
            self.text_k_proj,
            self.text_v_proj,
            self.sampling_offsets,
            self.output_proj,
        ):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
        if self.use_layernorm:
            nn.init.constant_(self.norm.weight, 1.0)
            nn.init.constant_(self.norm.bias, 0.0)
        if self.use_ffn:
            for layer in self.ffn:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)

    def _sample_visual_features(self, visual_feats, spatial_shapes, sampling_locations, level_start_index):
        bs, num_query, num_heads, num_levels, num_points, _ = sampling_locations.shape
        sampled_feats = []
        for lvl, (height, width) in enumerate(spatial_shapes):
            spatial_feat = visual_feats[:, level_start_index[lvl]: level_start_index[lvl] + height * width, :]
            spatial_feat = spatial_feat.view(bs, height, width, self.num_heads, -1).permute(0, 3, 4, 1, 2)
            sampling_grid = 2 * sampling_locations[:, :, :, lvl].reshape(bs * num_heads, num_query * num_points, 2) - 1
            sampled_feat = F.grid_sample(
                spatial_feat.reshape(bs * num_heads, -1, height, width),
                sampling_grid.unsqueeze(2),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).view(bs, num_heads, -1, num_query, num_points).permute(0, 3, 1, 4, 2)
            sampled_feats.append(sampled_feat)

        return torch.stack(sampled_feats, dim=3).reshape(
            bs, num_query, num_heads, num_levels * num_points, self.embed_dims // num_heads
        )

    def forward(self, query, query_pos, spatial_shapes, level_start_index, reference_points, text_feats):
        bs, num_query, _ = query.shape
        head_dim = self.embed_dims // self.num_heads
        query_with_pos = query + query_pos
        query_proj = self.visual_q_proj(query_with_pos)
        key_visual = self.visual_k_proj(query_with_pos)
        value_visual = self.visual_v_proj(query_with_pos)

        text_feat_mean = text_feats.mean(dim=1, keepdim=True).repeat(1, num_query, 1)
        combined_feat = torch.cat([query_with_pos, text_feat_mean], dim=-1)
        sampling_offsets = self.sampling_offsets(combined_feat).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        offset_normalizer = spatial_shapes[:, [1, 0]].reshape(1, 1, 1, self.num_levels, 1, 2)
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer

        sampled_key_visual = self._sample_visual_features(
            key_visual, spatial_shapes, sampling_locations, level_start_index
        )
        sampled_value_visual = self._sample_visual_features(
            value_visual, spatial_shapes, sampling_locations, level_start_index
        )
        sampled_key_visual = sampled_key_visual.permute(0, 1, 3, 2, 4).reshape(
            bs, num_query, self.num_levels * self.num_points, self.embed_dims
        )
        sampled_value_visual = sampled_value_visual.permute(0, 1, 3, 2, 4).reshape(
            bs, num_query, self.num_levels * self.num_points, self.embed_dims
        )

        key_text = self.text_k_proj(text_feats).unsqueeze(1).repeat(1, num_query, 1, 1)
        value_text = self.text_v_proj(text_feats).unsqueeze(1).repeat(1, num_query, 1, 1)
        key_combined = torch.cat([sampled_key_visual, key_text], dim=2)
        value_combined = torch.cat([sampled_value_visual, value_text], dim=2)

        query_proj = query_proj.view(bs, num_query, self.num_heads, head_dim)
        num_kv = key_combined.size(2)
        key_combined = key_combined.view(bs, num_query, num_kv, self.num_heads, head_dim)
        value_combined = value_combined.view(bs, num_query, num_kv, self.num_heads, head_dim)

        attn_weights = torch.einsum("bqhd,bqkhd->bqhk", query_proj, key_combined) * (head_dim ** -0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.einsum("bqhk,bqkhd->bqhd", attn_weights, value_combined).reshape(bs, num_query, self.embed_dims)

        output = self.output_proj(output) + query
        if self.use_layernorm:
            output = self.norm(output)
        if self.use_ffn:
            output = output + self.ffn(output)
            if self.use_layernorm:
                output = self.norm(output)
        return self.dropout(output)
