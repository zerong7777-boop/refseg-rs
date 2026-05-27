from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch.nn as nn

from ..checkpoint import load_prefixed_state_dict
from .decoding import XeMapSegSimpleDecoding
from .fusion import SparseTransformerFusion
from .segmentation import UNetSegHead


@dataclass
class LoadReport:
    prefix: str
    missing: List[str]
    unexpected: List[str]


class NativeRefSegSkeleton(nn.Module):
    """Portable container for the currently migrated native submodules."""

    def __init__(
        self,
        embed_dims: int = 256,
        unet_channels: Optional[List[int]] = None,
        with_unet_head: bool = True,
        with_sparse_fusion: bool = True,
        with_xemap_decoder: bool = False,
        with_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        if with_unet_head:
            self.U_net_seg_head = UNetSegHead(unet_channels or [embed_dims] * 4, embed_dims)
        if with_sparse_fusion:
            self.sparse_transformer_fusion = SparseTransformerFusion(
                embed_dims=embed_dims, num_heads=8, num_levels=4, num_points=4
            )
        if with_xemap_decoder:
            self.XeMap_seg_decoder = XeMapSegSimpleDecoding(c4_dims=embed_dims)
        if with_encoder:
            try:
                from .portable_transformer import PortableSparseGateTransformerEncoder
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "PortableSparseGateTransformerEncoder requires optional dependencies "
                    f"that are not available in the current environment: {exc}"
                ) from exc
            self.encoder = PortableSparseGateTransformerEncoder(
                num_layers=12,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=embed_dims,
                        num_heads=8,
                        dropout=0.0,
                        batch_first=True,
                        num_levels=4,
                    ),
                    ffn_cfg=dict(
                        embed_dims=embed_dims,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                    norm_cfg=dict(type="LN"),
                ),
                text_layer_cfg=dict(
                    self_attn_cfg=dict(embed_dims=embed_dims, num_heads=4, dropout=0.0, batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=embed_dims,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                    norm_cfg=dict(type="LN"),
                ),
                fusion_layer_cfg=dict(
                    v_dim=embed_dims,
                    l_dim=embed_dims,
                    embed_dim=1024,
                    num_heads=4,
                    dropout=0.1,
                    drop_path=0.0,
                    init_values=1e-4,
                ),
                num_feature_levels=4,
            )

    def load_from_checkpoint(self, checkpoint_path: str, strict: bool = True) -> Dict[str, LoadReport]:
        reports: Dict[str, LoadReport] = {}
        mapping = {
            "U_net_seg_head": getattr(self, "U_net_seg_head", None),
            "encoder.sparse_transformer_fusion": getattr(self, "sparse_transformer_fusion", None),
            "XeMap_seg_decoder": getattr(self, "XeMap_seg_decoder", None),
            "encoder": getattr(self, "encoder", None),
        }
        for prefix, module in mapping.items():
            if module is None:
                continue
            try:
                missing, unexpected = load_prefixed_state_dict(module, checkpoint_path, prefix, strict=strict)
            except RuntimeError as exc:
                reports[prefix] = LoadReport(prefix=prefix, missing=[str(exc)], unexpected=[])
                continue
            reports[prefix] = LoadReport(prefix=prefix, missing=missing, unexpected=unexpected)
        return reports
