from __future__ import annotations
from typing import Any, Dict, List

import torch
import torch.nn as nn

from ..checkpoint import load_prefixed_state_dict
from ..models import PortableChannelMapper


class PortableSwinFeatureExtractor(nn.Module):
    """Portable Swin backbone plus pure torch ChannelMapper neck."""

    def __init__(
        self,
        optional_site_packages: str = "",
        backbone_embed_dims: int = 128,
        backbone_depths: List[int] | None = None,
        backbone_num_heads: List[int] | None = None,
        backbone_window_size: int = 12,
        backbone_drop_path_rate: float = 0.3,
        neck_in_channels: List[int] | None = None,
        neck_out_channels: int | None = None,
        neck_num_outs: int = 4,
    ) -> None:
        super().__init__()
        del optional_site_packages
        try:
            from ..models.portable_swin import PortableSwinTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError("PortableSwinTransformer requires optional runtime dependencies.") from exc

        resolved_neck_in_channels = neck_in_channels or [
            backbone_embed_dims * 2,
            backbone_embed_dims * 4,
            backbone_embed_dims * 8,
        ]
        resolved_neck_out_channels = neck_out_channels or (backbone_embed_dims * 2)

        self.backbone = PortableSwinTransformer(
            embed_dims=backbone_embed_dims,
            depths=tuple(backbone_depths or [2, 2, 18, 2]),
            num_heads=tuple(backbone_num_heads or [4, 8, 16, 32]),
            window_size=backbone_window_size,
            drop_path_rate=backbone_drop_path_rate,
            out_indices=(1, 2, 3),
        )
        self.neck = PortableChannelMapper(
            in_channels=resolved_neck_in_channels,
            out_channels=resolved_neck_out_channels,
            kernel_size=1,
            num_outs=neck_num_outs,
            act=False,
            bias=True,
        )

    def forward(self, images: torch.Tensor):
        return list(self.neck(self.backbone(images)))

    def load_partial_checkpoint(self, checkpoint_path: str, strict: bool = False) -> Dict[str, Dict[str, List[str]]]:
        reports: Dict[str, Dict[str, List[str]]] = {}
        for prefix, module in (("backbone", self.backbone), ("neck", self.neck)):
            try:
                missing, unexpected = load_prefixed_state_dict(module, checkpoint_path, prefix, strict=strict)
            except RuntimeError as exc:
                reports[prefix] = {"missing": [str(exc)], "unexpected": []}
                continue
            reports[prefix] = {"missing": missing, "unexpected": unexpected}
        return reports


def build_portable_swin_feature_extractor(**kwargs: Any) -> PortableSwinFeatureExtractor:
    return PortableSwinFeatureExtractor(**kwargs)
