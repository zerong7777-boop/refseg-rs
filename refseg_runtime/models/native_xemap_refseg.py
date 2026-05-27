from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..checkpoint import (
    get_tensor,
    load_prefixed_state_dict_from_state_dict,
    load_state_dict_from_path,
)
from .decoding import XeMapSegSimpleDecoding
from .positional_encoding import PortableSinePositionalEncoding
from .segmentation import RefSegLoss, UNetSegHead, convert_to_two_channel_logits


@dataclass
class NativeForwardOutput:
    logits: torch.Tensor
    features: List[torch.Tensor]
    encoded_features: List[torch.Tensor]
    text_memory: Optional[torch.Tensor]
    text_attention_mask: Optional[torch.Tensor]


class NativeXeMapRefSeg(nn.Module):
    """Portable RefSeg wrapper around the migrated native modules."""

    def __init__(
        self,
        embed_dims: int = 256,
        decoder_type: str = "unet",
        with_encoder: bool = False,
        text_max_tokens: int = 32,
        visual_extractor: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        query_gate_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.decoder_type = decoder_type
        self.with_encoder = with_encoder

        if visual_extractor is None:
            raise ValueError(
                "NativeXeMapRefSeg requires an explicit visual_extractor for the delivery path."
            )
        resolved_visual = visual_extractor
        if hasattr(resolved_visual, "backbone") and hasattr(resolved_visual, "neck"):
            self.backbone = resolved_visual.backbone
            self.neck = resolved_visual.neck
        else:
            self.backbone = resolved_visual
            self.neck = None

        if text_encoder is None:
            raise ValueError(
                "NativeXeMapRefSeg requires an explicit text_encoder/language_model for the delivery path."
            )
        else:
            self.language_model = text_encoder
        self.refseg_loss = RefSegLoss()
        self.query_gate_cfg = copy.deepcopy(query_gate_cfg) if query_gate_cfg is not None else {}
        self.U_net_seg_head = UNetSegHead([embed_dims] * 4, embed_dims, query_gate_cfg=self.query_gate_cfg)
        self.enable_query_gate = self.U_net_seg_head.enable_query_gate
        self.XeMap_seg_decoder = XeMapSegSimpleDecoding(c4_dims=embed_dims)
        self.positional_encoding = PortableSinePositionalEncoding(
            num_feats=embed_dims // 2,
            normalize=True,
            offset=0.0,
            temperature=20,
        )
        self.level_embed = nn.Parameter(torch.zeros(4, embed_dims))
        nn.init.normal_(self.level_embed)

        if with_encoder:
            try:
                from .portable_transformer import PortableSparseGateTransformerEncoder
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "NativeXeMapRefSeg(with_encoder=True) requires optional runtime dependencies."
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

    @property
    def visual_extractor(self):
        return self._extract_visual_features

    @property
    def text_encoder(self):
        return self.language_model

    @staticmethod
    def _build_spatial_meta(features: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = features[0].shape[0]
        device = features[0].device
        spatial_shapes = torch.tensor(
            [[feature.shape[2], feature.shape[3]] for feature in features],
            dtype=torch.long,
            device=device,
        )
        level_sizes = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        level_start_index = torch.cat([torch.zeros(1, dtype=torch.long, device=device), level_sizes.cumsum(0)[:-1]])
        valid_ratios = torch.ones((batch_size, len(features), 2), dtype=features[0].dtype, device=device)
        return {
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
            "valid_ratios": valid_ratios,
        }

    def _flatten_features(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat([feature.flatten(2).transpose(1, 2) for feature in features], dim=1)

    def _make_query_pos(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        pos_embeds = []
        for level, feature in enumerate(features):
            pos = self.positional_encoding(None, input=feature)
            pos = pos.view(feature.shape[0], self.embed_dims, -1).permute(0, 2, 1)
            pos = pos + self.level_embed[level].view(1, 1, -1)
            pos_embeds.append(pos)
        return torch.cat(pos_embeds, dim=1)

    @staticmethod
    def _pool_text_features(
        text_memory: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if text_memory is None:
            return None
        if text_attention_mask is None:
            return text_memory.mean(dim=1)
        valid_mask = (~text_attention_mask.bool()).to(dtype=text_memory.dtype)
        valid_mask = valid_mask.unsqueeze(-1)
        denom = valid_mask.sum(dim=1).clamp(min=1.0)
        return (text_memory * valid_mask).sum(dim=1) / denom

    def _unflatten_features(self, memory: torch.Tensor, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        split_sizes = [feature.shape[2] * feature.shape[3] for feature in features]
        chunks = torch.split(memory, split_sizes, dim=1)
        output = []
        for chunk, feature in zip(chunks, features):
            batch_size, _, height, width = feature.shape
            output.append(chunk.transpose(1, 2).reshape(batch_size, self.embed_dims, height, width))
        return output

    def _decode(
        self,
        features: Sequence[torch.Tensor],
        text_memory: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.decoder_type == "xemap":
            return self.XeMap_seg_decoder(list(features))
        query_vec = None
        if self.enable_query_gate:
            query_vec = self._pool_text_features(text_memory, text_attention_mask)
        return self.U_net_seg_head(list(features), query_vec=query_vec)

    def _extract_visual_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        if self.neck is not None:
            return list(self.neck(self.backbone(images)))
        return list(self.backbone(images))

    def _encode_text_inputs(self, texts: Sequence[str], device: torch.device) -> Dict[str, torch.Tensor]:
        return self.language_model(texts, device=device)

    def load_partial_checkpoint(self, checkpoint_path: str, strict: bool = False) -> Dict[str, Dict[str, List[str]]]:
        reports: Dict[str, Dict[str, List[str]]] = {}
        try:
            state_dict = load_state_dict_from_path(checkpoint_path)
        except RuntimeError as exc:
            return {"checkpoint": {"missing": [str(exc)], "unexpected": []}}
        prefix_mapping = {
            "backbone": getattr(self, "backbone", None),
            "neck": getattr(self, "neck", None),
            "language_model": getattr(self, "language_model", None),
            "U_net_seg_head": getattr(self, "U_net_seg_head", None),
            "XeMap_seg_decoder": getattr(self, "XeMap_seg_decoder", None),
            "encoder": getattr(self, "encoder", None),
        }
        for prefix, module in prefix_mapping.items():
            if module is None:
                continue
            if prefix == "language_model":
                if hasattr(module, "load_partial_checkpoint"):
                    reports[prefix] = module.load_partial_checkpoint(
                        checkpoint_path,
                        strict=strict,
                        state_dict=state_dict,
                    )
                continue
            try:
                missing, unexpected = load_prefixed_state_dict_from_state_dict(
                    module,
                    state_dict,
                    prefix,
                    strict=strict,
                )
            except RuntimeError as exc:
                reports[prefix] = {"missing": [str(exc)], "unexpected": []}
                continue
            reports[prefix] = {"missing": missing, "unexpected": unexpected}
        level_embed = get_tensor(state_dict, "level_embed")
        if level_embed is None:
            reports["level_embed"] = {
                "missing": [f"No keys found for parameter 'level_embed' in {checkpoint_path}"],
                "unexpected": [],
            }
        elif tuple(level_embed.shape) != tuple(self.level_embed.shape):
            reports["level_embed"] = {
                "missing": [
                    "Checkpoint parameter 'level_embed' has incompatible shape "
                    f"{tuple(level_embed.shape)} != {tuple(self.level_embed.shape)}"
                ],
                "unexpected": [],
            }
        else:
            with torch.no_grad():
                self.level_embed.copy_(level_embed)
            reports["level_embed"] = {"missing": [], "unexpected": []}
        return reports

    def encode(self, images: torch.Tensor, texts: Optional[Sequence[str]] = None) -> NativeForwardOutput:
        if texts is None:
            texts = [""] * images.shape[0]
        features = self._extract_visual_features(images)
        encoded_features = list(features)
        text_memory = None
        text_attention_mask = None
        if self.with_encoder or self.enable_query_gate:
            text_inputs = self._encode_text_inputs(texts, device=images.device)
            memory_text = text_inputs.get("memory_text", text_inputs.get("embedded"))
            text_attention_mask = text_inputs.get("text_attention_mask")
            if text_attention_mask is None and "text_token_mask" in text_inputs:
                text_attention_mask = ~text_inputs["text_token_mask"]
            if self.with_encoder:
                text_self_attention_masks = text_inputs.get("text_self_attention_masks", text_inputs.get("masks"))
                spatial_meta = self._build_spatial_meta(features)
                flat_features = self._flatten_features(features)
                query_pos = self._make_query_pos(features)
                key_padding_mask = torch.zeros(
                    (flat_features.shape[0], flat_features.shape[1]),
                    dtype=torch.bool,
                    device=images.device,
                )
                encoded_memory, text_memory = self.encoder(
                    query=flat_features,
                    query_pos=query_pos,
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_meta["spatial_shapes"],
                    level_start_index=spatial_meta["level_start_index"],
                    valid_ratios=spatial_meta["valid_ratios"],
                    memory_text=memory_text,
                    text_attention_mask=text_attention_mask,
                    pos_text=text_inputs.get("pos_text"),
                    text_self_attention_masks=text_self_attention_masks,
                    position_ids=text_inputs.get("position_ids"),
                )
                encoded_features = self._unflatten_features(encoded_memory, features)
            else:
                text_memory = memory_text
        logits = self._decode(encoded_features, text_memory=text_memory, text_attention_mask=text_attention_mask)
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return NativeForwardOutput(
            logits=logits,
            features=list(features),
            encoded_features=encoded_features,
            text_memory=text_memory,
            text_attention_mask=text_attention_mask,
        )

    def forward(self, images: torch.Tensor, texts: Optional[Sequence[str]] = None) -> torch.Tensor:
        return self.encode(images, texts=texts).logits

    def predict(self, images: torch.Tensor, texts: Optional[Sequence[str]] = None) -> torch.Tensor:
        logits = self.forward(images, texts=texts)
        return torch.sigmoid(logits).squeeze(1)

    def loss(self, images: torch.Tensor, masks: torch.Tensor, texts: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
        logits = self.forward(images, texts=texts).squeeze(1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp_(-20.0, 20.0)
        target = (masks.squeeze(1) > 0.5).long()
        loss_seg = self.refseg_loss(convert_to_two_channel_logits(logits), target)
        return {"loss_seg": loss_seg}
