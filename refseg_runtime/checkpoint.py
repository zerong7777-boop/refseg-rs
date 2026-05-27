from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


def extract_state_dict(checkpoint_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            maybe = checkpoint_obj.get(key)
            if isinstance(maybe, dict):
                return maybe
        if all(isinstance(v, torch.Tensor) for v in checkpoint_obj.values()):
            return checkpoint_obj
    return {}


def resolve_runtime_checkpoint_path(checkpoint_path: str) -> str:
    if not checkpoint_path:
        return checkpoint_path
    if not checkpoint_path.endswith('.state_dict.pth'):
        raise ValueError(
            f'Checkpoint must be a sanitized .state_dict.pth file, got: {checkpoint_path}'
        )
    return checkpoint_path


def _torch_load_maybe_weights_only(checkpoint_path: str) -> Any:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This runtime expects a torch build that supports weights_only checkpoint loading. "
            "Use a sanitized .state_dict.pth checkpoint or upgrade the runtime torch version."
        ) from exc


def load_state_dict_from_path(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint_path = resolve_runtime_checkpoint_path(checkpoint_path)
    try:
        checkpoint_obj = _torch_load_maybe_weights_only(checkpoint_path)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load checkpoint weights from {checkpoint_path}. "
            "Use a sanitized .state_dict.pth checkpoint."
        ) from exc

    state_dict = extract_state_dict(checkpoint_obj)
    if not state_dict:
        raise RuntimeError(
            f"No parseable state_dict found in {checkpoint_path}. "
            "Use a sanitized .state_dict.pth checkpoint."
        )
    return state_dict


def get_tensor(state_dict: Dict[str, torch.Tensor], key: str) -> Optional[torch.Tensor]:
    if key in state_dict and isinstance(state_dict[key], torch.Tensor):
        return state_dict[key]
    module_key = f"module.{key}"
    if module_key in state_dict and isinstance(state_dict[module_key], torch.Tensor):
        return state_dict[module_key]
    return None


def strip_prefix_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    if prefix and not prefix.endswith("."):
        prefix = prefix + "."
    stripped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped[key[len(prefix):]] = value
        elif prefix and key.startswith("module." + prefix):
            stripped[key[len("module." + prefix):]] = value
    return stripped


def load_prefixed_state_dict(
    module: torch.nn.Module,
    checkpoint_path: str,
    prefix: str,
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    checkpoint_path = resolve_runtime_checkpoint_path(checkpoint_path)
    state_dict = load_state_dict_from_path(checkpoint_path)
    return load_prefixed_state_dict_from_state_dict(module, state_dict, prefix, strict=strict)


def load_prefixed_state_dict_from_state_dict(
    module: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    submodule_state = strip_prefix_from_state_dict(state_dict, prefix)
    if not submodule_state:
        raise RuntimeError(f"No keys found for prefix '{prefix}' in provided state_dict")
    incompatible = module.load_state_dict(submodule_state, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


@dataclass
class CheckpointArchitecture:
    checkpoint_path: str
    backbone_embed_dims: Optional[int]
    model_embed_dims: Optional[int]
    neck_out_channels: Optional[int]
    neck_in_channels: List[int]


@dataclass
class CheckpointSummary:
    checkpoint_path: str
    num_tensors: int
    backbone_embed_dims: Optional[int]
    model_embed_dims: Optional[int]
    neck_out_channels: Optional[int]
    has_unet_head: bool
    has_sparse_gate: bool
    example_keys: List[str]

    @property
    def embed_dims(self) -> Optional[int]:
        return self.backbone_embed_dims


def infer_checkpoint_architecture(checkpoint_path: str) -> CheckpointArchitecture:
    checkpoint_path = resolve_runtime_checkpoint_path(checkpoint_path)
    state_dict = load_state_dict_from_path(checkpoint_path)

    patch_weight = get_tensor(state_dict, "backbone.patch_embed.projection.weight")
    backbone_embed_dims = int(patch_weight.shape[0]) if patch_weight is not None and patch_weight.ndim >= 1 else None

    neck_in_channels: List[int] = []
    neck_out_channels = None
    for idx in range(3):
        neck_weight = get_tensor(state_dict, f"neck.convs.{idx}.conv.weight")
        if neck_weight is None:
            break
        neck_in_channels.append(int(neck_weight.shape[1]))
        if neck_out_channels is None:
            neck_out_channels = int(neck_weight.shape[0])

    model_embed_dims = None
    encoder_norm = get_tensor(state_dict, "encoder.fusion_layers.0.layer_norm_v.weight")
    if encoder_norm is not None and encoder_norm.ndim >= 1:
        model_embed_dims = int(encoder_norm.shape[0])
    if model_embed_dims is None and neck_out_channels is not None:
        model_embed_dims = neck_out_channels
    if model_embed_dims is None:
        head_weight = get_tensor(state_dict, "U_net_seg_head.conv_layers.0.0.weight")
        if head_weight is not None and head_weight.ndim >= 1:
            model_embed_dims = int(head_weight.shape[0])

    return CheckpointArchitecture(
        checkpoint_path=checkpoint_path,
        backbone_embed_dims=backbone_embed_dims,
        model_embed_dims=model_embed_dims,
        neck_out_channels=neck_out_channels,
        neck_in_channels=neck_in_channels,
    )


def summarize_checkpoint(checkpoint_path: str) -> CheckpointSummary:
    checkpoint_path = resolve_runtime_checkpoint_path(checkpoint_path)
    state_dict = load_state_dict_from_path(checkpoint_path)
    arch = infer_checkpoint_architecture(checkpoint_path)
    has_unet_head = any(
        k.startswith("U_net_seg_head.") or k.startswith("module.U_net_seg_head.")
        for k in state_dict
    )
    has_sparse_gate = any(
        "encoder.gate_modules." in k or "module.encoder.gate_modules." in k
        for k in state_dict
    )
    return CheckpointSummary(
        checkpoint_path=checkpoint_path,
        num_tensors=len(state_dict),
        backbone_embed_dims=arch.backbone_embed_dims,
        model_embed_dims=arch.model_embed_dims,
        neck_out_channels=arch.neck_out_channels,
        has_unet_head=has_unet_head,
        has_sparse_gate=has_sparse_gate,
        example_keys=list(state_dict.keys())[:20],
    )
