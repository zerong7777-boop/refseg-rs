from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from refseg_runtime.checkpoint import summarize_checkpoint
else:
    from .checkpoint import summarize_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a RefSeg checkpoint in the pure runtime. Use a sanitized checkpoint ending with .state_dict.pth."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path. Use a sanitized .state_dict.pth checkpoint.",
    )
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    resolved_checkpoint = args.checkpoint
    if not resolved_checkpoint.endswith(".state_dict.pth"):
        raise ValueError(f"Checkpoint must be a sanitized .state_dict.pth file, got: {resolved_checkpoint}")
    summary = summarize_checkpoint(args.checkpoint)
    state_dict = {}
    try:
        import torch
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            state_dict = {}
    except Exception:
        state_dict = {}

    inferred = {
        "backbone_embed_dims": summary.backbone_embed_dims,
        "model_embed_dims": summary.model_embed_dims,
        "neck_out_channels": summary.neck_out_channels,
        "has_unet_head": summary.has_unet_head,
        "has_sparse_gate": summary.has_sparse_gate,
        "with_xemap_decoder": any(k.startswith("XeMap_seg_decoder.") or k.startswith("module.XeMap_seg_decoder.") for k in state_dict),
    }
    if inferred["model_embed_dims"] is None and inferred["neck_out_channels"] is not None:
        inferred["model_embed_dims"] = inferred["neck_out_channels"]
    if inferred["model_embed_dims"] is None:
        inferred["model_embed_dims"] = inferred["backbone_embed_dims"]

    payload = {
        "requested_checkpoint": args.checkpoint,
        "resolved_checkpoint": resolved_checkpoint,
        "summary": {
            "checkpoint_path": summary.checkpoint_path,
            "num_tensors": summary.num_tensors,
            "embed_dims": summary.embed_dims,
            "backbone_embed_dims": inferred["backbone_embed_dims"],
            "model_embed_dims": inferred["model_embed_dims"],
            "neck_out_channels": inferred["neck_out_channels"],
            "has_unet_head": inferred["has_unet_head"],
            "has_sparse_gate": inferred["has_sparse_gate"],
            "with_xemap_decoder": inferred["with_xemap_decoder"],
            "example_keys": summary.example_keys,
        },
        "load_reports": {},
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.json:
        out_dir = os.path.dirname(args.json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
