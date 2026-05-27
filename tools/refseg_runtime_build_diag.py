from __future__ import annotations

import argparse
import json
import time

import torch

from refseg_runtime.backends.native_xemap import (
    _build_visual_extractor,
    _resolve_architecture,
    _select_device,
)
from refseg_runtime.models import NativeXeMapRefSeg, PortableGroundingBertTextEncoder


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    log("select device")
    device = _select_device(args.device)
    log(f"device={device}")

    log("resolve architecture")
    arch = _resolve_architecture(
        checkpoint_path=args.checkpoint,
        embed_dims=256,
        backbone_embed_dims=None,
        neck_out_channels=None,
        infer_arch_from_checkpoint=True,
    )
    log(json.dumps(arch, ensure_ascii=False))

    log("build visual extractor")
    visual_extractor = _build_visual_extractor(
        visual_backend="portable_swin",
        optional_site_packages="",
        backbone_embed_dims=int(arch["backbone_embed_dims"]),
        neck_out_channels=int(arch["neck_out_channels"]),
    )
    log("visual extractor built")

    log("build text encoder")
    text_encoder = PortableGroundingBertTextEncoder(
        name="bert-base-uncased",
        output_dims=int(arch["embed_dims"]),
        optional_site_packages="",
    )
    log("text encoder built")

    log("build model")
    model = NativeXeMapRefSeg(
        embed_dims=int(arch["embed_dims"]),
        decoder_type="unet",
        with_encoder=True,
        visual_extractor=visual_extractor,
        text_encoder=text_encoder,
        query_gate_cfg={"enable": True, "gate_channels": 256, "hidden_dim": 256},
    )
    log("model built")

    log("move model to device")
    model = model.to(device)
    log("model moved")

    log("load partial checkpoint")
    reports = model.load_partial_checkpoint(args.checkpoint, strict=False)
    log(f"load done keys={sorted(reports.keys())}")


if __name__ == "__main__":
    main()
