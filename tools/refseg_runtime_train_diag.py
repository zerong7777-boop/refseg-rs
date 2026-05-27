from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from refseg_runtime.backends.native_xemap import build_native_xemap_runtime
from refseg_runtime.dataset import RefSegDataset
from refseg_runtime.preprocess import GroundingImagePreprocessor, SegmentationMaskPreprocessor


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-path", required=True)
    parser.add_argument("--val-ann-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    image_transform = GroundingImagePreprocessor(scale=(800, 800), keep_ratio=False)
    mask_transform = SegmentationMaskPreprocessor(scale=(800, 800), keep_ratio=False)

    log("build train dataset")
    train_dataset = RefSegDataset(
        ann_path=args.ann_path,
        data_root=args.data_root,
        image_transform=image_transform,
        mask_transform=mask_transform,
        max_samples=args.max_samples,
    )
    log(f"train dataset len={len(train_dataset)}")

    log("build val dataset")
    val_dataset = RefSegDataset(
        ann_path=args.val_ann_path,
        data_root=args.data_root,
        image_transform=image_transform,
        mask_transform=mask_transform,
        max_samples=args.max_samples,
    )
    log(f"val dataset len={len(val_dataset)}")

    log("build dataloaders")
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=False)

    log("fetch first train batch")
    first_batch = next(iter(train_loader))
    log(
        "first batch shapes: "
        f"image={tuple(first_batch['image'].shape)} "
        f"mask={tuple(first_batch['mask'].shape)} "
        f"text={list(first_batch['text'])[:1]}"
    )

    log("build runtime bundle")
    bundle = build_native_xemap_runtime(
        dataloader=train_loader,
        val_dataloader=val_loader,
        work_dir=args.work_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
        lr=1e-5,
        optimizer_type="adamw",
        weight_decay=1e-4,
        freeze_backbone=True,
        freeze_language_model=True,
        max_steps=1,
        max_epochs=1,
        val_interval=1,
        with_encoder=True,
        query_gate_cfg={"enable": True, "gate_channels": 256, "hidden_dim": 256},
    )
    log("runtime bundle built")
    log(json.dumps(bundle.get("resolved_architecture", {}), ensure_ascii=False))
    log(f"load reports keys={sorted(bundle.get('load_reports', {}).keys())}")

    model = bundle["model"]
    log(f"model device={next(model.parameters()).device}")
    log("call train()")
    bundle["train"]()
    log("train() finished")


if __name__ == "__main__":
    main()
