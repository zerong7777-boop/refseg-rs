from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from refseg_runtime.dataset import RefSegDataset
    from refseg_runtime.factory import load_factory
    from refseg_runtime.preprocess import GroundingImagePreprocessor, SegmentationMaskPreprocessor
else:
    from .dataset import RefSegDataset
    from .factory import load_factory
    from .preprocess import GroundingImagePreprocessor, SegmentationMaskPreprocessor


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(
        description="Pure runtime RefSeg training scaffold. Use sanitized checkpoints ending with .state_dict.pth."
    )
    default_factory = "refseg_runtime.backends.native_xemap:build_native_xemap_runtime"
    parser.add_argument("--ann-path", required=True)
    parser.add_argument("--ann-format", default="auto", choices=["txt", "json", "auto"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--img-prefix", default="images")
    parser.add_argument("--mask-prefix", default="masked")
    parser.add_argument("--val-ann-path", default="")
    parser.add_argument("--val-ann-format", default="auto", choices=["txt", "json", "auto"])
    parser.add_argument("--val-data-root", default="")
    parser.add_argument("--val-img-prefix", default="images")
    parser.add_argument("--val-mask-prefix", default="masked")
    parser.add_argument("--val-max-samples", type=int, default=0)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--factory", default=default_factory,
                        help="module:function that builds a native training bundle.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--checkpoint", default="",
                        help="Checkpoint path. Use a sanitized .state_dict.pth checkpoint.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true",
                        help="Enable AMP during runtime training to reduce GPU memory usage.")
    parser.add_argument("--freeze-neck", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--val-every-steps", type=int, default=0)
    parser.add_argument("--val-max-batches", type=int, default=0)
    parser.add_argument("--val-pred-threshold", type=float, default=0.5)
    parser.add_argument("--val-gt-mode", default="gt_positive", choices=["gt_positive", "gt255", "gt1"])
    parser.add_argument("--val-iou-thresholds", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--best-metric", default="loss_seg", choices=["loss_seg", "miou", "oiou"])
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--config-json", default="",
                        help="Optional json blob passed to the training factory.")
    parser.add_argument("--resize-width", type=int, default=800)
    parser.add_argument("--resize-height", type=int, default=800)
    parser.add_argument("--keep-ratio", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    image_transform = GroundingImagePreprocessor(
        scale=(args.resize_width, args.resize_height),
        keep_ratio=args.keep_ratio,
    )
    mask_transform = SegmentationMaskPreprocessor(
        scale=(args.resize_width, args.resize_height),
        keep_ratio=args.keep_ratio,
    )
    dataset = RefSegDataset(
        ann_path=args.ann_path,
        ann_format=args.ann_format,
        data_root=args.data_root,
        img_prefix=args.img_prefix,
        mask_prefix=args.mask_prefix,
        image_transform=image_transform,
        mask_transform=mask_transform,
        offset=args.offset,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_dataloader = None
    if args.val_ann_path:
        val_dataset = RefSegDataset(
            ann_path=args.val_ann_path,
            ann_format=args.val_ann_format,
            data_root=args.val_data_root or args.data_root,
            img_prefix=args.val_img_prefix,
            mask_prefix=args.val_mask_prefix,
            image_transform=image_transform,
            mask_transform=mask_transform,
            offset=args.val_offset,
            max_samples=args.val_max_samples,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

    factory = load_factory(args.factory)
    factory_kwargs: Dict[str, Any] = {
        "checkpoint_path": args.checkpoint,
        "device": args.device,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "max_epochs": args.max_epochs,
        "save_every": args.save_every,
        "grad_clip_norm": args.grad_clip_norm,
        "use_amp": args.amp,
        "freeze_neck": args.freeze_neck,
        "freeze_encoder": args.freeze_encoder,
        "val_interval": args.val_interval,
        "val_every_steps": args.val_every_steps,
        "val_max_batches": args.val_max_batches,
        "val_pred_threshold": args.val_pred_threshold,
        "val_gt_mode": args.val_gt_mode,
        "val_iou_thresholds": args.val_iou_thresholds,
        "best_metric": args.best_metric,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "log_every": args.log_every,
    }
    if args.config_json:
        factory_kwargs.update(json.loads(args.config_json))
    bundle = factory(dataloader=dataloader, val_dataloader=val_dataloader, work_dir=args.work_dir, **factory_kwargs)
    requested_checkpoint = bundle.get("requested_checkpoint_path", factory_kwargs.get("checkpoint_path", ""))
    resolved_checkpoint = bundle.get("resolved_checkpoint_path", requested_checkpoint)
    if requested_checkpoint:
        print(json.dumps(
            {
                "requested_checkpoint": requested_checkpoint,
                "resolved_checkpoint": resolved_checkpoint,
            },
            ensure_ascii=False,
        ))
    train_fn = bundle.get("train")
    if not callable(train_fn):
        raise ValueError("Factory must return a dict with a callable 'train' entry.")
    train_fn()


if __name__ == "__main__":
    main()
