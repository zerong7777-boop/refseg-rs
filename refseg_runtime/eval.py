from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

if __package__ in (None, ""):
    package_dir = os.path.dirname(__file__)
    if sys.path and os.path.abspath(sys.path[0]) == os.path.abspath(package_dir):
        sys.path.pop(0)
    sys.path.insert(0, os.path.dirname(package_dir))

import numpy as np
import cv2
from PIL import Image

if __package__ in (None, ""):
    from refseg_runtime.annotations import build_samples, load_records
    from refseg_runtime.checkpoint import summarize_checkpoint
    from refseg_runtime.factory import load_factory
    from refseg_runtime.metrics import evaluate_predictions, sigmoid_np
    from refseg_runtime.visualize import save_overlay
else:
    from .annotations import build_samples, load_records
    from .checkpoint import summarize_checkpoint
    from .factory import load_factory
    from .metrics import evaluate_predictions, sigmoid_np
    from .visualize import save_overlay


def _parse_float_list(raw: str) -> List[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _load_prediction_map(path: str, apply_sigmoid: bool) -> np.ndarray:
    if path.endswith(".npy"):
        pred = np.load(path)
    else:
        pred = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if apply_sigmoid:
        pred = sigmoid_np(pred)
    return pred.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure runtime RefSeg evaluator. Use sanitized checkpoints ending with .state_dict.pth."
    )
    default_predictor_factory = "refseg_runtime.backends.native_xemap:create_native_xemap_predictor"
    parser.add_argument("--ann-path", required=True)
    parser.add_argument("--ann-format", default="auto", choices=["txt", "json", "auto"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--img-prefix", default="images")
    parser.add_argument("--mask-prefix", default="masked")
    parser.add_argument("--pred-dir", default="", help="Directory containing predicted masks or npy logits.")
    parser.add_argument("--pred-ext", default=".png", help="Prediction file extension inside pred-dir.")
    parser.add_argument("--pred-use-relpath", action="store_true",
                        help="Use GT-relative path stem under pred-dir instead of flat basename.")
    parser.add_argument("--predictor-factory", default=default_predictor_factory,
                        help="Optional module:function that returns an object with predict(image_path, text)->np.ndarray.")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint path. Use a sanitized .state_dict.pth checkpoint.",
    )
    parser.add_argument("--predictor-config-json", default="",
                        help="Optional json blob forwarded to the predictor factory.")
    parser.add_argument("--pred-threshold", type=float, default=0.5)
    parser.add_argument("--iou-thresholds", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--gt-mode", default="gt_positive", choices=["gt_positive", "gt255", "gt1"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--vis-dir", default="")
    parser.add_argument("--report-json", default="")
    args = parser.parse_args()

    records = load_records(args.ann_path, ann_format=args.ann_format)
    samples, missing_img, missing_gt = build_samples(
        records=records,
        data_root=args.data_root,
        img_prefix=args.img_prefix,
        mask_prefix=args.mask_prefix,
        offset=args.offset,
        max_samples=args.max_samples,
    )

    predictor = None
    resolved_checkpoint = args.checkpoint
    if not args.pred_dir:
        if not args.predictor_factory:
            raise ValueError("Either --pred-dir or --predictor-factory must be provided.")
        factory = load_factory(args.predictor_factory)
        predictor_kwargs = {}
        if args.predictor_config_json:
            predictor_kwargs.update(json.loads(args.predictor_config_json))
        predictor = factory(checkpoint_path=args.checkpoint, **predictor_kwargs)
        resolved_checkpoint = getattr(predictor, "resolved_checkpoint_path", args.checkpoint)

    predictions = []
    gt_maps = []
    for sample in samples:
        gt_map = np.array(Image.open(sample.gt_path).convert("L"), dtype=np.uint8)
        if predictor is not None:
            pred_map = predictor.predict(sample.img_path, sample.text)
            pred_map = np.asarray(pred_map, dtype=np.float32)
        else:
            if args.pred_use_relpath:
                rel = os.path.relpath(sample.gt_path, os.path.join(args.data_root, args.mask_prefix))
                pred_path = os.path.join(args.pred_dir, os.path.splitext(rel)[0] + args.pred_ext)
            else:
                pred_name = os.path.splitext(os.path.basename(sample.gt_path))[0] + args.pred_ext
                pred_path = os.path.join(args.pred_dir, pred_name)
            pred_map = _load_prediction_map(pred_path, apply_sigmoid=False)
        if pred_map.shape != gt_map.shape:
            gt_map = cv2.resize(
                gt_map,
                (pred_map.shape[1], pred_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        predictions.append(pred_map)
        gt_maps.append(gt_map)
        if args.save_vis and args.vis_dir:
            out_name = os.path.splitext(os.path.basename(sample.gt_path))[0] + ".jpg"
            save_overlay(sample.img_path, pred_map, os.path.join(args.vis_dir, out_name), gt_map, args.pred_threshold)

    p_at, miou, oiou = evaluate_predictions(
        predictions=predictions,
        gt_labels=gt_maps,
        iou_thresholds=_parse_float_list(args.iou_thresholds),
        gt_mode=args.gt_mode,
        pred_threshold=args.pred_threshold,
    )

    print("P@thresholds:", p_at)
    print("Mean IoU (mIoU):", miou)
    print("Overall IoU (oIoU):", oiou)
    if args.checkpoint:
        print(json.dumps(
            {
                "requested_checkpoint": args.checkpoint,
                "resolved_checkpoint": resolved_checkpoint,
            },
            ensure_ascii=False,
        ))

    if args.checkpoint:
        try:
            print("Checkpoint summary:", summarize_checkpoint(args.checkpoint))
        except Exception:
            pass

    if args.report_json:
        payload = {
            "checkpoint": {
                "requested": args.checkpoint,
                "resolved": resolved_checkpoint,
            },
            "dataset": {
                "loaded_records": len(records),
                "valid_samples": len(samples),
                "missing_img": missing_img,
                "missing_gt": missing_gt,
            },
            "results": {
                "primary": {
                    "pred_threshold": args.pred_threshold,
                    "p_at": p_at,
                    "miou": miou,
                    "oiou": oiou,
                }
            },
        }
        report_dir = os.path.dirname(args.report_json)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
