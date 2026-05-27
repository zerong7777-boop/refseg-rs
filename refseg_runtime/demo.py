from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from refseg_runtime.factory import load_factory
    from refseg_runtime.metrics import sigmoid_np
    from refseg_runtime.visualize import save_overlay
else:
    from .factory import load_factory
    from .metrics import sigmoid_np
    from .visualize import save_overlay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure runtime single-image RefSeg demo. Prefer sanitized checkpoints ending with .state_dict.pth."
    )
    default_predictor_factory = "refseg_runtime.backends.native_xemap:create_native_xemap_predictor"
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--prediction", default="",
                        help="Optional existing mask or npy logits to visualize.")
    parser.add_argument("--predictor-factory", default=default_predictor_factory,
                        help="Optional module:function that returns an object with predict(image_path, text)->np.ndarray.")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint path. Prefer .state_dict.pth. If a sibling sanitized checkpoint exists, runtime will use it automatically.",
    )
    parser.add_argument("--predictor-config-json", default="",
                        help="Optional json blob forwarded to the predictor factory.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pred-threshold", type=float, default=0.5)
    parser.add_argument("--apply-sigmoid", action="store_true")
    args = parser.parse_args()

    if args.prediction and args.predictor_factory != default_predictor_factory:
        raise ValueError("When --prediction is provided, do not also pass a custom --predictor-factory.")
    use_predictor = not bool(args.prediction)

    if not use_predictor:
        if args.prediction.endswith(".npy"):
            pred = np.load(args.prediction)
        else:
            pred = np.array(Image.open(args.prediction).convert("L"), dtype=np.float32) / 255.0
    else:
        factory = load_factory(args.predictor_factory)
        predictor_kwargs = {}
        if args.predictor_config_json:
            predictor_kwargs.update(json.loads(args.predictor_config_json))
        predictor = factory(checkpoint_path=args.checkpoint, **predictor_kwargs)
        pred = np.asarray(predictor.predict(args.image, args.text), dtype=np.float32)
        resolved_checkpoint = getattr(predictor, "resolved_checkpoint_path", args.checkpoint)
        if args.checkpoint:
            print(json.dumps(
                {
                    "requested_checkpoint": args.checkpoint,
                    "resolved_checkpoint": resolved_checkpoint,
                },
                ensure_ascii=False,
            ))

    if args.apply_sigmoid:
        pred = sigmoid_np(pred)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_overlay(args.image, pred, args.out, pred_threshold=args.pred_threshold)
    print(args.out)


if __name__ == "__main__":
    main()
