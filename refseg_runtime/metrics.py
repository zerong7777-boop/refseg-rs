from typing import Dict, Iterable, List, Tuple

import numpy as np


def binarize_gt(gt_labels: np.ndarray, gt_mode: str) -> np.ndarray:
    if gt_mode == "gt255":
        return (gt_labels == 255).astype(np.uint8)
    if gt_mode == "gt1":
        return (gt_labels == 1).astype(np.uint8)
    return (gt_labels > 0).astype(np.uint8)


def compute_iou(pred_seg: np.ndarray, gt_seg: np.ndarray) -> Tuple[float, float]:
    inter = np.logical_and(pred_seg, gt_seg).sum()
    union = np.logical_or(pred_seg, gt_seg).sum()
    return float(inter), float(union)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def evaluate_predictions(
    predictions: Iterable[np.ndarray],
    gt_labels: Iterable[np.ndarray],
    iou_thresholds: List[float],
    gt_mode: str,
    pred_threshold: float,
) -> Tuple[Dict[str, float], float, float]:
    p_at_total = {f"P@{t}": [] for t in iou_thresholds}
    iou_scores: List[float] = []
    overall_i = 0.0
    overall_u = 0.0

    for pred_map, gt_map in zip(predictions, gt_labels):
        gt_bin = binarize_gt(gt_map, gt_mode)
        pred_bin = (pred_map >= pred_threshold).astype(np.uint8)

        inter, union = compute_iou(pred_bin, gt_bin)
        overall_i += inter
        overall_u += union
        iou = 0.0 if union == 0 else inter / union
        iou_scores.append(iou)
        for t in iou_thresholds:
            p_at_total[f"P@{t}"].append(iou >= t)

    p_at = {
        key: float(np.mean(values) * 100.0) if values else 0.0
        for key, values in p_at_total.items()
    }
    miou = float(np.mean(iou_scores) * 100.0) if iou_scores else 0.0
    oiou = float((overall_i / overall_u) * 100.0) if overall_u > 0 else 0.0
    return p_at, miou, oiou
