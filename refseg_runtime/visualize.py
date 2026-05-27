import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def save_overlay(
    image_path: str,
    pred_map: np.ndarray,
    out_path: str,
    gt_map: Optional[np.ndarray] = None,
    pred_threshold: float = 0.5,
) -> None:
    img = np.array(Image.open(image_path).convert("RGB"))
    if img.shape[:2] != pred_map.shape:
        img = cv2.resize(img, (pred_map.shape[1], pred_map.shape[0]), interpolation=cv2.INTER_LINEAR)
    pred_bin = (pred_map >= pred_threshold).astype(np.uint8) * 255
    heatmap = cv2.applyColorMap(pred_bin, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.7, heatmap, 0.3, 0)

    panels = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR), heatmap, overlay]
    if gt_map is not None:
        gt_u8 = (gt_map > 0).astype(np.uint8) * 255
        panels.append(cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2BGR))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, np.concatenate(panels, axis=1))
