from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass
class ImageTransformMeta:
    original_size: Tuple[int, int]
    resized_size: Tuple[int, int]


class GroundingImagePreprocessor:
    def __init__(
        self,
        scale: Tuple[int, int] = (800, 800),
        keep_ratio: bool = True,
        mean: Sequence[float] = (123.675, 116.28, 103.53),
        std: Sequence[float] = (58.395, 57.12, 57.375),
    ) -> None:
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1) / 255.0
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1) / 255.0

    def _get_target_size(self, width: int, height: int) -> Tuple[int, int]:
        target_w, target_h = self.scale
        if not self.keep_ratio:
            return target_w, target_h
        scale_factor = min(target_w / max(width, 1), target_h / max(height, 1))
        new_w = max(1, int(round(width * scale_factor)))
        new_h = max(1, int(round(height * scale_factor)))
        return new_w, new_h

    def __call__(self, image: Union[Image.Image, np.ndarray]) -> tuple[torch.Tensor, ImageTransformMeta]:
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            original_size = image.size
            arr = np.asarray(image)
        else:
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected HxWx3 image array, got {image.shape}")
            arr = image
            original_size = (int(arr.shape[1]), int(arr.shape[0]))
        target_size = self._get_target_size(*original_size)
        if target_size != original_size:
            arr = cv2.resize(arr, target_size, interpolation=cv2.INTER_LINEAR)
        arr = arr.astype(np.float32, copy=False).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(arr)
        tensor = (tensor - self.mean) / self.std
        return tensor, ImageTransformMeta(
            original_size=(original_size[1], original_size[0]),
            resized_size=(target_size[1], target_size[0]),
        )


class SegmentationMaskPreprocessor:
    def __init__(
        self,
        scale: Tuple[int, int] = (800, 800),
        keep_ratio: bool = False,
    ) -> None:
        self.scale = scale
        self.keep_ratio = keep_ratio

    def _get_target_size(self, width: int, height: int) -> Tuple[int, int]:
        target_w, target_h = self.scale
        if not self.keep_ratio:
            return target_w, target_h
        scale_factor = min(target_w / max(width, 1), target_h / max(height, 1))
        new_w = max(1, int(round(width * scale_factor)))
        new_h = max(1, int(round(height * scale_factor)))
        return new_w, new_h

    def __call__(self, mask: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(mask, Image.Image):
            mask = mask.convert("L")
            original_size = mask.size
            arr = np.asarray(mask)
        else:
            if mask.ndim not in (2, 3):
                raise ValueError(f"Expected HxW or HxWx1 mask array, got {mask.shape}")
            arr = mask[:, :, 0] if mask.ndim == 3 else mask
            original_size = (int(arr.shape[1]), int(arr.shape[0]))
        target_size = self._get_target_size(*original_size)
        if target_size != original_size:
            arr = cv2.resize(arr, target_size, interpolation=cv2.INTER_NEAREST)
        arr = arr.astype(np.float32, copy=False)
        if arr.max() > 1:
            arr = arr / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
