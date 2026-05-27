from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .annotations import build_samples, load_records
from .types import RefSegSample


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = arr.transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr)


class RefSegDataset(Dataset):
    def __init__(
        self,
        ann_path: str,
        data_root: str,
        img_prefix: str = "images",
        mask_prefix: str = "masked",
        ann_format: str = "auto",
        image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        mask_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        offset: int = 0,
        max_samples: int = 0,
    ) -> None:
        records = load_records(ann_path, ann_format=ann_format)
        self.samples, self.missing_img, self.missing_gt = build_samples(
            records=records,
            data_root=data_root,
            img_prefix=img_prefix,
            mask_prefix=mask_prefix,
            offset=offset,
            max_samples=max_samples,
        )
        self.image_transform = image_transform or pil_to_tensor
        self.mask_transform = mask_transform or pil_to_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def _load_item(self, sample: RefSegSample) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(sample.img_path).convert("RGB")
        mask = Image.open(sample.gt_path).convert("L")
        image_value = self.image_transform(image)
        mask_value = self.mask_transform(mask)
        if isinstance(image_value, tuple):
            image_value = image_value[0]
        if isinstance(mask_value, tuple):
            mask_value = mask_value[0]
        return image_value, mask_value

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        image, mask = self._load_item(sample)
        return {
            "image": image,
            "mask": mask,
            "text": sample.text,
            "img_path": sample.img_path,
            "gt_path": sample.gt_path,
        }
