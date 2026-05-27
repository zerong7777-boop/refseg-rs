from dataclasses import dataclass


@dataclass(frozen=True)
class RefSegSample:
    img_path: str
    gt_path: str
    text: str
