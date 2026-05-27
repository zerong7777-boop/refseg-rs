import json
import os
from typing import Dict, List, Optional, Tuple

from .types import RefSegSample


def normalize_text(text: str) -> str:
    text = text.strip()
    if not text.endswith("."):
        text += "."
    return text


def strip_braces(value: str) -> str:
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        return value[1:-1]
    return value


def parse_txt_line(line: str) -> Optional[Dict[str, str]]:
    parts = line.strip().split("\t")
    if len(parts) < 5:
        return None
    return {
        "img_rel": strip_braces(parts[2]),
        "text": strip_braces(parts[3]),
        "gt_rel": strip_braces(parts[4]),
    }


def load_records_from_txt_file(txt_path: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with open(txt_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parsed = parse_txt_line(line)
            if parsed is not None:
                records.append(parsed)
    return records


def load_records_from_txt_folder(folder_path: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    txt_files = sorted(
        [name for name in os.listdir(folder_path) if name.endswith(".txt")],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x,
    )
    for txt_name in txt_files:
        with open(os.path.join(folder_path, txt_name), "r", encoding="utf-8-sig") as f:
            parsed = parse_txt_line(f.readline())
            if parsed is not None:
                records.append(parsed)
    return records


def load_records_from_json(json_path: str) -> List[Dict[str, str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        for key in ("items", "data", "samples", "annotations"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break

    if not isinstance(payload, list):
        raise ValueError(f"Unsupported json structure in {json_path}")

    records: List[Dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        img_rel = (
            item.get("img")
            or item.get("image")
            or item.get("image_path")
            or item.get("img_path")
            or item.get("image_rel")
        )
        gt_rel = (
            item.get("gt")
            or item.get("mask")
            or item.get("gt_path")
            or item.get("mask_path")
            or item.get("label")
            or item.get("label_path")
        )
        text = item.get("text") or item.get("expression") or item.get("caption")
        if img_rel and gt_rel and text:
            records.append(
                {
                    "img_rel": strip_braces(str(img_rel)),
                    "gt_rel": strip_braces(str(gt_rel)),
                    "text": str(text).strip(),
                }
            )
    return records


def load_records(ann_path: str, ann_format: str = "auto") -> List[Dict[str, str]]:
    if ann_format == "auto":
        if os.path.isdir(ann_path):
            ann_format = "txt"
        elif ann_path.endswith(".json"):
            ann_format = "json"
        else:
            ann_format = "txt"

    if ann_format == "json":
        return load_records_from_json(ann_path)

    if os.path.isdir(ann_path):
        return load_records_from_txt_folder(ann_path)
    return load_records_from_txt_file(ann_path)


def resolve_path(rel_or_abs: str, data_root: str, prefix: str) -> str:
    rel_or_abs = rel_or_abs.strip()
    if os.path.isabs(rel_or_abs):
        return rel_or_abs

    candidates = []
    if prefix:
        candidates.append(os.path.join(data_root, prefix, rel_or_abs))
    candidates.append(os.path.join(data_root, rel_or_abs))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def build_samples(
    records: List[Dict[str, str]],
    data_root: str,
    img_prefix: str,
    mask_prefix: str,
    offset: int = 0,
    max_samples: int = 0,
) -> Tuple[List[RefSegSample], int, int]:
    samples: List[RefSegSample] = []
    missing_img = 0
    missing_gt = 0

    for rec in records:
        img_path = resolve_path(rec["img_rel"], data_root, img_prefix)
        gt_path = resolve_path(rec["gt_rel"], data_root, mask_prefix)
        text = normalize_text(rec["text"])

        img_exists = os.path.exists(img_path)
        gt_exists = os.path.exists(gt_path)
        if not img_exists:
            missing_img += 1
        if not gt_exists:
            missing_gt += 1
        if not (img_exists and gt_exists):
            continue

        samples.append(RefSegSample(img_path=img_path, gt_path=gt_path, text=text))

    if offset > 0:
        samples = samples[offset:]
    if max_samples > 0:
        samples = samples[:max_samples]

    return samples, missing_img, missing_gt
