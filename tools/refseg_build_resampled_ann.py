#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a hard-example-resampled RefSeg annotation txt file.')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--mask-root', required=True)
    parser.add_argument('--out-ann', required=True)
    parser.add_argument('--summary-json', required=True)
    parser.add_argument('--small-threshold', type=float, default=0.01)
    parser.add_argument('--tiny-threshold', type=float, default=0.005)
    parser.add_argument('--source-bonus', default='2021LoveDA=1,aerialImageDataset=1')
    parser.add_argument('--small-bonus', type=int, default=1)
    parser.add_argument('--tiny-bonus', type=int, default=1)
    parser.add_argument('--combo-bonus', type=int, default=1)
    parser.add_argument('--max-copies', type=int, default=4)
    return parser.parse_args()


def parse_bonus_map(text: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if not text.strip():
        return mapping
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        key, value = item.split('=', 1)
        mapping[key.strip()] = int(value.strip())
    return mapping


def parse_line(line: str) -> Tuple[str, List[str]]:
    parts = line.rstrip('\n').split('\t')
    if len(parts) < 5:
        raise ValueError(f'expected >=5 columns, got {len(parts)}: {line!r}')
    cleaned = [part.strip().strip('{}') for part in parts[:5]]
    return line, cleaned


def load_mask_ratio(mask_path: Path) -> float:
    arr = np.array(Image.open(mask_path).convert('L'))
    return float((arr > 0).sum() / arr.size) if arr.size else 0.0


def main() -> None:
    args = parse_args()
    start = time.time()
    ann_path = Path(args.ann_file)
    mask_root = Path(args.mask_root)
    out_ann = Path(args.out_ann)
    summary_json = Path(args.summary_json)
    out_ann.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    source_bonus = parse_bonus_map(args.source_bonus)

    input_rows = 0
    output_rows = 0
    missing_masks = 0
    source_counter = Counter()
    weighted_source_counter = Counter()
    area_counter = Counter()
    weighted_area_counter = Counter()
    copy_counter = Counter()

    with ann_path.open('r', encoding='utf-8') as src, out_ann.open('w', encoding='utf-8') as dst:
        for line in src:
            if not line.strip():
                continue
            raw_line, parts = parse_line(line)
            _, _, img_rel, _, mask_rel = parts
            source = Path(img_rel).parts[0] if Path(img_rel).parts else 'unknown'
            mask_path = mask_root / mask_rel
            if not mask_path.exists():
                missing_masks += 1
                continue
            area_ratio = load_mask_ratio(mask_path)
            area_bucket = 'large'
            if area_ratio < args.small_threshold:
                area_bucket = 'small'
            elif area_ratio < 0.05:
                area_bucket = 'medium'

            copies = 1
            copies += source_bonus.get(source, 0)
            if area_ratio < args.small_threshold:
                copies += args.small_bonus
            if area_ratio < args.tiny_threshold:
                copies += args.tiny_bonus
            if source in source_bonus and area_ratio < args.small_threshold:
                copies += args.combo_bonus
            copies = max(1, min(copies, args.max_copies))

            input_rows += 1
            output_rows += copies
            source_counter[source] += 1
            weighted_source_counter[source] += copies
            area_counter[area_bucket] += 1
            weighted_area_counter[area_bucket] += copies
            copy_counter[copies] += 1
            dst.write(raw_line)
            for _ in range(copies - 1):
                dst.write(raw_line)

    payload = {
        'meta': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'elapsed_sec': round(time.time() - start, 4),
            'ann_file': str(ann_path.resolve()),
            'mask_root': str(mask_root.resolve()),
            'out_ann': str(out_ann.resolve()),
            'small_threshold': args.small_threshold,
            'tiny_threshold': args.tiny_threshold,
            'source_bonus': source_bonus,
            'small_bonus': args.small_bonus,
            'tiny_bonus': args.tiny_bonus,
            'combo_bonus': args.combo_bonus,
            'max_copies': args.max_copies,
        },
        'counts': {
            'input_rows': input_rows,
            'output_rows': output_rows,
            'missing_masks': missing_masks,
            'expansion_ratio': round(output_rows / input_rows, 6) if input_rows else 0.0,
        },
        'source_counts': dict(source_counter),
        'weighted_source_counts': dict(weighted_source_counter),
        'area_counts': dict(area_counter),
        'weighted_area_counts': dict(weighted_area_counter),
        'copy_histogram': {str(k): v for k, v in sorted(copy_counter.items())},
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[INFO] out_ann={out_ann}')
    print(f'[INFO] summary_json={summary_json}')
    print(f'[INFO] input_rows={input_rows} output_rows={output_rows} expansion_ratio={payload["counts"]["expansion_ratio"]}')


if __name__ == '__main__':
    main()
