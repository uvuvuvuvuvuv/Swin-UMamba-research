import os
import json
import argparse
import numpy as np
import cv2
from collections import Counter


def stem(x: str) -> str:
    return os.path.splitext(os.path.basename(x))[0]


def load_ids(split_json: str):
    with open(split_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        ids = list(data.keys())
    elif isinstance(data, list):
        ids = data
    else:
        raise ValueError(f"Unsupported json format: {type(data)}")
    return [stem(x) for x in ids]


def find_by_stem(folder: str, s: str, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    for ext in exts:
        p = os.path.join(folder, s + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find file for id='{s}' in {folder}")


def read_mask(mask_dir: str, img_id: str):
    p = find_by_stem(mask_dir, img_id, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {p}")
    return m, p


def convert_mask(m: np.ndarray, mode: str):
    """
    mode:
      - ours: pseudo 0/1/255 原样统计
      - baseline: pseudo 0/1/255 但把 255 视为 0
      - upper: gt 0/255 或 0/1 -> 强制二值 0/1
    """
    if mode == "baseline":
        m2 = m.copy()
        m2[m2 == 255] = 0
        return m2
    if mode == "upper":
        return (m > 0).astype(np.uint8)  # 0/255 or 0/1 -> 0/1
    return m  # ours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--mask_dir", type=str, required=True)
    ap.add_argument("--mode", type=str, default="ours", choices=["ours", "baseline", "upper"])
    ap.add_argument("--max_items", type=int, default=0, help="0=all, else only first N")
    args = ap.parse_args()

    ids = load_ids(args.split_json)
    if args.max_items > 0:
        ids = ids[:args.max_items]

    total_pixels = 0
    value_counter = Counter()
    file_missing = 0
    unique_sets = Counter()

    for img_id in ids:
        try:
            m, _ = read_mask(args.mask_dir, img_id)
        except FileNotFoundError:
            file_missing += 1
            continue

        m = convert_mask(m, args.mode)

        u, c = np.unique(m, return_counts=True)
        unique_sets[tuple(u.tolist())] += 1

        for vv, cc in zip(u.tolist(), c.tolist()):
            value_counter[int(vv)] += int(cc)
        total_pixels += int(m.size)

    print("\n==============================")
    print(f"Mode        : {args.mode}")
    print(f"Split json  : {args.split_json}")
    print(f"Mask dir    : {args.mask_dir}")
    print(f"Num IDs     : {len(ids)}")
    print(f"Missing     : {file_missing}")
    print(f"Total pixels: {total_pixels}")
    print("------------------------------")

    if total_pixels == 0:
        print("No masks read. Check paths.")
        return

    # 打印 value 占比
    for k in sorted(value_counter.keys()):
        ratio = value_counter[k] / total_pixels
        print(f"Value {k:>3}: {value_counter[k]:>12}  ({ratio*100:6.2f}%)")

    print("------------------------------")
    print("Unique value sets (top 10):")
    for uset, cnt in unique_sets.most_common(10):
        print(f"  {uset} : {cnt}")

    print("==============================\n")


if __name__ == "__main__":
    main()
