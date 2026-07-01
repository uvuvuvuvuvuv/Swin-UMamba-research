
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import argparse
from typing import Any, Dict, List

try:
    import pandas as pd
except Exception as e:
    raise SystemExit("This script requires pandas. Please install pandas in the current env.") from e


def flatten_scalars(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                nested = flatten_scalars(v, key)
                out.update(nested)
            else:
                out[key] = v
    elif isinstance(obj, list):
        # Keep only short scalar lists as joined text; ignore nested structures
        if all(not isinstance(x, (dict, list)) for x in obj):
            out[prefix] = ",".join(map(str, obj))
    return out


def safe_load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_meta(path: str) -> Dict[str, str]:
    parts = path.replace("\\", "/").split("/")
    # .../work_dir/<mode>/<dataset>/<fold>/<eval_dir>/<file>
    meta = {"mode": "", "dataset": "", "fold": "", "task_type": "", "summary_file": os.path.basename(path)}
    try:
        i = parts.index("work_dir")
        meta["mode"] = parts[i + 1]
        meta["dataset"] = parts[i + 2]
        meta["fold"] = parts[i + 3]
    except Exception:
        pass
    if "/eval_2d/" in path.replace("\\", "/"):
        meta["task_type"] = "2D"
    elif "/eval_3d/" in path.replace("\\", "/"):
        meta["task_type"] = "3D"
    else:
        meta["task_type"] = "unknown"
    return meta


def pick_metric(flat: Dict[str, Any], candidates: List[str]):
    for k in candidates:
        if k in flat:
            return flat[k]
    return None


def build_row(path: str) -> Dict[str, Any]:
    obj = safe_load_json(path)
    flat = flatten_scalars(obj)
    meta = infer_meta(path)

    row: Dict[str, Any] = {
        "mode": meta["mode"],
        "dataset": meta["dataset"],
        "fold": meta["fold"],
        "task_type": meta["task_type"],
        "summary_path": path,
    }

    # Common preferred fields
    row["mean_dice"] = pick_metric(flat, [
        "mean_dice", "dice_mean", "macro_dice", "overall.mean_dice", "summary.mean_dice",
        "summary.macro_dice", "results.mean_dice", "results.macro_dice", "DSC", "mean_DSC"
    ])
    row["std_dice"] = pick_metric(flat, [
        "std_dice", "dice_std", "summary.std_dice", "results.std_dice"
    ])
    row["mean_iou"] = pick_metric(flat, [
        "mean_iou", "iou_mean", "summary.mean_iou", "results.mean_iou"
    ])
    row["std_iou"] = pick_metric(flat, [
        "std_iou", "iou_std", "summary.std_iou", "results.std_iou"
    ])
    row["mean_mae"] = pick_metric(flat, [
        "mean_mae", "mae_mean", "summary.mean_mae", "results.mean_mae"
    ])
    row["std_mae"] = pick_metric(flat, [
        "std_mae", "mae_std", "summary.std_mae", "results.std_mae"
    ])
    row["mean_hd95_mm"] = pick_metric(flat, [
        "mean_hd95", "mean_hd95_mm", "macro_hd95", "HD95(mm)", "HD95", "summary.mean_hd95", "summary.macro_hd95"
    ])
    row["std_hd95_mm"] = pick_metric(flat, [
        "std_hd95", "std_hd95_mm", "summary.std_hd95"
    ])
    row["mean_assd_mm"] = pick_metric(flat, [
        "mean_assd", "mean_assd_mm", "macro_assd", "ASSD(mm)", "ASSD", "summary.mean_assd", "summary.macro_assd"
    ])
    row["std_assd_mm"] = pick_metric(flat, [
        "std_assd", "std_assd_mm", "summary.std_assd"
    ])

    # Also keep all flattened scalars in case the schema differs
    for k, v in flat.items():
        if k not in row:
            row[k] = v
    return row


def main():
    parser = argparse.ArgumentParser(description="Merge all eval summaries into one table.")
    parser.add_argument(
        "--work_dir",
        type=str,
        default="/storage/baiyuting/data/Swin-UMamba-main/work_dir",
        help="Root work_dir containing baseline/upper/<dataset>/<fold>/eval_2d or eval_3d"
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="all_eval_results_table.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--out_xlsx",
        type=str,
        default="all_eval_results_table.xlsx",
        help="Output XLSX path"
    )
    args = parser.parse_args()

    work_dir = args.work_dir
    files = sorted(glob.glob(os.path.join(work_dir, "**", "eval_2d", "eval_summary.json"), recursive=True))
    files += sorted(glob.glob(os.path.join(work_dir, "**", "eval_3d", "*summary*.json"), recursive=True))
    files = sorted(set(files))

    if not files:
        raise SystemExit(f"No eval summary files found under: {work_dir}")

    rows = [build_row(p) for p in files]
    df = pd.DataFrame(rows)

    preferred_cols = [
        "mode", "dataset", "fold", "task_type",
        "mean_dice", "std_dice",
        "mean_iou", "std_iou",
        "mean_mae", "std_mae",
        "mean_hd95_mm", "std_hd95_mm",
        "mean_assd_mm", "std_assd_mm",
        "summary_path",
    ]
    other_cols = [c for c in df.columns if c not in preferred_cols]
    df = df[preferred_cols + other_cols]

    df = df.sort_values(by=["task_type", "dataset", "mode", "fold"], na_position="last").reset_index(drop=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(args.out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="all_results")
        # Also provide a compact sheet with only common columns
        compact_cols = [c for c in preferred_cols if c in df.columns]
        df[compact_cols].to_excel(writer, index=False, sheet_name="compact")

    print(f"[OK] found {len(files)} summary files")
    print(f"[OK] csv  -> {args.out_csv}")
    print(f"[OK] xlsx -> {args.out_xlsx}")


if __name__ == "__main__":
    main()
