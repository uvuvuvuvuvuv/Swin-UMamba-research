#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from stage_timer_utils import StageTimer


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_2d_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    else:
        import imageio.v2 as imageio
        arr = imageio.imread(path)
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array at {path}, got shape={arr.shape}")
    return arr


def try_get(d: Dict[str, Any], keys: Sequence[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def infer_native_hw(rec: Dict[str, Any], geom: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    if "orig_h" in rec and "orig_w" in rec:
        return int(rec["orig_h"]), int(rec["orig_w"])
    if geom is not None:
        hw = try_get(geom, ["native_hw"], None)
        if hw is not None and len(hw) == 2:
            return int(hw[0]), int(hw[1])
        inv = try_get(geom, ["student_to_native"], None)
        if isinstance(inv, dict) and "to_h" in inv and "to_w" in inv:
            return int(inv["to_h"]), int(inv["to_w"])
    raise KeyError("Cannot infer native_hw from manifest / geometry_meta")


def restore_with_inverse_geometry(mask_2d: np.ndarray, inv_geom: Dict[str, Any]) -> np.ndarray:
    crop_h = int(inv_geom["crop_h"])
    crop_w = int(inv_geom["crop_w"])
    offset_x = int(inv_geom["offset_x"])
    offset_y = int(inv_geom["offset_y"])
    to_h = int(inv_geom["to_h"])
    to_w = int(inv_geom["to_w"])
    cropped = mask_2d[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
    if cropped.size == 0:
        raise ValueError("Empty crop while restoring by student_to_native geometry")
    restored = cv2.resize(cropped.astype(np.uint8), (to_w, to_h), interpolation=cv2.INTER_NEAREST)
    return restored.astype(mask_2d.dtype)


def restore_pred_to_native(pred_2d: np.ndarray, rec: Dict[str, Any], geom: Optional[Dict[str, Any]], allow_resize_debug: bool) -> np.ndarray:
    native_h, native_w = infer_native_hw(rec, geom)
    if pred_2d.shape == (native_h, native_w):
        return pred_2d
    if geom is not None:
        inv = try_get(geom, ["student_to_native"], None)
        if isinstance(inv, dict):
            try:
                restored = restore_with_inverse_geometry(pred_2d, inv)
                if restored.shape == (native_h, native_w):
                    return restored
            except Exception:
                pass
    if allow_resize_debug:
        return cv2.resize(pred_2d.astype(np.uint8), (native_w, native_h), interpolation=cv2.INTER_NEAREST).astype(pred_2d.dtype)
    raise ValueError(
        f"Prediction shape {pred_2d.shape} cannot be restored to native {(native_h, native_w)}. "
        f"Enable --allow_resize_pred_to_native_for_debug only for debugging."
    )


def dice_binary(pred: np.ndarray, gt: np.ndarray, valid_mask: Optional[np.ndarray] = None, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if valid_mask is not None:
        pred = pred & valid_mask
        gt = gt & valid_mask
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return np.nan
    return float((2.0 * inter + eps) / (denom + eps))


def iou_binary(pred: np.ndarray, gt: np.ndarray, valid_mask: Optional[np.ndarray] = None, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if valid_mask is not None:
        pred = pred & valid_mask
        gt = gt & valid_mask
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return np.nan
    return float((inter + eps) / (union + eps))


def mae_binary(pred: np.ndarray, gt: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> float:
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    if valid_mask is None:
        return float(np.mean(np.abs(pred - gt)))
    if valid_mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(pred[valid_mask] - gt[valid_mask])))


def infer_eval_labels(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> List[int]:
    if num_classes and num_classes > 1:
        return list(range(1, num_classes))
    vals = sorted(set(np.unique(gt).tolist()) | set(np.unique(pred).tolist()))
    vals = [int(v) for v in vals if int(v) not in (0, 255)]
    if not vals:
        return [1]
    return vals


def format_float(x: float) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0)), "n": int(arr.size)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol-aligned 2D evaluation in native space.")
    parser.add_argument("--fold_root", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num_classes", type=int, default=0, help="0=auto infer from masks; >1 means labels 1..num_classes-1")
    parser.add_argument("--report_mae", action="store_true", help="Enable MAE reporting (recommended for polyp tasks)")
    parser.add_argument("--require_native_gt", action="store_true", help="Formal protocol mode. Require native_gt in manifest.")
    parser.add_argument("--allow_resize_pred_to_native_for_debug", action="store_true",
                        help="Debug only. If geometry restore is unavailable, nearest-resize pred to native size.")
    args = parser.parse_args()

    fold_name = Path(args.fold_root).name
    dataset_name = Path(args.fold_root).parent.name
    run_root = str(Path(args.pred_dir).parent)

    mode = ""
    parts = Path(args.pred_dir).parts
    if "baseline" in parts:
        mode = "baseline"
    elif "upper" in parts:
        mode = "upper"

    stage_name = f"eval_{mode}" if mode else "eval_2d"
    stage_time_path = os.path.join(run_root, f"stage_time_{stage_name}.json")

    ensure_dir(args.save_dir)
    with StageTimer(
            save_path=stage_time_path,
            stage_name=stage_name,
            dataset=dataset_name,
            fold=fold_name,
            mode=mode,
            split=args.split,
    ) as timer:
        manifest_path = os.path.join(args.fold_root, "meta", "manifest.json")
        geometry_path = os.path.join(args.fold_root, "meta", "geometry_meta.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        manifest = load_json(manifest_path)
        geometry_meta = load_json(geometry_path) if os.path.exists(geometry_path) else {}

        eval_records = [x for x in manifest if x.get("split") == args.split]
        if not eval_records:
            raise RuntimeError(f"No records found for split={args.split} in {manifest_path}")

        per_case_rows: List[Dict[str, Any]] = []
        all_case_dice: List[float] = []
        all_case_iou: List[float] = []
        all_case_mae: List[float] = []
        per_class_dice: Dict[int, List[float]] = {}
        per_class_iou: Dict[int, List[float]] = {}

        for rec in eval_records:
            slice_name = rec["slice_name"]
            pred_path = os.path.join(args.pred_dir, slice_name)
            if not os.path.exists(pred_path):
                raise FileNotFoundError(f"Prediction not found: {pred_path}")

            gt_rel = rec.get("native_gt", None)
            if gt_rel is None:
                if args.require_native_gt:
                    raise FileNotFoundError(f"native_gt missing in manifest for {slice_name}")
                gt_rel = rec.get("student_gt", None)
                if gt_rel is None:
                    raise FileNotFoundError(f"Neither native_gt nor student_gt found in manifest for {slice_name}")

            gt_path = os.path.join(args.fold_root, gt_rel)
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"GT path not found: {gt_path}")

            pred = load_2d_array(pred_path).astype(np.uint8)
            gt = load_2d_array(gt_path).astype(np.uint8)
            geom = geometry_meta.get(rec.get("geometry_key", slice_name), None)

            if gt_rel == rec.get("student_gt"):
                pred_native = pred
            else:
                pred_native = restore_pred_to_native(pred, rec, geom, args.allow_resize_pred_to_native_for_debug)

            if pred_native.shape != gt.shape:
                raise ValueError(f"Native-space mismatch for {slice_name}: pred={pred_native.shape}, gt={gt.shape}")

            valid = gt != 255
            labels = infer_eval_labels(gt[valid], pred_native[valid], args.num_classes)
            row: Dict[str, Any] = {"slice_name": slice_name, "case_id": rec.get("case_id", os.path.splitext(slice_name)[0])}

            dice_vals = []
            iou_vals = []
            for c in labels:
                pred_c = pred_native == c
                gt_c = gt == c
                d = dice_binary(pred_c, gt_c, valid)
                j = iou_binary(pred_c, gt_c, valid)
                row[f"dice_class_{c}"] = format_float(d)
                row[f"iou_class_{c}"] = format_float(j)
                if not np.isnan(d):
                    dice_vals.append(float(d))
                    per_class_dice.setdefault(int(c), []).append(float(d))
                if not np.isnan(j):
                    iou_vals.append(float(j))
                    per_class_iou.setdefault(int(c), []).append(float(j))

            row["dice_macro"] = format_float(float(np.mean(dice_vals)) if dice_vals else np.nan)
            row["iou_macro"] = format_float(float(np.mean(iou_vals)) if iou_vals else np.nan)
            if row["dice_macro"] is not None:
                all_case_dice.append(float(row["dice_macro"]))
            if row["iou_macro"] is not None:
                all_case_iou.append(float(row["iou_macro"]))

            if args.report_mae:
                mae = mae_binary((pred_native > 0).astype(np.uint8), (gt > 0).astype(np.uint8), valid)
                row["mae_fg"] = format_float(mae)
                if row["mae_fg"] is not None:
                    all_case_mae.append(float(row["mae_fg"]))

            per_case_rows.append(row)
            timer.set_outputs(len(per_case_rows))

        csv_path = os.path.join(args.save_dir, "eval_per_case.csv")
        fieldnames = sorted({k for row in per_case_rows for k in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_case_rows)

        summary = {
            "fold_root": args.fold_root,
            "pred_dir": args.pred_dir,
            "split": args.split,
            "num_eval_slices": len(per_case_rows),
            "dice_macro": summarize(all_case_dice),
            "iou_macro": summarize(all_case_iou),
            "mae_fg": summarize(all_case_mae) if args.report_mae else None,
            "per_class": {},
            "protocol_notes": {
                "evaluation_space": "native",
                "2d_primary_metrics": ["Dice/DSC", "IoU", "MAE(optional_for_polyp)"],
                "geometry_source": "geometry_meta.json",
                "debug_resize_enabled": bool(args.allow_resize_pred_to_native_for_debug),
            },
        }
        for c in sorted(set(per_class_dice.keys()) | set(per_class_iou.keys())):
            summary["per_class"][str(c)] = {
                "dice": summarize(per_class_dice.get(c, [])),
                "iou": summarize(per_class_iou.get(c, [])),
            }

        save_json(summary, os.path.join(args.save_dir, "eval_summary.json"))
        save_json(per_case_rows, os.path.join(args.save_dir, "eval_per_case.json"))
        timer.set_outputs(len(per_case_rows))
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
