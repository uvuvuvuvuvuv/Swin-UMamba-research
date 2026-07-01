#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

from stage_timer_utils import StageTimer

DATASET_LABELS = {
    "btcv": {
        1: "spleen",
        2: "right_kidney",
        3: "left_kidney",
        4: "gallbladder",
        5: "esophagus",
        6: "liver",
        7: "stomach",
        8: "aorta",
        9: "ivc",
        10: "portal_vein_splenic_vein",
        11: "pancreas",
        12: "right_adrenal_gland",
        13: "left_adrenal_gland",
    },
    "acdc": {
        1: "rv",
        2: "myo",
        3: "lv",
    },

    "synapse": {
        1: "aorta",
        2: "gallbladder",
        3: "left_kidney",
        4: "right_kidney",
        5: "liver",
        6: "pancreas",
        7: "spleen",
        8: "stomach",
    },

     "prostate158": {
        1: "central_gland",
        2: "peripheral_zone",
    },
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def try_get(d: Dict[str, Any], keys: Sequence[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


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
    return arr.astype(np.uint8)


def extract_numeric_suffix(name: str) -> Optional[int]:
    stem = os.path.splitext(os.path.basename(name))[0]
    nums = re.findall(r"(\d+)", stem)
    return int(nums[-1]) if nums else None


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
        raise ValueError("Empty crop while restoring pred by student_to_native geometry")
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


def normalize_spacing(rec: Dict[str, Any], geom: Optional[Dict[str, Any]], default=(1.0, 1.0, 1.0)) -> Tuple[float, float, float]:
    sp = try_get(rec, ["spacing", "spacing_xyz", "spacing_zyx"], None)
    if sp is None and geom is not None:
        sp = try_get(geom, ["spacing", "spacing_xyz", "spacing_zyx"], None)
    if sp is None:
        return tuple(map(float, default))
    if isinstance(sp, dict):
        if all(k in sp for k in ["z", "y", "x"]):
            return (float(sp["z"]), float(sp["y"]), float(sp["x"]))
        sp = try_get(sp, ["spacing_zyx", "spacing_xyz", "spacing"], default)
    if isinstance(sp, (list, tuple)) and len(sp) == 3:
        vals = [float(v) for v in sp]
        if "spacing_zyx" in rec or (geom is not None and "spacing_zyx" in geom):
            return (vals[0], vals[1], vals[2])
        return (vals[2], vals[1], vals[0])
    return tuple(map(float, default))


def dice_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    ps = int(pred.sum())
    gs = int(gt.sum())
    if ps == 0 and gs == 0:
        return np.nan
    inter = int(np.logical_and(pred, gt).sum())
    return float((2.0 * inter + eps) / (ps + gs + eps))


def _surface(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    footprint = generate_binary_structure(mask.ndim, connectivity)
    eroded = binary_erosion(mask, structure=footprint, border_value=0)
    return np.logical_xor(mask, eroded)


def _surface_distances(result: np.ndarray, reference: np.ndarray, spacing_zyx: Tuple[float, float, float], connectivity: int = 1) -> np.ndarray:
    result = result.astype(bool)
    reference = reference.astype(bool)
    result_surface = _surface(result, connectivity)
    ref_surface = _surface(reference, connectivity)
    if result_surface.sum() == 0 or ref_surface.sum() == 0:
        return np.asarray([], dtype=np.float64)
    dt = distance_transform_edt(~ref_surface, sampling=spacing_zyx)
    return dt[result_surface]


def hd95_binary(pred: np.ndarray, gt: np.ndarray, spacing_zyx: Tuple[float, float, float], connectivity: int = 1, empty_policy: str = "nan") -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return np.nan
    if pred.sum() == 0 or gt.sum() == 0:
        if empty_policy == "diag":
            z, y, x = spacing_zyx
            dz = pred.shape[0] * z
            dy = pred.shape[1] * y
            dx = pred.shape[2] * x
            return float(np.sqrt(dz * dz + dy * dy + dx * dx))
        return np.nan
    d1 = _surface_distances(pred, gt, spacing_zyx, connectivity)
    d2 = _surface_distances(gt, pred, spacing_zyx, connectivity)
    if d1.size == 0 or d2.size == 0:
        return np.nan
    return float(np.percentile(np.hstack([d1, d2]), 95))


def assd_binary(pred: np.ndarray, gt: np.ndarray, spacing_zyx: Tuple[float, float, float], connectivity: int = 1, empty_policy: str = "nan") -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return np.nan
    if pred.sum() == 0 or gt.sum() == 0:
        if empty_policy == "diag":
            z, y, x = spacing_zyx
            dz = pred.shape[0] * z
            dy = pred.shape[1] * y
            dx = pred.shape[2] * x
            return float(np.sqrt(dz * dz + dy * dy + dx * dx))
        return np.nan
    d1 = _surface_distances(pred, gt, spacing_zyx, connectivity)
    d2 = _surface_distances(gt, pred, spacing_zyx, connectivity)
    if d1.size == 0 or d2.size == 0:
        return np.nan
    return float((d1.mean() + d2.mean()) / 2.0)


def format_float(x: float) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0)), "n": int(arr.size)}


def get_slice_idx(rec: Dict[str, Any], order_fallback: int) -> int:
    val = try_get(rec, ["slice_idx", "slice_index", "z_index", "index"], None)
    if val is not None:
        return int(val)
    fidx = extract_numeric_suffix(rec["slice_name"])
    if fidx is not None:
        return int(fidx)
    return int(order_fallback)


def infer_class_ids(records: List[Dict[str, Any]], num_classes: int, fold_root: str) -> List[int]:
    if num_classes and num_classes > 1:
        return list(range(1, num_classes))
    label_meta_path = os.path.join(fold_root, "meta", "label_meta.json")
    if os.path.exists(label_meta_path):
        meta = load_json(label_meta_path)
        labels = [int(x) for x in meta.get("unique_labels", []) if int(x) not in (0, 255)]
        if labels:
            return sorted(labels)
    vals = set()
    for rec in records[: min(16, len(records))]:
        gt_rel = rec.get("native_gt") or rec.get("student_gt")
        if gt_rel is None:
            continue
        gt = load_2d_array(os.path.join(fold_root, gt_rel))
        vals |= {int(x) for x in np.unique(gt).tolist() if int(x) not in (0, 255)}
    return sorted(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol-aligned 3D reconstruction and evaluation in native physical space.")
    parser.add_argument("--fold_root", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num_classes", type=int, default=0, help="Total classes including background. 0 means infer from label_meta / GTs.")
    parser.add_argument("--connectivity", type=int, default=1)
    parser.add_argument("--empty_distance_policy", type=str, default="nan", choices=["nan", "diag"])
    parser.add_argument("--require_native_gt", action="store_true")
    parser.add_argument("--allow_resize_pred_to_native_for_debug", action="store_true")
    parser.add_argument("--save_case_volumes", action="store_true")
    args = parser.parse_args()

    fold_name = Path(args.fold_root).name
    run_root = str(Path(args.pred_dir).parent)

    mode = ""
    parts = Path(args.pred_dir).parts
    if "baseline" in parts:
        mode = "baseline"
    elif "upper" in parts:
        mode = "upper"

    stage_name = f"eval_{mode}" if mode else "eval_3d"
    stage_time_path = os.path.join(run_root, f"stage_time_{stage_name}.json")

    ensure_dir(args.save_dir)
    with StageTimer(
        save_path=stage_time_path,
        stage_name=stage_name,
        dataset=Path(args.fold_root).parent.name,
        fold=fold_name,
        mode=mode,
        split=args.split,
    ) as timer:
        manifest_path = os.path.join(args.fold_root, "meta", "manifest.json")
        geometry_path = os.path.join(args.fold_root, "meta", "geometry_meta.json")
        split_meta_path = os.path.join(args.fold_root, "meta", "split_meta.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        manifest = load_json(manifest_path)
        geometry_meta = load_json(geometry_path) if os.path.exists(geometry_path) else {}
        split_meta = load_json(split_meta_path) if os.path.exists(split_meta_path) else {}
        dataset_name = str(split_meta.get("dataset_name") or split_meta.get("dataset") or Path(args.fold_root).parent.name)
        label_names = DATASET_LABELS.get(dataset_name, {})

        records = [x for x in manifest if x.get("split") == args.split]
        if not records:
            raise RuntimeError(f"No records found for split={args.split} in {manifest_path}")

        cases: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for i, rec in enumerate(records):
            rec = dict(rec)
            rec["_slice_idx"] = get_slice_idx(rec, i)
            cases[str(rec["case_id"])] .append(rec)

        class_ids = infer_class_ids(records, args.num_classes, args.fold_root)
        per_case_class_rows: List[Dict[str, Any]] = []
        per_case_macro_rows: List[Dict[str, Any]] = []
        per_class_dice: Dict[int, List[float]] = {c: [] for c in class_ids}
        per_class_hd95: Dict[int, List[float]] = {c: [] for c in class_ids}
        per_class_assd: Dict[int, List[float]] = {c: [] for c in class_ids}

        if args.save_case_volumes:
            ensure_dir(os.path.join(args.save_dir, "case_volumes"))

        for case_idx, (case_id, recs) in enumerate(sorted(cases.items()), 1):
            recs = sorted(recs, key=lambda x: x["_slice_idx"])
            pred_slices: List[np.ndarray] = []
            gt_slices: List[np.ndarray] = []
            spacing_zyx = None

            for rec in recs:
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
                    raise FileNotFoundError(f"GT not found: {gt_path}")

                pred = load_2d_array(pred_path)
                gt = load_2d_array(gt_path)
                geom = geometry_meta.get(rec.get("geometry_key", slice_name), None)
                if gt_rel == rec.get("student_gt"):
                    pred_native = pred
                else:
                    pred_native = restore_pred_to_native(pred, rec, geom, args.allow_resize_pred_to_native_for_debug)

                if pred_native.shape != gt.shape:
                    raise ValueError(f"Native mismatch for {slice_name}: pred={pred_native.shape}, gt={gt.shape}")

                pred_slices.append(pred_native.astype(np.uint8))
                gt_slices.append(gt.astype(np.uint8))
                if spacing_zyx is None:
                    spacing_zyx = normalize_spacing(rec, geom)

            pred_vol = np.stack(pred_slices, axis=0)
            gt_vol = np.stack(gt_slices, axis=0)
            spacing_zyx = tuple(map(float, spacing_zyx if spacing_zyx is not None else (1.0, 1.0, 1.0)))

            case_dice_vals: List[float] = []
            case_hd95_vals: List[float] = []
            case_assd_vals: List[float] = []
            print(f"[{case_idx}/{len(cases)}] evaluating {case_id} with spacing_zyx={spacing_zyx}", flush=True)

            for c in class_ids:
                pred_c = pred_vol == c
                gt_c = gt_vol == c
                d = dice_binary(pred_c, gt_c)
                h = hd95_binary(pred_c, gt_c, spacing_zyx, connectivity=args.connectivity, empty_policy=args.empty_distance_policy)
                a = assd_binary(pred_c, gt_c, spacing_zyx, connectivity=args.connectivity, empty_policy=args.empty_distance_policy)
                if not np.isnan(d):
                    case_dice_vals.append(float(d))
                    per_class_dice[c].append(float(d))
                if not np.isnan(h):
                    case_hd95_vals.append(float(h))
                    per_class_hd95[c].append(float(h))
                if not np.isnan(a):
                    case_assd_vals.append(float(a))
                    per_class_assd[c].append(float(a))

                per_case_class_rows.append({
                    "case_id": case_id,
                    "class_id": c,
                    "label_name": label_names.get(c, f"class_{c}"),
                    "num_slices": int(pred_vol.shape[0]),
                    "spacing_z": float(spacing_zyx[0]),
                    "spacing_y": float(spacing_zyx[1]),
                    "spacing_x": float(spacing_zyx[2]),
                    "dice": format_float(d),
                    "hd95_mm": format_float(h),
                    "assd_mm": format_float(a),
                })

            per_case_macro_rows.append({
                "case_id": case_id,
                "num_slices": int(pred_vol.shape[0]),
                "spacing_z": float(spacing_zyx[0]),
                "spacing_y": float(spacing_zyx[1]),
                "spacing_x": float(spacing_zyx[2]),
                "dice_macro": format_float(float(np.mean(case_dice_vals)) if case_dice_vals else np.nan),
                "hd95_macro_mm": format_float(float(np.mean(case_hd95_vals)) if case_hd95_vals else np.nan),
                "assd_macro_mm": format_float(float(np.mean(case_assd_vals)) if case_assd_vals else np.nan),
            })
            timer.set_outputs(len(per_case_macro_rows))

            if args.save_case_volumes:
                np.save(os.path.join(args.save_dir, "case_volumes", f"{case_id}_pred.npy"), pred_vol)
                np.save(os.path.join(args.save_dir, "case_volumes", f"{case_id}_gt.npy"), gt_vol)

        class_csv = os.path.join(args.save_dir, "per_case_per_class.csv")
        with open(class_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_case_class_rows[0].keys()) if per_case_class_rows else [])
            writer.writeheader()
            writer.writerows(per_case_class_rows)

        macro_csv = os.path.join(args.save_dir, "per_case_macro.csv")
        with open(macro_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_case_macro_rows[0].keys()) if per_case_macro_rows else [])
            writer.writeheader()
            writer.writerows(per_case_macro_rows)

        class_summary: Dict[str, Any] = {}
        mean_dice_all: List[float] = []
        mean_hd95_all: List[float] = []
        mean_assd_all: List[float] = []
        for c in class_ids:
            ds = summarize(per_class_dice[c])
            hs = summarize(per_class_hd95[c])
            ass = summarize(per_class_assd[c])
            class_summary[str(c)] = {
                "label_name": label_names.get(c, f"class_{c}"),
                "dice": ds,
                "hd95_mm": hs,
                "assd_mm": ass,
            }
            if ds["mean"] is not None:
                mean_dice_all.append(float(ds["mean"]))
            if hs["mean"] is not None:
                mean_hd95_all.append(float(hs["mean"]))
            if ass["mean"] is not None:
                mean_assd_all.append(float(ass["mean"]))

        summary = {
            "dataset": dataset_name,
            "fold_root": args.fold_root,
            "pred_dir": args.pred_dir,
            "split": args.split,
            "num_eval_cases": len(per_case_macro_rows),
            "class_metrics": class_summary,
            "overall_macro_mean": {
                "dice": float(np.mean(mean_dice_all)) if mean_dice_all else None,
                "hd95_mm": float(np.mean(mean_hd95_all)) if mean_hd95_all else None,
                "assd_mm": float(np.mean(mean_assd_all)) if mean_assd_all else None,
            },
            "case_wise_macro": {
                "dice": summarize([x["dice_macro"] for x in per_case_macro_rows if x["dice_macro"] is not None]),
                "hd95_mm": summarize([x["hd95_macro_mm"] for x in per_case_macro_rows if x["hd95_macro_mm"] is not None]),
                "assd_mm": summarize([x["assd_macro_mm"] for x in per_case_macro_rows if x["assd_macro_mm"] is not None]),
            },
            "protocol_notes": {
                "evaluation_space": "native",
                "volume_reconstruction": "all slices sorted by slice_idx and stacked in native space",
                "primary_metrics_3d": ["DSC", "HD95(mm)", "ASSD(mm)"],
                "distance_unit": "mm",
                "connectivity": int(args.connectivity),
                "empty_distance_policy": args.empty_distance_policy,
                "debug_resize_enabled": bool(args.allow_resize_pred_to_native_for_debug),
            },
        }

        save_json(summary, os.path.join(args.save_dir, "eval_3d_summary.json"))
        save_json(per_case_macro_rows, os.path.join(args.save_dir, "per_case_macro.json"))
        timer.set_outputs(len(per_case_macro_rows))
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
