#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.serif": ["Times New Roman", "Times New Roman PS MT", "DejaVu Serif"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from skimage.measure import find_contours as sk_find_contours
except Exception:
    sk_find_contours = None

DATASETS_3D = {"btcv", "synapse", "acdc", "prostate158"}
MULTICLASS_REVIEW_DATASETS = {"btcv", "synapse", "acdc", "prostate158"}

DEFAULT_MULTICLASS_PALETTE = {
    0: {"name": "background", "rgb": [0, 0, 0]},
    1: {"name": "spleen", "rgb": [220, 20, 60]},
    2: {"name": "r_kidney", "rgb": [65, 105, 225]},
    3: {"name": "l_kidney", "rgb": [0, 191, 255]},
    4: {"name": "gallbladder", "rgb": [154, 205, 50]},
    5: {"name": "esophagus", "rgb": [255, 140, 0]},
    6: {"name": "liver", "rgb": [34, 139, 34]},
    7: {"name": "stomach", "rgb": [138, 43, 226]},
    8: {"name": "aorta", "rgb": [255, 69, 0]},
    9: {"name": "ivc", "rgb": [70, 130, 180]},
    10: {"name": "veins_psv", "rgb": [255, 105, 180]},
    11: {"name": "pancreas", "rgb": [0, 206, 209]},
    12: {"name": "r_adrenal", "rgb": [160, 82, 45]},
    13: {"name": "l_adrenal", "rgb": [0, 139, 139]},
    255: {"name": "ignore", "rgb": [255, 236, 179]},
}
SYNAPSE_CLASS_MAP = {
    0: {"name": "background", "rgb": [0, 0, 0]},
    1: {"name": "spleen", "rgb": [220, 20, 60]},
    2: {"name": "r_kidney", "rgb": [65, 105, 225]},
    3: {"name": "l_kidney", "rgb": [0, 191, 255]},
    4: {"name": "gallbladder", "rgb": [154, 205, 50]},
    5: {"name": "liver", "rgb": [34, 139, 34]},
    6: {"name": "stomach", "rgb": [138, 43, 226]},
    7: {"name": "aorta", "rgb": [255, 69, 0]},
    8: {"name": "pancreas", "rgb": [0, 206, 209]},
    255: {"name": "ignore", "rgb": [255, 236, 179]},
}
ACDC_CLASS_MAP = {
    0: {"name": "background", "rgb": [0, 0, 0]},
    1: {"name": "rv", "rgb": [220, 20, 60]},
    2: {"name": "myo", "rgb": [65, 105, 225]},
    3: {"name": "lv", "rgb": [34, 139, 34]},
    255: {"name": "ignore", "rgb": [255, 236, 179]},
}
PROSTATE158_CLASS_MAP = {
    0: {"name": "background", "rgb": [0, 0, 0]},
    1: {"name": "central_gland", "rgb": [220, 20, 60]},
    2: {"name": "peripheral_zone", "rgb": [65, 105, 225]},
    255: {"name": "ignore", "rgb": [255, 236, 179]},
}
BINARY_CLASS_MAP = {
    0: {"name": "background", "rgb": [0, 0, 0]},
    1: {"name": "foreground", "rgb": [220, 20, 60]},
    255: {"name": "ignore", "rgb": [255, 236, 179]},
}
DATASET_CLASS_MAPS = {
    "btcv": DEFAULT_MULTICLASS_PALETTE,
    "synapse": SYNAPSE_CLASS_MAP,
    "acdc": ACDC_CLASS_MAP,
    "prostate158": PROSTATE158_CLASS_MAP,
    "kvasirseg": BINARY_CLASS_MAP,
    "cvc_clinicdb": BINARY_CLASS_MAP,
    "tn3k": BINARY_CLASS_MAP,
    "tg3k": BINARY_CLASS_MAP,
    "ddti": BINARY_CLASS_MAP,
    "otu_2d": BINARY_CLASS_MAP,
    "monuseg": BINARY_CLASS_MAP,
    "ph2": BINARY_CLASS_MAP,
    "drive": BINARY_CLASS_MAP,
    "chasedb1": BINARY_CLASS_MAP,
    "hrf": BINARY_CLASS_MAP,
}
ACTIVATION_KEYS = [
    "activation_panel",
    "activation_label",
    "under_score",
    "over_score",
    "pred_gt_area_ratio",
    "gt_fg_pixels",
    "pseudo_fg_pixels",
    "under_pixels",
    "over_pixels",
    "empty_positive_flag",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv_rows(rows: List[Dict[str, Any]], path: Path, fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def palette_jsonable(palette: Dict[int, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(k): v for k, v in palette.items()}


def palette_for_dataset(dataset: str) -> Dict[int, Dict[str, Any]]:
    return DATASET_CLASS_MAPS.get(dataset.lower(), BINARY_CLASS_MAP)


def class_name(palette: Dict[int, Dict[str, Any]], cid: int) -> str:
    return str(palette.get(cid, {"name": f"class_{cid}"})["name"])


def class_color_rgb(palette: Dict[int, Dict[str, Any]], cid: int) -> Tuple[int, int, int]:
    if cid in palette:
        rgb = palette[cid]["rgb"]
    else:
        cmap = plt.get_cmap("tab20")
        rgba = cmap(int(cid) % 20)
        rgb = [int(round(255 * rgba[0])), int(round(255 * rgba[1])), int(round(255 * rgba[2]))]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def multiclass_legend_layout(num_handles: int) -> Tuple[int, float, float]:
    if num_handles <= 0:
        return 0, 0.02, 11.0
    ncol = min(num_handles, 5)
    nrows = int(math.ceil(num_handles / max(ncol, 1)))
    bottom = 0.10 + 0.065 * nrows
    fontsize = 12.0 if num_handles <= 6 else (10.5 if num_handles <= 10 else 9.5)
    return ncol, bottom, fontsize


def normalize_key(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().replace("\\", "/")
    return os.path.basename(s)


def normalize_stem(s: Any) -> str:
    return os.path.splitext(normalize_key(s))[0]


def try_float(x: Any) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def find_files(root: Path, suffixes: Tuple[str, ...]) -> List[Path]:
    out: List[Path] = []
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            out.append(p)
    return sorted(out)


def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def maybe_read_table(path: Path):
    if pd is None:
        raise RuntimeError("student screening requires pandas in the active environment")
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return pd.DataFrame(obj)
            if isinstance(obj, dict):
                for key in ["files", "cases", "records", "rows", "items"]:
                    if key in obj and isinstance(obj[key], list) and obj[key] and isinstance(obj[key][0], dict):
                        return pd.DataFrame(obj[key])
            return None
    except Exception:
        return None
    return None


ID_CANDIDATES = [
    "sample_id",
    "slice_name",
    "filename",
    "file_name",
    "image",
    "image_id",
    "id",
    "case_id",
    "case",
    "name",
]


def detect_id_column(df) -> Optional[str]:
    mapping = {norm_col(c): c for c in df.columns}
    for name in ID_CANDIDATES:
        if name in mapping:
            return mapping[name]
    for c in df.columns:
        if str(df[c].dtype) == "object":
            return c
    return None


def metric_aliases() -> Dict[str, List[str]]:
    return {
        "dice": ["dice", "dsc", "mean_dice", "avg_dice", "dice_score"],
        "iou": ["iou", "miou", "mean_iou"],
        "mae": ["mae", "mean_absolute_error"],
        "hd95": ["hd95", "hausdorff95", "hausdorff_95", "hd_95"],
        "assd": ["assd", "asd", "avg_surface_distance", "average_surface_distance"],
        "class": ["class", "class_id", "label", "label_id", "organ", "organ_name"],
    }


def find_metric_column(df, aliases: List[str]) -> Optional[str]:
    mapping = {norm_col(c): c for c in df.columns}
    for alias in aliases:
        if alias in mapping:
            return mapping[alias]
    return None


def discover_metric_rows(eval_root: Path, kind: str) -> List[Dict[str, Any]]:
    if pd is None:
        raise RuntimeError("student screening requires pandas in the active environment")

    rows: List[Dict[str, Any]] = []
    files = find_files(eval_root, (".csv", ".json"))
    aliases = metric_aliases()

    for fp in files:
        df = maybe_read_table(fp)
        if df is None or df.empty:
            continue

        id_col = detect_id_column(df)
        dice_col = find_metric_column(df, aliases["dice"])
        iou_col = find_metric_column(df, aliases["iou"])
        mae_col = find_metric_column(df, aliases["mae"])
        hd95_col = find_metric_column(df, aliases["hd95"])
        assd_col = find_metric_column(df, aliases["assd"])
        class_col = find_metric_column(df, aliases["class"])

        if id_col is None:
            continue
        if not any([dice_col, iou_col, mae_col, hd95_col, assd_col]):
            continue

        for _, row in df.iterrows():
            rid = row.get(id_col)
            if pd.isna(rid):
                continue
            rows.append(
                {
                    "source_file": str(fp),
                    "raw_id": str(rid),
                    "id_norm": normalize_key(rid),
                    "stem_norm": normalize_stem(rid),
                    "dice": try_float(row.get(dice_col)) if dice_col else None,
                    "iou": try_float(row.get(iou_col)) if iou_col else None,
                    "mae": try_float(row.get(mae_col)) if mae_col else None,
                    "hd95": try_float(row.get(hd95_col)) if hd95_col else None,
                    "assd": try_float(row.get(assd_col)) if assd_col else None,
                    "class": None if class_col is None else str(row.get(class_col)),
                }
            )

    best: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
    for row in rows:
        key = (row["stem_norm"], row["class"])
        score = sum(row.get(name) is not None for name in ["dice", "iou", "mae", "hd95", "assd"])
        if key not in best:
            best[key] = row
            continue
        prev = best[key]
        prev_score = sum(prev.get(name) is not None for name in ["dice", "iou", "mae", "hd95", "assd"])
        if score > prev_score:
            best[key] = row
    return list(best.values())


def merge_mode_rows(baseline_rows: List[Dict[str, Any]], upper_rows: List[Dict[str, Any]], kind: str):
    if pd is None:
        raise RuntimeError("student screening requires pandas in the active environment")

    def aggregate(rows: List[Dict[str, Any]]):
        if not rows:
            return pd.DataFrame(columns=["key"])
        df = pd.DataFrame(rows)
        out_rows = []
        for key, group in df.groupby("stem_norm"):
            no_class = group[group["class"].isna()]
            gg = no_class if len(no_class) > 0 else group
            out_rows.append(
                {
                    "key": key,
                    "dice": gg["dice"].dropna().mean() if "dice" in gg else np.nan,
                    "iou": gg["iou"].dropna().mean() if "iou" in gg else np.nan,
                    "mae": gg["mae"].dropna().mean() if "mae" in gg else np.nan,
                    "hd95": gg["hd95"].dropna().mean() if "hd95" in gg else np.nan,
                    "assd": gg["assd"].dropna().mean() if "assd" in gg else np.nan,
                    "source_examples": list(gg["raw_id"].astype(str).head(5)),
                }
            )
        return pd.DataFrame(out_rows)

    bdf = aggregate(baseline_rows).add_prefix("baseline_")
    udf = aggregate(upper_rows).add_prefix("upper_")
    merged = pd.merge(bdf, udf, left_on="baseline_key", right_on="upper_key", how="outer")
    merged["key"] = merged["baseline_key"].fillna(merged["upper_key"])

    for metric in ["dice", "iou", "mae", "hd95", "assd"]:
        bcol = f"baseline_{metric}"
        ucol = f"upper_{metric}"
        if bcol not in merged:
            merged[bcol] = np.nan
        if ucol not in merged:
            merged[ucol] = np.nan

    merged["baseline_badness"] = 0.0
    merged["upper_badness"] = 0.0

    if merged["baseline_dice"].notna().any():
        merged["baseline_badness"] += 1 - merged["baseline_dice"].fillna(0)
    if merged["upper_dice"].notna().any():
        merged["upper_badness"] += 1 - merged["upper_dice"].fillna(0)
    if merged["baseline_iou"].notna().any():
        merged["baseline_badness"] += 1 - merged["baseline_iou"].fillna(0)
    if merged["upper_iou"].notna().any():
        merged["upper_badness"] += 1 - merged["upper_iou"].fillna(0)

    for prefix in ["baseline", "upper"]:
        for metric in ["mae", "hd95", "assd"]:
            col = f"{prefix}_{metric}"
            if not merged[col].notna().any():
                continue
            vals = merged[col].fillna(merged[col].max())
            denom = max(float(vals.quantile(0.95)), 1e-6)
            merged[f"{prefix}_badness"] += np.minimum(vals, denom) / denom

    merged["dice_gap"] = merged["upper_dice"].fillna(np.nan) - merged["baseline_dice"].fillna(np.nan)
    return merged


def classify_buckets(df, kind: str) -> Dict[str, Any]:
    if pd is None:
        raise RuntimeError("student screening requires pandas in the active environment")
    if df.empty:
        return {name: df.copy() for name in ["pseudo_suspect", "model_suspect", "both_bad", "boundary_suspect"]}

    if kind == "2d":
        b_bad = (
            (df["baseline_dice"].fillna(1) < 0.70)
            | (df["baseline_iou"].fillna(1) < 0.55)
            | (df["baseline_badness"] >= df["baseline_badness"].quantile(0.85))
        )
        u_good = (df["upper_dice"].fillna(0) >= 0.80) | (df["upper_iou"].fillna(0) >= 0.65)
        u_bad = (
            (df["upper_dice"].fillna(1) < 0.75)
            | (df["upper_iou"].fillna(1) < 0.60)
            | (df["upper_badness"] >= df["upper_badness"].quantile(0.85))
        )
        boundary = pd.Series(False, index=df.index)
    else:
        b_bad = (
            (df["baseline_dice"].fillna(1) < 0.75)
            | (df["baseline_hd95"].fillna(0) > 20)
            | (df["baseline_assd"].fillna(0) > 5)
            | (df["baseline_badness"] >= df["baseline_badness"].quantile(0.85))
        )
        u_good = (df["upper_dice"].fillna(0) >= 0.82) & (
            (df["upper_hd95"].fillna(0) < 12) | df["upper_hd95"].isna()
        )
        u_bad = (
            (df["upper_dice"].fillna(1) < 0.78)
            | (df["upper_hd95"].fillna(0) > 18)
            | (df["upper_assd"].fillna(0) > 5)
            | (df["upper_badness"] >= df["upper_badness"].quantile(0.85))
        )
        boundary = (
            (df["upper_dice"].fillna(0) >= 0.75)
            & ((df["upper_hd95"].fillna(0) > 20) | (df["upper_assd"].fillna(0) > 4))
        )

    pseudo_suspect = df[b_bad & u_good & (df["dice_gap"].fillna(0) >= 0.10)].copy()
    model_suspect = df[u_bad].copy()
    both_bad = df[b_bad & u_bad].copy()
    boundary_suspect = df[boundary].copy()

    pseudo_suspect = pseudo_suspect.sort_values(["dice_gap", "baseline_badness"], ascending=[False, False])
    model_suspect = model_suspect.sort_values(["upper_badness"], ascending=[False])
    both_bad = both_bad.sort_values(["upper_badness", "baseline_badness"], ascending=[False, False])
    if not boundary_suspect.empty:
        cols = [c for c in ["upper_hd95", "upper_assd"] if c in boundary_suspect.columns]
        if cols:
            boundary_suspect = boundary_suspect.sort_values(cols, ascending=[False] * len(cols))

    return {
        "pseudo_suspect": pseudo_suspect,
        "model_suspect": model_suspect,
        "both_bad": both_bad,
        "boundary_suspect": boundary_suspect,
    }


def write_dataframe_csv(df, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 4:
        img = np.squeeze(img)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=-1)
    elif img.ndim == 3:
        if img.shape[0] == 3 and img.shape[-1] != 3:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] > 3:
            img = img[:, :, :3]
    else:
        raise ValueError(f"Unexpected image ndim={img.ndim}")
    if img.max() <= 1.5:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def load_npy_2d(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array from {path}, got {arr.shape}")
    return arr


def load_npy_img(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=-1)
    return arr


def load_manifest_index(fold_root: Path) -> Dict[str, Dict[str, Any]]:
    manifest = load_json(fold_root / "meta" / "manifest.json")
    return {item["slice_name"]: item for item in manifest}


def load_geometry_meta(fold_root: Path) -> Dict[str, Dict[str, Any]]:
    return load_json(fold_root / "meta" / "geometry_meta.json")


def geom_sizes(geom: Dict[str, Any]) -> Tuple[int, int, int, int]:
    return int(geom["orig_h"]), int(geom["orig_w"]), int(geom["new_h"]), int(geom["new_w"])


def restore_teacher_to_native(mask_teacher: np.ndarray, geom: Dict[str, Any]) -> np.ndarray:
    orig_h, orig_w, new_h, new_w = geom_sizes(geom)
    cropped = mask_teacher[:new_h, :new_w]
    return cv2.resize(cropped.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def restore_teacher_to_native_rgb(img_teacher_rgb: np.ndarray, geom: Dict[str, Any]) -> np.ndarray:
    orig_h, orig_w, new_h, new_w = geom_sizes(geom)
    cropped = img_teacher_rgb[:new_h, :new_w]
    return cv2.resize(cropped.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)


def restore_student_to_native(mask_student: np.ndarray, geom: Dict[str, Any]) -> np.ndarray:
    native_to_student = geom["native_to_student"]
    target_h = int(native_to_student["target_h"])
    target_w = int(native_to_student["target_w"])
    new_h = int(native_to_student["new_h"])
    new_w = int(native_to_student["new_w"])
    ox = int(native_to_student.get("offset_x", 0))
    oy = int(native_to_student.get("offset_y", 0))
    orig_h = int(geom["orig_h"])
    orig_w = int(geom["orig_w"])
    if mask_student.shape != (target_h, target_w):
        mask_student = cv2.resize(mask_student.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    crop = mask_student[oy:oy + new_h, ox:ox + new_w]
    return cv2.resize(crop.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def load_native_image_and_gt(fold_root: Path, item: Dict[str, Any], geom: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    teacher_img = to_uint8_rgb(load_npy_img(fold_root / item["teacher_img"]))
    native_rgb = restore_teacher_to_native_rgb(teacher_img, geom)
    native_gt_rel = item.get("native_gt")
    if native_gt_rel:
        native_gt_path = fold_root / native_gt_rel
        if native_gt_path.exists():
            return native_rgb, load_npy_2d(native_gt_path)
    teacher_gt = load_npy_2d(fold_root / item["teacher_gt"])
    return native_rgb, restore_teacher_to_native(teacher_gt, geom)


def load_teacher_pseudo_native(fold_root: Path, split: str, filename: str, geom: Dict[str, Any]) -> np.ndarray:
    pseudo = load_npy_2d(fold_root / "pseudo_teacher" / f"tri_{split}" / filename)
    return restore_teacher_to_native(pseudo, geom)


def load_student_pred_native(pred_path: Path, gt_native: np.ndarray, geom: Dict[str, Any]) -> np.ndarray:
    pred = load_npy_2d(pred_path)
    if pred.shape == gt_native.shape:
        return pred.astype(np.uint8)
    return restore_student_to_native(pred, geom)


def line_width_from_hw(h: int, w: int) -> int:
    m = max(h, w)
    return 2 if m <= 320 else (3 if m <= 768 else 4)


def font_scale_from_hw(h: int, w: int) -> float:
    m = max(h, w)
    return 0.45 if m <= 320 else (0.55 if m <= 768 else 0.75)


def mpl_linewidth_from_hw(h: int, w: int) -> float:
    m = max(h, w)
    return 1.0 if m <= 512 else (1.15 if m <= 1024 else 1.3)


def grayscale_rgb(native_rgb: np.ndarray) -> np.ndarray:
    rgb = to_uint8_rgb(native_rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.repeat(gray[:, :, None], 3, axis=-1)


def touches_image_border(box: Sequence[int], h: int, w: int, pad: int = 0) -> bool:
    x1, y1, x2, y2 = [int(v) for v in box]
    return x1 <= pad or y1 <= pad or x2 >= (w - 1 - pad) or y2 >= (h - 1 - pad)


def component_drop_reason(dataset: str, bbox: Sequence[int], area: int, shape: Tuple[int, int]) -> Optional[str]:
    dataset_key = dataset.lower()
    h, w = shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    if dataset_key == "tg3k":
        if x1 == 0 and y1 == 0 and x2 <= 15 and y2 <= 15 and area <= 128:
            return "tg3k_top_left_systematic_artifact"
        if y1 == 0 and bh <= 2 and area <= 4:
            return "tg3k_top_edge_singleton"
    if dataset_key == "kvasirseg":
        if x1 == 0 and y1 == 0 and x2 <= 7 and y2 <= 7 and area <= 16:
            return "kvasir_top_left_jpeg_artifact"
        if (x1 == 0 or y1 == 0) and area <= 8 and max(bw, bh) <= 8:
            return "kvasir_border_jpeg_artifact"
    if touches_image_border(bbox, h, w) and area <= 4 and max(bw, bh) <= 4:
        return "tiny_border_noise"
    return None


def analyze_binary_gt_components(dataset: str, gt_native: np.ndarray) -> Dict[str, Any]:
    fg = ((gt_native > 0) & (gt_native != 255)).astype(np.uint8)
    out: Dict[str, Any] = {
        "raw_components": 0,
        "display_components": 0,
        "largest_area": 0,
        "removed_components": [],
        "display_mask": fg,
    }
    if fg.sum() == 0:
        return out
    num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    cleaned = np.zeros_like(fg)
    components: List[Dict[str, Any]] = []
    largest_comp_id = None
    largest_area = -1
    for comp_id in range(1, num_labels):
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        cw = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        ch = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        bbox = [x, y, x + cw - 1, y + ch - 1]
        if area > largest_area:
            largest_area = area
            largest_comp_id = comp_id
        reason = component_drop_reason(dataset, bbox, area, fg.shape)
        component = {"component_id": int(comp_id), "area": area, "bbox": bbox, "drop_reason": reason}
        components.append(component)
        if reason is None:
            cleaned[label_map == comp_id] = 1
    if cleaned.sum() == 0 and largest_comp_id is not None:
        cleaned[label_map == largest_comp_id] = 1
    out.update(
        {
            "raw_components": max(0, num_labels - 1),
            "display_components": int(len(np.unique(label_map[cleaned > 0]))),
            "largest_area": int(max(largest_area, 0)),
            "removed_components": [x for x in components if x["drop_reason"] is not None],
            "display_mask": cleaned.astype(np.uint8),
        }
    )
    return out


def display_gt_mask(dataset: str, gt_native: np.ndarray) -> np.ndarray:
    return analyze_binary_gt_components(dataset, gt_native)["display_mask"]


def filter_binary_prompt_boxes(
    dataset: str,
    prompt_boxes: Sequence[Dict[str, Any]],
    shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    boxes = [dict(info) for info in prompt_boxes]
    if not boxes:
        return []
    kept: List[Dict[str, Any]] = []
    for info in boxes:
        area = int(info.get("area", 0))
        if component_drop_reason(dataset, info["bbox"], area, shape) is not None:
            continue
        kept.append(info)
    if kept:
        return kept
    largest = max(boxes, key=lambda info: int(info.get("area", 0)))
    return [largest]


def save_figure(fig: Any, path: Path, dpi: int = 320) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)


def finalize_axis(ax: Any, h: int, w: int) -> None:
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_floating_text(
    ax: Any,
    text: str,
    *,
    fg: str = "white",
    bg_alpha: float = 0.38,
    fontsize: float = 8.0,
    x: float = 0.018,
    y: float = 0.982,
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=fg,
        family="sans-serif",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": (0.0, 0.0, 0.0, bg_alpha),
            "edgecolor": "none",
        },
    )


def mask_rgba(mask: np.ndarray, rgb: Tuple[int, int, int], alpha: float) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    out[..., 0] = rgb[0] / 255.0
    out[..., 1] = rgb[1] / 255.0
    out[..., 2] = rgb[2] / 255.0
    out[..., 3] = mask.astype(np.float32) * float(alpha)
    return out


def extract_prompt_boxes(prompt_entry: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not prompt_entry or not isinstance(prompt_entry.get("instances"), list):
        return []
    boxes: List[Dict[str, Any]] = []
    for ins in prompt_entry["instances"]:
        bbox_native = ins.get("bbox_native", ins.get("bbox"))
        if bbox_native is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_native]
        label_id = int(ins.get("label_id", 1))
        area = int(ins.get("area", max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)))
        boxes.append({"bbox": [x1, y1, x2, y2], "label_id": label_id, "area": max(area, 0)})
    return sorted(boxes, key=lambda item: (item["label_id"], -item["area"], item["bbox"][1], item["bbox"][0]))


def prompt_caption(prompt_entry: Optional[Dict[str, Any]], palette: Dict[int, Dict[str, Any]]) -> Optional[str]:
    tags: List[str] = []
    for info in extract_prompt_boxes(prompt_entry):
        tag = f"{class_name(palette, info['label_id'])}({info['label_id']})"
        if tag not in tags:
            tags.append(tag)
    if not tags:
        return None
    shown = tags[:4]
    if len(tags) > 4:
        shown.append("...")
    return "Prompt: " + ", ".join(shown)


def add_boxes_to_axis(
    ax: Any,
    bbox_records: Sequence[Dict[str, Any]],
    *,
    palette: Dict[int, Dict[str, Any]],
    default_color: Optional[str] = None,
    linewidth: float = 1.15,
) -> None:
    for info in bbox_records:
        x1, y1, x2, y2 = [float(v) for v in info["bbox"]]
        if default_color is None:
            rgb = class_color_rgb(palette, int(info.get("label_id", 1)))
            edgecolor = tuple(v / 255.0 for v in rgb)
        else:
            edgecolor = default_color
        ax.add_patch(
            Rectangle(
                (x1, y1),
                max(1.0, x2 - x1 + 1.0),
                max(1.0, y2 - y1 + 1.0),
                fill=False,
                edgecolor=edgecolor,
                linewidth=linewidth,
                joinstyle="miter",
            )
        )


def contour_polylines(mask: np.ndarray) -> List[np.ndarray]:
    mask_u8 = mask.astype(np.uint8)
    if sk_find_contours is not None:
        polys = []
        for contour in sk_find_contours(mask_u8, 0.5):
            if contour.shape[0] < 2:
                continue
            polys.append(np.stack([contour[:, 1], contour[:, 0]], axis=1))
        return polys
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt[:, 0, :].astype(float) for cnt in contours if cnt.shape[0] >= 2]


def draw_mask_contours(ax: Any, mask: np.ndarray, color: Tuple[int, int, int], linewidth: float) -> None:
    color_float = tuple(v / 255.0 for v in color)
    for poly in contour_polylines(mask):
        ax.plot(poly[:, 0], poly[:, 1], color=color_float, linewidth=linewidth, solid_joinstyle="round")


def draw_binary_gt_fill(ax: Any, mask: np.ndarray, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    mask = mask.astype(np.uint8)
    h, w = mask.shape[:2]
    ax.imshow(np.zeros((h, w, 3), dtype=np.uint8), interpolation="nearest")
    if mask.sum() == 0:
        return
    # NOTE:
    # Border-touching connected components may produce open contours; filling open
    # polygons can create artificial diagonal wedges. Render binary fill directly
    # from raster mask to keep GT geometry faithful to source labels.
    ax.imshow(mask_rgba(mask, color, alpha=1.0), interpolation="nearest")


def render_label_canvas(label_map: np.ndarray, palette: Dict[int, Dict[str, Any]]) -> np.ndarray:
    canvas = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    for cid in sorted(int(x) for x in np.unique(label_map) if int(x) != 0):
        canvas[label_map == cid] = class_color_rgb(palette, cid)
    return canvas


def present_class_ids(
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    palette: Dict[int, Dict[str, Any]],
    gt_boxes: Sequence[Dict[str, Any]] = (),
    pred_boxes: Sequence[Dict[str, Any]] = (),
) -> List[int]:
    present = {int(x) for x in np.unique(gt_map) if int(x) != 0}
    present.update(int(x) for x in np.unique(pred_map) if int(x) != 0)
    present.update(int(info.get("label_id", 0)) for info in gt_boxes if int(info.get("label_id", 0)) != 0)
    present.update(int(info.get("label_id", 0)) for info in pred_boxes if int(info.get("label_id", 0)) != 0)
    return [cid for cid in sorted(present) if cid in palette]


def build_multiclass_triptych(
    dataset: str,
    sample_tag: str,
    native_rgb: np.ndarray,
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    palette: Dict[int, Dict[str, Any]],
    *,
    gt_boxes: Sequence[Dict[str, Any]] = (),
    pred_boxes: Sequence[Dict[str, Any]] = (),
    sample_note: Optional[str] = None,
) -> Any:
    rgb = grayscale_rgb(native_rgb)
    h, w = gt_map.shape[:2]
    present = present_class_ids(gt_map, pred_map, palette, gt_boxes=gt_boxes, pred_boxes=pred_boxes)
    handles = [Patch(facecolor=np.array(class_color_rgb(palette, cid)) / 255.0, edgecolor="none", label=class_name(palette, cid)) for cid in present]
    ncol, bottom, legend_fontsize = multiclass_legend_layout(len(handles))
    legend_rows = 0 if ncol == 0 else int(math.ceil(len(handles) / ncol))
    fig_h = 3.2 + 0.24 * legend_rows
    fig_w = max(6.6, fig_h * (w / max(h, 1)) * 3.0)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), gridspec_kw={"wspace": 0.02}, facecolor="white")
    lw = mpl_linewidth_from_hw(h, w)

    raw_ax, gt_ax, pred_ax = axes
    raw_ax.imshow(rgb, interpolation="nearest")
    raw_ax.set_title("Raw", fontsize=14, family="sans-serif", pad=6)
    raw_text = sample_tag if not sample_note else f"{sample_tag}\n{sample_note}"
    add_floating_text(raw_ax, raw_text, fg="white", bg_alpha=0.40, fontsize=8.0)
    finalize_axis(raw_ax, h, w)

    gt_ax.set_facecolor("black")
    gt_ax.imshow(render_label_canvas(gt_map, palette), interpolation="nearest")
    add_boxes_to_axis(gt_ax, gt_boxes, palette=palette, linewidth=lw)
    gt_ax.set_title("Ground Truth (GT)", fontsize=14, family="sans-serif", pad=6)
    finalize_axis(gt_ax, h, w)

    pred_ax.imshow(rgb, interpolation="nearest")
    for cid in sorted(int(x) for x in np.unique(pred_map) if int(x) != 0):
        mask = (pred_map == cid).astype(np.uint8)
        if mask.sum() == 0:
            continue
        color = class_color_rgb(palette, cid)
        alpha = 0.22 if cid == 255 else 0.35
        pred_ax.imshow(mask_rgba(mask, color, alpha=alpha), interpolation="nearest")
        draw_mask_contours(pred_ax, mask, color, lw)
    add_boxes_to_axis(pred_ax, pred_boxes, palette=palette, linewidth=lw)
    pred_ax.set_title("Prediction", fontsize=14, family="sans-serif", pad=6)
    finalize_axis(pred_ax, h, w)

    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=bottom, wspace=0.02)
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            frameon=False,
            fontsize=legend_fontsize,
            ncol=ncol,
            handlelength=1.2,
            columnspacing=0.8,
        )
    return fig


def build_binary_triptych(
    dataset: str,
    sample_tag: str,
    native_rgb: np.ndarray,
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    *,
    prompt_boxes: Sequence[Dict[str, Any]] = (),
    prompt_text: Optional[str] = None,
    gt_title: str = "GT + Prompt",
    pred_title: str = "Pseudo",
    sample_note: Optional[str] = None,
    box_color: str = "#2CA02C",
    display_mask: Optional[np.ndarray] = None,
    display_boxes: Optional[Sequence[Dict[str, Any]]] = None,
) -> Any:
    h, w = gt_map.shape[:2]
    fig_h = 2.9 if max(h, w) <= 768 else 3.4
    fig_w = max(6.0, fig_h * (w / max(h, 1)) * 3.0)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), gridspec_kw={"wspace": 0.02}, facecolor="white")
    lw = mpl_linewidth_from_hw(h, w)
    shown_mask = display_gt_mask(dataset, gt_map) if display_mask is None else display_mask.astype(np.uint8)
    shown_boxes = list(prompt_boxes if display_boxes is None else display_boxes)

    raw_ax, gt_ax, pred_ax = axes

    raw_ax.imshow(native_rgb, interpolation="nearest")
    raw_text = f"Raw\n{sample_tag}"
    if sample_note:
        raw_text = raw_text + f"\n{sample_note}"
    add_floating_text(raw_ax, raw_text, fg="white", bg_alpha=0.40, fontsize=8.0)
    finalize_axis(raw_ax, h, w)

    gt_ax.set_facecolor("black")
    draw_binary_gt_fill(gt_ax, shown_mask)
    add_boxes_to_axis(gt_ax, shown_boxes, palette=BINARY_CLASS_MAP, default_color=box_color, linewidth=lw)
    gt_text = gt_title
    if prompt_text:
        gt_text = gt_text + "\n" + prompt_text
    add_floating_text(gt_ax, gt_text, fg="white", bg_alpha=0.34, fontsize=8.0)
    finalize_axis(gt_ax, h, w)

    pred_fg = ((pred_map > 0) & (pred_map != 255)).astype(np.uint8)
    ignore_mask = (pred_map == 255).astype(np.uint8)
    pred_ax.imshow(native_rgb, interpolation="nearest")
    if np.any(ignore_mask):
        pred_ax.imshow(mask_rgba(ignore_mask, (255, 236, 179), alpha=0.22), interpolation="nearest")
    if np.any(pred_fg):
        pred_ax.imshow(mask_rgba(pred_fg, (214, 39, 40), alpha=0.46), interpolation="nearest")
        draw_mask_contours(pred_ax, pred_fg, (214, 39, 40), lw)
    add_boxes_to_axis(pred_ax, shown_boxes, palette=BINARY_CLASS_MAP, default_color=box_color, linewidth=lw)
    add_floating_text(pred_ax, pred_title, fg="white", bg_alpha=0.34, fontsize=8.0)
    finalize_axis(pred_ax, h, w)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02)
    return fig


def build_case_summary_figure(
    dataset: str,
    rows: Sequence[Dict[str, Any]],
    palette: Dict[int, Dict[str, Any]],
) -> Any:
    if not rows:
        raise ValueError("rows must not be empty")
    h, w = rows[0]["gt_map"].shape[:2]
    nrows = len(rows)
    lw = mpl_linewidth_from_hw(h, w)
    fig_row_h = 2.1 if max(h, w) <= 768 else 2.5
    fig_w = max(6.8, fig_row_h * (w / max(h, 1)) * 3.0)
    bottom = 0.02
    legend_fontsize = 11.0
    legend_ncol = 0
    legend_rows = 0
    if dataset.lower() in MULTICLASS_REVIEW_DATASETS:
        present_preview: set[int] = set()
        for row in rows:
            present_preview.update(present_class_ids(row["gt_map"], row["pred_map"], palette))
        legend_ncol, bottom, legend_fontsize = multiclass_legend_layout(len(present_preview))
        legend_rows = 0 if legend_ncol == 0 else int(math.ceil(len(present_preview) / legend_ncol))
    fig_h = fig_row_h * nrows + (0.22 * legend_rows if legend_rows > 0 else 0.08)
    fig, axes = plt.subplots(
        nrows,
        3,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"wspace": 0.02, "hspace": 0.02},
        facecolor="white",
    )

    present: set[int] = set()
    multiclass = dataset.lower() in MULTICLASS_REVIEW_DATASETS
    for row_idx, row in enumerate(rows):
        raw_ax, gt_ax, pred_ax = axes[row_idx]
        sample_tag = row["sample_tag"]
        sample_note = row.get("sample_note")
        native_rgb = row["native_rgb"]
        gt_map = row["gt_map"]
        pred_map = row["pred_map"]
        if multiclass:
            gray = grayscale_rgb(native_rgb)
            raw_ax.imshow(gray, interpolation="nearest")
            gt_ax.set_facecolor("black")
            gt_ax.imshow(render_label_canvas(gt_map, palette), interpolation="nearest")
            pred_ax.imshow(gray, interpolation="nearest")
            for cid in sorted(int(x) for x in np.unique(pred_map) if int(x) != 0):
                mask = (pred_map == cid).astype(np.uint8)
                if mask.sum() == 0:
                    continue
                color = class_color_rgb(palette, cid)
                alpha = 0.22 if cid == 255 else 0.35
                pred_ax.imshow(mask_rgba(mask, color, alpha=alpha), interpolation="nearest")
                draw_mask_contours(pred_ax, mask, color, lw)
            present.update(present_class_ids(gt_map, pred_map, palette))
            if row_idx == 0:
                raw_ax.set_title("Raw", fontsize=14, family="sans-serif", pad=6)
                gt_ax.set_title("Ground Truth (GT)", fontsize=14, family="sans-serif", pad=6)
                pred_ax.set_title("Prediction", fontsize=14, family="sans-serif", pad=6)
        else:
            raw_ax.imshow(native_rgb, interpolation="nearest")
            gt_ax.set_facecolor("black")
            draw_binary_gt_fill(gt_ax, display_gt_mask(dataset, gt_map))
            pred_ax.imshow(native_rgb, interpolation="nearest")
            pred_fg = ((pred_map > 0) & (pred_map != 255)).astype(np.uint8)
            ignore_mask = (pred_map == 255).astype(np.uint8)
            if np.any(ignore_mask):
                pred_ax.imshow(mask_rgba(ignore_mask, (255, 236, 179), alpha=0.22), interpolation="nearest")
            if np.any(pred_fg):
                pred_ax.imshow(mask_rgba(pred_fg, (214, 39, 40), alpha=0.46), interpolation="nearest")
                draw_mask_contours(pred_ax, pred_fg, (214, 39, 40), lw)
        add_floating_text(raw_ax, sample_tag if not sample_note else f"{sample_tag}\n{sample_note}", fg="white", bg_alpha=0.38, fontsize=7.8)
        finalize_axis(raw_ax, h, w)
        finalize_axis(gt_ax, h, w)
        finalize_axis(pred_ax, h, w)

    fig.subplots_adjust(left=0, right=1, top=0.98, bottom=bottom, wspace=0.02, hspace=0.02)
    if multiclass and present:
        handles = [Patch(facecolor=np.array(class_color_rgb(palette, cid)) / 255.0, edgecolor="none", label=class_name(palette, cid)) for cid in sorted(present)]
        legend_ncol, _, legend_fontsize = multiclass_legend_layout(len(handles))
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            frameon=False,
            fontsize=legend_fontsize,
            ncol=legend_ncol,
            handlelength=1.1,
            columnspacing=0.75,
        )
    return fig


def infer_kind(dataset: str) -> str:
    return "3d" if dataset.lower() in DATASETS_3D else "2d"


def should_use_multiclass_style(dataset: str) -> bool:
    return dataset.lower() in MULTICLASS_REVIEW_DATASETS


def review_triptych_figure(
    dataset: str,
    sample_tag: str,
    native_rgb: np.ndarray,
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    palette: Dict[int, Dict[str, Any]],
    *,
    prompt_entry: Optional[Dict[str, Any]] = None,
    sample_note: Optional[str] = None,
    mode: str,
) -> Any:
    prompt_boxes = extract_prompt_boxes(prompt_entry)
    if should_use_multiclass_style(dataset):
        pred_boxes = prompt_boxes if mode == "teacher" else []
        gt_boxes = prompt_boxes if mode == "teacher" else []
        return build_multiclass_triptych(
            dataset=dataset,
            sample_tag=sample_tag,
            native_rgb=native_rgb,
            gt_map=gt_map,
            pred_map=pred_map,
            palette=palette,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            sample_note=sample_note,
        )
    prompt_text = prompt_caption(prompt_entry, palette) if mode == "teacher" else None
    gt_title = "GT + Prompt" if mode == "teacher" else "Ground Truth (GT)"
    pred_title = "Pseudo" if mode == "teacher" else "Prediction"
    display_mask = display_gt_mask(dataset, gt_map)
    display_boxes = filter_binary_prompt_boxes(dataset, prompt_boxes, gt_map.shape) if mode == "teacher" else []
    return build_binary_triptych(
        dataset=dataset,
        sample_tag=sample_tag,
        native_rgb=native_rgb,
        gt_map=gt_map,
        pred_map=pred_map,
        prompt_boxes=display_boxes,
        prompt_text=prompt_text,
        gt_title=gt_title,
        pred_title=pred_title,
        sample_note=sample_note,
        display_mask=display_mask,
        display_boxes=display_boxes,
    )


def find_prompt_json(fold_root: Path, split: str) -> Path:
    candidates = [fold_root / f"prompts_{split}.json", fold_root / "prompts" / f"prompts_{split}.json"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Prompt json not found for split={split}")


def build_activation_stats(gt_native: np.ndarray, pseudo_native: np.ndarray) -> Dict[str, Any]:
    gt_fg = (gt_native > 0) & (gt_native != 255)
    pseudo_fg = (pseudo_native > 0) & (pseudo_native != 255)
    ignore_mask = gt_native == 255

    under_mask = gt_fg & (~pseudo_fg)
    over_mask = (~gt_fg) & (~ignore_mask) & pseudo_fg

    gt_fg_pixels = int(gt_fg.sum())
    pseudo_fg_pixels = int(pseudo_fg.sum())
    under_pixels = int(under_mask.sum())
    over_pixels = int(over_mask.sum())
    empty_positive_flag = bool(gt_fg_pixels > 0 and pseudo_fg_pixels == 0)

    denom = max(gt_fg_pixels, 1)
    under_score = float(under_pixels / denom)
    over_score = float(over_pixels / denom)
    pred_gt_area_ratio = float(pseudo_fg_pixels / denom)

    if (empty_positive_flag or under_score >= 0.20) and (over_score >= 0.20 and pred_gt_area_ratio >= 1.15):
        activation_label = "both_activation"
    elif empty_positive_flag or under_score >= 0.20:
        activation_label = "under_activation"
    elif over_score >= 0.20 and pred_gt_area_ratio >= 1.15:
        activation_label = "over_activation"
    else:
        activation_label = "normal"

    return {
        "activation_label": activation_label,
        "under_score": round(under_score, 6),
        "over_score": round(over_score, 6),
        "pred_gt_area_ratio": round(pred_gt_area_ratio, 6),
        "gt_fg_pixels": gt_fg_pixels,
        "pseudo_fg_pixels": pseudo_fg_pixels,
        "under_pixels": under_pixels,
        "over_pixels": over_pixels,
        "empty_positive_flag": empty_positive_flag,
    }


def should_review(stats: Dict[str, Any]) -> bool:
    return stats["activation_label"] in {"under_activation", "over_activation", "both_activation"}


def activation_record(index_record: Dict[str, Any], stats: Dict[str, Any], rel_path: Optional[str]) -> Dict[str, Any]:
    out = dict(index_record.get("activation", {}))
    out.update(stats)
    out["activation_panel"] = rel_path
    return out


def ranking_row(record: Dict[str, Any]) -> Dict[str, Any]:
    activation = record["activation"]
    return {
        "sample_id": record.get("sample_id"),
        "case_id": record.get("case_id"),
        "slice_idx": record.get("slice_idx"),
        "activation_label": activation["activation_label"],
        "under_score": activation["under_score"],
        "over_score": activation["over_score"],
        "pred_gt_area_ratio": activation["pred_gt_area_ratio"],
        "gt_fg_pixels": activation["gt_fg_pixels"],
        "pseudo_fg_pixels": activation["pseudo_fg_pixels"],
        "under_pixels": activation["under_pixels"],
        "over_pixels": activation["over_pixels"],
        "empty_positive_flag": activation["empty_positive_flag"],
        "activation_panel": activation["activation_panel"],
    }


def teacher_csv_fieldnames() -> List[str]:
    return [
        "sample_id",
        "case_id",
        "slice_idx",
        "activation_label",
        "under_score",
        "over_score",
        "pred_gt_area_ratio",
        "gt_fg_pixels",
        "pseudo_fg_pixels",
        "under_pixels",
        "over_pixels",
        "empty_positive_flag",
        "activation_panel",
    ]


def unique_ordered(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def pick_teacher_preview_records(
    records: Sequence[Dict[str, Any]],
    preview_count: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    if preview_count <= 0 or not records:
        return []

    def score(record: Dict[str, Any]) -> Tuple[float, float, float]:
        stats = record["stats"]
        analysis = record["component_analysis"]
        return (
            float(stats["under_score"] + stats["over_score"]),
            float(analysis["raw_components"] - analysis["display_components"]),
            float(stats["gt_fg_pixels"]),
        )

    clean_candidates = [
        r
        for r in records
        if r["stats"]["activation_label"] == "normal"
        and r["component_analysis"]["display_components"] == 1
        and not r["component_analysis"]["removed_components"]
    ]
    split_candidates = [
        r
        for r in records
        if len(r["display_boxes"]) >= 2 or r["component_analysis"]["display_components"] >= 2
    ]
    artifact_candidates = [
        r
        for r in records
        if r["component_analysis"]["removed_components"]
        or r["stats"]["activation_label"] in {"under_activation", "over_activation", "both_activation"}
    ]

    selected: List[Tuple[str, Dict[str, Any]]] = []
    if clean_candidates:
        selected.append(("clean", max(clean_candidates, key=lambda r: r["stats"]["gt_fg_pixels"])))
    if split_candidates:
        selected.append(
            (
                "split",
                max(
                    split_candidates,
                    key=lambda r: (len(r["display_boxes"]), r["component_analysis"]["display_components"], r["stats"]["gt_fg_pixels"]),
                ),
            )
        )
    if artifact_candidates:
        selected.append(("artifact_sensitive", max(artifact_candidates, key=score)))

    if len(selected) < preview_count:
        ordered = sorted(records, key=score, reverse=True)
        existing = {rec["sample_id"] for _, rec in selected}
        for rec in ordered:
            if rec["sample_id"] in existing:
                continue
            selected.append(("preview", rec))
            existing.add(rec["sample_id"])
            if len(selected) >= preview_count:
                break

    return selected[:preview_count]


def discover_teacher_datasets(processed_root: Path, datasets_arg: str) -> List[str]:
    if datasets_arg.strip():
        return [x.strip() for x in datasets_arg.split(",") if x.strip()]
    out = []
    for path in sorted(processed_root.iterdir()):
        if (path / "fold_0").exists():
            out.append(path.name)
    return out


def discover_student_datasets(work_root: Path, datasets_arg: str) -> List[str]:
    if datasets_arg.strip():
        return [x.strip() for x in datasets_arg.split(",") if x.strip()]
    names = set()
    for mode in ["baseline", "upper"]:
        mode_root = work_root / mode
        if not mode_root.exists():
            continue
        for path in mode_root.iterdir():
            if (path / "fold_0").exists():
                names.add(path.name)
    return sorted(names)


def screen_teacher_dataset(
    dataset: str,
    processed_root: Path,
    review_root: Path,
    split: str,
    overwrite: bool,
    preview_only: bool,
    preview_count: int,
) -> None:
    fold_root = processed_root / dataset / "fold_0"
    output_dataset_root = review_root / "pseudo" / dataset
    activation_dir = output_dataset_root / "activation"
    preview_dir = output_dataset_root / "preview"

    if not fold_root.exists():
        raise FileNotFoundError(f"Missing fold_root: {fold_root}")

    manifest = load_manifest_index(fold_root)
    geom_meta = load_geometry_meta(fold_root)
    prompts = load_json(find_prompt_json(fold_root, split))
    palette = palette_for_dataset(dataset)

    ensure_dir(output_dataset_root)
    ensure_dir(activation_dir)
    ensure_dir(preview_dir)
    save_json(palette_jsonable(palette), output_dataset_root / "palette.json")

    summary_counts = {"normal": 0, "under_activation": 0, "over_activation": 0, "both_activation": 0}
    updated_records: List[Dict[str, Any]] = []
    preview_candidates: List[Dict[str, Any]] = []

    manifest_items = [item for item in manifest.values() if item.get("split") == split]
    for item in manifest_items:
        sample_id = item["slice_name"]
        geom = geom_meta.get(sample_id)
        prompt_entry = prompts.get(sample_id)
        if geom is None or prompt_entry is None:
            continue

        native_rgb, native_gt = load_native_image_and_gt(fold_root, item, geom)
        pseudo_native = load_teacher_pseudo_native(fold_root, split, sample_id, geom)
        stats = build_activation_stats(native_gt, pseudo_native)
        prompt_boxes = extract_prompt_boxes(prompt_entry)
        display_boxes = filter_binary_prompt_boxes(dataset, prompt_boxes, native_gt.shape)
        component_analysis = analyze_binary_gt_components(dataset, native_gt)

        rel_activation: Optional[str] = None
        if not preview_only and should_review(stats):
            stem = Path(sample_id).stem
            rel_activation = f"activation/{stats['activation_label']}__{stem}__activation.png"
            abs_activation = output_dataset_root / rel_activation
            if overwrite or (not abs_activation.exists()):
                fig = review_triptych_figure(
                    dataset=dataset,
                    sample_tag=stem,
                    native_rgb=native_rgb,
                    gt_map=native_gt,
                    pred_map=pseudo_native,
                    palette=palette,
                    prompt_entry=prompt_entry,
                    mode="teacher",
                )
                save_figure(fig, abs_activation)

        record = {
            "sample_id": sample_id,
            "source": "teacher",
            "split": split,
            "case_id": item.get("case_id"),
            "slice_idx": item.get("slice_idx"),
            "class_ids": unique_ordered([str(info["label_id"]) for info in prompt_boxes]),
            "activation": activation_record({}, stats, rel_activation),
        }
        updated_records.append(record)
        summary_counts[stats["activation_label"]] += 1
        preview_candidates.append(
            {
                "sample_id": sample_id,
                "item": item,
                "prompt_entry": prompt_entry,
                "native_rgb": native_rgb,
                "native_gt": native_gt,
                "pseudo_native": pseudo_native,
                "stats": stats,
                "display_boxes": display_boxes,
                "component_analysis": component_analysis,
            }
        )

    preview_specs = pick_teacher_preview_records(preview_candidates, preview_count)
    preview_rows: List[Dict[str, Any]] = []
    for rank, (tag, rec) in enumerate(preview_specs, start=1):
        stem = Path(rec["sample_id"]).stem
        rel_preview = f"preview/preview_{rank:02d}__{tag}__{stem}.png"
        abs_preview = output_dataset_root / rel_preview
        if overwrite or (not abs_preview.exists()):
            fig = review_triptych_figure(
                dataset=dataset,
                sample_tag=stem,
                native_rgb=rec["native_rgb"],
                gt_map=rec["native_gt"],
                pred_map=rec["pseudo_native"],
                palette=palette,
                prompt_entry=rec["prompt_entry"],
                sample_note=tag,
                mode="teacher",
            )
            save_figure(fig, abs_preview)
        preview_rows.append(
            {
                "rank": rank,
                "preview_tag": tag,
                "sample_id": rec["sample_id"],
                "activation_label": rec["stats"]["activation_label"],
                "preview_panel": rel_preview,
                "removed_components": rec["component_analysis"]["removed_components"],
                "display_box_count": len(rec["display_boxes"]),
            }
        )

    save_json({"dataset": dataset, "split": split, "files": preview_rows}, output_dataset_root / "preview_index.json")

    if preview_only:
        save_json(
            {
                "dataset": dataset,
                "split": split,
                "num_records": len(updated_records),
                "num_preview_files": len(preview_rows),
                "counts": summary_counts,
                "output_root": str(output_dataset_root),
                "preview_only": True,
            },
            output_dataset_root / "preview_summary.json",
        )
        return

    index_data = {"dataset": dataset, "split": split, "palette_file": "palette.json", "files": updated_records}
    save_json(index_data, output_dataset_root / "index.json")

    ranking_rows = [
        ranking_row(record)
        for record in updated_records
        if isinstance(record.get("activation"), dict) and record["activation"].get("activation_panel")
    ]
    under_rows = sorted(
        [row for row in ranking_rows if row["activation_label"] in {"under_activation", "both_activation"}],
        key=lambda row: (not row["empty_positive_flag"], -row["under_score"], -row["gt_fg_pixels"]),
    )
    over_rows = sorted(
        [row for row in ranking_rows if row["activation_label"] in {"over_activation", "both_activation"}],
        key=lambda row: (-row["over_score"], -row["pred_gt_area_ratio"], -row["over_pixels"]),
    )
    both_rows = sorted(
        [row for row in ranking_rows if row["activation_label"] == "both_activation"],
        key=lambda row: (-(row["under_score"] + row["over_score"]), -row["under_pixels"], -row["over_pixels"]),
    )

    save_json(under_rows, output_dataset_root / "activation_under_ranking.json")
    save_json(over_rows, output_dataset_root / "activation_over_ranking.json")
    save_json(both_rows, output_dataset_root / "activation_both_ranking.json")
    write_csv_rows(under_rows, output_dataset_root / "activation_under_ranking.csv", teacher_csv_fieldnames())
    write_csv_rows(over_rows, output_dataset_root / "activation_over_ranking.csv", teacher_csv_fieldnames())
    write_csv_rows(both_rows, output_dataset_root / "activation_both_ranking.csv", teacher_csv_fieldnames())

    summary = {
        "dataset": dataset,
        "split": split,
        "space": "native_formal",
        "num_records": len(updated_records),
        "num_review_records": len(ranking_rows),
        "num_preview_files": len(preview_rows),
        "counts": summary_counts,
        "output_root": str(output_dataset_root),
    }
    save_json(summary, output_dataset_root / "activation_summary.json")


def build_manifest_stem_lookup(manifest: Dict[str, Dict[str, Any]], split: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for sample_id, item in manifest.items():
        if item.get("split") != split:
            continue
        out[normalize_stem(sample_id)] = item
    return out


def build_case_lookup(manifest: Dict[str, Dict[str, Any]], split: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for sample_id, item in manifest.items():
        if item.get("split") != split:
            continue
        case_id = str(item.get("case_id", "unknown_case"))
        out.setdefault(case_id, []).append(sample_id)
    for case_id in list(out.keys()):
        out[case_id] = sorted(out[case_id], key=lambda sid: int(manifest[sid].get("slice_idx", 0)))
    return out


def dice_fg(gt: np.ndarray, pred: np.ndarray) -> float:
    g = (gt > 0) & (gt != 255)
    p = pred > 0
    inter = float((g & p).sum())
    denom = float(g.sum() + p.sum())
    return 1.0 if denom == 0 else 2.0 * inter / denom


def render_student_2d_case(
    dataset: str,
    row: Any,
    out_dir: Path,
    fold_root: Path,
    manifest: Dict[str, Dict[str, Any]],
    geom_meta: Dict[str, Dict[str, Any]],
    stem_lookup: Dict[str, Dict[str, Any]],
    work_root: Path,
    overwrite: bool,
) -> None:
    ensure_dir(out_dir)
    key = str(row["key"])
    item = stem_lookup.get(key)
    if item is None:
        save_json({"sample_key": key, "missing_render_source": True}, out_dir / "meta.json")
        return

    sample_id = item["slice_name"]
    geom = geom_meta.get(sample_id)
    if geom is None:
        save_json({"sample_key": key, "sample_id": sample_id, "missing_geometry": True}, out_dir / "meta.json")
        return

    native_rgb, native_gt = load_native_image_and_gt(fold_root, item, geom)
    palette = palette_for_dataset(dataset)
    baseline_pred_path = work_root / "baseline" / dataset / "fold_0" / "pred_test" / sample_id
    upper_pred_path = work_root / "upper" / dataset / "fold_0" / "pred_test" / sample_id

    if baseline_pred_path.exists():
        baseline_pred = load_student_pred_native(baseline_pred_path, native_gt, geom)
        if overwrite or not (out_dir / "baseline_triptych.png").exists():
            fig = review_triptych_figure(
                dataset=dataset,
                sample_tag=Path(sample_id).stem,
                native_rgb=native_rgb,
                gt_map=native_gt,
                pred_map=baseline_pred,
                palette=palette,
                mode="student",
            )
            save_figure(fig, out_dir / "baseline_triptych.png")

    if upper_pred_path.exists():
        upper_pred = load_student_pred_native(upper_pred_path, native_gt, geom)
        if overwrite or not (out_dir / "upper_triptych.png").exists():
            fig = review_triptych_figure(
                dataset=dataset,
                sample_tag=Path(sample_id).stem,
                native_rgb=native_rgb,
                gt_map=native_gt,
                pred_map=upper_pred,
                palette=palette,
                mode="student",
            )
            save_figure(fig, out_dir / "upper_triptych.png")

    meta = {
        "sample_key": key,
        "sample_id": sample_id,
        "case_id": item.get("case_id"),
        "slice_idx": item.get("slice_idx"),
        "baseline_dice": None if pd.isna(row.get("baseline_dice")) else float(row["baseline_dice"]),
        "baseline_iou": None if pd.isna(row.get("baseline_iou")) else float(row["baseline_iou"]),
        "baseline_mae": None if pd.isna(row.get("baseline_mae")) else float(row["baseline_mae"]),
        "upper_dice": None if pd.isna(row.get("upper_dice")) else float(row["upper_dice"]),
        "upper_iou": None if pd.isna(row.get("upper_iou")) else float(row["upper_iou"]),
        "upper_mae": None if pd.isna(row.get("upper_mae")) else float(row["upper_mae"]),
        "dice_gap": None if pd.isna(row.get("dice_gap")) else float(row["dice_gap"]),
        "note": "No same-sample pseudo is available for test split under the current train-only pseudo protocol.",
    }
    save_json(meta, out_dir / "meta.json")


def dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def collect_case_slice_stats(
    fold_root: Path,
    manifest: Dict[str, Dict[str, Any]],
    geom_meta: Dict[str, Dict[str, Any]],
    files: Sequence[str],
    pred_dir: Path,
) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    for sample_id in files:
        item = manifest[sample_id]
        geom = geom_meta.get(sample_id)
        if geom is None or not (pred_dir / sample_id).exists():
            continue
        _, native_gt = load_native_image_and_gt(fold_root, item, geom)
        pred_native = load_student_pred_native(pred_dir / sample_id, native_gt, geom)
        stats.append(
            {
                "sample_id": sample_id,
                "slice_idx": int(item.get("slice_idx", 0)),
                "has_fg": bool((native_gt > 0).any()),
                "dice_fg": round(float(dice_fg(native_gt, pred_native)), 6),
            }
        )
    return stats


def select_summary_sample_ids(slice_stats: Sequence[Dict[str, Any]]) -> List[str]:
    positive = [row for row in slice_stats if row["has_fg"]]
    if not positive:
        return []
    positive = sorted(positive, key=lambda row: int(row["slice_idx"]))
    first_fg = positive[0]["sample_id"]
    mid_fg = positive[len(positive) // 2]["sample_id"]
    last_fg = positive[-1]["sample_id"]
    worst_fg = min(positive, key=lambda row: row["dice_fg"])["sample_id"]
    return dedupe_preserve([first_fg, mid_fg, last_fg, worst_fg])


def build_case_render_rows(
    dataset: str,
    fold_root: Path,
    manifest: Dict[str, Dict[str, Any]],
    geom_meta: Dict[str, Dict[str, Any]],
    sample_ids: Sequence[str],
    pred_dir: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample_id in sample_ids:
        item = manifest[sample_id]
        geom = geom_meta[sample_id]
        native_rgb, native_gt = load_native_image_and_gt(fold_root, item, geom)
        pred_native = load_student_pred_native(pred_dir / sample_id, native_gt, geom)
        rows.append(
            {
                "sample_tag": f"{str(item.get('case_id', 'case'))} | z={int(item.get('slice_idx', 0)):03d}",
                "sample_note": f"dice_fg={dice_fg(native_gt, pred_native):.4f}",
                "native_rgb": native_rgb,
                "gt_map": native_gt,
                "pred_map": pred_native,
            }
        )
    return rows


def render_student_3d_case(
    dataset: str,
    row: Any,
    out_dir: Path,
    fold_root: Path,
    manifest: Dict[str, Dict[str, Any]],
    geom_meta: Dict[str, Dict[str, Any]],
    case_lookup: Dict[str, List[str]],
    work_root: Path,
    overwrite: bool,
) -> None:
    ensure_dir(out_dir)
    case_id = str(row["key"])
    files = case_lookup.get(case_id, [])
    if not files:
        save_json({"case_id": case_id, "missing_render_source": True}, out_dir / "meta.json")
        return

    palette = palette_for_dataset(dataset)
    baseline_pred_dir = work_root / "baseline" / dataset / "fold_0" / "pred_test"
    upper_pred_dir = work_root / "upper" / dataset / "fold_0" / "pred_test"

    baseline_stats = collect_case_slice_stats(fold_root, manifest, geom_meta, files, baseline_pred_dir)
    upper_stats = collect_case_slice_stats(fold_root, manifest, geom_meta, files, upper_pred_dir)
    baseline_selected = select_summary_sample_ids(baseline_stats)
    upper_selected = select_summary_sample_ids(upper_stats)

    if baseline_selected and (overwrite or not (out_dir / "baseline_summary.png").exists()):
        rows = build_case_render_rows(dataset, fold_root, manifest, geom_meta, baseline_selected, baseline_pred_dir)
        fig = build_case_summary_figure(dataset, rows, palette)
        save_figure(fig, out_dir / "baseline_summary.png")
    if upper_selected and (overwrite or not (out_dir / "upper_summary.png").exists()):
        rows = build_case_render_rows(dataset, fold_root, manifest, geom_meta, upper_selected, upper_pred_dir)
        fig = build_case_summary_figure(dataset, rows, palette)
        save_figure(fig, out_dir / "upper_summary.png")

    meta = {
        "case_id": case_id,
        "baseline_dice": None if pd.isna(row.get("baseline_dice")) else float(row["baseline_dice"]),
        "baseline_hd95": None if pd.isna(row.get("baseline_hd95")) else float(row["baseline_hd95"]),
        "baseline_assd": None if pd.isna(row.get("baseline_assd")) else float(row["baseline_assd"]),
        "upper_dice": None if pd.isna(row.get("upper_dice")) else float(row["upper_dice"]),
        "upper_hd95": None if pd.isna(row.get("upper_hd95")) else float(row["upper_hd95"]),
        "upper_assd": None if pd.isna(row.get("upper_assd")) else float(row["upper_assd"]),
        "dice_gap": None if pd.isna(row.get("dice_gap")) else float(row["dice_gap"]),
        "baseline_selected_slices": baseline_selected,
        "upper_selected_slices": upper_selected,
        "note": "No same-case pseudo is available for test split under the current train-only pseudo protocol.",
    }
    save_json(meta, out_dir / "meta.json")


def screen_student_dataset(
    dataset: str,
    processed_root: Path,
    work_root: Path,
    review_root: Path,
    topk: int,
    overwrite: bool,
) -> None:
    if pd is None:
        raise RuntimeError("student screening requires pandas in the active environment")

    kind = infer_kind(dataset)
    eval_baseline = work_root / "baseline" / dataset / "fold_0" / ("eval_3d" if kind == "3d" else "eval_2d")
    eval_upper = work_root / "upper" / dataset / "fold_0" / ("eval_3d" if kind == "3d" else "eval_2d")
    fold_root = processed_root / dataset / "fold_0"
    if not fold_root.exists():
        raise FileNotFoundError(f"Missing fold_root: {fold_root}")

    baseline_rows = discover_metric_rows(eval_baseline, kind)
    upper_rows = discover_metric_rows(eval_upper, kind)
    merged = merge_mode_rows(baseline_rows, upper_rows, kind)
    buckets = classify_buckets(merged, kind)

    ds_root = review_root / "student" / dataset
    ensure_dir(ds_root)
    save_json(palette_jsonable(palette_for_dataset(dataset)), ds_root / "palette.json")

    manifest = load_manifest_index(fold_root)
    geom_meta = load_geometry_meta(fold_root)
    stem_lookup = build_manifest_stem_lookup(manifest, split="test")
    case_lookup = build_case_lookup(manifest, split="test")

    summary = {
        "dataset": dataset,
        "kind": kind,
        "num_baseline_rows": len(baseline_rows),
        "num_upper_rows": len(upper_rows),
        "num_merged": int(len(merged)),
        "buckets": {},
    }

    for bucket_name, df in buckets.items():
        out_bucket = ds_root / bucket_name
        ensure_dir(out_bucket)
        if len(df) == 0:
            write_dataframe_csv(df, out_bucket / "ranking.csv")
            save_json([], out_bucket / "ranking.json")
            summary["buckets"][bucket_name] = 0
            continue

        top_df = df.head(topk).copy()
        write_dataframe_csv(top_df, out_bucket / "ranking.csv")
        save_json(top_df.to_dict(orient="records"), out_bucket / "ranking.json")

        for _, row in top_df.iterrows():
            case_dir = out_bucket / str(row["key"])
            if kind == "2d":
                render_student_2d_case(
                    dataset=dataset,
                    row=row,
                    out_dir=case_dir,
                    fold_root=fold_root,
                    manifest=manifest,
                    geom_meta=geom_meta,
                    stem_lookup=stem_lookup,
                    work_root=work_root,
                    overwrite=overwrite,
                )
            else:
                render_student_3d_case(
                    dataset=dataset,
                    row=row,
                    out_dir=case_dir,
                    fold_root=fold_root,
                    manifest=manifest,
                    geom_meta=geom_meta,
                    case_lookup=case_lookup,
                    work_root=work_root,
                    overwrite=overwrite,
                )
        summary["buckets"][bucket_name] = int(len(top_df))

    save_json(summary, ds_root / "screen_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified teacher/student screening and bad-case visualization entrypoint.")
    parser.add_argument("--target", choices=["teacher", "student", "both"], default="both")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--review_root", default="/storage/baiyuting/data/MedSAM-main/data/vis")
    parser.add_argument("--output_root", default="")
    parser.add_argument("--processed_root", default="/storage/baiyuting/data/MedSAM-main/data/processed")
    parser.add_argument("--work_root", default="/storage/baiyuting/data/Swin-UMamba-main/work_dir")
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--preview_only", action="store_true")
    parser.add_argument("--preview_count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root)
    work_root = Path(args.work_root)
    review_root = Path(args.output_root or args.review_root)

    if args.target in {"teacher", "both"}:
        teacher_datasets = discover_teacher_datasets(processed_root, args.datasets)
        for dataset in teacher_datasets:
            print(f"[TEACHER] dataset={dataset}")
            try:
                screen_teacher_dataset(
                    dataset=dataset,
                    processed_root=processed_root,
                    review_root=review_root,
                    split=args.split,
                    overwrite=args.overwrite,
                    preview_only=args.preview_only,
                    preview_count=args.preview_count,
                )
            except FileNotFoundError as exc:
                print(f"[SKIP][TEACHER] dataset={dataset} reason={exc}")

    if args.target in {"student", "both"}:
        student_datasets = discover_student_datasets(work_root, args.datasets)
        for dataset in student_datasets:
            print(f"[STUDENT] dataset={dataset}")
            try:
                screen_student_dataset(
                    dataset=dataset,
                    processed_root=processed_root,
                    work_root=work_root,
                    review_root=review_root,
                    topk=args.topk,
                    overwrite=args.overwrite,
                )
            except FileNotFoundError as exc:
                print(f"[SKIP][STUDENT] dataset={dataset} reason={exc}")

    print(f"[DONE] screen_cases finished. review_root={review_root}")


if __name__ == "__main__":
    main()
