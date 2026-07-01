#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
timer.py

What this version adds over the original timer.py:
1) stale-file detection for inferred stages
2) optional filtering so inferred stages only count files that belong to the current run window
3) paper-ready summary export

Important:
- train_* still uses exact elapsed_sec from train_log.csv when available
- non-training stages remain estimates unless your pipeline writes explicit stage_time.json files
- current-run filtering is heuristic; it greatly helps infer/eval pollution, but explicit stage logs are still the gold standard
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------- basic utils -------------------------- #

def sec_to_hms(seconds: Optional[float]) -> str:
    if seconds is None or (isinstance(seconds, float) and (math.isnan(seconds) or math.isinf(seconds))):
        return ""
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60.0
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_markdown_table(rows: List[Dict[str, Any]], headers: List[Tuple[str, str]], out_md: Path, intro: str = "") -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    if intro:
        lines.append(intro.strip())
        lines.append("")
    lines.append("| " + " | ".join(display for _, display in headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        vals = [str(r.get(key, "")).replace("|", r"\|") for key, _ in headers]
        lines.append("| " + " | ".join(vals) + " |")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def collect_files(paths: List[Path], recursive: bool = True) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if not p.exists():
            continue
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            if recursive:
                out.extend([x for x in p.rglob("*") if x.is_file()])
            else:
                out.extend([x for x in p.glob("*") if x.is_file()])
    return sorted(set(out))


def mtime_range(files: List[Path]) -> Tuple[Optional[float], Optional[float]]:
    if not files:
        return None, None
    ts = [x.stat().st_mtime for x in files if x.exists() and x.is_file()]
    if not ts:
        return None, None
    return min(ts), max(ts)


def filter_files_by_window(
    files: List[Path],
    min_ts: Optional[float] = None,
    max_ts: Optional[float] = None,
) -> Tuple[List[Path], List[Path]]:
    kept: List[Path] = []
    dropped: List[Path] = []
    for p in files:
        if not p.exists() or not p.is_file():
            continue
        ts = p.stat().st_mtime
        ok = True
        if min_ts is not None and ts < min_ts:
            ok = False
        if max_ts is not None and ts > max_ts:
            ok = False
        if ok:
            kept.append(p)
        else:
            dropped.append(p)
    return kept, dropped


def stage_record(
    stage_name: str,
    files: List[Path],
    exact_sec: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    min_ts: Optional[float] = None,
    max_ts: Optional[float] = None,
    restrict_to_current_run: bool = False,
) -> Optional[Dict[str, Any]]:
    all_start_ts, all_end_ts = mtime_range(files)

    used_files = files
    dropped_files: List[Path] = []
    if restrict_to_current_run and exact_sec is None and files:
        used_files, dropped_files = filter_files_by_window(files, min_ts=min_ts, max_ts=max_ts)
        if not used_files:
            # never silently drop everything; fall back to all files and mark the stage suspicious
            used_files = files

    start_ts, end_ts = mtime_range(used_files)
    if exact_sec is None:
        if start_ts is None or end_ts is None:
            return None
        duration_sec = float(max(0.0, end_ts - start_ts))
        duration_source = "filtered_file_mtime_estimate" if (restrict_to_current_run and (min_ts is not None or max_ts is not None)) else "file_mtime_estimate"
    else:
        duration_sec = float(max(0.0, exact_sec))
        duration_source = "exact_log"
        if end_ts is not None and start_ts is None:
            start_ts = end_ts - duration_sec
        elif start_ts is not None and end_ts is None:
            end_ts = start_ts + duration_sec
        elif start_ts is not None and end_ts is not None:
            start_ts = end_ts - duration_sec

    rec = {
        "stage": stage_name,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_sec": duration_sec,
        "duration_hms": sec_to_hms(duration_sec),
        "duration_source": duration_source,
        "num_files": len(used_files),
        "num_files_total": len(files),
        "num_files_filtered_out": len(dropped_files),
        "stale_files_detected": bool(dropped_files),
        "window_min_ts": min_ts,
        "window_max_ts": max_ts,
        "all_files_start_ts": all_start_ts,
        "all_files_end_ts": all_end_ts,
    }
    if extra:
        rec.update(extra)
    return rec


def read_train_log_csv(csv_path: Path) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty train_log.csv: {csv_path}")
    return rows


def load_train_config(cfg_path: Path) -> Dict[str, Any]:
    if not cfg_path.exists():
        return {}
    return load_json(cfg_path)

def load_explicit_stage_time(stage_json_path: Path, expected_stage_name: str) -> Optional[Dict[str, Any]]:
    if not stage_json_path.exists():
        return None

    obj = load_json(stage_json_path)

    if str(obj.get("stage_name", "")) != expected_stage_name:
        return None

    start_ts = obj.get("start_ts", None)
    end_ts = obj.get("end_ts", None)
    duration_sec = obj.get("duration_sec", None)

    if duration_sec is None and start_ts is not None and end_ts is not None:
        duration_sec = max(0.0, float(end_ts) - float(start_ts))

    if duration_sec is None:
        return None

    return {
        "stage": expected_stage_name,
        "start_ts": float(start_ts) if start_ts is not None else None,
        "end_ts": float(end_ts) if end_ts is not None else None,
        "duration_sec": float(duration_sec),
        "duration_hms": sec_to_hms(float(duration_sec)),
        "duration_source": "explicit_stage_json",
        "num_files": int(obj.get("num_outputs", 0)),
        "num_files_total": int(obj.get("num_outputs", 0)),
        "num_files_filtered_out": 0,
        "stale_files_detected": False,
        "window_min_ts": None,
        "window_max_ts": None,
        "all_files_start_ts": float(start_ts) if start_ts is not None else None,
        "all_files_end_ts": float(end_ts) if end_ts is not None else None,
        "status": obj.get("status", "success"),
        "notes": obj.get("notes", ""),
        "host": obj.get("host", ""),
        "pid": obj.get("pid", ""),
    }


def infer_dataset_name_from_fold_root(fold_root: Path) -> str:
    split_meta_path = fold_root / "meta" / "split_meta.json"
    if split_meta_path.exists():
        try:
            split_meta = load_json(split_meta_path)
            ds = split_meta.get("dataset_name") or split_meta.get("dataset")
            if ds:
                return str(ds)
        except Exception:
            pass
    return fold_root.parent.name


def infer_num_classes_from_split_meta(fold_root: Path) -> Any:
    split_meta_path = fold_root / "meta" / "split_meta.json"
    if not split_meta_path.exists():
        return ""
    try:
        sm = load_json(split_meta_path)
        if "num_classes" in sm:
            return sm["num_classes"]
        return 14 if sm.get("multi_class_preserved", False) else 2
    except Exception:
        return ""


def read_train_stage(run_dir: Path, stage_name: str) -> Optional[Dict[str, Any]]:
    explicit = load_explicit_stage_time(
        run_dir / f"stage_time_{stage_name}.json",
        stage_name,
    )
    if explicit is not None:
        return explicit

    log_path = run_dir / "train_log.csv"
    cfg_path = run_dir / "train_config.json"
    if not log_path.exists():
        return None


# -------------------------- stage scanners -------------------------- #

def scan_preprocess_stage(fold_root: Path) -> Optional[Dict[str, Any]]:
    explicit = load_explicit_stage_time(
        fold_root / "meta" / "stage_time_preprocess.json",
        "preprocess",
    )
    if explicit is not None:
        return explicit

    candidates = [
        fold_root / "meta" / "manifest.json",
        fold_root / "meta" / "geometry_meta.json",
        fold_root / "meta" / "split_meta.json",
        fold_root / "teacher_npy",
        fold_root / "student_npy",
    ]
    files = collect_files(candidates, recursive=True)
    return stage_record("preprocess", files)


def scan_prompt_stage(
    fold_root: Path,
    min_ts: Optional[float],
    tolerance_sec: float,
    restrict_to_current_run: bool,
) -> Optional[Dict[str, Any]]:
    explicit = load_explicit_stage_time(
        fold_root / "meta" / "stage_time_prompts.json",
        "prompts",
    )
    if explicit is not None:
        return explicit

    candidates = [
        fold_root / "prompts" / "prompts_train.json",
        fold_root / "prompts" / "prompts_test.json",
        fold_root / "prompts_train.json",
        fold_root / "prompts_test.json",
        fold_root / "meta" / "prompt_build_stats.json",
    ]
    files = collect_files(candidates, recursive=False)
    return stage_record(
        "prompts",
        files,
        min_ts=None if min_ts is None else (min_ts - tolerance_sec),
        restrict_to_current_run=restrict_to_current_run,
    )


def scan_pseudo_train_stage(
    fold_root: Path,
    min_ts: Optional[float],
    tolerance_sec: float,
    restrict_to_current_run: bool,
) -> Optional[Dict[str, Any]]:
    explicit = load_explicit_stage_time(
        fold_root / "meta" / "stage_time_pseudo_train.json",
        "pseudo_train",
    )
    if explicit is not None:
        return explicit

    candidates = [
        fold_root / "pseudo_teacher" / "tri_train",
        fold_root / "pseudo_teacher" / "vis_train",
        fold_root / "pseudo_student" / "tri_train",
        fold_root / "pseudo_student" / "vis_train",
        fold_root / "pseudo_v4_stats_train.json",
        fold_root / "pseudo_v3_stats_train.json",
        fold_root / "pseudo_v2_stats_train.json",
        fold_root / "meta" / "pseudo_box255_error_train.log",
        fold_root / "meta" / "pseudo_v4_inference_error_train.log",
        fold_root / "meta" / "pseudo_v3_inference_error_train.log",
        fold_root / "meta" / "pseudo_v2_inference_error_train.log",
    ]
    files = collect_files(candidates, recursive=True)
    return stage_record(
        "pseudo_train",
        files,
        min_ts=None if min_ts is None else (min_ts - tolerance_sec),
        restrict_to_current_run=restrict_to_current_run,
    )


def scan_infer_stage(
    run_dir: Path,
    stage_name: str,
    min_ts: Optional[float],
    tolerance_sec: float,
    restrict_to_current_run: bool,
) -> Optional[Dict[str, Any]]:
    explicit = load_explicit_stage_time(
        run_dir / f"stage_time_{stage_name}.json",
        stage_name,
    )
    if explicit is not None:
        return explicit

    pred_dirs = [
        run_dir / "pred_test_native",
        run_dir / "pred_test",
    ]
    candidates = pred_dirs + [run_dir / "infer_config.json"]
    files = collect_files(candidates, recursive=True)
    return stage_record(
        stage_name,
        files,
        min_ts=None if min_ts is None else (min_ts - tolerance_sec),
        restrict_to_current_run=restrict_to_current_run,
    )


def scan_eval_stage(
    run_dir: Path,
    stage_name: str,
    min_ts: Optional[float],
    tolerance_sec: float,
    restrict_to_current_run: bool,
) -> Optional[Dict[str, Any]]:
    explicit = load_explicit_stage_time(
        run_dir / f"stage_time_{stage_name}.json",
        stage_name,
    )
    if explicit is not None:
        return explicit

    eval_dirs = [x for x in run_dir.glob("eval*") if x.is_dir()]
    candidates = eval_dirs[:]
    for name in ["btc_eval_summary.json", "eval_summary.json", "eval_per_case.json", "per_case_macro.json"]:
        p = run_dir / name
        if p.exists():
            candidates.append(p)
    files = collect_files(candidates, recursive=True)
    return stage_record(
        stage_name,
        files,
        min_ts=None if min_ts is None else (min_ts - tolerance_sec),
        restrict_to_current_run=restrict_to_current_run,
    )


# -------------------------- aggregation -------------------------- #

STAGE_ORDER = [
    "preprocess",
    "prompts",
    "pseudo_train",
    "train_baseline",
    "infer_baseline",
    "eval_baseline",
    "train_upper",
    "infer_upper",
    "eval_upper",
]


def scan_run_branch(
    work_dir: Path,
    mode: str,
    dataset: str,
    fold: str,
    tolerance_sec: float,
    restrict_to_current_run: bool,
) -> Dict[str, Optional[Dict[str, Any]]]:
    run_dir = work_dir / mode / dataset / fold
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    if mode == "baseline":
        out["train_baseline"] = read_train_stage(run_dir, "train_baseline")
        infer_min_ts = out["train_baseline"]["end_ts"] if out["train_baseline"] else None
        out["infer_baseline"] = scan_infer_stage(
            run_dir,
            "infer_baseline",
            min_ts=infer_min_ts,
            tolerance_sec=tolerance_sec,
            restrict_to_current_run=restrict_to_current_run,
        )
        eval_min_ts = None
        if out["infer_baseline"] and out["infer_baseline"].get("end_ts") is not None:
            eval_min_ts = out["infer_baseline"]["end_ts"]
        elif out["train_baseline"]:
            eval_min_ts = out["train_baseline"]["end_ts"]
        out["eval_baseline"] = scan_eval_stage(
            run_dir,
            "eval_baseline",
            min_ts=eval_min_ts,
            tolerance_sec=tolerance_sec,
            restrict_to_current_run=restrict_to_current_run,
        )
    elif mode == "upper":
        out["train_upper"] = read_train_stage(run_dir, "train_upper")
        infer_min_ts = out["train_upper"]["end_ts"] if out["train_upper"] else None
        out["infer_upper"] = scan_infer_stage(
            run_dir,
            "infer_upper",
            min_ts=infer_min_ts,
            tolerance_sec=tolerance_sec,
            restrict_to_current_run=restrict_to_current_run,
        )
        eval_min_ts = None
        if out["infer_upper"] and out["infer_upper"].get("end_ts") is not None:
            eval_min_ts = out["infer_upper"]["end_ts"]
        elif out["train_upper"]:
            eval_min_ts = out["train_upper"]["end_ts"]
        out["eval_upper"] = scan_eval_stage(
            run_dir,
            "eval_upper",
            min_ts=eval_min_ts,
            tolerance_sec=tolerance_sec,
            restrict_to_current_run=restrict_to_current_run,
        )
    return out


def find_fold_roots(processed_root: Path, dataset_filter: List[str]) -> List[Path]:
    fold_roots: List[Path] = []
    dataset_dirs = [p for p in processed_root.iterdir() if p.is_dir()] if processed_root.exists() else []
    for ds_dir in dataset_dirs:
        ds_name = ds_dir.name
        if dataset_filter and ds_name not in dataset_filter:
            continue
        for fold_root in sorted(ds_dir.glob("fold_*")):
            if fold_root.is_dir():
                fold_roots.append(fold_root)
    return fold_roots


def compute_order_violations(stage_details: Dict[str, Dict[str, Any]], tolerance_sec: float) -> List[str]:
    msgs: List[str] = []
    pairs = [
        ("preprocess", "prompts"),
        ("prompts", "pseudo_train"),
        ("train_baseline", "infer_baseline"),
        ("infer_baseline", "eval_baseline"),
        ("train_upper", "infer_upper"),
        ("infer_upper", "eval_upper"),
    ]
    for prev_stage, cur_stage in pairs:
        prev_rec = stage_details.get(prev_stage)
        cur_rec = stage_details.get(cur_stage)
        if not prev_rec or not cur_rec:
            continue
        prev_end = prev_rec.get("end_ts")
        cur_start = cur_rec.get("start_ts")
        if prev_end is None or cur_start is None:
            continue
        if float(cur_start) < float(prev_end) - tolerance_sec:
            msgs.append(f"{cur_stage} starts earlier than {prev_stage} ends")
    return msgs


def aggregate_dataset_fold(
    fold_root: Path,
    work_dir: Path,
    tolerance_sec: float,
    restrict_to_current_run: bool,
) -> Dict[str, Any]:
    dataset = infer_dataset_name_from_fold_root(fold_root)
    fold = fold_root.name

    preprocess = scan_preprocess_stage(fold_root)
    prompts_min_ts = preprocess["end_ts"] if preprocess else None
    prompts = scan_prompt_stage(fold_root, min_ts=prompts_min_ts, tolerance_sec=tolerance_sec, restrict_to_current_run=restrict_to_current_run)
    pseudo_min_ts = prompts["end_ts"] if prompts else None
    pseudo_train = scan_pseudo_train_stage(fold_root, min_ts=pseudo_min_ts, tolerance_sec=tolerance_sec, restrict_to_current_run=restrict_to_current_run)

    stages: Dict[str, Optional[Dict[str, Any]]] = {
        "preprocess": preprocess,
        "prompts": prompts,
        "pseudo_train": pseudo_train,
    }
    stages.update(scan_run_branch(work_dir, "baseline", dataset, fold, tolerance_sec, restrict_to_current_run))
    stages.update(scan_run_branch(work_dir, "upper", dataset, fold, tolerance_sec, restrict_to_current_run))

    row: Dict[str, Any] = {
        "dataset": dataset,
        "fold": fold,
        "fold_root": str(fold_root),
        "num_classes_hint": infer_num_classes_from_split_meta(fold_root),
    }

    start_candidates: List[float] = []
    end_candidates: List[float] = []
    total_stage_sum_sec = 0.0
    exact_stage_sum_sec = 0.0
    estimated_stage_sum_sec = 0.0
    stale_stage_count = 0

    for stage_name in STAGE_ORDER:
        rec = stages.get(stage_name)
        exists = rec is not None
        row[f"{stage_name}_exists"] = bool(exists)
        if not exists:
            row[f"{stage_name}_duration_sec"] = ""
            row[f"{stage_name}_duration_hms"] = ""
            row[f"{stage_name}_duration_source"] = ""
            row[f"{stage_name}_num_files"] = ""
            row[f"{stage_name}_stale"] = ""
            continue

        dur = float(rec["duration_sec"])
        row[f"{stage_name}_duration_sec"] = round(dur, 6)
        row[f"{stage_name}_duration_hms"] = rec["duration_hms"]
        row[f"{stage_name}_duration_source"] = rec["duration_source"]
        row[f"{stage_name}_num_files"] = rec["num_files"]
        row[f"{stage_name}_stale"] = bool(rec.get("stale_files_detected", False))

        if rec.get("stale_files_detected", False):
            stale_stage_count += 1

        if rec.get("start_ts") is not None:
            start_candidates.append(float(rec["start_ts"]))
        if rec.get("end_ts") is not None:
            end_candidates.append(float(rec["end_ts"]))

        total_stage_sum_sec += dur
        if rec["duration_source"] == "exact_log":
            exact_stage_sum_sec += dur
        else:
            estimated_stage_sum_sec += dur

    wall_clock_span_sec = ""
    if start_candidates and end_candidates:
        wall_clock_span_sec = round(max(end_candidates) - min(start_candidates), 6)

    row["total_stage_sum_sec"] = round(total_stage_sum_sec, 6)
    row["total_stage_sum_hms"] = sec_to_hms(total_stage_sum_sec)
    row["exact_stage_sum_sec"] = round(exact_stage_sum_sec, 6)
    row["exact_stage_sum_hms"] = sec_to_hms(exact_stage_sum_sec)
    row["estimated_stage_sum_sec"] = round(estimated_stage_sum_sec, 6)
    row["estimated_stage_sum_hms"] = sec_to_hms(estimated_stage_sum_sec)
    row["wall_clock_span_sec"] = wall_clock_span_sec
    row["wall_clock_span_hms"] = sec_to_hms(float(wall_clock_span_sec)) if wall_clock_span_sec != "" else ""

    stage_details = {k: v for k, v in stages.items() if v is not None}
    order_violations = compute_order_violations(stage_details, tolerance_sec=tolerance_sec)

    row["stale_stage_count"] = stale_stage_count
    row["stage_order_violation_count"] = len(order_violations)
    row["suspicious_timing_detected"] = bool(stale_stage_count or order_violations)
    row["_stage_details"] = stage_details
    row["_order_violations"] = order_violations
    return row


def aggregate_per_dataset(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        ds = r["dataset"]
        b = buckets.setdefault(ds, {
            "dataset": ds,
            "num_folds": 0,
            "folds": [],
            "total_stage_sum_sec": 0.0,
            "exact_stage_sum_sec": 0.0,
            "estimated_stage_sum_sec": 0.0,
            "stale_stage_count": 0,
            "suspicious_timing_detected": False,
            "wall_clock_earliest": None,
            "wall_clock_latest": None,
        })
        b["num_folds"] += 1
        b["folds"].append(r["fold"])
        b["total_stage_sum_sec"] += float(r["total_stage_sum_sec"])
        b["exact_stage_sum_sec"] += float(r["exact_stage_sum_sec"])
        b["estimated_stage_sum_sec"] += float(r["estimated_stage_sum_sec"])
        b["stale_stage_count"] += int(r.get("stale_stage_count", 0))
        b["suspicious_timing_detected"] = b["suspicious_timing_detected"] or bool(r.get("suspicious_timing_detected", False))

        detail = r.get("_stage_details", {})
        for rec in detail.values():
            st = rec.get("start_ts")
            ed = rec.get("end_ts")
            if st is not None:
                b["wall_clock_earliest"] = st if b["wall_clock_earliest"] is None else min(b["wall_clock_earliest"], st)
            if ed is not None:
                b["wall_clock_latest"] = ed if b["wall_clock_latest"] is None else max(b["wall_clock_latest"], ed)

    out: List[Dict[str, Any]] = []
    for ds, b in sorted(buckets.items()):
        span = ""
        if b["wall_clock_earliest"] is not None and b["wall_clock_latest"] is not None:
            span = round(float(b["wall_clock_latest"]) - float(b["wall_clock_earliest"]), 6)

        out.append({
            "dataset": ds,
            "num_folds": b["num_folds"],
            "folds": ",".join(sorted(b["folds"])),
            "total_stage_sum_sec": round(b["total_stage_sum_sec"], 6),
            "total_stage_sum_hms": sec_to_hms(b["total_stage_sum_sec"]),
            "exact_stage_sum_sec": round(b["exact_stage_sum_sec"], 6),
            "exact_stage_sum_hms": sec_to_hms(b["exact_stage_sum_sec"]),
            "estimated_stage_sum_sec": round(b["estimated_stage_sum_sec"], 6),
            "estimated_stage_sum_hms": sec_to_hms(b["estimated_stage_sum_sec"]),
            "wall_clock_span_sec": span,
            "wall_clock_span_hms": sec_to_hms(float(span)) if span != "" else "",
            "stale_stage_count": b["stale_stage_count"],
            "suspicious_timing_detected": b["suspicious_timing_detected"],
        })
    return out


def build_paper_ready_rows(fold_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in fold_rows:
        flagged_stages = []
        for stage_name in ["infer_baseline", "eval_baseline", "infer_upper", "eval_upper", "pseudo_train", "prompts", "preprocess"]:
            if r.get(f"{stage_name}_stale", False):
                flagged_stages.append(stage_name)
        rows.append({
            "dataset": r["dataset"],
            "fold": r["fold"],
            "baseline_train_hms": r.get("train_baseline_duration_hms", ""),
            "upper_train_hms": r.get("train_upper_duration_hms", ""),
            "total_training_exact_hms": r.get("exact_stage_sum_hms", ""),
            "coarse_pipeline_span_hms": r.get("wall_clock_span_hms", ""),
            "nontraining_timing_status": "stale_detected" if flagged_stages or r.get("stage_order_violation_count", 0) else "coarse_reference",
            "flagged_stages": ",".join(flagged_stages),
        })
    return rows


def printable_table(rows: List[Dict[str, Any]]) -> None:
    print("=" * 220)
    print(
        f"{'dataset':<16} {'fold':<8} "
        f"{'preprocess':<12} {'prompts':<12} {'pseudo':<12} "
        f"{'train(base)':<14} {'infer(base)':<14} {'eval(base)':<14} "
        f"{'train(upper)':<14} {'infer(upper)':<14} {'eval(upper)':<14} "
        f"{'sum':<12} {'wall':<12} {'stale':<7} {'viol':<6}"
    )
    print("=" * 220)
    for r in rows:
        print(
            f"{r['dataset']:<16} {r['fold']:<8} "
            f"{str(r.get('preprocess_duration_hms','')):<12} "
            f"{str(r.get('prompts_duration_hms','')):<12} "
            f"{str(r.get('pseudo_train_duration_hms','')):<12} "
            f"{str(r.get('train_baseline_duration_hms','')):<14} "
            f"{str(r.get('infer_baseline_duration_hms','')):<14} "
            f"{str(r.get('eval_baseline_duration_hms','')):<14} "
            f"{str(r.get('train_upper_duration_hms','')):<14} "
            f"{str(r.get('infer_upper_duration_hms','')):<14} "
            f"{str(r.get('eval_upper_duration_hms','')):<14} "
            f"{str(r.get('total_stage_sum_hms','')):<12} "
            f"{str(r.get('wall_clock_span_hms','')):<12} "
            f"{str(r.get('stale_stage_count','')):<7} "
            f"{str(r.get('stage_order_violation_count','')):<6}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", type=str, default="/storage/baiyuting/data/MedSAM-main/data/processed")
    parser.add_argument("--work_dir", type=str, default="/storage/baiyuting/data/Swin-UMamba-main/work_dir")
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--mtime_tolerance_sec", type=float, default=120.0, help="Time tolerance when filtering current-run files.")
    parser.add_argument("--disable_current_run_filter", action="store_true", help="Turn off mtime window filtering for inferred stages.")
    parser.add_argument("--save_csv", type=str, default="")
    parser.add_argument("--save_dataset_csv", type=str, default="")
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument("--save_paper_csv", type=str, default="")
    parser.add_argument("--save_paper_md", type=str, default="")
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    work_dir = Path(args.work_dir)
    restrict_to_current_run = not args.disable_current_run_filter

    dataset_filter: List[str] = []
    if args.datasets.strip().lower() != "all":
        dataset_filter = [x.strip() for x in args.datasets.split(",") if x.strip()]

    fold_roots = find_fold_roots(processed_root, dataset_filter)
    if not fold_roots:
        print("[WARN] No fold roots found.")
        return

    fold_rows = [
        aggregate_dataset_fold(
            fr,
            work_dir,
            tolerance_sec=args.mtime_tolerance_sec,
            restrict_to_current_run=restrict_to_current_run,
        )
        for fr in fold_roots
    ]
    fold_rows_sorted = sorted(fold_rows, key=lambda x: (x["dataset"], x["fold"]))
    printable_table(fold_rows_sorted)

    dataset_rows = aggregate_per_dataset(fold_rows_sorted)
    paper_rows = build_paper_ready_rows(fold_rows_sorted)

    print("\n" + "=" * 120)
    print(f"{'dataset':<16} {'num_folds':<10} {'train_exact':<14} {'wall':<14} {'stale':<8} {'suspicious':<12}")
    print("=" * 120)
    for r in dataset_rows:
        print(
            f"{r['dataset']:<16} {str(r['num_folds']):<10} "
            f"{r['exact_stage_sum_hms']:<14} {r['wall_clock_span_hms']:<14} "
            f"{str(r['stale_stage_count']):<8} {str(r['suspicious_timing_detected']):<12}"
        )

    if args.save_csv:
        csv_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in fold_rows_sorted]
        save_csv(csv_rows, Path(args.save_csv))
        print(f"\n[DONE] saved fold csv -> {args.save_csv}")

    if args.save_dataset_csv:
        save_csv(dataset_rows, Path(args.save_dataset_csv))
        print(f"[DONE] saved dataset csv -> {args.save_dataset_csv}")

    if args.save_json:
        payload = {
            "processed_root": str(processed_root),
            "work_dir": str(work_dir),
            "mtime_tolerance_sec": args.mtime_tolerance_sec,
            "restrict_to_current_run": restrict_to_current_run,
            "fold_rows": fold_rows_sorted,
            "dataset_rows": dataset_rows,
            "paper_rows": paper_rows,
        }
        save_json(payload, Path(args.save_json))
        print(f"[DONE] saved json -> {args.save_json}")

    if args.save_paper_csv:
        save_csv(paper_rows, Path(args.save_paper_csv))
        print(f"[DONE] saved paper csv -> {args.save_paper_csv}")

    if args.save_paper_md:
        save_markdown_table(
            paper_rows,
            headers=[
                ("dataset", "Dataset"),
                ("fold", "Fold"),
                ("baseline_train_hms", "Baseline train"),
                ("upper_train_hms", "Upper train"),
                ("total_training_exact_hms", "Training total (exact)"),
                ("coarse_pipeline_span_hms", "Pipeline span (coarse)"),
                ("nontraining_timing_status", "Status"),
            ],
            out_md=Path(args.save_paper_md),
            intro=(
                "# Paper-ready timing table\n\n"
                "Use **Training total (exact)** as the formal timing metric. "
                "Use **Pipeline span (coarse)** only as a coarse workflow-span reference, "
                "because non-training stages are still mtime-based unless explicit stage logs are available."
            ),
        )
        print(f"[DONE] saved paper markdown -> {args.save_paper_md}")


if __name__ == "__main__":
    main()
