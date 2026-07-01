#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_KEYS = ["mode", "dataset", "fold", "task_type"]
DEFAULT_METRICS = [
    "mean_dice",
    "mean_iou",
    "mean_mae",
    "mean_hd95_mm",
    "mean_assd_mm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare eval summary tables before and after rerun.")
    parser.add_argument("--before_csv", required=True, help="CSV snapshot before rerun.")
    parser.add_argument("--after_csv", required=True, help="CSV snapshot after rerun.")
    parser.add_argument("--out_csv", required=True, help="Merged comparison CSV output path.")
    parser.add_argument("--out_md", default="", help="Optional markdown summary output path.")
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset whitelist. Empty means keep all rows found in both tables.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_numeric_if_possible(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def format_metric(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    before = pd.read_csv(args.before_csv)
    after = pd.read_csv(args.after_csv)

    if args.datasets.strip():
        keep = {x.strip() for x in args.datasets.split(",") if x.strip()}
        before = before[before["dataset"].isin(keep)].copy()
        after = after[after["dataset"].isin(keep)].copy()

    keep_cols = DEFAULT_KEYS + [c for c in DEFAULT_METRICS if c in before.columns or c in after.columns]
    before = before[[c for c in keep_cols if c in before.columns]].copy()
    after = after[[c for c in keep_cols if c in after.columns]].copy()

    to_numeric_if_possible(before, DEFAULT_METRICS)
    to_numeric_if_possible(after, DEFAULT_METRICS)

    merged = before.merge(after, on=DEFAULT_KEYS, how="outer", suffixes=("_before", "_after"), indicator=True)

    for metric in DEFAULT_METRICS:
        col_before = f"{metric}_before"
        col_after = f"{metric}_after"
        if col_before in merged.columns and col_after in merged.columns:
            merged[f"{metric}_delta"] = merged[col_after] - merged[col_before]

    sort_cols = [c for c in ["task_type", "dataset", "mode", "fold"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    out_csv = Path(args.out_csv)
    ensure_parent(out_csv)
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] comparison csv -> {out_csv}")

    if args.out_md:
        out_md = Path(args.out_md)
        ensure_parent(out_md)
        lines: List[str] = []
        lines.append("# Eval Snapshot Comparison")
        lines.append("")
        lines.append(f"- before: `{args.before_csv}`")
        lines.append(f"- after: `{args.after_csv}`")
        lines.append("")

        for task_type in ["2D", "3D"]:
            block = merged[merged["task_type"] == task_type].copy() if "task_type" in merged.columns else merged.copy()
            if block.empty:
                continue
            lines.append(f"## {task_type}")
            lines.append("")
            lines.append("| mode | dataset | mean_dice_before | mean_dice_after | mean_dice_delta | mean_iou_delta | mean_mae_delta |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
            for _, row in block.iterrows():
                lines.append(
                    "| {mode} | {dataset} | {dice_b} | {dice_a} | {dice_d} | {iou_d} | {mae_d} |".format(
                        mode=row.get("mode", "-"),
                        dataset=row.get("dataset", "-"),
                        dice_b=format_metric(row.get("mean_dice_before")),
                        dice_a=format_metric(row.get("mean_dice_after")),
                        dice_d=format_metric(row.get("mean_dice_delta")),
                        iou_d=format_metric(row.get("mean_iou_delta")),
                        mae_d=format_metric(row.get("mean_mae_delta")),
                    )
                )
            lines.append("")

        out_md.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] comparison md  -> {out_md}")


if __name__ == "__main__":
    main()
