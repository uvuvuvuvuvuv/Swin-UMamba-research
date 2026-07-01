#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

THREE_D_DATASETS = {"btcv", "synapse", "acdc", "prostate158"}
TWO_D_DATASETS = {
    "kvasirseg", "cvc_clinicdb", "tn3k", "tg3k", "ddti", "otu_2d",
    "drive", "ph2",
}
POLYP_DATASETS = {"kvasirseg", "cvc_clinicdb"}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_dataset_name(fold_root: str) -> str:
    p = os.path.join(fold_root, "meta", "split_meta.json")
    if os.path.exists(p):
        meta = load_json(p)
        ds = meta.get("dataset_name") or meta.get("dataset")
        if ds:
            return str(ds)
    return Path(fold_root).parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch to protocol-aligned 2D or 3D evaluator by dataset name.")
    parser.add_argument("--fold_root", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--require_native_gt", action="store_true")
    parser.add_argument("--allow_resize_pred_to_native_for_debug", action="store_true")
    parser.add_argument("--save_case_volumes", action="store_true")
    parser.add_argument("--eval_2d_script", type=str, default="eval_2d.py")
    parser.add_argument("--eval_3d_script", type=str, default="eval_3d.py")
    args = parser.parse_args()

    dataset = infer_dataset_name(args.fold_root)
    common = [
        "--fold_root", args.fold_root,
        "--pred_dir", args.pred_dir,
        "--save_dir", args.save_dir,
        "--split", args.split,
    ]
    if args.require_native_gt:
        common.append("--require_native_gt")
    if args.allow_resize_pred_to_native_for_debug:
        common.append("--allow_resize_pred_to_native_for_debug")

    if dataset in THREE_D_DATASETS:
        cmd = [sys.executable, args.eval_3d_script, *common]
        if args.num_classes > 0:
            cmd += ["--num_classes", str(args.num_classes)]
        if args.save_case_volumes:
            cmd.append("--save_case_volumes")
    elif dataset in TWO_D_DATASETS:
        cmd = [sys.executable, args.eval_2d_script, *common]
        if args.num_classes > 0:
            cmd += ["--num_classes", str(args.num_classes)]
        if dataset in POLYP_DATASETS:
            cmd.append("--report_mae")
    else:
        raise KeyError(f"Unknown dataset group for dataset={dataset}. Please extend dispatcher mapping.")

    print("[DISPATCH] dataset=", dataset)
    print("[DISPATCH] cmd=", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
