import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple
from stage_timer_utils import StageTimer

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_SWIN_UMAMBA_PKG_ROOT = _PROJECT_ROOT / "swin_umamba"
if str(_SWIN_UMAMBA_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_SWIN_UMAMBA_PKG_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from student_patch_dataset import build_student_patch_dataset
from train_student import build_model


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def try_get(d: Dict[str, Any], keys: Sequence[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def infer_dataset_name(fold_root: str, dataset_arg: str | None) -> str:
    if dataset_arg:
        return dataset_arg
    split_meta_path = os.path.join(fold_root, "meta", "split_meta.json")
    if os.path.exists(split_meta_path):
        split_meta = load_json(split_meta_path)
        ds = split_meta.get("dataset_name") or split_meta.get("dataset")
        if ds:
            return str(ds)
    return Path(fold_root).parent.name


def pad_to_multiple(x: torch.Tensor, multiple: int = 32):
    _, _, h, w = x.shape
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    out = torch.zeros((1, x.shape[1], new_h, new_w), dtype=x.dtype, device=x.device)
    out[:, :, :h, :w] = x
    return out, (h, w)


def infer_native_hw(rec: Dict[str, Any], geom: Dict[str, Any] | None) -> Tuple[int, int]:
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


def restore_pred_to_native(pred_2d: np.ndarray, rec: Dict[str, Any], geom: Dict[str, Any] | None) -> np.ndarray:
    native_h, native_w = infer_native_hw(rec, geom)
    if pred_2d.shape == (native_h, native_w):
        return pred_2d
    if geom is not None:
        inv = try_get(geom, ["student_to_native"], None)
        if isinstance(inv, dict):
            restored = restore_with_inverse_geometry(pred_2d, inv)
            if restored.shape == (native_h, native_w):
                return restored
    raise ValueError(f"Prediction shape {pred_2d.shape} cannot be restored to native {(native_h, native_w)}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--mode", type=str, required=True, choices=["upper", "baseline"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--save_student_space", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deep_supervision", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and (device.type == "cuda")
    dataset_name = infer_dataset_name(args.fold_root, args.dataset.strip() or None)
    fold_name = Path(args.fold_root).name
    run_root = str(_PROJECT_ROOT / "work_dir" / args.mode / dataset_name / fold_name)
    stage_name = f"infer_{args.mode}"
    stage_time_path = os.path.join(run_root, f"stage_time_{stage_name}.json")
    if len(args.out_dir) == 0:
        args.out_dir = os.path.join(run_root, "pred_test_native")
    ensure_dir(args.out_dir)
    student_out_dir = os.path.join(run_root, "pred_test_student") if args.save_student_space else ""
    if args.save_student_space:
        ensure_dir(student_out_dir)

    with StageTimer(
            save_path=stage_time_path,
            stage_name=stage_name,
            dataset=dataset_name,
            fold=fold_name,
            mode=args.mode,
            split="test",
    ) as timer:
        manifest_path = os.path.join(args.fold_root, "meta", "manifest.json")
        geometry_path = os.path.join(args.fold_root, "meta", "geometry_meta.json")
        manifest = load_json(manifest_path)
        geometry_meta = load_json(geometry_path) if os.path.exists(geometry_path) else {}
        rec_map = {rec["slice_name"]: rec for rec in manifest if rec.get("split") == "test"}

        test_ds = build_student_patch_dataset(
            fold_root=args.fold_root,
            dataset_name=dataset_name,
            mode=args.mode,
            split="test",
            crop_size=None,
            test_resize=None,
        )

        num_classes = test_ds.num_classes
        print(f"[INFO] dataset={dataset_name}, mode={args.mode}, test_samples={len(test_ds)}")
        print(f"[INFO] num_classes={num_classes}")
        print(f"[INFO] ckpt={args.ckpt}")
        print(f"[INFO] out_dir(native)={args.out_dir}")
        if args.save_student_space:
            print(f"[INFO] out_dir(student)={student_out_dir}")

        model = build_model(
            num_classes=num_classes,
            in_chans=3,
            deep_supervision=args.deep_supervision,
            pretrained_ckpt=None,
        ).to(device)

        ckpt = torch.load(args.ckpt, map_location=device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        saved_count = 0

        for i in range(len(test_ds)):
            sample = test_ds[i]
            image = sample["image"].unsqueeze(0).to(device)
            orig_h, orig_w = map(int, sample["orig_hw"].tolist())
            slice_name = sample["slice_name"]
            rec = rec_map[slice_name]
            geom = geometry_meta.get(rec.get("geometry_key", slice_name), None)

            image_pad, (h0, w0) = pad_to_multiple(image, multiple=32)
            with autocast(enabled=amp):
                logits = model(image_pad)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
            logits = logits[:, :, :h0, :w0]
            pred_student = torch.argmax(logits, dim=1)[0]
            pred_student = pred_student[:orig_h, :orig_w].detach().cpu().numpy().astype(np.uint8)

            if args.save_student_space:
                np.save(os.path.join(student_out_dir, slice_name), pred_student)

            pred_native = restore_pred_to_native(pred_student, rec, geom)
            np.save(os.path.join(args.out_dir, slice_name), pred_native.astype(np.uint8))

            saved_count += 1
            timer.set_outputs(saved_count)

            if (i + 1) % 20 == 0 or (i + 1) == len(test_ds):
                print(f"[{i + 1}/{len(test_ds)}] saved -> {os.path.join(args.out_dir, slice_name)}")

        save_json(
            {
                "dataset": dataset_name,
                "mode": args.mode,
                "fold_root": args.fold_root,
                "ckpt": args.ckpt,
                "num_test_samples": len(test_ds),
                "native_pred_dir": args.out_dir,
                "student_pred_dir": student_out_dir if args.save_student_space else None,
            },
            os.path.join(run_root, "infer_config.json"),
        )

        timer.set_outputs(saved_count)
        print("[DONE] inference finished.")


if __name__ == "__main__":
    main()
