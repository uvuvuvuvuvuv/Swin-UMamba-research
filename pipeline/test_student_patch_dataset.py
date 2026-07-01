from pathlib import Path
from torch.utils.data import DataLoader
import argparse
import json
import os

from student_patch_dataset import build_student_patch_dataset


def resolve_crop_from_split_meta(fold_root: str):
    p = os.path.join(fold_root, "meta", "split_meta.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        h = meta.get("student_target_h")
        w = meta.get("student_target_w")
        if h is not None and w is not None:
            return int(h), int(w)
    raise FileNotFoundError(f"student_target_h/w not found in {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--student_pseudo_name",
        type=str,
        default="tri_train",
        help="Subdirectory name under pseudo_student/ used for baseline train labels.",
    )
    args = parser.parse_args()

    os.environ["STUDENT_PSEUDO_NAME"] = str(args.student_pseudo_name)

    dataset_name = args.dataset or Path(args.fold_root).parent.name
    print(f"[INFO] student_pseudo_name={args.student_pseudo_name}")
    print(f"[INFO] student_pseudo_rel_dir=pseudo_student/{args.student_pseudo_name}")
    crop_size = resolve_crop_from_split_meta(args.fold_root)

    for mode in ["baseline", "upper"]:
        print("=" * 80)
        print(f"[MODE] {mode}")
        ds = build_student_patch_dataset(
            fold_root=args.fold_root,
            dataset_name=dataset_name,
            mode=mode,
            split=args.split,
            crop_size=crop_size,
            fg_sample_prob=0.5,
            min_valid_ratio_for_baseline=0.10,
        )

        print("len(ds) =", len(ds))
        print("num_classes =", ds.num_classes)
        print("crop_size =", crop_size)

        loader = DataLoader(ds, batch_size=2, shuffle=(args.split == "train"), num_workers=0)
        batch = next(iter(loader))
        print("image:", batch["image"].shape, batch["image"].dtype)
        print("mask :", batch["mask"].shape, batch["mask"].dtype)
        print("slice_name:", batch["slice_name"][:2])

        import torch
        mask = batch["mask"]
        uniq = torch.unique(mask)
        print("mask unique:", uniq.tolist())
        if mode == "baseline" and args.split == "train":
            print("num_ignore_255 =", int((mask == 255).sum().item()))
        else:
            print("num_ignore_255 =", 0)
        if args.split == "test":
            print("orig_hw:", batch["orig_hw"][:2])


if __name__ == "__main__":
    main()
