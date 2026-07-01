import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from stage_timer_utils import StageTimer

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_SWIN_UMAMBA_PKG_ROOT = _PROJECT_ROOT / "swin_umamba"
if str(_SWIN_UMAMBA_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_SWIN_UMAMBA_PKG_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from student_patch_dataset import build_student_patch_dataset
from nnunetv2.nets.SwinUMambaD import SwinUMambaD, load_pretrained_ckpt
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

try:
    from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
    from nnunetv2.utilities.network_initialization import InitWeights_He
    HAS_NNUNET_INIT = True
except Exception:
    HAS_NNUNET_INIT = False


DATASET_CROP_PRESETS = {
    # Fallbacks only. Preferred source of truth: fold_root/meta/split_meta.json
    "kvasirseg": (352, 352),
    "cvc_clinicdb": (352, 352),
    "tn3k": (256, 256),
    "tg3k": (256, 256),
    "ddti": (256, 256),
    "otu_2d": (256, 256),
    "monuseg": (512, 512),
    "ph2": (256, 256),
    "btcv": (512, 512),
    "synapse": (512, 512),
    "acdc": (320, 320),
    "prostate158": (320, 320),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_history_csv(path: str, max_epoch: int | None = None) -> list[Dict]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read history csv {path}: {e}")
        return []

    rows: list[Dict] = []
    for _, row in df.iterrows():
        try:
            epoch = int(row["epoch"])
        except Exception:
            continue
        if max_epoch is not None and epoch > int(max_epoch):
            continue
        rows.append({
            "epoch": int(epoch),
            "train_loss": float(row["train_loss"]),
            "lr": float(row["lr"]),
            "freeze_encoder": int(row["freeze_encoder"]),
            "elapsed_sec": float(row["elapsed_sec"]),
        })
    return rows


def resolve_crop_size(fold_root: str, dataset_name: str, crop_h: int | None, crop_w: int | None) -> Tuple[int, int]:
    if crop_h is not None and crop_w is not None:
        return int(crop_h), int(crop_w)

    split_meta_path = os.path.join(fold_root, "meta", "split_meta.json")
    if os.path.exists(split_meta_path):
        split_meta = load_json(split_meta_path)
        h = split_meta.get("student_target_h")
        w = split_meta.get("student_target_w")
        if h is not None and w is not None:
            return int(h), int(w)

    if dataset_name not in DATASET_CROP_PRESETS:
        raise KeyError(
            f"No crop preset for dataset={dataset_name}, and split_meta.json does not provide student_target_h/w."
        )
    return tuple(int(x) for x in DATASET_CROP_PRESETS[dataset_name])


def build_model(
    num_classes: int,
    in_chans: int = 3,
    deep_supervision: bool = False,
    pretrained_ckpt: str | None = None,
) -> torch.nn.Module:
    vss_args = dict(
        in_chans=in_chans,
        patch_size=4,
        depths=[2, 2, 9, 2],
        dims=96,
        drop_path_rate=0.2,
    )
    decoder_args = dict(
        num_classes=num_classes,
        deep_supervision=deep_supervision,
        features_per_stage=[96, 192, 384, 768],
        drop_path_rate=0.2,
        d_state=16,
    )
    model = SwinUMambaD(vss_args, decoder_args)
    if HAS_NNUNET_INIT:
        model.apply(InitWeights_He(1e-2))
        model.apply(init_last_bn_before_add_to_0)

    if pretrained_ckpt:
        if not os.path.exists(pretrained_ckpt):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_ckpt}")
        model = load_pretrained_ckpt(
            model,
            num_input_channels=in_chans,
            ckpt_path=pretrained_ckpt,
        )
    return model


def build_loss(mode: str, device: torch.device) -> torch.nn.Module:
    ignore_label = 255 if mode == "baseline" else None
    loss = DC_and_CE_loss(
        soft_dice_kwargs={
            "batch_dice": True,
            "smooth": 1e-5,
            "do_bg": False,
            "ddp": False,
        },
        ce_kwargs={},
        weight_ce=1.0,
        weight_dice=1.0,
        ignore_label=ignore_label,
        dice_class=MemoryEfficientSoftDiceLoss,
    )
    return loss.to(device)


def maybe_freeze_encoder(model: torch.nn.Module, freeze: bool) -> None:
    if freeze:
        model.freeze_encoder()
    else:
        model.unfreeze_encoder()


def _resize_target_for_logits(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if target.shape[-2:] == logits.shape[-2:]:
        return target
    target_rs = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
    return target_rs.long()


def compute_ds_loss(outputs, target, criterion):
    if not isinstance(outputs, (list, tuple)):
        return criterion(outputs, target)
    weights = [1.0 / (2 ** i) for i in range(len(outputs))]
    s = sum(weights)
    weights = [w / s for w in weights]
    total = 0.0
    for w, out in zip(weights, outputs):
        target_i = _resize_target_for_logits(target, out)
        total = total + w * criterion(out, target_i)
    return total


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    amp,
    grad_clip,
) -> float:
    model.train()
    loss_meter = 0.0
    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        images = batch["image"]
        masks = batch["mask"].unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            outputs = model(images)
            loss = compute_ds_loss(outputs, masks, criterion)

        scaler.scale(loss).backward()
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_meter += float(loss.detach().item())
        num_batches += 1

    return loss_meter / max(1, num_batches)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--mode", type=str, required=True, choices=["upper", "baseline"])
    parser.add_argument(
        "--student_pseudo_name",
        type=str,
        default="tri_train",
        help="Subdirectory name under pseudo_student/ used for baseline train labels. "
             "Default keeps frozen baseline behavior: pseudo_student/tri_train.",
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--freeze_encoder_epochs", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deep_supervision", action="store_true")

    parser.add_argument("--crop_h", type=int, default=None)
    parser.add_argument("--crop_w", type=int, default=None)
    parser.add_argument("--fg_sample_prob", type=float, default=0.5)
    parser.add_argument("--min_valid_ratio_for_baseline", type=float, default=0.10)

    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--save_every", type=int, default=20)

    args = parser.parse_args()

    # Keep frozen baseline default behavior, while allowing IdeaX experiments
    # to explicitly choose pseudo_student/<student_pseudo_name>.
    os.environ["STUDENT_PSEUDO_NAME"] = str(args.student_pseudo_name)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and (device.type == "cuda")

    dataset_name = infer_dataset_name(args.fold_root, args.dataset.strip() or None)
    crop_size = resolve_crop_size(
        fold_root=args.fold_root,
        dataset_name=dataset_name,
        crop_h=args.crop_h,
        crop_w=args.crop_w,
    )

    if not args.out_dir:
        args.out_dir = str(_PROJECT_ROOT / "work_dir" / args.mode / dataset_name / Path(args.fold_root).name)
    ensure_dir(args.out_dir)

    fold_name = Path(args.fold_root).name
    stage_name = f"train_{args.mode}"
    stage_time_path = os.path.join(args.out_dir, f"stage_time_{stage_name}.json")

    run_cfg = dict(vars(args))
    run_cfg["dataset"] = dataset_name
    run_cfg["student_pseudo_rel_dir"] = os.path.join("pseudo_student", str(args.student_pseudo_name))
    run_cfg["resolved_crop_h"] = int(crop_size[0])
    run_cfg["resolved_crop_w"] = int(crop_size[1])
    save_json(run_cfg, os.path.join(args.out_dir, "train_config.json"))
    with StageTimer(
            save_path=stage_time_path,
            stage_name=stage_name,
            dataset=dataset_name,
            fold=fold_name,
            mode=args.mode,
            split="train",
    ) as timer:
        print(f"[INFO] student_pseudo_name={args.student_pseudo_name}")
        print(f"[INFO] student_pseudo_rel_dir=pseudo_student/{args.student_pseudo_name}")

        train_ds = build_student_patch_dataset(
            fold_root=args.fold_root,
            dataset_name=dataset_name,
            mode=args.mode,
            split="train",
            crop_size=crop_size,
            fg_sample_prob=args.fg_sample_prob,
            min_valid_ratio_for_baseline=args.min_valid_ratio_for_baseline,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
        )

        num_classes = train_ds.num_classes
        print(f"[INFO] dataset={dataset_name}, mode={args.mode}")
        print(f"[INFO] num_classes={num_classes}, crop_size={crop_size}")
        print(f"[INFO] train_samples={len(train_ds)}")

        model = build_model(
            num_classes=num_classes,
            in_chans=3,
            deep_supervision=args.deep_supervision,
            pretrained_ckpt=args.pretrained_ckpt if args.pretrained_ckpt else None,
        ).to(device)
        criterion = build_loss(args.mode, device)
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            eps=1e-5, betas=(0.9, 0.999),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = GradScaler(enabled=amp)

        print(f"[INFO] device={device}, amp={amp}")
        print(f"[INFO] out_dir={args.out_dir}")

        train_log_path = os.path.join(args.out_dir, "train_log.csv")
        checkpoint_latest_path = os.path.join(args.out_dir, "checkpoint_latest.pth")

        history: list[Dict] = []
        start_epoch = 0
        resumed_elapsed_sec = 0.0

        if args.resume_ckpt:
            if not os.path.exists(args.resume_ckpt):
                raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_ckpt}")
            ckpt = torch.load(args.resume_ckpt, map_location=device)
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=True)

            if isinstance(ckpt, dict):
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler"])
                if "scaler" in ckpt and ckpt["scaler"] is not None:
                    try:
                        scaler.load_state_dict(ckpt["scaler"])
                    except Exception as e:
                        print(f"[WARN] failed to restore GradScaler state: {e}")
                start_epoch = int(ckpt.get("epoch", 0))

            history = load_history_csv(train_log_path, max_epoch=start_epoch)
            if not history and isinstance(ckpt, dict) and isinstance(ckpt.get("history"), list):
                history = [row for row in ckpt["history"] if int(row.get("epoch", 0)) <= start_epoch]
            if history:
                resumed_elapsed_sec = float(history[-1].get("elapsed_sec", 0.0) or 0.0)

            print(f"[INFO] resume_ckpt={args.resume_ckpt}")
            print(f"[INFO] resume_epoch={start_epoch}")
            print(f"[INFO] resumed_elapsed_sec={resumed_elapsed_sec:.3f}")

        if start_epoch >= args.epochs and os.path.exists(os.path.join(args.out_dir, "last.pth")):
            print(f"[INFO] training already complete at epoch={start_epoch}, skip.")
            timer.set_outputs(args.epochs)
            return

        start_time = time.time() - resumed_elapsed_sec
        for epoch in range(start_epoch, args.epochs):
            freeze_now = epoch < args.freeze_encoder_epochs
            maybe_freeze_encoder(model, freeze_now)
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                criterion=criterion,
                device=device,
                amp=amp,
                grad_clip=args.grad_clip,
            )
            scheduler.step()

            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "lr": lr_now,
                "freeze_encoder": int(freeze_now),
                "elapsed_sec": elapsed,
            }
            history.append(row)
            print(
                f"[Epoch {epoch + 1:03d}/{args.epochs:03d}] loss={train_loss:.6f} lr={lr_now:.8f} freeze={freeze_now}")

            ckpt_state = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if amp else None,
                "args": run_cfg,
                "history": history,
            }
            torch.save(ckpt_state, checkpoint_latest_path)

            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                torch.save(ckpt_state, os.path.join(args.out_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

            pd.DataFrame(history).to_csv(train_log_path, index=False)
            timer.set_outputs(epoch + 1)

        ckpt_last = {
            "epoch": args.epochs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if amp else None,
            "args": run_cfg,
            "history": history,
        }
        torch.save(ckpt_last, os.path.join(args.out_dir, "last.pth"))
        timer.set_outputs(args.epochs)
        print(f"[DONE] saved: {os.path.join(args.out_dir, 'last.pth')}")


if __name__ == "__main__":
    main()
