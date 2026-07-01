import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from stage_timer_utils import StageTimer

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_SWIN_UMAMBA_PKG_ROOT = _PROJECT_ROOT / "swin_umamba"
if str(_SWIN_UMAMBA_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_SWIN_UMAMBA_PKG_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from train_student import build_model, ensure_dir, infer_dataset_name, load_json, save_json, set_seed


SKIP_PRETRAIN_KEYS = {
    "norm.weight",
    "norm.bias",
    "head.weight",
    "head.bias",
}


@dataclass
class MIMSampleRecord:
    slice_name: str
    split: str
    student_img: str


def ensure_3ch_float_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape[-1] == 3:
            pass
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported image ndim: {arr.ndim}")
    return arr.astype(np.float32)


def resize_image_keep_range(img: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def normalize_to_unit_range(img: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    vmin = float(img.min())
    vmax = float(img.max())
    if vmax > 1.0 or vmin < 0.0:
        denom = max(vmax - vmin, 1e-6)
        img = (img - vmin) / denom
    return np.clip(img, 0.0, 1.0)


class StudentMIMImageDataset(Dataset):
    def __init__(
        self,
        fold_root: str,
        split: str,
        image_size: int,
        max_samples: int,
        seed: int,
    ):
        self.fold_root = fold_root
        self.split = split
        self.image_size = int(image_size)

        manifest_path = os.path.join(fold_root, "meta", "manifest.json")
        manifest = load_json(manifest_path)
        if not isinstance(manifest, list):
            raise ValueError(f"manifest must be a list, got: {type(manifest)}")

        samples: List[MIMSampleRecord] = []
        for rec in manifest:
            if rec.get("split") != split:
                continue

            student_img = rec.get("student_img")
            if not isinstance(student_img, str):
                raise ValueError(f"manifest record has invalid student_img: {rec}")
            student_img_norm = student_img.replace("\\", "/")
            if not student_img_norm.startswith("student_npy/imgs/"):
                raise ValueError(
                    f"Only student_npy/imgs is allowed for MIM input, got: {student_img}"
                )

            img_abs = os.path.join(self.fold_root, student_img)
            if not os.path.exists(img_abs):
                raise FileNotFoundError(f"Image not found: {img_abs}")

            samples.append(
                MIMSampleRecord(
                    slice_name=str(rec.get("slice_name", "")),
                    split=str(rec.get("split", "")),
                    student_img=student_img,
                )
            )

        if not samples:
            raise RuntimeError(f"No samples found for split={split} in {manifest_path}")

        if max_samples > 0 and len(samples) > max_samples:
            rng = np.random.default_rng(seed)
            keep_idx = rng.choice(len(samples), size=max_samples, replace=False)
            samples = [samples[int(i)] for i in sorted(keep_idx.tolist())]

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.samples[idx]
        img = np.load(os.path.join(self.fold_root, rec.student_img))
        img = ensure_3ch_float_image(img)
        img = resize_image_keep_range(img, (self.image_size, self.image_size))
        img = normalize_to_unit_range(img)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)

        return {
            "image": torch.from_numpy(img),
            "slice_name": rec.slice_name,
        }


class OneLayerReconHead(nn.Module):
    def __init__(self, in_channels: int = 768, out_channels: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        out = self.proj(x)
        if out.shape[-2:] != out_hw:
            out = F.interpolate(out, size=out_hw, mode="bilinear", align_corners=False)
        return out


def make_patch_mask(
    batch_size: int,
    image_h: int,
    image_w: int,
    patch_size: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    if image_h % patch_size != 0 or image_w % patch_size != 0:
        raise ValueError(
            f"image_size ({image_h}, {image_w}) must be divisible by mask_patch_size={patch_size}"
        )

    grid_h = image_h // patch_size
    grid_w = image_w // patch_size
    num_patches = grid_h * grid_w
    num_masked = int(round(num_patches * mask_ratio))
    num_masked = max(1, min(num_patches, num_masked))

    noise = torch.rand((batch_size, num_patches), device=device)
    order = torch.argsort(noise, dim=1)
    patch_mask = torch.zeros((batch_size, num_patches), device=device, dtype=torch.float32)
    patch_mask.scatter_(1, order[:, :num_masked], 1.0)
    patch_mask = patch_mask.view(batch_size, 1, grid_h, grid_w)
    pixel_mask = patch_mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    return pixel_mask


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(pred - target) * mask
    denom = mask.sum() * target.shape[1]
    denom = torch.clamp(denom, min=1.0)
    return diff.sum() / denom


def clone_state_dict_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def set_encoder_trainable(encoder: nn.Module, trainable: bool) -> None:
    for p in encoder.parameters():
        p.requires_grad = bool(trainable)


def map_pretrain_key_to_encoder_key(pretrain_key: str) -> Optional[str]:
    if pretrain_key in SKIP_PRETRAIN_KEYS:
        return None
    m = re.match(r"layers\.(\d+)\.downsample\.(.+)", pretrain_key)
    if m is not None:
        return f"downsamples.{m.group(1)}.{m.group(2)}"
    return pretrain_key


def map_encoder_key_to_pretrain_key(encoder_key: str) -> str:
    m = re.match(r"downsamples\.(\d+)\.(.+)", encoder_key)
    if m is not None:
        return f"layers.{m.group(1)}.downsample.{m.group(2)}"
    return encoder_key


def collect_pretrain_match_info(
    encoder: nn.Module,
    pretrained_ckpt: str,
    num_input_channels: int,
) -> Dict[str, Any]:
    ckpt = torch.load(pretrained_ckpt, map_location="cpu")
    if "model" not in ckpt or not isinstance(ckpt["model"], dict):
        raise ValueError(f"Invalid pretrained checkpoint format: {pretrained_ckpt}")

    pretrain_model = ckpt["model"]
    if "patch_embed.proj.weight" not in pretrain_model:
        raise KeyError("pretrained checkpoint missing key: patch_embed.proj.weight")

    pretrain_in_chans = int(pretrain_model["patch_embed.proj.weight"].shape[1])
    encoder_sd = encoder.state_dict()

    matched: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for src_key, src_tensor in pretrain_model.items():
        if src_key in SKIP_PRETRAIN_KEYS:
            skipped.append({"src_key": src_key, "reason": "skip_head_or_norm"})
            continue

        if "patch_embed" in src_key and pretrain_in_chans != num_input_channels:
            skipped.append({"src_key": src_key, "reason": "input_channel_mismatch"})
            continue

        dst_key = map_pretrain_key_to_encoder_key(src_key)
        if dst_key is None:
            skipped.append({"src_key": src_key, "reason": "mapped_to_none"})
            continue

        if dst_key not in encoder_sd:
            raise KeyError(f"pretrain key {src_key} mapped to missing encoder key {dst_key}")

        dst_tensor = encoder_sd[dst_key]
        if tuple(src_tensor.shape) != tuple(dst_tensor.shape):
            raise RuntimeError(
                f"shape mismatch for {src_key} -> {dst_key}: {tuple(src_tensor.shape)} vs {tuple(dst_tensor.shape)}"
            )

        matched.append(
            {
                "src_key": src_key,
                "dst_key": dst_key,
                "shape": list(src_tensor.shape),
            }
        )

    return {
        "pretrained_ckpt": pretrained_ckpt,
        "total_pretrain_keys": int(len(pretrain_model)),
        "matched_count": int(len(matched)),
        "skipped_count": int(len(skipped)),
        "matched": matched,
        "skipped": skipped,
    }


def export_encoder_checkpoints(
    encoder: nn.Module,
    pretrained_ckpt: str,
    adapted_encoder_path: str,
    adapted_for_train_path: str,
) -> None:
    encoder_state = clone_state_dict_cpu(encoder.state_dict())
    torch.save({"model": encoder_state}, adapted_encoder_path)

    base_ckpt = torch.load(pretrained_ckpt, map_location="cpu")
    if "model" not in base_ckpt or not isinstance(base_ckpt["model"], dict):
        raise ValueError(f"Invalid pretrained checkpoint format: {pretrained_ckpt}")

    out_model: Dict[str, torch.Tensor] = {}
    for k, v in base_ckpt["model"].items():
        out_model[k] = v.detach().cpu().clone()

    for enc_key, enc_value in encoder_state.items():
        train_key = map_encoder_key_to_pretrain_key(enc_key)
        if train_key in out_model:
            if tuple(out_model[train_key].shape) != tuple(enc_value.shape):
                raise RuntimeError(
                    f"Export shape mismatch for key {train_key}: "
                    f"{tuple(out_model[train_key].shape)} vs {tuple(enc_value.shape)}"
                )
        out_model[train_key] = enc_value

    torch.save({"model": out_model}, adapted_for_train_path)


def train_one_epoch(
    encoder: nn.Module,
    recon_head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp: bool,
    mask_patch_size: int,
    mask_ratio: float,
    mask_fill: str,
) -> float:
    encoder.train()
    recon_head.train()

    loss_meter = 0.0
    num_batches = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)

        mask = make_patch_mask(
            batch_size=images.shape[0],
            image_h=images.shape[-2],
            image_w=images.shape[-1],
            patch_size=mask_patch_size,
            mask_ratio=mask_ratio,
            device=device,
        )

        if mask_fill == "zero":
            masked_images = images * (1.0 - mask)
        else:
            raise ValueError(f"Unsupported mask_fill: {mask_fill}")

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            feats = encoder(masked_images)
            latent = feats[-1]
            recon = recon_head(latent, out_hw=(images.shape[-2], images.shape[-1]))
            loss = masked_l1_loss(recon, images, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter += float(loss.detach().item())
        num_batches += 1

    return loss_meter / max(1, num_batches)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)

    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument("--mask_patch_size", type=int, default=16)
    parser.add_argument("--mask_fill", type=str, default="zero", choices=["zero"])
    parser.add_argument("--image_size", type=int, default=192)

    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "pretrained" / "vmamba" / "vmamba_tiny_e292.pth"),
    )
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--freeze_encoder_epochs", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=20)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if not os.path.exists(args.fold_root):
        raise FileNotFoundError(f"fold_root not found: {args.fold_root}")
    if not os.path.exists(args.pretrained_ckpt):
        raise FileNotFoundError(f"pretrained_ckpt not found: {args.pretrained_ckpt}")
    if args.epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {args.epochs}")
    if not (0.0 < args.mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio must be in (0, 1], got {args.mask_ratio}")
    if args.mask_patch_size <= 0:
        raise ValueError(f"mask_patch_size must be > 0, got {args.mask_patch_size}")
    if args.image_size <= 0:
        raise ValueError(f"image_size must be > 0, got {args.image_size}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")

    dataset_name = infer_dataset_name(args.fold_root, args.dataset.strip() or None)
    fold_name = Path(args.fold_root).name

    if not args.out_dir:
        args.out_dir = str(_PROJECT_ROOT / "work_dir" / "adapt" / dataset_name / fold_name)
    ensure_dir(args.out_dir)

    stage_time_path = os.path.join(args.out_dir, "stage_time_adapt.json")

    run_cfg = dict(vars(args))
    run_cfg["dataset"] = dataset_name
    run_cfg["resolved_amp"] = amp
    save_json(run_cfg, os.path.join(args.out_dir, "args.json"))

    dataset = StudentMIMImageDataset(
        fold_root=args.fold_root,
        split=args.split,
        image_size=args.image_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    base_model = build_model(
        num_classes=2,
        in_chans=3,
        deep_supervision=False,
        pretrained_ckpt=args.pretrained_ckpt,
    ).to(device)
    encoder = base_model.vssm_encoder
    recon_head = OneLayerReconHead(in_channels=encoder.num_features, out_channels=3).to(device)

    match_info = collect_pretrain_match_info(
        encoder=encoder,
        pretrained_ckpt=args.pretrained_ckpt,
        num_input_channels=3,
    )
    save_json(match_info, os.path.join(args.out_dir, "matched_pretrain_keys.json"))

    optimizer = AdamW(
        list(encoder.parameters()) + list(recon_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-5,
        betas=(0.9, 0.999),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=amp)

    print(f"[INFO] dataset={dataset_name}")
    print(f"[INFO] split={args.split}, train_samples={len(dataset)}")
    print(f"[INFO] image_size={args.image_size}, mask_ratio={args.mask_ratio}, patch={args.mask_patch_size}")
    print(f"[INFO] device={device}, amp={amp}")
    print(f"[INFO] out_dir={args.out_dir}")

    history: List[Dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_encoder_state: Optional[Dict[str, torch.Tensor]] = None
    best_recon_head_state: Optional[Dict[str, torch.Tensor]] = None
    start_time = time.time()

    with StageTimer(
        save_path=stage_time_path,
        stage_name="adapt",
        dataset=dataset_name,
        fold=fold_name,
        mode="mim",
        split=args.split,
    ) as timer:
        for epoch in range(args.epochs):
            freeze_now = epoch < args.freeze_encoder_epochs
            set_encoder_trainable(encoder, trainable=(not freeze_now))

            train_loss = train_one_epoch(
                encoder=encoder,
                recon_head=recon_head,
                loader=loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                amp=amp,
                mask_patch_size=args.mask_patch_size,
                mask_ratio=args.mask_ratio,
                mask_fill=args.mask_fill,
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
                f"[Epoch {epoch + 1:03d}/{args.epochs:03d}] "
                f"loss={train_loss:.6f} lr={lr_now:.8f} freeze={freeze_now}"
            )

            last_ckpt = {
                "epoch": epoch + 1,
                "encoder": encoder.state_dict(),
                "recon_head": recon_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": run_cfg,
            }
            torch.save(last_ckpt, os.path.join(args.out_dir, "last.pth"))

            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch + 1
                best_encoder_state = clone_state_dict_cpu(encoder.state_dict())
                best_recon_head_state = clone_state_dict_cpu(recon_head.state_dict())
                best_ckpt = {
                    "epoch": best_epoch,
                    "best_train_loss": best_loss,
                    "encoder": best_encoder_state,
                    "recon_head": best_recon_head_state,
                    "args": run_cfg,
                }
                torch.save(best_ckpt, os.path.join(args.out_dir, "best.pth"))

            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                save_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                torch.save(last_ckpt, save_path)

            pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "train_log.csv"), index=False)
            timer.set_outputs(epoch + 1)

    if best_encoder_state is None:
        best_encoder_state = clone_state_dict_cpu(encoder.state_dict())
    encoder.load_state_dict(best_encoder_state, strict=True)

    if best_recon_head_state is not None:
        recon_head.load_state_dict(best_recon_head_state, strict=True)

    export_encoder_checkpoints(
        encoder=encoder,
        pretrained_ckpt=args.pretrained_ckpt,
        adapted_encoder_path=os.path.join(args.out_dir, "adapted_encoder.pth"),
        adapted_for_train_path=os.path.join(args.out_dir, "adapted_encoder_for_train.pth"),
    )

    summary = {
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_loss),
        "num_samples": int(len(dataset)),
    }
    save_json(summary, os.path.join(args.out_dir, "adapt_summary.json"))

    print(f"[DONE] best_epoch={best_epoch}, best_train_loss={best_loss:.6f}")
    print(f"[DONE] saved: {os.path.join(args.out_dir, 'adapted_encoder_for_train.pth')}")


if __name__ == "__main__":
    main()
