import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SampleRecord:
    slice_name: str
    case_id: str
    split: str
    fold: str
    student_img: str
    student_gt: str
    label_path: Optional[str] = None


ALLOWED_MODES = {"upper", "baseline"}
ALLOWED_SPLITS = {"train", "test"}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def ensure_label_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Unsupported label shape: {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"Label must be 2D, got {arr.shape}")
    return arr


def resize_image_keep_range(img: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def resize_mask_nearest(mask: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def crop_with_pad(
    img: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    crop_h: int,
    crop_w: int,
    pad_value_img: float = 0.0,
    pad_value_mask: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    bottom = top + crop_h
    right = left + crop_w

    src_top = max(top, 0)
    src_left = max(left, 0)
    src_bottom = min(bottom, h)
    src_right = min(right, w)

    dst_top = src_top - top
    dst_left = src_left - left
    dst_bottom = dst_top + (src_bottom - src_top)
    dst_right = dst_left + (src_right - src_left)

    out_img = np.full((crop_h, crop_w, img.shape[2]), pad_value_img, dtype=img.dtype)
    out_mask = np.full((crop_h, crop_w), pad_value_mask, dtype=mask.dtype)

    out_img[dst_top:dst_bottom, dst_left:dst_right] = img[src_top:src_bottom, src_left:src_right]
    out_mask[dst_top:dst_bottom, dst_left:dst_right] = mask[src_top:src_bottom, src_left:src_right]
    return out_img, out_mask


class StudentPatchDataset(Dataset):
    """
    Modes:
      - upper:    supervised upper bound, read student_gt on train
      - baseline: weakly supervised main baseline, read pseudo_student/tri_train on train

    Test/inference:
      always returns full student_img and uses student_gt only as a shape placeholder;
      inference code never feeds the test mask into the model.
    """

    def __init__(
        self,
        fold_root: str,
        mode: str,
        split: str,
        crop_size: Tuple[int, int],
        dataset_name: Optional[str] = None,
        fg_sample_prob: float = 0.5,
        min_valid_ratio_for_baseline: float = 0.10,
        min_fg_pixels: int = 16,
        max_resample_trials: int = 20,
        test_resize: Optional[Tuple[int, int]] = None,

    ):
        if mode not in ALLOWED_MODES:
            raise ValueError(f"mode must be one of {sorted(ALLOWED_MODES)}, got {mode}")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}, got {split}")

        self.fold_root = fold_root
        self.mode = mode
        self.split = split
        self.crop_h, self.crop_w = crop_size
        self.dataset_name = dataset_name
        self.fg_sample_prob = float(fg_sample_prob)
        self.min_valid_ratio_for_baseline = float(min_valid_ratio_for_baseline)
        self.min_fg_pixels = int(min_fg_pixels)
        self.max_resample_trials = int(max_resample_trials)
        self.test_resize = test_resize

        self.meta_dir = os.path.join(fold_root, "meta")
        self.manifest_path = os.path.join(self.meta_dir, "manifest.json")
        self.label_meta_path = os.path.join(self.meta_dir, "label_meta.json")
        self.split_meta_path = os.path.join(self.meta_dir, "split_meta.json")

        self.manifest = load_json(self.manifest_path)
        self.label_meta = load_json(self.label_meta_path)
        self.split_meta = load_json(self.split_meta_path)

        if self.dataset_name is None:
            self.dataset_name = self.split_meta.get("dataset_name") or self.split_meta.get("dataset") or "unknown"

        self.num_classes = self._infer_num_classes()
        self.samples: List[SampleRecord] = self._build_samples()



    def _infer_num_classes(self) -> int:
        unique_labels = self.label_meta.get("unique_labels", [])
        if not unique_labels:
            return 2
        return int(max(unique_labels)) + 1

    def _resolve_label_relpath(self, rec: dict) -> Optional[str]:
        slice_name = rec["slice_name"]

        # test 阶段只用 student_gt 占位，infer_student.py 不会把 test mask 喂进前向
        if self.split == "test":
            return rec["student_gt"]

        if self.mode == "upper":
            return rec["student_gt"]

        if self.mode == "baseline":
            return os.path.join("pseudo_student", os.environ.get("STUDENT_PSEUDO_NAME", "tri_train"), slice_name)

        raise ValueError(f"Unknown mode: {self.mode}")

    def _build_samples(self) -> List[SampleRecord]:
        out: List[SampleRecord] = []
        for rec in self.manifest:
            if rec["split"] != self.split:
                continue

            label_rel = self._resolve_label_relpath(rec)
            label_abs = os.path.join(self.fold_root, label_rel) if label_rel is not None else None

            if label_abs is not None and not os.path.exists(label_abs):
                raise FileNotFoundError(f"Label not found: {label_abs}")

            img_abs = os.path.join(self.fold_root, rec["student_img"])
            gt_abs = os.path.join(self.fold_root, rec["student_gt"])
            if not os.path.exists(img_abs):
                raise FileNotFoundError(f"Image not found: {img_abs}")
            if not os.path.exists(gt_abs):
                raise FileNotFoundError(f"GT not found: {gt_abs}")

            out.append(
                SampleRecord(
                    slice_name=rec["slice_name"],
                    case_id=rec["case_id"],
                    split=rec["split"],
                    fold=rec["fold"],
                    student_img=rec["student_img"],
                    student_gt=rec["student_gt"],
                    label_path=label_rel,
                )
            )
        return out


    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_mask(self, idx: int) -> Tuple[np.ndarray, np.ndarray, SampleRecord]:
        rec = self.samples[idx]

        img = np.load(os.path.join(self.fold_root, rec.student_img))
        mask = np.load(os.path.join(self.fold_root, rec.label_path))

        img = ensure_3ch_float_image(img)
        mask = ensure_label_2d(mask)

        if img.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image/label shape mismatch for {rec.slice_name}: "
                f"image={img.shape[:2]}, mask={mask.shape[:2]}, "
                f"img_path={os.path.join(self.fold_root, rec.student_img)}, "
                f"label_path={os.path.join(self.fold_root, rec.label_path)}"
            )

        return img, mask, rec

    def _get_valid_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.mode == "baseline":
            return mask != 255
        return np.ones_like(mask, dtype=bool)

    def _get_fg_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.mode == "baseline":
            return mask == 1 if self.num_classes == 2 else ((mask > 0) & (mask != 255))
        return mask > 0

    def _random_crop_coords(self, h: int, w: int) -> Tuple[int, int]:
        top = random.randint(0, max(0, h - self.crop_h)) if h > self.crop_h else random.randint(-(self.crop_h - h), 0)
        left = random.randint(0, max(0, w - self.crop_w)) if w > self.crop_w else random.randint(-(self.crop_w - w), 0)
        return top, left

    def _foreground_biased_crop_coords(self, fg_mask: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        ys, xs = np.where(fg_mask)
        if len(xs) == 0:
            return None, None

        idx = random.randint(0, len(xs) - 1)
        cy, cx = ys[idx], xs[idx]
        top = cy - self.crop_h // 2
        left = cx - self.crop_w // 2
        top = random.randint(top - self.crop_h // 8, top + self.crop_h // 8)
        left = random.randint(left - self.crop_w // 8, left + self.crop_w // 8)
        return top, left

    def _is_valid_train_patch(self, patch_mask: np.ndarray) -> bool:
        if self.mode == "baseline":
            valid_ratio = float(self._get_valid_mask(patch_mask).mean())
            if valid_ratio < self.min_valid_ratio_for_baseline:
                return False
        return True

    def _sample_train_patch(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        fg_mask = self._get_fg_mask(mask)

        for _ in range(self.max_resample_trials):
            use_fg = fg_mask.sum() >= self.min_fg_pixels and (random.random() < self.fg_sample_prob)
            if use_fg:
                top, left = self._foreground_biased_crop_coords(fg_mask)
                if top is None or left is None:
                    top, left = self._random_crop_coords(h, w)
            else:
                top, left = self._random_crop_coords(h, w)

            pad_value_mask = 255 if self.mode == "baseline" else 0
            patch_img, patch_mask = crop_with_pad(
                img, mask, top, left, self.crop_h, self.crop_w,
                pad_value_img=0.0, pad_value_mask=pad_value_mask,
            )
            if self._is_valid_train_patch(patch_mask):
                return patch_img, patch_mask

        top, left = self._random_crop_coords(h, w)
        pad_value_mask = 255 if self.mode == "baseline" else 0
        return crop_with_pad(
            img, mask, top, left, self.crop_h, self.crop_w,
            pad_value_img=0.0, pad_value_mask=pad_value_mask,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:


        img, mask, rec = self._load_image_mask(idx)
        orig_h, orig_w = img.shape[:2]

        if self.split == "train":
            img, mask = self._sample_train_patch(img, mask)
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            mask = mask.astype(np.int64)
            return {
                "image": torch.from_numpy(img),
                "mask": torch.from_numpy(mask),
                "slice_name": rec.slice_name,
                "case_id": rec.case_id,
            }

        if self.test_resize is not None:
            img = resize_image_keep_range(img, self.test_resize)
            mask = resize_mask_nearest(mask, self.test_resize)

        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        mask = mask.astype(np.int64)
        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask),
            "slice_name": rec.slice_name,
            "case_id": rec.case_id,
            "orig_hw": torch.tensor([orig_h, orig_w], dtype=torch.int64),
        }


# Fallbacks only. Preferred source of truth is fold_root/meta/split_meta.json.
DATASET_CROP_PRESETS = {
    "kvasirseg": (352, 352),
    "cvc_clinicdb": (352, 352),
    "tn3k": (256, 256),
    "tg3k": (256, 256),
    "ddti": (256, 256),
    "otu_2d": (256, 256),
    "drive": (512, 512),
    "chasedb1": (512, 512),
    "hrf": (1024, 1024),
    "monuseg": (512, 512),
    "ph2": (256, 256),
    "btcv": (512, 512),
    "synapse": (512, 512),
    "acdc": (320, 320),
    "prostate158": (320, 320),
}


def build_student_patch_dataset(
    fold_root: str,
    dataset_name: str,
    mode: str,
    split: str,
    crop_size: Optional[Tuple[int, int]] = None,
    fg_sample_prob: float = 0.5,
    min_valid_ratio_for_baseline: float = 0.10,
    test_resize: Optional[Tuple[int, int]] = None,
) -> StudentPatchDataset:
    if crop_size is None:
        if dataset_name not in DATASET_CROP_PRESETS:
            raise KeyError(f"Unknown dataset_name={dataset_name}, please specify crop_size manually.")
        crop_size = DATASET_CROP_PRESETS[dataset_name]

    return StudentPatchDataset(
        fold_root=fold_root,
        dataset_name=dataset_name,
        mode=mode,
        split=split,
        crop_size=crop_size,
        fg_sample_prob=fg_sample_prob,
        min_valid_ratio_for_baseline=min_valid_ratio_for_baseline,
        test_resize=test_resize,
    )
