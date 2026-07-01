# src/dataset.py
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ----------------------------
# Split reader (兼容 dict / list 多种格式)
# ----------------------------
def _read_split(split_json_path: str):
    """
    兼容以下格式：

    1) list[str]
       ["cju2xxx", "cju3xxx", ...]

    2) list[dict]
       [{"id": "..."}] / [{"name": "..."}] / [{"image": "..."}] ...

    3) dict with common keys -> list
       {"ids":[...]} / {"train":[...]} / {"val":[...]} / {"data":[...]} ...

    4) dict where KEYS ARE IDS  ✅ 你现在这种
       {
         "cju76o55n...": {...},
         "cjz14qsk...": {...}
       }
       -> 返回 dict.keys()
    """
    with open(split_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Case A: list ---
    if isinstance(data, list):
        if len(data) == 0:
            return []
        if isinstance(data[0], str):
            return data
        if isinstance(data[0], dict):
            ids = []
            for it in data:
                for k in ["id", "name", "image", "img", "filename", "file"]:
                    if k in it:
                        ids.append(str(it[k]))
                        break
            if len(ids) == 0:
                raise ValueError("list[dict] split json exists but cannot find id/name/image keys.")
            return ids
        raise ValueError(f"Unsupported list element type: {type(data[0])}")

    # --- Case B: dict ---
    if isinstance(data, dict):
        # 1) 常见字段承载 list 的情况
        for k in ["ids", "images", "data", "train", "val", "test", "files", "items"]:
            if k in data:
                v = data[k]
                if isinstance(v, list):
                    if len(v) == 0:
                        return []
                    if isinstance(v[0], str):
                        return [str(x) for x in v]
                    if isinstance(v[0], dict):
                        ids = []
                        for it in v:
                            for kk in ["id", "name", "image", "img", "filename", "file"]:
                                if kk in it:
                                    ids.append(str(it[kk]))
                                    break
                        if len(ids) == 0:
                            raise ValueError(f"dict[{k}] is list[dict] but cannot find id/name/image keys.")
                        return ids
                    raise ValueError(f"dict[{k}] list element type unsupported: {type(v[0])}")

                # 2) {"ids": {"0":"xxx","1":"yyy"}} 这种
                if isinstance(v, dict):
                    vv = list(v.values())
                    if len(vv) == 0:
                        return []
                    if isinstance(vv[0], str):
                        return [str(x) for x in vv]

        # ✅ 3) 你这种：dict 的 key 本身就是 id
        # 只要 key 都是 str，直接拿 keys 当 ids
        if len(data) > 0 and all(isinstance(k, str) for k in data.keys()):
            return list(data.keys())

        # 4) 兜底：如果 value 全是 str，就当作 ids
        all_vals = list(data.values())
        if len(all_vals) > 0 and all(isinstance(x, str) for x in all_vals):
            return [str(x) for x in all_vals]

        raise ValueError(f"Unsupported split json format(dict keys={list(data.keys())[:10]})")

    raise ValueError(f"Unsupported split json format: {type(data)}")


# ----------------------------
# Path resolver
# ----------------------------
def _resolve_path(base_dir, _id):
    candidates = [
        os.path.join(base_dir, _id),
        os.path.join(base_dir, _id + ".png"),
        os.path.join(base_dir, _id + ".jpg"),
        os.path.join(base_dir, _id + ".jpeg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ----------------------------
# Mask sanitizers
# ----------------------------
def sanitize_gt(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise ValueError("mask is None")
    if mask.ndim != 2:
        mask = mask.squeeze()
    mask = mask.astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)  # GT -> {0,1}
    return mask


def sanitize_pseudo(mask: np.ndarray) -> np.ndarray:
    """
    Pseudo 最终必须是 {0,1,255}
    - 200..255 -> 255 (ignore)
    - 1..199   -> 1   (fg)
    - 0        -> 0   (bg)
    """
    if mask is None:
        raise ValueError("mask is None")
    if mask.ndim != 2:
        mask = mask.squeeze()
    mask = mask.astype(np.uint8)

    out = np.zeros_like(mask, dtype=np.uint8)
    out[mask >= 200] = 255
    out[(mask >= 1) & (mask < 200)] = 1
    return out


# ----------------------------
# Albumentations transforms
# ----------------------------
def get_train_transform(img_size=352):
    return A.Compose(
        [
            A.Resize(
                img_size, img_size,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST
            ),
            A.Affine(
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                scale=(0.90, 1.10),
                rotate=(-15, 15),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.7,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_val_transform(img_size=352):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


# ----------------------------
# Dataset
# ----------------------------
class PolypDataset(Dataset):
    def __init__(
        self,
        img_dir,
        mask_dir,
        split_json_path,
        transform=None,
        mask_type="auto",  # auto / gt / pseudo
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # --- auto infer mask_type ---
        if mask_type == "auto":
            md = str(mask_dir).lower()
            if ("gt" in md) or ("gts" in md) or (("mask" in md) and ("pseudo" not in md)):
                mask_type = "gt"
            else:
                mask_type = "pseudo"
        if mask_type not in ["gt", "pseudo"]:
            raise ValueError(f"mask_type must be one of ['auto','gt','pseudo'], got {mask_type}")
        self.mask_type = mask_type

        self.ids = _read_split(split_json_path)
        if len(self.ids) == 0:
            raise ValueError(f"Split json empty: {split_json_path}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]

        img_path = _resolve_path(self.img_dir, _id)
        if img_path is None:
            raise FileNotFoundError(f"Image not found for id={_id} in {self.img_dir}")

        mask_path = _resolve_path(self.mask_dir, _id)
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for id={_id} in {self.mask_dir}")

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")

        # sanitize BEFORE aug
        mask = sanitize_gt(mask) if self.mask_type == "gt" else sanitize_pseudo(mask)

        # mask: uint8, shape [H,W]
        if self.mask_type == "gt":
            # 任何 >0 都当作前景，强制转为 0/1
            mask = (mask > 0).astype("uint8")

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]
        else:
            image = ToTensorV2()(image=image)["image"]

        # sanitize AFTER aug (防插值污染)
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask = sanitize_gt(mask) if self.mask_type == "gt" else sanitize_pseudo(mask)

        mask = torch.from_numpy(mask).long()
        return image, mask
