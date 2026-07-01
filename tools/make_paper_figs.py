import os
import json
import argparse
import random
import cv2
import numpy as np


def stem(x: str) -> str:
    return os.path.splitext(os.path.basename(x))[0]


def load_ids(split_json: str):
    with open(split_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        ids = list(data.keys())
    elif isinstance(data, list):
        ids = data
    else:
        raise ValueError(f"Unsupported json format: {type(data)}")
    return [stem(x) for x in ids]


def find_by_stem(folder: str, s: str, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    for ext in exts:
        p = os.path.join(folder, s + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find file for id='{s}' in {folder}")


def read_rgb(img_dir: str, img_id: str):
    p = find_by_stem(img_dir, img_id, exts=(".jpg", ".png", ".jpeg", ".bmp"))
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {p}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, p


def read_gray(mask_dir: str, img_id: str):
    p = find_by_stem(mask_dir, img_id, exts=(".png", ".jpg", ".jpeg", ".bmp"))
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {p}")
    return m, p


def mask_to_01(m: np.ndarray):
    return (m > 0).astype(np.uint8)


def pseudo_to_color(pseudo: np.ndarray):
    """
    pseudo: 0/1/255
    0 black, 1 red, 255 yellow
    """
    h, w = pseudo.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[pseudo == 1] = (255, 0, 0)
    out[pseudo == 255] = (255, 255, 0)
    return out


def draw_boundary(rgb: np.ndarray, mask01: np.ndarray, color=(0, 255, 0), thickness=2):
    """
    在 RGB 上画边界（color 是 RGB）
    """
    img = rgb.copy()
    m = (mask01 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV 画图用 BGR，所以转一下颜色
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, contours, -1, (color[2], color[1], color[0]), thickness)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay_mask(rgb: np.ndarray, mask01: np.ndarray, color=(255, 0, 0), alpha=0.35):
    out = rgb.copy()
    overlay = np.zeros_like(out, dtype=np.uint8)
    overlay[mask01 > 0] = color
    out = (out * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return out


def make_grid_2x2(a, b, c, d):
    top = np.concatenate([a, b], axis=1)
    bot = np.concatenate([c, d], axis=1)
    return np.concatenate([top, bot], axis=0)


def put_text(rgb: np.ndarray, text: str):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def dice(pred01: np.ndarray, gt01: np.ndarray):
    pred = pred01.astype(np.float32)
    gt = gt01.astype(np.float32)
    smooth = 1e-5
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    return float((2 * inter + smooth) / (union + smooth))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--img_dir", type=str, required=True)
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--pseudo_dir", type=str, default="")
    ap.add_argument("--pred_dir", type=str, required=True, help="inference.py 输出的 pred_masks 目录")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resize", type=int, default=352, help="输出统一尺寸，便于拼图")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ids = load_ids(args.split_json)
    random.seed(args.seed)
    random.shuffle(ids)
    ids = ids[:min(args.num, len(ids))]

    saved = 0
    missing_pred = 0

    for img_id in ids:
        try:
            img, _ = read_rgb(args.img_dir, img_id)
            gt, _ = read_gray(args.gt_dir, img_id)
            pred, _ = read_gray(args.pred_dir, img_id)
        except FileNotFoundError as e:
            print(f"[Skip] {img_id}: {e}")
            continue

        # resize 统一
        if args.resize > 0:
            img = cv2.resize(img, (args.resize, args.resize), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (args.resize, args.resize), interpolation=cv2.INTER_NEAREST)
            pred = cv2.resize(pred, (args.resize, args.resize), interpolation=cv2.INTER_NEAREST)

        gt01 = mask_to_01(gt)
        pred01 = mask_to_01(pred)

        d = dice(pred01, gt01)

        # pseudo 可选
        if args.pseudo_dir and os.path.isdir(args.pseudo_dir):
            try:
                pseudo, _ = read_gray(args.pseudo_dir, img_id)
                if args.resize > 0:
                    pseudo = cv2.resize(pseudo, (args.resize, args.resize), interpolation=cv2.INTER_NEAREST)
                pseudo_color = pseudo_to_color(pseudo)
            except FileNotFoundError:
                pseudo_color = np.zeros_like(img, dtype=np.uint8)
        else:
            pseudo_color = np.zeros_like(img, dtype=np.uint8)

        # 4格：
        # A 原图
        A = img
        # B GT 边界叠加（绿）
        B = draw_boundary(img, gt01, color=(0, 255, 0), thickness=2)
        # C Pseudo 可视化（红/黄）
        C = pseudo_color
        # D Pred 边界叠加（红） + 可选 mask 半透明
        D = draw_boundary(img, pred01, color=(255, 0, 0), thickness=2)
        D = overlay_mask(D, pred01, color=(255, 0, 0), alpha=0.20)

        grid = make_grid_2x2(A, B, C, D)
        grid = put_text(grid, f"{img_id} | Dice={d:.4f}")

        out_path = os.path.join(args.out_dir, f"{img_id}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        saved += 1

    print(f"\n✅ Saved {saved} figures to: {args.out_dir}")


if __name__ == "__main__":
    main()
