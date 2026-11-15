# Kiểm tra biên độ làm giàu ảnh trước pilot R3 (không huấn luyện, không đọc nhãn hay Validation):
#   (1) Hình học: lề phổi tới mép ảnh đã căn (mặt nạ CheXmask dịch theo dx của preprocess/align_midline.py) và tỉ lệ mẫu có góc
#       khung bao phổi bị đẩy ra ngoài ảnh dưới xoay / scale / tịnh tiến (Monte Carlo; khung bao chữ nhật nên là cận trên).
#   (2) Cường độ: % pixel bão hòa trắng và dải động p1-p99 sau RandomGamma + ColorJitter, trên ảnh 512 px đã căn.
# Chỉ dùng ảnh tập huấn luyện (split train_fold_*), mặt nạ có mask_status = ok.
#   python analysis/augmentation_range_check.py
#   python analysis/augmentation_range_check.py --geometry-images 5000 --intensity-images 500

# Khai báo utilities
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Khai báo thư viện
import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from preprocess.align_midline import MASK_COLUMNS, rle_decode
from utilities.data import RandomGamma

# Cấu hình hình học: (tên, góc xoay tối đa, tịnh tiến tối đa, khoảng scale)
GEOMETRY_SETTINGS = [
    ("cu",                  7,  0.04, (0.90, 1.10)),
    ("r3_translate_0.06",   10, 0.06, (0.85, 1.15)),
    ("r3_translate_0.08",   10, 0.08, (0.85, 1.15)),
    ("r3_translate_0.10",   10, 0.10, (0.85, 1.15)),
]

# Cấu hình cường độ: (tên, brightness, contrast, khoảng gamma hoặc None)
INTENSITY_SETTINGS = [
    ("cu",                  0.1, 0.1, None),
    ("r3_0.2_gamma",        0.2, 0.2, (0.8, 1.25)),
    ("0.3_gamma",           0.3, 0.3, (0.8, 1.25)),
    ("0.3_khong_gamma",     0.3, 0.3, None),
]

# ============================================================
# Khung bao phổi (x0, x1, y0, y1, tỉ lệ cạnh ảnh) trong tọa độ ảnh đã căn
def lung_boxes(alignment, mask_csv, count, seed, max_rows):
    wanted = set(alignment.index)
    parts, read = [], 0
    for chunk in pd.read_csv(mask_csv, usecols = MASK_COLUMNS, chunksize = 5000):
        parts.append(chunk[chunk["Image Index"].isin(wanted)])
        read += len(chunk)
        if read >= max_rows:
            break
    masks = pd.concat(parts).drop_duplicates(subset = ["Image Index"])
    masks = masks.sample(min(count, len(masks)), random_state = seed)

    boxes = []
    for row in masks.itertuples(index = False):
        height, width = int(row.Height), int(row.Width)
        lung = rle_decode(row[2], height, width) | rle_decode(row[3], height, width)
        ys, xs = np.nonzero(lung)
        if xs.size == 0:
            continue
        dx = float(alignment.loc[row[0], "dx"])
        boxes.append(((xs.min() + dx) / width, (xs.max() + dx) / width, ys.min() / height, ys.max() / height))
    return np.asarray(boxes)

# ============================================================
# Tỉ lệ mẫu có góc khung bao ra ngoài ảnh và mức vượt mép trung bình (tỉ lệ cạnh ảnh)
def out_of_frame(boxes, degrees, translate, scale, draws, rng):
    outside, overshoot = [], []
    for x0, x1, y0, y1 in boxes:
        cx = np.array([x0, x1, x0, x1]) - 0.5
        cy = np.array([y0, y0, y1, y1]) - 0.5
        angle = np.deg2rad(rng.uniform(-degrees, degrees, draws))[:, None]
        factor = rng.uniform(*scale, draws)[:, None]
        tx, ty = rng.uniform(-translate, translate, draws)[:, None], rng.uniform(-translate, translate, draws)[:, None]
        x = factor * (np.cos(angle) * cx - np.sin(angle) * cy) + tx
        y = factor * (np.sin(angle) * cx + np.cos(angle) * cy) + ty
        reach = np.maximum(np.abs(x), np.abs(y)).max(1) - 0.5
        outside.append((reach > 0).mean())
        overshoot.append(np.maximum(reach, 0).mean())
    return float(np.mean(outside)), float(np.mean(overshoot))

# ============================================================
# % pixel bão hòa trắng (trung bình, p95 theo mẫu) và dải động p1-p99 trung bình
def saturation(images, brightness, contrast, gamma, draws):
    steps = ([RandomGamma(gamma)] if gamma is not None else []) + [transforms.ColorJitter(brightness = brightness, contrast = contrast)]
    transform = transforms.Compose(steps)
    white, spread = [], []
    for image in images:
        for _ in range(draws):
            channel = transform(image)[0].numpy()
            white.append((channel >= 0.999).mean())
            spread.append(np.percentile(channel, 99) - np.percentile(channel, 1))
    white = np.asarray(white)
    return float(white.mean()), float(np.percentile(white, 95)), float(np.mean(spread))

# ============================================================
def main():
    parser = argparse.ArgumentParser(description = "Kiểm tra biên độ làm giàu ảnh: lề phổi / tịnh tiến và bão hòa cường độ")
    parser.add_argument("--geometry-images", type = int, default = 3000, help = "Số ảnh đo lề phổi")
    parser.add_argument("--geometry-draws", type = int, default = 200, help = "Số lần lấy mẫu biến đổi cho mỗi ảnh")
    parser.add_argument("--mask-rows", type = int, default = 30000, help = "Số dòng đầu của CheXmask được đọc (tệp lớn)")
    parser.add_argument("--intensity-images", type = int, default = 300, help = "Số ảnh 512 px đo bão hòa")
    parser.add_argument("--intensity-draws", type = int, default = 8, help = "Số lần lấy mẫu biến đổi cho mỗi ảnh")
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--output", default = "training_results/explore/augmentation_check", help = "Thư mục kết quả")
    args = parser.parse_args()

    output = ROOT_DIR / args.output
    output.mkdir(parents = True, exist_ok = True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    result, lines = {"args": vars(args)}, []

    # (1) Hình học
    alignment = pd.read_csv(ROOT_DIR / "dataset" / "alignment_midline.csv")
    alignment = alignment[(alignment["mask_status"] == "ok") & alignment["split"].str.startswith("train")].set_index("Image Index")
    boxes = lung_boxes(alignment, ROOT_DIR / "original_dataset" / "CheXmask" / "OriginalResolution" / "ChestX-Ray8.csv", args.geometry_images, args.seed, args.mask_rows)
    margins = np.stack([boxes[:, 0], 1 - boxes[:, 1], boxes[:, 2], 1 - boxes[:, 3]], axis = 1)
    names = ["trai", "phai", "tren", "duoi"]
    result["margins"] = {name: {"p5": float(np.percentile(margins[:, i], 5)), "median": float(np.median(margins[:, i]))} for i, name in enumerate(names)}
    result["margins"]["nho_nhat"] = {"p5": float(np.percentile(margins.min(1), 5)), "median": float(np.median(margins.min(1)))}
    lines.append(f"LỀ PHỔI tới mép ảnh đã căn ({len(boxes)} ảnh huấn luyện, tỉ lệ cạnh ảnh)")
    for name, value in result["margins"].items():
        lines.append(f"  {name:9s} p5 {value['p5']:.3f} | trung vị {value['median']:.3f}")

    lines.append(f"GÓC KHUNG BAO PHỔI RA NGOÀI ẢNH ({args.geometry_draws} lần / ảnh; cận trên)")
    result["geometry"] = {}
    for name, degrees, translate, scale in GEOMETRY_SETTINGS:
        share, overshoot = out_of_frame(boxes, degrees, translate, scale, args.geometry_draws, rng)
        result["geometry"][name] = {"degrees": degrees, "translate": translate, "scale": scale, "out_share": share, "mean_overshoot": overshoot}
        lines.append(f"  {name:20s} ±{degrees}°, {translate:.0%}, {scale} | mẫu ra ngoài {share:6.1%} | vượt mép TB {overshoot:.2%}")

    # (2) Cường độ
    files = sorted((ROOT_DIR / "dataset" / "images_512_aligned").glob("*.png"))
    train_names = set(alignment.index)
    files = [f for f in files if f.name in train_names]
    picked = np.random.default_rng(args.seed).choice(len(files), size = min(args.intensity_images, len(files)), replace = False)
    images = [transforms.functional.to_tensor(Image.open(files[i]).convert("RGB")) for i in picked]
    pixels = torch.stack(images)[:, 0].numpy()
    result["pixels"] = {"median": float(np.median(pixels)), "p99": float(np.percentile(pixels, 99)), "white_share": float((pixels >= 0.999).mean())}
    lines.append(f"BÃO HÒA CƯỜNG ĐỘ ({len(images)} ảnh 512 px, {args.intensity_draws} lần / ảnh)")
    lines.append(f"  ảnh gốc: trung vị pixel {result['pixels']['median']:.2f}, p99 {result['pixels']['p99']:.2f}, pixel trắng {result['pixels']['white_share']:.2%}")
    result["intensity"] = {}
    for name, brightness, contrast, gamma in INTENSITY_SETTINGS:
        mean_white, p95_white, spread = saturation(images, brightness, contrast, gamma, args.intensity_draws)
        result["intensity"][name] = {"brightness": brightness, "contrast": contrast, "gamma": gamma, "white_mean": mean_white, "white_p95": p95_white, "spread_p1_p99": spread}
        lines.append(f"  {name:20s} pixel trắng TB {mean_white:6.2%} | p95 theo mẫu {p95_white:6.2%} | dải động {spread:.2f}")

    text = "\n".join(lines)
    print(text)
    (output / "summary.txt").write_text(text + "\n", encoding = "utf-8")
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii = False, indent = 2), encoding = "utf-8")

if __name__ == "__main__":
    main()
