# Khai báo utilities
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Khai báo thư viện
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import config
from utilities.log import TqdmComma

# ============================================================
# Căn đường giữa cơ thể về giữa ảnh bằng mặt nạ phổi CheXmask (ChestX-Ray8.csv, thư mục OriginalResolution), rồi resize.
#   x_mid = trung bình hoành độ trọng tâm phổi trái và phổi phải (ảnh gốc), dx = round(W / 2 - x_mid),
#   dịch ngang dx pixel, đệm 0 (không cuộn vòng), resize LANCZOS về <kích thước> như resize_images.py.
# Mặt nạ có Dice RCA (Mean) < --rca-min, thiếu dòng CheXmask hoặc mặt nạ rỗng: không dịch (dx = 0), ghi lý do.
# Kết quả: dataset/images_<kích thước>_aligned/ (huấn luyện bằng IMAGE_VARIANT = "aligned") và dataset/alignment_midline.csv
# (dx theo pixel ảnh gốc: áp cùng phép dịch cho hộp bao NIH / ChestX-Det khi phân tích vị trí).
#
# Quy ước RLE của CheXmask (cặp start, length; start tính từ 1; trải phẳng theo hàng) CHƯA được xác minh từ mã gốc.
# Chạy --check trước: lưới 5 x 10 ảnh phủ mặt nạ và thống kê vị trí / diện tích phổi; sai thì đổi --rle-order / --rle-zero-based.
#   python preprocess/align_midline.py 512 --chexmask <đường dẫn>/ChestX-Ray8.csv --check 50
#   python preprocess/align_midline.py 512 --chexmask <đường dẫn>/ChestX-Ray8.csv
# Chạy lại đúng lệnh sau khi bị ngắt: ảnh đã có được bỏ qua (alignment_midline.csv được tính lại toàn bộ).
SOURCE_DIRECTORY = (config.ORIGINAL_DATASET_DIRECTORY / "NIH_ChestXRay14" / "images").resolve()
ALIGNMENT_CSV = config.DATASET_DIRECTORY / "alignment_midline.csv"
MASK_COLUMNS = ["Image Index", "Dice RCA (Mean)", "Left Lung", "Right Lung", "Height", "Width"]


# ============================================================
# Giải mã RLE (start, length, start, length, ...) thành mặt nạ (height, width) kiểu bool
def rle_decode(rle, height, width, one_based = True, order = "C"):
    runs = np.asarray(str(rle).split(), dtype = np.int64)
    if runs.size % 2:
        raise ValueError("RLE phải gồm các cặp (start, length)")
    starts = runs[0::2] - (1 if one_based else 0)
    ends = starts + runs[1::2]
    if starts.size and (starts.min() < 0 or ends.max() > height * width):
        raise ValueError(f"RLE vượt kích thước mặt nạ {height} x {width} (thử đổi --rle-zero-based)")
    delta = np.zeros(height * width + 1, dtype = np.int32)
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends, -1)
    return (np.cumsum(delta[:-1]) > 0).reshape((height, width), order = order)


# Mã hóa ngược (cùng quy ước), dùng cho kiểm thử vòng tròn
def rle_encode(mask, one_based = True, order = "C"):
    flat = np.concatenate([[0], np.asarray(mask, dtype = np.int8).flatten(order = order), [0]])
    changes = np.flatnonzero(np.diff(flat))
    starts, ends = changes[0::2], changes[1::2]
    return " ".join(f"{s + (1 if one_based else 0)} {e - s}" for s, e in zip(starts, ends))


# ============================================================
# Dịch ngang dx pixel (dx > 0: sang phải), phần trống đệm 0
def shift_horizontal(image, dx):
    if dx == 0:
        return image
    shifted = np.zeros_like(image)
    if abs(dx) >= image.shape[1]:
        return shifted
    if dx > 0:
        shifted[:, dx:] = image[:, :-dx]
    else:
        shifted[:, :dx] = image[:, -dx:]
    return shifted


# ============================================================
# Đường giữa từ một dòng CheXmask: trả về (x_mid theo pixel ảnh gốc hoặc None, thông tin mặt nạ)
def midline(row, image_width, rca_min, one_based, order):
    if row is None:
        return None, {"mask_status": "missing"}
    if not np.isfinite(row["Dice RCA (Mean)"]) or row["Dice RCA (Mean)"] < rca_min:
        return None, {"mask_status": "low_rca", "rca_mean": row["Dice RCA (Mean)"]}
    height, width = int(row["Height"]), int(row["Width"])
    left = rle_decode(row["Left Lung"], height, width, one_based, order) if isinstance(row["Left Lung"], str) else np.zeros((height, width), bool)
    right = rle_decode(row["Right Lung"], height, width, one_based, order) if isinstance(row["Right Lung"], str) else np.zeros((height, width), bool)
    if not left.any() or not right.any():
        return None, {"mask_status": "empty", "rca_mean": row["Dice RCA (Mean)"]}
    scale = image_width / width                                             # Mặt nạ khác kích thước ảnh: quy về pixel ảnh
    left_x, right_x = np.nonzero(left)[1].mean() * scale, np.nonzero(right)[1].mean() * scale
    return 0.5 * (left_x + right_x), {
        "mask_status": "ok", "rca_mean": row["Dice RCA (Mean)"],
        "left_lung_centroid_x": left_x, "right_lung_centroid_x": right_x,
        "left_lung_area": left.mean(), "right_lung_area": right.mean(),
    }


# ============================================================
# Một ảnh: đọc ảnh gốc, tính dx, dịch, resize, lưu (tệp tạm rồi đổi tên). Trả về dòng của alignment_midline.csv
def align_image(job):
    name, source, target, size, row, rca_min, one_based, order = job
    with Image.open(source) as image:
        pixels = np.asarray(image.convert("L"))
    x_mid, info = midline(row, pixels.shape[1], rca_min, one_based, order)
    dx = 0 if x_mid is None else int(round(pixels.shape[1] / 2 - x_mid))
    if not target.exists():
        aligned = Image.fromarray(shift_horizontal(pixels, dx)).resize((size, size), Image.Resampling.LANCZOS)
        temporary = target.with_name(f"{target.stem}.part.png")
        aligned.save(temporary, format = "PNG")
        os.replace(temporary, target)
    return {config.IMAGE_ID_COLUMN: name, "image_width": pixels.shape[1], "image_height": pixels.shape[0], "x_mid": x_mid, "dx": dx,
            "dx_fraction": dx / pixels.shape[1], **info}


# ============================================================
# Đọc các dòng CheXmask cần dùng (tệp lớn: đọc theo khối, chỉ giữ các cột và ảnh cần thiết)
def read_chexmask(path, names):
    header = pd.read_csv(path, nrows = 0).columns.tolist()
    missing = [c for c in MASK_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{path}: thiếu cột {missing}. Các cột có trong tệp: {header}")
    wanted, parts = set(names), []
    for chunk in TqdmComma(pd.read_csv(path, usecols = MASK_COLUMNS, chunksize = 2000), desc = "Đọc CheXmask", unit = " khối"):
        parts.append(chunk[chunk["Image Index"].isin(wanted)])
    frame = pd.concat(parts).drop_duplicates(subset = ["Image Index"])
    return {row["Image Index"]: row for row in frame.to_dict("records")}


# ============================================================
# Kiểm tra quy ước RLE trên ảnh thật: phổi phải của bệnh nhân nằm ở nửa TRÁI ảnh (quy ước PA / AP), diện tích mỗi phổi
# 5-25% ảnh, và lưới ảnh phủ mặt nạ (đỏ: phổi phải, xanh dương: phổi trái, xanh lá: đường giữa ước tính, xám: giữa ảnh)
def check_masks(images, masks, count, args):
    picked = images.sample(n = min(count, len(images)), random_state = 2026)
    rows, thumbnails, thumb = [], [], 160
    for name, original in zip(picked[config.IMAGE_ID_COLUMN], picked["original_path"]):
        row = masks.get(name)
        with Image.open(SOURCE_DIRECTORY / str(original)) as image:
            pixels = np.asarray(image.convert("L"))
        x_mid, info = midline(row, pixels.shape[1], -np.inf, not args.rle_zero_based, args.rle_order)
        rows.append({"name": name, **info, "x_mid": x_mid, "dx": None if x_mid is None else pixels.shape[1] / 2 - x_mid})
        canvas = Image.fromarray(pixels).convert("RGB").resize((thumb, thumb))
        if row is not None and info["mask_status"] == "ok":
            height, width = int(row["Height"]), int(row["Width"])
            overlay = np.zeros((height, width, 3), np.uint8)
            overlay[rle_decode(row["Right Lung"], height, width, not args.rle_zero_based, args.rle_order)] = (255, 0, 0)
            overlay[rle_decode(row["Left Lung"], height, width, not args.rle_zero_based, args.rle_order)] = (0, 80, 255)
            canvas = Image.blend(canvas, Image.fromarray(overlay).resize((thumb, thumb)), 0.35)
        draw = ImageDraw.Draw(canvas)
        draw.line([(thumb // 2, 0), (thumb // 2, thumb)], fill = (160, 160, 160))
        if x_mid is not None:
            x = x_mid / pixels.shape[1] * thumb
            draw.line([(x, 0), (x, thumb)], fill = (0, 255, 0))
        thumbnails.append(canvas)

    grid = Image.new("RGB", (10 * thumb, ((len(thumbnails) + 9) // 10) * thumb))
    for i, canvas in enumerate(thumbnails):
        grid.paste(canvas, ((i % 10) * thumb, (i // 10) * thumb))
    output = config.DATASET_DIRECTORY / f"alignment_check_{args.rle_order}_{'0' if args.rle_zero_based else '1'}based.png"
    grid.save(output)

    frame = pd.DataFrame(rows)
    ok = frame[frame["mask_status"] == "ok"]
    half = ok["right_lung_centroid_x"] < ok["left_lung_centroid_x"]
    area = ok[["left_lung_area", "right_lung_area"]].stack()
    print(f"Quy ước RLE: start tính từ {0 if args.rle_zero_based else 1}, trải phẳng order = {args.rle_order}")
    print(f"Có mặt nạ: {len(ok)}/{len(frame)} ảnh | trạng thái: {frame['mask_status'].value_counts().to_dict()}")
    print(f"Phổi phải nằm bên trái phổi trái trên ảnh: {half.mean():.1%} (cần ~100%)")
    print(f"Diện tích mỗi phổi trong 5-25% ảnh: {area.between(0.05, 0.25).mean():.1%} | trung vị {area.median():.1%}")
    print(f"|dx| trung vị: {ok['dx'].abs().median():.1f} px ảnh gốc")
    print(f"Lưới kiểm tra bằng mắt: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Căn đường giữa (CheXmask) rồi resize ảnh NIH ChestX-ray14 vào dataset/images_<kích thước>_aligned/")
    parser.add_argument("size", type = int, help = "Kích thước ảnh, chia hết cho 32 (ví dụ 512)")
    parser.add_argument("--chexmask", type = Path, required = True, help = "Đường dẫn ChestX-Ray8.csv của CheXmask (OriginalResolution)")
    parser.add_argument("--rca-min", type = float, default = 0.7, help = "Ngưỡng Dice RCA (Mean) để tin mặt nạ")
    parser.add_argument("--rle-order", choices = ["C", "F"], default = "C", help = "Thứ tự trải phẳng RLE: C (theo hàng) | F (theo cột)")
    parser.add_argument("--rle-zero-based", action = "store_true", help = "start của RLE tính từ 0 (mặc định tính từ 1)")
    parser.add_argument("--check", type = int, default = 0, metavar = "N", help = "Chỉ kiểm tra quy ước RLE trên N ảnh ngẫu nhiên, không tạo bộ ảnh")
    parser.add_argument("--workers", type = int, default = max(1, (os.cpu_count() or 2) - 2), help = "Số tiến trình song song")
    args = parser.parse_args()
    if args.size <= 0 or args.size % 32:
        parser.error("Kích thước phải là số dương chia hết cho 32")

    # Mọi ảnh của train_folds.csv và test.csv, mỗi ảnh một lần (cùng danh sách với resize_images.py)
    train_df, test_df = pd.read_csv(config.TRAIN_FOLDS_DIRECTORY), pd.read_csv(config.TEST_DIRECTORY)
    train_df["split"] = np.where(train_df["Fold"] == 1, "validation_fold_1", "train_fold_1")
    test_df["split"] = "test"
    images = pd.concat([train_df, test_df]).drop_duplicates(subset = [config.IMAGE_ID_COLUMN])
    masks = read_chexmask(args.chexmask, images[config.IMAGE_ID_COLUMN])
    print(f"CheXmask: {len(masks):,}/{len(images):,} ảnh có dòng mặt nạ")

    if args.check:
        check_masks(images, masks, args.check, args)
        sys.exit(0)

    target_directory = config.DATASET_DIRECTORY / f"images_{args.size}_aligned"
    target_directory.mkdir(parents = True, exist_ok = True)
    for partial in target_directory.glob("*.part.png"):
        partial.unlink()
    jobs = [(name, SOURCE_DIRECTORY / str(original), target_directory / str(name), args.size, masks.get(name), args.rca_min, not args.rle_zero_based, args.rle_order)
            for name, original in zip(images[config.IMAGE_ID_COLUMN], images["original_path"])]
    missing = [str(job[1]) for job in jobs if not job[1].is_file()]
    if missing:
        raise FileNotFoundError(f"Thiếu {len(missing):,} ảnh gốc, ví dụ: {missing[0]}")

    with ProcessPoolExecutor(max_workers = args.workers) as pool:
        records = list(TqdmComma(pool.map(align_image, jobs, chunksize = 32), total = len(jobs), desc = f"Căn giữa + resize {args.size} px"))
    alignment = pd.DataFrame(records).merge(images[[config.IMAGE_ID_COLUMN, "split"]], on = config.IMAGE_ID_COLUMN)
    alignment.to_csv(ALIGNMENT_CSV, index = False)

    # Báo cáo: tỉ lệ ảnh không dịch được theo tập, |dx| trung vị, các ảnh |dx| > 15% chiều rộng (cần xem bằng mắt)
    print(f"\nĐã lưu {ALIGNMENT_CSV} | ảnh: {target_directory}")
    print(alignment.groupby("split")["mask_status"].value_counts(normalize = True).unstack(fill_value = 0).round(4).to_string())
    print(f"|dx| trung vị: {alignment['dx'].abs().median():.0f} px | |dx| > 15% chiều rộng: {(alignment['dx_fraction'].abs() > 0.15).sum():,} ảnh")
