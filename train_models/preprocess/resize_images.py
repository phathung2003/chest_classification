# Khai báo utilities
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Khai báo thư viện
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from PIL import Image

import config
from utilities.log import TqdmComma

# ============================================================
# Tạo bộ ảnh ở kích thước khác từ ảnh gốc NIH, cùng cách xử lý với notebook tiền xử lý (ảnh xám, LANCZOS, PNG).
# Ảnh lưu ở dataset/images_<kích thước>/ cùng tên tệp; bộ ảnh images/, train_folds.csv, test.csv và các fold giữ nguyên.
# Huấn luyện ở kích thước mới: đặt IMAGE_SIZE trong config.py, load_dataset() tự đổi image_path sang thư mục tương ứng.
#   python preprocess/resize_images.py 448
# Chạy lại đúng lệnh sau khi bị ngắt: ảnh đã có được bỏ qua.
SOURCE_DIRECTORY = (config.ORIGINAL_DATASET_DIRECTORY / "NIH_ChestXRay14" / "images").resolve()


# ============================================================
# Resize một ảnh, ghi ra tệp tạm rồi đổi tên để lần chạy bị ngắt không để lại ảnh ghi dở. Trả về 1 nếu tạo mới, 0 nếu đã có
def resize_image(job):
    source, target, size = job
    if target.exists():
        return 0
    with Image.open(source) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    temporary = target.with_name(f"{target.stem}.part.png")
    image.save(temporary, format = "PNG")
    os.replace(temporary, target)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Tạo bộ ảnh NIH ChestX-ray14 ở kích thước khác trong dataset/images_<kích thước>/")
    parser.add_argument("size", type = int, help = "Kích thước ảnh, chia hết cho 32 (ví dụ 448)")
    parser.add_argument("--workers", type = int, default = max(1, (os.cpu_count() or 2) - 2), help = "Số tiến trình resize song song")
    args = parser.parse_args()

    base_size = int(json.loads(config.DATASET_INFO_DIRECTORY.read_text(encoding = "utf-8"))["image_size"])
    if args.size <= 0 or args.size % 32:
        parser.error("Kích thước phải là số dương chia hết cho 32")
    if args.size == base_size:
        parser.error(f"Ảnh {base_size} px là bộ ảnh gốc của tiền xử lý, đã có trong {config.DATASET_DIRECTORY / 'images'}")

    # Mọi ảnh của train_folds.csv và test.csv, mỗi ảnh một lần
    images = pd.concat([pd.read_csv(config.TRAIN_FOLDS_DIRECTORY), pd.read_csv(config.TEST_DIRECTORY)]).drop_duplicates(subset = [config.IMAGE_ID_COLUMN])
    target_directory = config.DATASET_DIRECTORY / f"images_{args.size}"
    target_directory.mkdir(parents = True, exist_ok = True)
    for partial in target_directory.glob("*.part.png"):
        partial.unlink()
    jobs = [(SOURCE_DIRECTORY / str(original), target_directory / str(name), args.size) for name, original in zip(images[config.IMAGE_ID_COLUMN], images["original_path"])]

    # Kiểm tra đủ ảnh gốc trước khi chạy (so tên tệp với danh sách thư mục, nhanh hơn kiểm tra từng tệp)
    listed = {}
    for source, _, _ in jobs:
        if source.parent not in listed:
            listed[source.parent] = set(os.listdir(source.parent)) if source.parent.is_dir() else set()
    missing = [str(source) for source, _, _ in jobs if source.name not in listed[source.parent]]
    if missing:
        raise FileNotFoundError(f"Thiếu {len(missing):,} ảnh gốc, ví dụ: {missing[0]}")

    with ProcessPoolExecutor(max_workers = args.workers) as pool:
        created = sum(TqdmComma(pool.map(resize_image, jobs, chunksize = 64), total = len(jobs), desc = f"Resize {args.size} px"))
    print(f"Xong: tạo mới {created:,} ảnh, đã có sẵn {len(jobs) - created:,} ảnh | {target_directory}")
