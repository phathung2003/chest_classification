import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import config


# ============================================================
# Đọc và kiểm tra tập dữ liệu đã tiền xử lý: dataset_info.json, train_folds.csv (đã gán Fold) và test.csv
def load_dataset():
    for path in (config.DATASET_INFO_DIRECTORY, config.TRAIN_FOLDS_DIRECTORY, config.TEST_DIRECTORY):
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy:\n {path}")

    with open(config.DATASET_INFO_DIRECTORY, "r", encoding = "utf-8") as f:
        dataset_config = json.load(f)
    name = dataset_config["dataset_name"]
    classes = list(dataset_config["classes"])
    number_of_folds = int(dataset_config["number_of_folds"])
    train_df = pd.read_csv(config.TRAIN_FOLDS_DIRECTORY)
    test_df = pd.read_csv(config.TEST_DIRECTORY)

    validate_data(train_df, test_df, classes, number_of_folds)
    return train_df, test_df, name, classes, number_of_folds


# ============================================================
# Kiểm tra trước khi huấn luyện: đủ cột, đủ fold, nhãn nhị phân, không rò rỉ ảnh / bệnh nhân
def validate_data(train_df, test_df, classes, number_of_folds):

    # Đủ cột bắt buộc, không rỗng
    missing_train = [c for c in ["image_path", "Fold", *classes] if c not in train_df.columns]
    missing_test = [c for c in ["image_path", *classes] if c not in test_df.columns]
    if missing_train or missing_test:
        raise ValueError(f"Thiếu cột. train_folds.csv: {missing_train} | test.csv: {missing_test}")
    if train_df.empty or test_df.empty:
        raise ValueError("Tập huấn luyện và tập Test không được rỗng")

    # Đủ fold
    actual_folds = set(train_df["Fold"].astype(int).unique())
    expected_folds = set(range(1, int(number_of_folds) + 1))
    if actual_folds != expected_folds:
        raise ValueError(f"Số fold thực tế: {actual_folds}\nSố fold mong đợi: {expected_folds}")

    # Nhãn nhị phân
    for name, frame in (("train", train_df), ("test", test_df)):
        if not np.isin(frame[classes].to_numpy(), [0, 1]).all():
            raise ValueError(f"Tập {name} có nhãn không phải 0/1 hoặc bị thiếu")

    # Không trùng ảnh giữa huấn luyện và Test
    if set(train_df["image_path"].astype(str)) & set(test_df["image_path"].astype(str)):
        raise ValueError("Phát hiện ảnh nằm ở cả tập huấn luyện và tập Test")

    # Không trùng bệnh nhân giữa huấn luyện / Test và giữa các fold
    patient = config.PATIENT_ID_COLUMN
    if patient not in train_df or patient not in test_df:
        print(f"CẢNH BÁO: không có cột {patient}, không kiểm tra được việc chia theo bệnh nhân.")
        return
    if train_df[patient].isna().any() or test_df[patient].isna().any():
        raise ValueError("Mã bệnh nhân không được thiếu")
    if set(train_df[patient].astype(str)) & set(test_df[patient].astype(str)):
        raise ValueError("Phát hiện bệnh nhân nằm ở cả tập huấn luyện và tập Test")
    if (train_df.groupby(patient)["Fold"].nunique() > 1).any():
        raise ValueError("Phát hiện bệnh nhân nằm ở nhiều fold")


# ============================================================
# Biến đổi ảnh: làm giàu ảnh cho tập huấn luyện, chỉ chuẩn hóa cho Validation / Test
def transform_setting():
    normalize = transforms.Normalize(mean = config.TRANSFORM_NORMALIZE_MEAN, std = config.TRANSFORM_NORMALIZE_STD)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p = config.FLIP_CHANCE),
        transforms.RandomAffine(degrees = config.DEGREES, translate = config.TRANSLATE, scale = config.SCALE, fill = 0, interpolation = InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness = config.BRIGHTNESS, contrast = config.CONTRAST),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


# ============================================================
# Seed cho từng worker: PyTorch đặt seed torch theo generator của DataLoader, hàm này đồng bộ thêm numpy / random
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# Tạo DataLoader. Seed quyết định thứ tự xáo trộn và seed của worker (tức là augmentation).
# Train loader được tạo lại mỗi epoch với seed riêng và persistent_workers = False để worker nhận seed mới.
def create_dataloader(dataset_class, classes, device, dataframe, transform, shuffle, *,
                      batch_size, seed, persistent_workers = None):
    dataset = dataset_class(dataframe = dataframe, classes = classes, transform = transform)

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    workers = int(config.NUMBER_OF_WORKERS)
    if persistent_workers is None:
        persistent_workers = bool(config.PERSISTENT_WORKERS)

    loader = DataLoader(dataset, batch_size = int(batch_size), shuffle = shuffle, num_workers = workers, prefetch_factor = int(config.PREFETCH_FACTOR) if workers > 0 else None, pin_memory = device.type == "cuda", drop_last = False, worker_init_fn = _seed_worker, generator = generator, persistent_workers = bool(persistent_workers and workers > 0))
    return dataset, loader
