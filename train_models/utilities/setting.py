import os
import random

import numpy as np
import torch

import config
from utilities.log import print_heading, print_message


# ============================================================
# Chọn thiết bị huấn luyện và in thông tin GPU
def device_info():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_heading("Thiết bị")
    print_message("Huấn luyện trên", "GPU CUDA" if device.type == "cuda" else "CPU")
    if device.type == "cuda":
        print_message("GPU sử dụng", torch.cuda.get_device_name(0))
        print_message("VRAM GPU", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    return device


# ============================================================
# Đặt seed cho Python / NumPy / PyTorch và bật chế độ tất định theo config.py.
# CUBLAS_WORKSPACE_CONFIG cần có trước khi CUDA khởi tạo (các tệp mô hình đặt ở đầu tệp).
def set_seed(seed = None):
    seed = int(config.SEED if seed is None else seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(config.CUDNN_DETERMINISTIC)
    torch.backends.cudnn.benchmark = bool(config.CUDNN_BENCHMARK)
    torch.use_deterministic_algorithms(bool(config.DETERMINISTIC_ALGORITHMS))


# ============================================================
# Danh sách fold cần chạy: tất cả, hoặc các fold trong SELECTED_FOLD
def set_fold(number_of_folds):
    number_of_folds = int(number_of_folds)
    if number_of_folds < 2:
        raise ValueError("NUMBER_OF_FOLDS phải >= 2")
    if config.TRAIN_ALL_FOLD:
        return list(range(1, number_of_folds + 1))

    selected = config.SELECTED_FOLD
    folds = [int(f) for f in selected] if isinstance(selected, (list, tuple)) else [int(selected)]
    if not folds or len(set(folds)) != len(folds) or any(f not in range(1, number_of_folds + 1) for f in folds):
        raise ValueError("SELECTED_FOLD không hợp lệ: dùng một số nguyên hoặc danh sách fold không trùng lặp")
    return folds
