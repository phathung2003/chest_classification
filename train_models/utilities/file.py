# ============================================================
# Đọc / ghi tệp (File I/O)
# Mục tiêu: Ghi tệp kết quả (JSON, CSV, checkpoint)
# ============================================================

import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import torch

# ============================================================
# Ghi tệp an toàn (Atomic Write)
def atomic_write(path, writer):
    
    ## Kiểm tra writer phải là hàm
    if not callable(writer):
        raise TypeError(
            "[atomic_write]: writer phải là hàm nhận đường dẫn tệp tạm.\n"
            f"Nhận được: writer = {type(writer).__name__}"
        )

    # Chuyển đổi dữ liệu
    path = Path(path)

    # Tạo thư mục chứa tệp đích
    path.parent.mkdir(parents = True, exist_ok = True)

    # Tạo tệp tạm cùng thư mục
    fd, temporary = tempfile.mkstemp(prefix = f".{path.name}.", suffix = ".tmp", dir = path.parent)
    os.close(fd)

    try:

        # Ghi dữ liệu vào tệp tạm
        writer(Path(temporary))

        # Đẩy dữ liệu xuống đĩa (fsync)
        with open(temporary, "rb+") as file:
            os.fsync(file.fileno())

        # Thay thế tệp đích bằng tệp tạm
        os.replace(temporary, path)

        # Đồng bộ metadata thư mục (POSIX)
        if os.name == "posix":
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

    # Dọn tệp tạm nếu xảy ra lỗi
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

# ============================================================
# Chuẩn hóa dữ liệu trước khi ghi JSON
# Trả về: Dữ liệu chỉ gồm các kiểu hợp lệ của JSON
def json_clean(data):

    # dict, list, tuple: Chuẩn hóa từng phần tử
    if isinstance(data, dict):
        return {str(key): json_clean(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [json_clean(value) for value in data]

    # Kiểu NumPy --> kiểu Python
    if isinstance(data, np.ndarray):
        return json_clean(data.tolist())
    if isinstance(data, np.generic):
        return json_clean(data.item())

    # NaN / Inf không hợp lệ trong JSON --> null
    if isinstance(data, float) and not np.isfinite(data):
        return None

    # set --> danh sách đã sắp xếp (thứ tự cố định), Path --> chuỗi
    if isinstance(data, (set, frozenset)):
        return sorted(json_clean(value) for value in data)
    if isinstance(data, Path):
        return str(data)

    return data

# ============================================================
# Ghi JSON (an toàn, UTF-8, không cho phép NaN)
def save_json(path, data):

    # Hàm ghi dữ liệu vào tệp tạm
    def write(temporary):
        with temporary.open("w", encoding = "utf-8") as file:
            json.dump(json_clean(data), file, ensure_ascii = False, indent = 2, allow_nan = False)

    # Ghi an toàn
    atomic_write(path, write)

# ============================================================
# Đọc JSON
# Trả về: Nội dung tệp JSON
def read_json(path):
    with Path(path).open("r", encoding = "utf-8") as file:
        return json.load(file)

# ============================================================
# Ghi checkpoint PyTorch (an toàn)
def save_torch(path, payload):
    atomic_write(path, lambda temporary: torch.save(payload, temporary))

# ============================================================
# Đọc checkpoint PyTorch
# Trả về: Nội dung checkpoint
def load_torch(path, device = "cpu"):
    return torch.load(path, map_location = device, weights_only = True)

# ============================================================
# Ghi CSV
def save_csv(path, frame):
    atomic_write(path, lambda temporary: frame.to_csv(temporary, index = False))

# ============================================================
# Đổi tên sang dạng snake_case ("CNN SymMamba v1 sym4" --> "cnn_symmamba_v1_sym4")
# Trả về: Chuỗi snake_case
def to_snake_case(text):
    
    # Kiểm tra text phải là chuỗi
    if not isinstance(text, str):
        raise TypeError(
            "[to_snake_case]: text phải là chuỗi.\n"
            f"Nhận được: text = {text!r} ({type(text).__name__})"
        )

    # Bỏ dấu gạch ngang, chuyển chữ thường
    text = text.replace("-", "").lower()

    # Thay chuỗi ký tự không phải chữ / số bằng "_"
    text = re.sub(r"[^a-z0-9]+", "_", text)

    # Bỏ "_" ở hai đầu
    return text.strip("_")
