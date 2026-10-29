import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Ghi tệp an toàn: ghi vào tệp tạm rồi thay thế, nên tệp đích không bao giờ bị ghi dở khi chương trình dừng giữa chừng
def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    # Tạo tệp tạm cùng thư mục
    fd, temporary = tempfile.mkstemp(prefix = f".{path.name}.", suffix = ".tmp", dir = path.parent)
    os.close(fd)

    try:
        # Ghi dữ liệu rồi thay thế tệp gốc
        writer(Path(temporary))
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
# Chuẩn hóa dữ liệu trước khi ghi JSON: NumPy --> Python, NaN / Inf --> null, Path --> str
def json_clean(data):
    if isinstance(data, dict):
        return {str(k): json_clean(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [json_clean(v) for v in data]
    if isinstance(data, np.ndarray):
        return json_clean(data.tolist())
    if isinstance(data, np.generic):
        return json_clean(data.item())
    if isinstance(data, float) and not np.isfinite(data):
        return None
    if isinstance(data, (set, frozenset)):
        return sorted(json_clean(v) for v in data)
    if isinstance(data, Path):
        return str(data)
    return data


# ============================================================
# Đọc / ghi JSON
def save_json(path, data):
    def write(tmp):
        with tmp.open("w", encoding = "utf-8") as f:
            json.dump(json_clean(data), f, ensure_ascii = False, indent = 2, allow_nan = False)
    atomic_write(path, write)


def read_json(path):
    with Path(path).open("r", encoding = "utf-8") as f:
        return json.load(f)


# ============================================================
# Đọc / ghi checkpoint PyTorch (weights_only = True: không thực thi mã khi tải)
def save_torch(path, payload):
    atomic_write(path, lambda tmp: torch.save(payload, tmp))


def load_torch(path, device = "cpu"):
    return torch.load(path, map_location = device, weights_only = True)


# ============================================================
# Ghi CSV
def save_csv(path, frame):
    atomic_write(path, lambda tmp: frame.to_csv(tmp, index = False))


# ============================================================
# Đổi tên sang dạng snake_case (dùng làm tên thư mục kết quả)
def to_snake_case(text):
    text = text.replace("-", "").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")
