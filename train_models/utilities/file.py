import os
import json
import torch
import tempfile
import numpy as np
from pathlib import Path

# ============================================================
# Quy trình lưu file
def atomic_write(path, writer):
    
    # Tạo thư mục
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Tạo file tạm
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    
    try:
        # Ghi dữ liệu vào file tạm
        writer(Path(temporary))
        
        # Đồng bộ dữ liệu
        with open(temporary, "rb") as stream:
            stream.flush()
            # os.fsync(stream.fileno())
        
        # Thay thế file gốc
        os.replace(temporary, path)
        
        # Đồng bộ metadata thư mục.
        if os.name == "posix":
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    
    # Dọn file tạm nếu xảy ra lỗi
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

# ============================================================
# Lưu file JSON  
def save_json(path, data):
    
    # Ghi dữ liệu vào file JSON
    def write(tmp):
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(json_clean(data), f, ensure_ascii = False, indent = 2, allow_nan = False)
    
    # Lưu dữ liệu
    atomic_write(path, write)

# ============================================================
# Đọc file JSON
def read_json(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)

# ============================================================
# Chuẩn hóa dữ liệu file JSON
def json_clean(data):
    # Dữ liệu thuộc dạng Dictionary
    if isinstance(data, dict):
        return {str(k): json_clean(v) for k, v in data.items()}
    
    # Dữ liệu thuộc dạng list, tuple
    if isinstance(data, (list, tuple)):
        return [json_clean(v) for v in data]
    
    # Dữ liệu thuộc dạng mảng NumPy
    if isinstance(data, np.ndarray):
        return json_clean(data.tolist())
    
    # Giá trị thuộc dạng scalar (Numpy)
    if isinstance(data, np.generic):
        return json_clean(data.item())
    
    # Giá trị thuộc dạng số thực (float)
    if isinstance(data, float) and not np.isfinite(data):
        return None
    
    # Giá trị thuộc dạng đường dẫn
    if isinstance(data, Path):
        return str(data)
    
    return data

# ============================================================
# Lưu checkpoint
def save_torch(path, payload):
    atomic_write(path, lambda tmp: torch.save(payload, tmp))

# ============================================================
# Tải checkpoint
def load_torch(path, device='cpu'):
    return torch.load(path, map_location = device, weights_only=True)

# ============================================================
# Lưu file CSV
def save_csv(path, frame):
    atomic_write(path, lambda tmp: frame.to_csv(tmp, index=False))
