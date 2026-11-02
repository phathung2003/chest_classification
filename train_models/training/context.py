# ============================================================
# Thông tin thí nghiệm dùng chung (Experiment Context)
# Mục tiêu: Gom mọi thông tin cố định của một lượt chạy (mô hình, dữ liệu, cấu hình, thiết bị) vào một đối tượng để truyền cho các hàm huấn luyện / đánh giá, thay vì truyền hàng chục tham số rời rạc.
# ============================================================

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable

import torch

# ============================================================
# Thông tin huấn luyện 
@dataclass(frozen = True)                       # frozen = True: Không sửa được sau khi tạo
class ExperimentContext:
    
    # Thông tin lần chạy
    output_directory    : Path                  # Thư mục lượt chạy
    signature           : str                   # Chữ ký cấu hình
    
    # Cấu hình mô hình
    model_name          : str                   # Tên mô hình
    model_factory       : Callable              # Lớp mô hình: model_factory(num_classes = ..., **model_kwargs)
    model_kwargs        : dict                  # Tham số mô hình
    
    # Thông tin tập dữ liệu
    dataset_class       : Callable              # Lớp Dataset: dataset_class(dataframe = ..., classes = ..., transform = ...)
    classes             : list                  # Danh sách Class
    train_transform     : Any                   # Cấu hình làm giàu ảnh (tập huấn luyện)
    eval_transform      : Any                   # Cấu hình làm giàu ảnh (kiểm thử/kiểm tra)
    
    # Cấu hình huấn luyện
    loss_config         : dict                  # Cấu hình hàm Loss
    training_config     : dict                  # Kết quả của build_training_config()
    device              : torch.device          # Thiết bị huấn luyện
    
    # Số lớp trong tập dữ liệu(Class)
    @property
    def number_of_classes(self):
        return len(self.classes)

    # Batch size
    @property
    def batch_size(self):
        return int(self.training_config["batch_size"])

    # Thư mục kết quả của một Fold
    def fold_directory(self, fold):
        return self.output_directory / f"fold_{int(fold)}"
