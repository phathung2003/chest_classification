from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

# ============================================================
# Thông tin huấn luyện
@dataclass(frozen = True)
class ExperimentContext:
    output_directory    : Path                  # Thư mục lượt chạy: training_results/<dataset>/<model>/run_<n>
    signature           : str                   # Chữ ký cấu hình, dùng để chặn chạy tiếp bằng cấu hình khác
    model_name          : str
    model_factory       : Callable              # Lớp mô hình: model_factory(num_classes = ..., **model_kwargs)
    model_kwargs        : dict
    dataset_class       : Callable              # Lớp Dataset: dataset_class(dataframe = ..., classes = ..., transform = ...)
    classes             : list
    loss_config         : dict
    training_config     : dict                  # Kết quả của build_training_config()
    device              : torch.device
    train_transform     : Any
    eval_transform      : Any

    @property
    def number_of_classes(self):
        return len(self.classes)

    @property
    def batch_size(self):
        return int(self.training_config["batch_size"])

    def fold_directory(self, fold):
        return self.output_directory / f"fold_{int(fold)}"
