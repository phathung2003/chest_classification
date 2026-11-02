# ============================================================
# Mô hình cơ sở DenseNet-121 (Baseline)
# Mục tiêu: Mô hình đối chứng cho mọi kiến trúc đề xuất: DenseNet-121 tiền huấn luyện ImageNet + GAP + Linear.

# Kiến trúc (ảnh 224 x 224):
# - DenseNet-121: Bản đồ đặc trưng [B, 1024, 7, 7].
# - GAP (trung bình toàn cục) --> [B, 1024] --> Linear --> logits [B, 14].
# ============================================================

# Khai báo utilities: Thêm thư mục train_models vào sys.path để chạy trực tiếp bằng python models/densenet121.py
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Chế độ tất định của cuBLAS: Phải đặt trước khi CUDA khởi tạo
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import densenet121, DenseNet121_Weights

import config
from training.experiment import start_train
from utilities.dataset import NIHChestXrayDataset

# ============================================================
# Cài đặt mô hình
MODEL_NAME = "DenseNet121 Baseline_v1"                                      # Tên mô hình (tên thư mục kết quả)
dataset_class = NIHChestXrayDataset                                         # Lớp Dataset

# ============================================================
# Mô hình: DenseNet-121 --> ReLU --> GAP --> Linear
class DenseNetBaseline(nn.Module):

    # Khởi tạo mô hình
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = densenet121(weights = DenseNet121_Weights.IMAGENET1K_V1).features      # Phần trích đặc trưng (bỏ classifier gốc)
        self.classifier = nn.Linear(1024, num_classes)                                      # Lớp phân loại: 1024 đặc trưng --> num_classes logits

    # Lan truyền xuôi (Forward)
    # Trả về: logits [batch_size, num_classes]
    def forward(self, x):

        # Kiểm tra dữ liệu đầu vào
        ## Ảnh đầu vào phải có dạng [batch_size, 3, H, W]
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError(
                "[DenseNetBaseline]: Ảnh đầu vào phải có dạng [batch_size, 3, H, W].\n"
                f"Nhận được: x = {tuple(x.shape)}"
            )

        # Trích đặc trưng --> GAP --> phân loại
        x = F.relu(self.cnn(x))                                # [B, 1024, H/32, W/32]
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)             # [B, 1024]
        return self.classifier(x)

# ============================================================
# Chạy huấn luyện
if __name__ == "__main__":
    start_train(
        model_name     = MODEL_NAME,
        dataset_class  = NIHChestXrayDataset,
        model_factory  = DenseNetBaseline,
        model_kwargs   = {},
        **config.EXPERIMENT_PROTOCOL,
    )
