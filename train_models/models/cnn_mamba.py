# ============================================================
# Mô hình CNN Mamba v1 (DenseNet-121 + Mamba)
# Hướng đi: Thêm khối Mamba sau DenseNet-121 để trộn thông tin giữa các vùng ảnh (token) trước khi phân loại.

# Kiến trúc (ảnh 224 x 224):
# - DenseNet-121: [B, 1024, 7, 7] --> chiếu xuống [B, 256, 7, 7] --> chuỗi 49 token [B, 49, 256].
# - Mamba 2 lớp --> gộp trung bình + cực đại theo token [B, 512] --> Linear --> logits [B, 14].
# ============================================================

# Khai báo utilities
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Chế độ tất định của cuBLAS
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import densenet121, DenseNet121_Weights
from mambapy.mamba import Mamba, MambaConfig

import config
from training.experiment import start_train
from utilities.dataset import NIHChestXrayDataset

# ============================================================
# Cài đặt mô hình
MODEL_NAME = "CNN Mamba_v1.2.1"                                             # Tên mô hình
MAMBA_DIM = 256                                                             # Số chiều của token
MAMBA_LAYERS = 2                                                            # Số lớp Mamba
DROPOUT = 0.35                                                              # Dropout trước classifier
PRETRAINED_CNN = True                                                       # DenseNet-121 dùng trọng số ImageNet
dataset_class = NIHChestXrayDataset                                         # Lớp Dataset

# ============================================================
# Mô hình
class CNNMamba(nn.Module):

    # Khởi tạo mô hình
    def __init__(self, num_classes):
        super().__init__()

        # DenseNet121
        weights = (DenseNet121_Weights.IMAGENET1K_V1 if PRETRAINED_CNN else None)
        backbone = densenet121(weights = weights)

        # Bỏ classifier gốc (Output: [B, 1024, 7, 7] khi input = 224 x 224.)
        self.cnn = (backbone.features)

        # Projection: 1024 kênh --> MAMBA_DIM kênh
        self.projection = nn.Conv2d(
            in_channels = 1024,
            out_channels = MAMBA_DIM,
            kernel_size = 1,
            bias = False,
        )

        # Chuẩn hóa sau projection
        self.projection_norm = (nn.BatchNorm2d(MAMBA_DIM))

        # Mamba
        mamba_config = MambaConfig(
            d_model = MAMBA_DIM,
            n_layers = MAMBA_LAYERS,
        )

        self.mamba = Mamba(mamba_config)
        self.final_norm = (nn.LayerNorm(MAMBA_DIM))

        # Classifier: Đầu vào là [gộp trung bình, gộp cực đại] --> 2 * MAMBA_DIM
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(MAMBA_DIM * 2, num_classes)

    # Lan truyền xuôi (Forward)
    # Trả về: logits [batch_size, num_classes]
    def forward(self, x):

        # Kiểm tra dữ liệu đầu vào
        ## Ảnh đầu vào phải có dạng [batch_size, 3, H, W]
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError(
                "[CNNMamba]: Ảnh đầu vào phải có dạng [batch_size, 3, H, W].\n"
                f"Nhận được: x = {tuple(x.shape)}"
            )

        # CNN feature extractor (Expected: [B, 1024, 7, 7])
        x = self.cnn(x)
        x = F.relu(x, inplace = False)

        # Projection (Expected: [B, 256, 7, 7])
        x = self.projection(x)
        x = self.projection_norm(x)
        x = F.silu(x)

        # 2D feature map -> sequence | [B, D, H, W] -> [B, H*W, D] (Expected: [B, 49, 256])
        x = x.flatten(start_dim = 2)
        x = x.transpose(1, 2)

        # Mamba
        x = self.mamba(x)
        x = self.final_norm(x)

        # Sequence pooling (Expected: [B, 256])
        avg_pool = x.mean(dim = 1)
        max_pool = x.amax(dim = 1)

        # Ghép gộp trung bình và gộp cực đại (Expected: [B, 512])
        x = torch.cat([avg_pool, max_pool], dim = 1,)

        # Classification
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits

# ============================================================
# Chạy huấn luyện
if __name__ == "__main__":
    start_train(
        model_name     = MODEL_NAME,
        dataset_class  = NIHChestXrayDataset,
        model_factory  = CNNMamba,
        model_kwargs   = {},
        **config.EXPERIMENT_PROTOCOL,
    )
