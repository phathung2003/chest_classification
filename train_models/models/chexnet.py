# ============================================================
# Mô hình: CheXNet (Rajpurkar et al., 2017)
# Mục tiêu: Tái hiện mô hình CheXNet theo bài báo
# Kiến trúc: Ảnh 224 x 224 --> DenseNet-121 đầy đủ (features + GAP + classifier mới) --> logits [B, 14].
# ============================================================

# Khai báo utilities
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Chế độ tất định của cuBLAS
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Khai báo thư viện
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from training.experiment import start_train
from utilities.dataset import NIHChestXrayDataset

# ============================================================
# Cài đặt mô hình
MODEL_NAME = "CheXNet_Paper_14_v2"                                          # Tên mô hình

# ============================================================
# Mô hình
class CheXNet(nn.Module):

    # Khởi tạo mô hình
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = densenet121(weights = DenseNet121_Weights.IMAGENET1K_V1)                       # DenseNet-121 đầy đủ (features + GAP + classifier)
        self.cnn.classifier = nn.Linear(self.cnn.classifier.in_features, num_classes)             # Thay classifier 1000 lớp bằng num_classes lớp

    # Lan truyền xuôi (Forward)
    # Trả về: logits [batch_size, num_classes]
    def forward(self, x):

        # Kiểm tra dữ liệu đầu vào
        ## Ảnh đầu vào phải có dạng [batch_size, 3, H, W]
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError(
                "[CheXNet]: Ảnh đầu vào phải có dạng [batch_size, 3, H, W].\n"
                f"Nhận được: x = {tuple(x.shape)}"
            )

        return self.cnn(x)  # logits: [batch, num_classes]

# ============================================================
# Cấu hình huấn luyện
LOSS_CONFIG = {"name": "chexnet_bce"}                                       # BCE: Tổng theo lớp, trung bình theo batch

SCHEDULER_CONFIG = {
    "name": "reduce_on_plateau",
    "mode": "min",                                                          # Theo dõi validation loss
    "factor": 0.1,                                                          # Giảm learning rate 10 lần
}

OPTIMIZER_CONFIG = {
    "name": "adam",
    "betas": (0.9, 0.999),
}

SELECTION_METRIC = "validation_loss"

# ============================================================
# Chạy huấn luyện
if __name__ == "__main__":
    start_train(
        model_name = MODEL_NAME,
        dataset_class = NIHChestXrayDataset,
        model_factory = CheXNet,
        model_kwargs = {},
        loss_config = LOSS_CONFIG,
        scheduler_config = SCHEDULER_CONFIG,
        optimizer_config = OPTIMIZER_CONFIG,
        selection_metric = SELECTION_METRIC,
    )
