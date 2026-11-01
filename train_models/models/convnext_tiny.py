# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import timm
import torch.nn as nn

import config
from training.experiment import run_experiment
from utilities.dataset import NIHChestXrayDataset

# ============================================================
# MODEL SETTINGS
MODEL_NAME = "ConvNeXt-T Baseline_v1"

# ============================================================
# Mô hình: ConvNeXt-T đầy đủ 4 stage (ImageNet-1k, timm), GAP + LayerNorm của timm, Linear 14 lớp.
# Baseline E1 của nghiên cứu CLIS: cùng backbone với CLIS-Mamba nhưng giữ stage 4 thay cho các khối trộn token
class ConvNeXtTinyBaseline(nn.Module):
    PRETRAINED_MODULES = ("cnn",)                                  # Khối dùng trọng số ImageNet (nhóm LR backbone)

    def __init__(self, num_classes, pretrained = True):
        super().__init__()
        self.cnn = timm.create_model("convnext_tiny.fb_in1k", pretrained = pretrained, num_classes = 0)
        self.classifier = nn.Linear(self.cnn.num_features, num_classes)

    def forward(self, x):
        return self.classifier(self.cnn(x))                        # cnn: [B, 768] sau GAP + LayerNorm

if __name__ == "__main__":
    from utilities.setting import apply_config

    apply_config(config.CLIS_CONFIG)
    run_experiment(
        model_name     = f"{MODEL_NAME} CLIS {config.IMAGE_SIZE}px",
        dataset_class  = NIHChestXrayDataset,
        model_factory  = ConvNeXtTinyBaseline,
        model_kwargs   = {},
        **config.CLIS_PROTOCOL,
    )
