
# Khai báo utilities
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


import models.chexnet as chexnet
import models.cnn_bimamba as bimamba
import models.cnn_mamba as mamba
import models.densenet121 as densenet

# Khai báo thư viện
import torch
import torch.nn as nn

from torchvision.models import densenet121, DenseNet121_Weights

import config
from training.experiment import run_experiment
from utilities.dataset import NIHChestXrayDataset

# ============================================================
# MODEL SETTINGS
MODEL_NAME = "CheXNet_v1"
PRETRAINED_CNN = True

dataset_class = NIHChestXrayDataset

# ============================================================
# Loss, optimizer, scheduler, tiêu chí chọn mô hình: dùng GIAO THỨC CHUNG trong config.py
# để so sánh công bằng giữa các kiến trúc. Không khai báo riêng ở đây.

# ============================================================
# Run Experiment: cùng giao thức, chỉ khác kiến trúc
EXPERIMENTS = [
    ("DenseNet121 Baseline_v1", densenet.DenseNetBaseline),
    (mamba.MODEL_NAME, mamba.CNNMamba),
    (bimamba.MODEL_NAME, bimamba.CNNMamba),
]

if __name__ == "__main__":
    for model_name, model_factory in EXPERIMENTS:
        run_experiment(
            model_name = model_name,
            dataset_class = NIHChestXrayDataset,
            model_factory = model_factory,
            model_kwargs = {},
            **config.EXPERIMENT_PROTOCOL,
        )
