import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

from training.experiment import run_experiment
from utilities.dataset import NIHChestXrayDataset

class CheXNet(nn.Module):
    def __init__(self, num_classes = 14, weights = "imagenet", checkpoint_path = None):
        super().__init__()
        if num_classes != 14:
            raise ValueError("Checkpoint/kiến trúc này dùng 14 bệnh của NIH ChestXray14.")
        if weights not in ("imagenet", "chexnet", "random"):
            raise ValueError("weights phải là 'imagenet', 'chexnet' hoặc 'random'.")

        imagenet_weights = (DenseNet121_Weights.IMAGENET1K_V1
                            if weights == "imagenet" else None)
        self.densenet121 = densenet121(weights = imagenet_weights)
        in_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(in_features, num_classes)

        if weights == "chexnet":
            if checkpoint_path is None:
                raise ValueError("weights='chexnet' yêu cầu checkpoint_path.")
            load_original_chexnet_weights(self.densenet121, checkpoint_path)
        elif checkpoint_path is not None:
            raise ValueError("checkpoint_path chỉ dùng khi weights='chexnet'.")

    def forward(self, x):
        # Batch ảnh đơn: [B,3,H,W] -> logits [B,14].
        if x.ndim == 4:
            return self.densenet121(x)

        # Dành cho ten-crop inference: [B,10,3,224,224].
        # Trung bình XÁC SUẤT trên 10 crop như bản code CheXNet gốc;
        # đổi ngược về logits để không bị sigmoid lần thứ hai.
        if x.ndim == 5:
            if x.shape[1] != 10:
                raise ValueError("Dữ liệu nhiều crop phải có đúng 10 crops.")
            summed_prob = None
            for crop_id in range(x.shape[1]):
                logits = self.densenet121(x[:, crop_id])
                prob = torch.sigmoid(logits.float())
                summed_prob = prob if summed_prob is None else summed_prob + prob
            mean_prob = summed_prob / x.shape[1]
            return torch.logit(mean_prob.clamp(1e-6, 1.0 - 1e-6))

        raise ValueError(f"Input phải có 4 hoặc 5 chiều; nhận {tuple(x.shape)}.")


MODEL_NAME = "CheXNet DenseNet121 Baseline_v1"
LOSS_CONFIG = {"name": "chexnet_bce"}

SCHEDULER_CONFIG = {
    "name": "reduce_on_plateau",
    "mode": "min",
    "factor": 0.1,
}

OPTIMIZER_CONFIG = {
    "name": "adam",
    "betas": (0.9, 0.999),
}

SELECTION_METRIC = "validation_loss"

if __name__ == "__main__":
    run_experiment(
        model_name = MODEL_NAME,
        dataset_class = NIHChestXrayDataset,
        model_factory = CheXNet,
        model_kwargs = {},
        loss_config = LOSS_CONFIG,
        scheduler_config = SCHEDULER_CONFIG,
        optimizer_config = OPTIMIZER_CONFIG,
        selection_metric = SELECTION_METRIC,
    )
