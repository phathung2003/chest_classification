import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from training.experiment import run_experiment
from utilities.dataset import NIHChestXrayDataset

MODEL_NAME = "CheXNet_Paper_14_v2"

class CheXNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = densenet121(weights = DenseNet121_Weights.IMAGENET1K_V1)
        self.cnn.classifier = nn.Linear(self.cnn.classifier.in_features, num_classes)

    def forward(self, x):
        return self.cnn(x)  # logits: [batch, num_classes]

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
