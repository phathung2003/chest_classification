# Khai báo utilities
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Khai báo thư viện
import json

import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import densenet121, DenseNet121_Weights
from mambapy.mamba import Mamba, MambaConfig

from utilities.config import TRAIN_FOLDS_DIRECTORY, TEST_DIRECTORY, DATASET_INFO_DIRECTORY, MODELS_OUTPUT_DIRECTORY
from utilities.setting import device_info, set_seed, set_fold
from utilities.check import check_file_directory, check_dataset
from utilities.models import dataset_info, transform_setting, run_experiment
from utilities.other import to_snake_case
from utilities.dataset import NIHChestXrayDataset

# ============================================================
# 5. MODEL SETTINGS
# ============================================================
MODEL_NAME = "CNN Mamba v1"
MAMBA_DIM = 256
MAMBA_LAYERS = 2
DROPOUT = 0.20
PRETRAINED_CNN = True
dataset_class = NIHChestXrayDataset

def setup():
    # ============================================================
    # Kiểm tra đường dẫn trong tập dữ liệu
    required_files = [TRAIN_FOLDS_DIRECTORY, TEST_DIRECTORY, DATASET_INFO_DIRECTORY]
    check_file_directory(required_files)

    # ============================================================
    # Đọc thông tin tập dữ liệu
    with open(DATASET_INFO_DIRECTORY, "r", encoding="utf-8") as file:
        dataset_config = json.load(file)

    # ============================================================
    # Thông tin thiết bị huấn luyện
    DEVICE = device_info()

    # ============================================================
    # Thiết lập seed
    set_seed(dataset_config["seed"])

    # ============================================================
    # Làm giàu ảnh
    train_transform, test_transform = transform_setting()
    
    # ============================================================
    # Thông tin tập dữ liệu
    train_folds_df, test_df, DATASET_NAME, CLASSES, NUMBER_OF_CLASSES, NUMBER_OF_FOLDS = dataset_info(dataset_config)
    check_dataset(CLASSES, train_folds_df, test_df)

    FOLD_TRAIN = set_fold(NUMBER_OF_FOLDS)
    OUTPUT_DIRECTORY = (MODELS_OUTPUT_DIRECTORY / to_snake_case(DATASET_NAME) / to_snake_case(MODEL_NAME))
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    run_experiment(
        DEVICE = DEVICE, 
        CLASSES = CLASSES, 
        dataset_class = NIHChestXrayDataset, 
        FOLD_TRAIN = FOLD_TRAIN, 
        OUTPUT_DIRECTORY = OUTPUT_DIRECTORY, 
        train_folds_df = train_folds_df, 
        test_df = test_df, 
        model_factory=CNNMamba,
        train_transform = train_transform,
        test_transform = test_transform,
        model_kwargs={

        }
    )

# ============================================================
# Mô hình: CNN + MAMBA MODEL

class CNNMamba(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        
        # DenseNet121
        weights = (DenseNet121_Weights.IMAGENET1K_V1 if PRETRAINED_CNN else None)
        backbone = densenet121(weights=weights)

        # Bỏ classifier gốc (Output: [B, 1024, 7, 7] khi input = 224 x 224.)
        self.cnn = (backbone.features)
        
        # Projection
        self.projection = nn.Conv2d(
            in_channels=1024,
            out_channels=MAMBA_DIM,
            kernel_size=1,
            bias=False,
        )
        
        self.projection_norm = (nn.BatchNorm2d(MAMBA_DIM))
        
        # Mamba
        mamba_config = MambaConfig(
            d_model=MAMBA_DIM,
            n_layers=MAMBA_LAYERS,
        )
        
        self.mamba = Mamba(mamba_config)
        self.final_norm = (nn.LayerNorm(MAMBA_DIM))
        
        # Classifier
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(MAMBA_DIM, num_classes)

    def forward(self, x):
        
        # CNN feature extractor (Expected: [B, 1024, 7, 7])
        x = self.cnn(x)
        x = F.relu(x, inplace=False)

        # Projection (Expected: [B, 256, 7, 7])
        x = self.projection(x)
        x = self.projection_norm(x)
        x = F.silu(x)
        
        # 2D feature map -> sequence | [B, D, H, W] -> [B, H*W, D] (Expected: [B, 49, 256])
        x = x.flatten(start_dim=2)
        x = x.transpose(1, 2)
        
        # Mamba
        x = self.mamba(x)
        x = self.final_norm(x)

        # Sequence pooling (Expected: [B, 256])
        x = x.mean(dim=1)
        
        # Classification
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits




if __name__ == "__main__":
    setup()

