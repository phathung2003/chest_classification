# Khai báo utilities
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import pandas as pd
import config as config
from torchvision import transforms
from torch.utils.data import DataLoader
from utilities.log import print_heading, print_message

# ============================================================
# Thông tin tập dữ liệu
def dataset_info(dataset_config):
    DATASET_NAME        = dataset_config["dataset_name"]
    CLASSES             = dataset_config["classes"]
    NUMBER_OF_CLASSES   = len(CLASSES)
    NUMBER_OF_FOLDS     = dataset_config["number_of_folds"]
    
    train_folds_df = pd.read_csv(config.TRAIN_FOLDS_DIRECTORY)
    test_df = pd.read_csv(config.TEST_DIRECTORY)
    
    print_heading("Thông tin tập dữ liệu")
    print_message("Tên tập dữ liệu"             , DATASET_NAME)
    print_message("Số ảnh tập huấn luyện"       , f"{len(train_folds_df):,} ảnh")
    print_message("Số ảnh tập kiểm tra"         , f"{len(test_df):,} ảnh")
    print_message("Kích cỡ ảnh"                 , f"{dataset_config['image_size']} x {dataset_config['image_size']}")
    print_message("Số phần được chia"           , NUMBER_OF_FOLDS)
    print_message("Số lớp"                      , NUMBER_OF_CLASSES)
    
    print("Các lớp trong tập dữ liệu")
    for index, class_name in enumerate(CLASSES, start=1):
        print( f"{index:>2}. "f"{class_name}")
    
    return train_folds_df, test_df, DATASET_NAME, CLASSES, NUMBER_OF_CLASSES, NUMBER_OF_FOLDS

# ============================================================
# Cấu hình làm giàu ảnh
def transform_setting():
    train_transform = transforms.Compose([

    transforms.RandomAffine(degrees = config.DERGEES, translate = config.TRANSLATE, scale = config.SCALE),
        transforms.ToTensor(),
        transforms.Normalize(mean = config.TRANSFORM_NORMALIZE_MEAN, std = config.TRANSFORM_NORMALIZE_STD),
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = config.TRANSFORM_NORMALIZE_MEAN, std = config.TRANSFORM_NORMALIZE_STD),
    ])
    
    return train_transform, eval_transform

# ============================================================
# Dataloader
def create_dataloader(dataset_class, CLASSES, DEVICE, dataframe, transform, shuffle):
    dataset = dataset_class(dataframe=dataframe, classes=CLASSES, transform=transform,)

    loader = DataLoader(
        dataset,
        batch_size      = config.BATCH_SIZE,
        shuffle         = shuffle,
        num_workers     = config.NUMBBER_OF_WORKERS,
        prefetch_factor = 2 if config.NUMBBER_OF_WORKERS > 0 else None,
        pin_memory      = (DEVICE.type == "cuda"),
        drop_last=False,
    )

    return (dataset, loader)

# ============================================================
# Pos Weight
def pos_weight(fold, DEVICE):
    
    if fold == 0:
        path = (config.DATASET_DIRECTORY / "pos_weight.npy").resolve()
    else:
        path = (config.FOLDS_DIRECTORY / f"fold_{fold}" / "pos_weight.npy").resolve()
    
    # Kiểm tra tệp tin 
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    
    weights = np.load(path, allow_pickle=False)
    
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)
