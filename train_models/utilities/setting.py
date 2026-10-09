# Khai báo utilities
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import torch
import random
import numpy as np
from utilities.notice import print_heading, print_message
from utilities.config import NUMBER_OF_FOLDS, TRAIN_ALL_FOLD, SELECTED_FOLD

def device_info():
    
    DEVICE = torch.device("cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    
    print_heading("Cấu hình huấn luyện")
    print_message("Huấn luyện trên", "GPU CUDA" if DEVICE == "cuda" else "CPU")


    if DEVICE.type == "cuda":
        gpu_model   = torch.cuda.get_device_name(0)
        memory      = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print_message("GPU sử dụng", gpu_model)
        print_message("VRAM GPU"   , f"{memory:.2f} GB")
    
    return DEVICE

def set_seed(SEED, CUDNN_DETERMINISTIC = True, CUDNN_BENCHMARK = False, DETERMINISTIC_ALGORITHMS = True):

    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # cuDNN
    torch.backends.cudnn.deterministic  = CUDNN_DETERMINISTIC 
    torch.backends.cudnn.benchmark      = CUDNN_BENCHMARK
    
    # PyTorch deterministic
    torch.use_deterministic_algorithms(DETERMINISTIC_ALGORITHMS)
    
    print_heading("Cấu hình thiết lập ngẫu nhiên")
    print_message("Giá trị khởi tạo"            , SEED)
    print_message("cuDNN deterministic"         , CUDNN_DETERMINISTIC)
    print_message("cuDNN benchmark"             , CUDNN_BENCHMARK)
    print_message("Deterministic algorithms"    , DETERMINISTIC_ALGORITHMS)
    
def set_fold(NUMBER_OF_FOLDS):
    if not isinstance(NUMBER_OF_FOLDS, int) or NUMBER_OF_FOLDS < 2:
        raise ValueError("NUMBER_OF_FOLDS phải >= 2")
    
    if TRAIN_ALL_FOLD == True:
        return list(range(1, NUMBER_OF_FOLDS + 1))

    if type(SELECTED_FOLD) is not int or not 1 <= SELECTED_FOLD <= NUMBER_OF_FOLDS:
        print("Fold 1 được chọn mặc định vì SELECTED_FOLD không hợp lệ")
        return 1
    
    return SELECTED_FOLD