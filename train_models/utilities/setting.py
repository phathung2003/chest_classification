import os
import random
import numpy as np
import torch
from utilities.log import print_heading, print_message
from utilities.runtime import runtime


def device_info():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_heading('Cấu hình huấn luyện')
    print_message('Huấn luyện trên', 'GPU CUDA' if device.type == 'cuda' else 'CPU')
    if device.type == 'cuda':
        print_message('GPU sử dụng', torch.cuda.get_device_name(0))
        print_message('VRAM GPU', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    print('\n')
    return device


def set_seed(seed=None, CUDNN_DETERMINISTIC=None,
             CUDNN_BENCHMARK=None, DETERMINISTIC_ALGORITHMS=None):
    seed = int(runtime.SEED if seed is None else seed)
    deterministic = bool(runtime.CUDNN_DETERMINISTIC if CUDNN_DETERMINISTIC is None else CUDNN_DETERMINISTIC)
    benchmark = bool(runtime.CUDNN_BENCHMARK if CUDNN_BENCHMARK is None else CUDNN_BENCHMARK)
    algorithms = bool(runtime.DETERMINISTIC_ALGORITHMS if DETERMINISTIC_ALGORITHMS is None else DETERMINISTIC_ALGORITHMS)
    # Set CUBLAS_WORKSPACE_CONFIG before importing torch in the Python process
    # for strongest CUDA reproducibility (set it in the shell before launch).
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.use_deterministic_algorithms(algorithms)


def set_fold(number_of_folds):
    if int(number_of_folds) < 2:
        raise ValueError('NUMBER_OF_FOLDS must be >= 2.')
    if runtime.TRAIN_ALL_FOLD:
        return list(range(1, int(number_of_folds) + 1))
    selected = runtime.SELECTED_FOLD
    if isinstance(selected, (list, tuple)):
        folds = [int(fold) for fold in selected]
    else:
        folds = [int(selected)]
    if not folds or len(set(folds)) != len(folds) or any(
        fold not in range(1, int(number_of_folds) + 1) for fold in folds
    ):
        raise ValueError('Invalid SELECTED_FOLD; use an integer or list of unique folds.')
    return folds
