# ============================================================
# Thiết lập môi trường chạy (Runtime Setting)
# Mục tiêu: Chọn thiết bị, cố định seed và chế độ tất định để kết quả huấn luyện tái lập được, giữ máy chạy liên tục trong lúc huấn luyện và chọn các Fold cần chạy.
# ============================================================

import os
import random

import numpy as np
import torch

import config
from utilities.log import print_heading, print_message

# ============================================================
# Thông tin thiết bị huấn luyện
# Trả về: torch.device
def device_info():

    # Chọn thiết bị (GPU CUDA nếu có, ngược lại là CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # In thông tin thiết bị
    print_heading("Thông số thiết bị")
    print_message("Huấn luyện trên", "GPU CUDA" if device.type == "cuda" else "CPU")

    # In tên GPU và dung lượng VRAM
    if device.type == "cuda":
        print_message("GPU sử dụng", torch.cuda.get_device_name(0))
        print_message("VRAM GPU", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

    return device

# ============================================================
# Đặt seed và bật chế độ tất định (theo config.py)
# - CUBLAS_WORKSPACE_CONFIG cần có trước khi CUDA khởi tạo (các tệp mô hình đặt ở đầu tệp).
def set_seed(seed = None):

    # Lấy dữ liệu
    random_seed = int(config.SEED if seed is None else seed)
    
    # Kiểm tra seed phải nằm trong khoảng [0, 2^32 - 1] (giới hạn của np.random.seed)
    if not 0 <= random_seed < 2 ** 32:
        raise ValueError(
            "[set_seed]: seed phải nằm trong khoảng [0, 2^32 - 1].\n"
            f"Nhận được: seed = {random_seed}"
        )

    # Đặt biến môi trường: Hash của Python và workspace của cuBLAS (cần cho chế độ tất định)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Đặt seed cho các bộ sinh ngẫu nhiên
    random.seed(random_seed)                                                    # Python
    np.random.seed(random_seed)                                                 # NumPy
    torch.manual_seed(random_seed)                                              # PyTorch

    # Đặt seed cho mọi GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Bật chế độ tất định của cuDNN và PyTorch
    torch.backends.cudnn.deterministic = bool(config.CUDNN_DETERMINISTIC)
    torch.backends.cudnn.benchmark = bool(config.CUDNN_BENCHMARK)
    torch.use_deterministic_algorithms(bool(config.DETERMINISTIC_ALGORITHMS))

# ============================================================
# Giữ máy chạy liên tục
def keep_system_awake():

    # Chỉ áp dụng cho Windows
    if os.name != "nt":
        return

    try:

        # Thư viện gọi Windows API
        import ctypes
        from ctypes import wintypes

        # Cấu trúc PROCESS_POWER_THROTTLING_STATE của Windows API
        class PowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        # Khai báo kiểu tham số / kết quả của các hàm Windows API
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel32.GetStdHandle.restype = wintypes.HANDLE
        kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]

        # 1. Không cho máy ngủ / ngủ đông (màn hình vẫn tắt được): ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

        # 2. Tắt Power Throttling (EcoQoS): Tiến trình không bị hạ hiệu năng khi cửa sổ bị thu nhỏ / chạy nền
        ## ProcessPowerThrottling = 4, PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1 (StateMask = 0: tắt giới hạn)
        state = PowerThrottlingState(1, 0x1, 0)
        kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))

        # 3. Tắt QuickEdit của console: Bấm chuột vào cửa sổ console sẽ chặn mọi lệnh in, làm treo huấn luyện
        ## Bỏ ENABLE_QUICK_EDIT_MODE (0x40), giữ các cờ khác, cần ENABLE_EXTENDED_FLAGS (0x80)
        handle = kernel32.GetStdHandle(-10 & 0xFFFFFFFF)       # STD_INPUT_HANDLE
        mode = wintypes.DWORD()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, (mode.value & ~0x40) | 0x80)

    # Không có console / Windows API (ví dụ chạy trong IDE) --> bỏ qua
    except (OSError, AttributeError):
        pass

# ============================================================
# Chọn danh sách Fold cần chạy
# Trả về: Danh sách Fold huấn luyện
def set_fold(number_of_folds):

    # Chuyển đổi dữ liệu
    number_of_folds = int(number_of_folds)
    
    ## Kiểm tra tổng số Fold phải có ít nhất 2 Fold
    if number_of_folds < 2:
        raise ValueError(
            "[set_fold]: NUMBER_OF_FOLDS phải lớn hơn hoặc bằng 2.\n"
            f"Nhận được: number_of_folds = {number_of_folds}"
        )

    # Chạy tất cả Fold
    if config.TRAIN_ALL_FOLD:
        return list(range(1, number_of_folds + 1))

    # Chạy các Fold trong SELECTED_FOLD (một số nguyên hoặc danh sách)
    selected_fold = config.SELECTED_FOLD
    folds = [int(fold) for fold in selected_fold] if isinstance(selected_fold, (list, tuple)) else [int(selected_fold)]
    
    ## Kiểm tra SELECTED_FOLD không được rỗng, không trùng lặp và phải nằm trong khoảng [1, number_of_folds]
    if not folds or len(set(folds)) != len(folds) or any(fold not in range(1, number_of_folds + 1) for fold in folds):
        raise ValueError(
            f"[set_fold]: SELECTED_FOLD phải là một số nguyên hoặc danh sách Fold không trùng lặp trong khoảng [1, {number_of_folds}].\n"
            f"Nhận được: SELECTED_FOLD = {selected_fold!r}"
        )

    return folds
