# ============================================================
# Truy vết thí nghiệm (Experiment Tracking)
# Mục tiêu: Ghi lại đủ thông tin để biết một kết quả đến từ mô hình, dữ liệu, cấu hình và mã nguồn nào (ghi vào train_info.json của lượt chạy).
# ============================================================

import hashlib
import inspect
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

import config
from utilities.file import json_clean

ROOT_DIRECTORY = Path(__file__).resolve().parents[1]                        # Thư mục huấn luyện (train_models)

# Mã nguồn ảnh hưởng tới kết quả (Không gồm config.py: Giá trị cấu hình đã nằm trong settings, SELECTED_FOLD được phép đổi giữa các lần chạy tiếp)
PIPELINE_SOURCE_FILES = ("training/*.py", "utilities/data.py", "utilities/dataset.py", "utilities/metric.py", "utilities/statistics.py", "utilities/setting.py")

# Cài đặt trong config.py không đưa vào chữ ký
CONFIG_EXCLUDED = {"ROOT_DIRECTORY", "ORIGINAL_DATASET_DIRECTORY", "DATASET_DIRECTORY", "FOLDS_DIRECTORY", "TRAIN_FOLDS_DIRECTORY", "TEST_DIRECTORY", "DATASET_INFO_DIRECTORY", "MODELS_OUTPUT_DIRECTORY", "TRAIN_ALL_FOLD", "SELECTED_FOLD", "RESUME_TRAINING", "REQUIRE_CLEAN_GIT", "NUMBER_OF_WORKERS", "PREFETCH_FACTOR", "PERSISTENT_WORKERS", "PROGRESS_SYNC_STEPS", "IMAGE_EXTENSION", "COMPARE_BOOTSTRAP_ITERATION", "INFERENCE_WARMUP_STEPS", "INFERENCE_TIMED_STEPS", "CONSOLE_WIDTH", "GPU_MONITOR_INTERVAL", "MIN_RESULT_WIDTH", "MIN_TITLE_WIDTH", "MIN_MESSAGE_WIDTH", "MIN_NUMBER_WIDTH", "EXPERIMENT_PROTOCOL"}

# ============================================================
# Chụp mọi hằng số config.py ảnh hưởng tới kết quả
# Trả về: dict {tên hằng số: giá trị}, sắp xếp theo tên
def config_snapshot():
    return {name: getattr(config, name) for name in sorted(dir(config)) if name.isupper() and name not in CONFIG_EXCLUDED}

# ============================================================
# Hash SHA-256 của dữ liệu JSON
# Trả về: Chuỗi hex
def stable_hash(data):

    # Chuỗi JSON chuẩn hóa (sắp xếp khóa, không khoảng trắng)
    value = json.dumps(json_clean(data), ensure_ascii = False, sort_keys = True, separators = (",", ":"), allow_nan = False)

    return hashlib.sha256(value.encode("utf-8")).hexdigest()

# ============================================================
# Hash SHA-256 của một tệp (đổi CRLF --> LF để Windows và Linux cho cùng chữ ký)
# Trả về: Chuỗi hex
def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()

# ============================================================
# Hash DataFrame
# Trả về: Chuỗi hex
def data_hash(frame):

    # Bộ băm SHA-256
    digest = hashlib.sha256()

    # Băm tên cột (theo thứ tự)
    digest.update(stable_hash(list(frame.columns)).encode())

    # Băm nội dung từng dòng (theo thứ tự)
    digest.update(pd.util.hash_pandas_object(frame.reset_index(drop = True), index = False)
                  .to_numpy(dtype = np.uint64).tobytes())

    return digest.hexdigest()

# ============================================================
# Hash tệp mã nguồn chứa một lớp / hàm (tệp mô hình, lớp Dataset)
# Trả về: Hash tệp mã nguồn; không tìm được tệp (ví dụ lớp có sẵn của thư viện) --> tên đối tượng
def source_hash(obj):
    try:

        # Tìm tệp mã nguồn của đối tượng
        path = inspect.getsourcefile(obj)
        return file_hash(path) if path and Path(path).is_file() else str(obj)

    # Không lấy được mã nguồn --> Dùng tên đối tượng
    except (TypeError, OSError):
        return str(obj)

# ============================================================
# Hash toàn bộ mã nguồn pipeline (PIPELINE_SOURCE_FILES)
# Trả về: Chuỗi hex
def pipeline_source_hash():

    # Danh sách tệp mã nguồn (sắp xếp, không trùng lặp)
    files = sorted({path for pattern in PIPELINE_SOURCE_FILES for path in ROOT_DIRECTORY.glob(pattern)})

    # Hash của bảng {đường dẫn tương đối: hash tệp}
    return stable_hash({path.relative_to(ROOT_DIRECTORY).as_posix(): file_hash(path) for path in files})

# ============================================================
# Tạo chữ ký thí nghiệm
# Trả về: (chữ ký, settings): settings là toàn bộ thông tin dùng để tính chữ ký (được ghi vào train_info.json)
def make_signature(model_name, model_factory, model_kwargs, dataset_name, dataset_class,
                   train_df, test_df, classes, folds, number_of_folds, loss_config, training_config):

    # Tập hợp thông tin của thí nghiệm
    settings = {
        # Mô hình
        "model_name"                : model_name,
        "model_kwargs"              : model_kwargs,
        "model_source_hash"         : source_hash(model_factory),

        # Dữ liệu
        "dataset_name"              : dataset_name,
        "classes"                   : list(classes),
        "number_of_folds"           : int(number_of_folds),
        "dataset_source_hash"       : source_hash(dataset_class),
        "train_df_hash"             : data_hash(train_df),
        "test_df_hash"              : data_hash(test_df),
        "image_size"                : int(config.IMAGE_SIZE),

        # Huấn luyện (training_config đã chứa đủ mọi giá trị mặc định)
        "folds"                     : list(folds),
        "loss_config"               : dict(loss_config),
        "training_config"           : dict(training_config),
        "use_amp"                   : bool(config.USE_AMP),
        "threshold"                 : float(config.THRESHOLD),

        # Làm giàu ảnh
        "augmentation"              : {
            "flip_chance"           : config.FLIP_CHANCE,
            "degrees"               : config.DEGREES,
            "translate"             : config.TRANSLATE,
            "scale"                 : config.SCALE,
            "brightness"            : config.BRIGHTNESS,
            "contrast"              : config.CONTRAST,
            "normalize_mean"        : config.TRANSFORM_NORMALIZE_MEAN,
            "normalize_std"         : config.TRANSFORM_NORMALIZE_STD,
        },

        # Tái lập
        "seed"                      : int(config.SEED),
        "loader_seed_policy"        : "epoch_sampler_per_image_seed_v3",
        "cudnn_deterministic"       : bool(config.CUDNN_DETERMINISTIC),
        "cudnn_benchmark"           : bool(config.CUDNN_BENCHMARK),
        "deterministic_algorithms"  : bool(config.DETERMINISTIC_ALGORITHMS),
        "torch_version"             : torch.__version__,
        "pipeline_source_hash"      : pipeline_source_hash(),

        # Đánh giá
        "evaluate_fold_models_on_test" : bool(config.EVALUATE_FOLD_MODELS_ON_TEST),
        "bootstrap_iteration"       : int(config.BOOTSTRAP_ITERATION),
        "bootstrap_seed"            : int(config.BOOTSTRAP_SEED),
        "enable_postprocess"        : bool(config.ENABLE_POSTPROCESS),
        "postprocess_calibration"   : str(config.POSTPROCESS_CALIBRATION),
        "threshold_beta"            : float(config.THRESHOLD_BETA),
        "threshold_min_recall"      : float(config.THRESHOLD_MIN_RECALL),

        # Toàn bộ giá trị config ảnh hưởng tới kết quả (xem CONFIG_EXCLUDED)
        "config"                    : config_snapshot(),
    }

    # Fold được chọn trong lần gọi này không thuộc chữ ký: Cho phép chạy [1], rồi [2, 3], rồi [4, 5] trong cùng lượt chạy
    signature_settings = {key: value for key, value in settings.items() if key != "folds"}

    # Hash thông tin thành chữ ký
    return stable_hash(signature_settings), settings

# ============================================================
# Chạy lệnh git trong thư mục huấn luyện
# Trả về: Kết quả in ra của lệnh; None nếu không có git, không phải repo hoặc lệnh lỗi
def _git(*args):

    # Chạy lệnh (không có git / quá thời gian --> None)
    try:
        result = subprocess.run(["git", *args], cwd = ROOT_DIRECTORY, capture_output = True, text = True, encoding = "utf-8", timeout = 30)
    except (OSError, subprocess.SubprocessError):
        return None

    # Lệnh thành công --> Kết quả in ra, ngược lại --> None
    return result.stdout.rstrip() if result.returncode == 0 else None

# ============================================================
# Lấy trạng thái Git
# Trả về: dict thông tin Git
def git_provenance():

    # Lấy commit hiện tại
    commit = (_git("rev-parse", "HEAD") or "").strip() or None

    # Không có git --> Các trường còn lại là None
    if commit is None:
        return {"git_available": False, "git_commit": None, "git_branch": None, "git_dirty": None, "git_dirty_files": []}

    # Khởi tạo danh sách tệp chưa commit
    dirty_files = []

    # Duyệt từng dòng của git status
    for line in (_git("status", "--porcelain", "--", ".") or "").splitlines():
        code, path = line[:2], line[3:].strip().strip('"')

        # Bỏ qua tệp mới không phải .py
        if code == "??" and not path.endswith(".py"):
            continue

        dirty_files.append(path)

    return {
        "git_available"     : True,
        "git_commit"        : commit,
        "git_branch"        : (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "git_dirty"         : bool(dirty_files),                                # Còn thay đổi chưa commit
        "git_dirty_files"   : dirty_files[:100],                                # Tối đa 100 tệp
    }

# ============================================================
# Lấy thông tin môi trường chạy
# Trả về: dict thông tin môi trường
def environment_info():

    # Hàm lấy phiên bản của một thư viện (chưa cài --> None)
    def version(package):
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

    # Thông tin GPU (không có CUDA --> None)
    gpu = None

    if torch.cuda.is_available():
        gpu = {"name": torch.cuda.get_device_name(0), "count": torch.cuda.device_count(),
               "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}

    return {
        "python"    : sys.version.split()[0],
        "platform"  : platform.platform(),
        "packages"  : {name: version(name) for name in ("torch", "torchvision", "numpy", "pandas", "scikit-learn",
                                                        "scipy", "pillow", "mambapy", "mamba-ssm")},
        "gpu"       : gpu,
    }

# ============================================================
# Ghi nhận một lần khởi chạy
# Trả về: dict thời điểm khởi chạy, trạng thái Git và môi trường
def launch_record():
    return {
        "started_at"    : datetime.now(timezone.utc).astimezone().isoformat(timespec = "seconds"),
        **git_provenance(),
        "environment"   : environment_info(),
    }

# ============================================================
# Đo thông số mô hình
# Trả về: dict thông số mô hình
def model_specification(model_factory, classes, device, model_kwargs, batch_size):

    # Khởi tạo mô hình (chế độ đánh giá)
    model = model_factory(num_classes = len(classes), **model_kwargs).to(device).eval()

    # Tạo ảnh đầu vào ngẫu nhiên kích thước [1, 3, IMAGE_SIZE, IMAGE_SIZE]
    input_tensor = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)

    # Đếm FLOPs của một lần forward (MACs = FLOPs / 2)
    with FlopCounterMode(display = False) as counter:
        model(input_tensor)
    flops = int(counter.get_total_flops())

    # Tổng hợp thông số
    specification = {
        "parameters"            : sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters"  : sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "macs"                  : flops // 2,
        "flops"                 : flops,
        **inference_speed(model, device, batch_size),
    }

    # Giải phóng mô hình
    del model
    return specification

# ============================================================
# Đo tốc độ suy luận thực tế
# Trả về: dict tốc độ suy luận
@torch.no_grad()                                                            # Tắt Gradient
def inference_speed(model, device, batch_size, warmup = None, repeats = None):

    # Lấy dữ liệu
    warmup_steps    = config.INFERENCE_WARMUP_STEPS if warmup is None else warmup
    timed_steps     = config.INFERENCE_TIMED_STEPS if repeats is None else repeats
    use_amp         = bool(config.USE_AMP and device.type == "cuda")

    # Hàm đo thời gian trung vị của một lần forward với batch cho trước (bỏ qua các lần làm nóng)
    def median_seconds(batch):

        # Ảnh đầu vào ngẫu nhiên và danh sách thời gian
        images = torch.randn(batch, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
        timings = []

        # Chạy forward nhiều lần (đồng bộ GPU trước và sau mỗi lần để đo đúng thời gian)
        with torch.autocast(device_type = device.type, dtype = torch.float16, enabled = use_amp):
            for step in range(warmup_steps + timed_steps):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                model(images)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if step >= warmup_steps:                                    # Chỉ ghi thời gian sau các lần làm nóng
                    timings.append(time.perf_counter() - started)

        return float(np.median(timings))

    # Đo độ trễ (batch 1) và thời gian một batch huấn luyện
    latency = median_seconds(1)
    batch_time = median_seconds(int(batch_size))

    return {
        "inference_precision"               : "fp16 AMP" if use_amp else "fp32",
        "inference_latency_ms_batch_1"      : latency * 1000,
        "inference_batch_size"              : int(batch_size),
        "inference_images_per_second"       : int(batch_size) / batch_time,
        "inference_warmup_steps"            : int(warmup_steps),
        "inference_timed_steps"             : int(timed_steps),
    }
