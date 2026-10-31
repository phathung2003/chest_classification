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

# Truy vết thí nghiệm: mọi thông tin được ghi vào train_info.json
#   - Chữ ký: hash của mô hình, dữ liệu, cấu hình và mã nguồn (quyết định chạy tiếp hay tạo lượt chạy mới)
#   - Lần khởi chạy: commit Git, tệp chưa commit, phiên bản thư viện, GPU
#   - Thông số mô hình: số tham số, MACs, FLOPs

ROOT_DIRECTORY = Path(__file__).resolve().parents[1]

# Mã nguồn ảnh hưởng tới kết quả số. Không gồm config.py: giá trị cấu hình đã nằm trong settings,
# và SELECTED_FOLD được phép đổi giữa các lần chạy tiếp của cùng một lượt chạy.
PIPELINE_SOURCE_FILES = ("training/*.py", "utilities/data.py", "utilities/dataset.py", "utilities/metric.py",
                         "utilities/statistics.py", "utilities/setting.py")

# Hằng số trong config.py KHÔNG đưa vào chữ ký: đường dẫn (phụ thuộc máy), điều khiển lượt chạy (được phép đổi khi chạy tiếp),
# hiển thị / hiệu năng / công cụ ngoài (không đổi kết quả huấn luyện), giao thức chung (loss_config và training_config thực tế
# đã nằm trong chữ ký). Tham số riêng của mô hình nằm trong tệp mô hình, đã vào chữ ký qua hash tệp mô hình.
# Số worker (từ 1 trở lên) không ảnh hưởng kết quả: thứ tự ảnh do EpochSampler quyết định, làm giàu ảnh dùng seed riêng
# của từng ảnh (0 worker chỉ lệch ở mức làm tròn số thực vì biến đổi ảnh chạy đa luồng trong tiến trình chính)
CONFIG_EXCLUDED = {"ROOT_DIRECTORY", "ORIGINAL_DATASET_DIRECTORY", "DATASET_DIRECTORY", "FOLDS_DIRECTORY", "TRAIN_FOLDS_DIRECTORY", "TEST_DIRECTORY", "DATASET_INFO_DIRECTORY", "MODELS_OUTPUT_DIRECTORY", "TRAIN_ALL_FOLD", "SELECTED_FOLD", "RESUME_TRAINING", "REQUIRE_CLEAN_GIT", "NUMBER_OF_WORKERS", "PREFETCH_FACTOR", "PERSISTENT_WORKERS", "PROGRESS_SYNC_STEPS", "IMAGE_EXTENSION", "COMPARE_BOOTSTRAP_ITERATIONS", "INFERENCE_WARMUP_STEPS", "INFERENCE_TIMED_STEPS", "CONSOLE_WIDTH", "GPU_MONITOR_INTERVAL", "MIN_RESULT_WIDTH", "MIN_TITLE_WIDTH", "MIN_MESSAGE_WIDTH", "MIN_NUMBER_WIDTH", "EXPERIMENT_PROTOCOL"}


# ============================================================
# Bản chụp mọi hằng số config.py ảnh hưởng tới kết quả: đổi bất kỳ giá trị nào --> chữ ký đổi --> không chạy tiếp lẫn cấu hình
def config_snapshot():
    return {name: getattr(config, name) for name in sorted(dir(config)) if name.isupper() and name not in CONFIG_EXCLUDED}


# ============================================================
# Hash SHA-256 của dữ liệu JSON (khóa sắp xếp cố định nên kết quả ổn định)
def stable_hash(data):
    value = json.dumps(json_clean(data), ensure_ascii = False, sort_keys = True, separators = (",", ":"), allow_nan = False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ============================================================
# Hash SHA-256 của một tệp mã nguồn. Đổi CRLF (git checkout trên Windows) thành LF (Linux) trước khi hash,
# để cùng một commit cho cùng chữ ký trên mọi hệ điều hành
def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ============================================================
# Hash DataFrame: tên cột, thứ tự cột và thứ tự dòng đều là một phần của thí nghiệm
def data_hash(frame):
    digest = hashlib.sha256()
    digest.update(stable_hash(list(frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame.reset_index(drop = True), index = False)
                  .to_numpy(dtype = np.uint64).tobytes())
    return digest.hexdigest()


# ============================================================
# Hash tệp mã nguồn chứa một lớp / hàm (tệp mô hình, lớp Dataset)
def source_hash(obj):
    try:
        path = inspect.getsourcefile(obj)
        return file_hash(path) if path and Path(path).is_file() else str(obj)
    except (TypeError, OSError):
        return str(obj)


# ============================================================
# Hash toàn bộ mã nguồn pipeline: sửa vòng huấn luyện, đánh giá, dữ liệu hay chỉ số đều đổi chữ ký,
# nên một lượt chạy không bao giờ được chạy tiếp bằng mã khác
def pipeline_source_hash():
    files = sorted({path for pattern in PIPELINE_SOURCE_FILES for path in ROOT_DIRECTORY.glob(pattern)})
    return stable_hash({path.relative_to(ROOT_DIRECTORY).as_posix(): file_hash(path) for path in files})


# ============================================================
# Chữ ký thí nghiệm. Chữ ký khác --> lượt chạy mới; chữ ký khớp --> được chạy tiếp lượt chạy dở
def make_signature(model_name, model_factory, model_kwargs, dataset_name, dataset_class,
                   train_df, test_df, classes, folds, number_of_folds, loss_config, training_config):
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
        "bootstrap_iterations"      : int(config.BOOTSTRAP_ITERATIONS),
        "bootstrap_seed"            : int(config.BOOTSTRAP_SEED),
        "enable_postprocess"        : bool(config.ENABLE_POSTPROCESS),
        "postprocess_calibration"   : str(config.POSTPROCESS_CALIBRATION),
        "threshold_beta"            : float(config.THRESHOLD_BETA),
        "threshold_min_recall"      : float(config.THRESHOLD_MIN_RECALL),

        # Toàn bộ giá trị config ảnh hưởng tới kết quả (xem CONFIG_EXCLUDED)
        "config"                    : config_snapshot(),
    }

    # Fold được chọn trong lần gọi này KHÔNG thuộc chữ ký: cho phép chạy [1], rồi [2, 3], rồi [4, 5] trong cùng lượt chạy
    signature_settings = {k: v for k, v in settings.items() if k != "folds"}
    return stable_hash(signature_settings), settings


# ============================================================
# Chạy lệnh git trong thư mục huấn luyện; None nếu không có git hoặc không phải repo
def _git(*args):
    try:
        result = subprocess.run(["git", *args], cwd = ROOT_DIRECTORY, capture_output = True, text = True, encoding = "utf-8", timeout = 30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.rstrip() if result.returncode == 0 else None


# ============================================================
# Trạng thái Git: commit hiện tại và các tệp còn thay đổi chưa commit
# (mọi tệp đã theo dõi bị sửa + tệp .py mới chưa theo dõi; dữ liệu và kết quả đã nằm trong .gitignore)
def git_provenance():
    commit = (_git("rev-parse", "HEAD") or "").strip() or None
    if commit is None:
        return {"git_available": False, "git_commit": None, "git_branch": None, "git_dirty": None, "git_dirty_files": []}

    dirty = []
    for line in (_git("status", "--porcelain", "--", ".") or "").splitlines():
        code, path = line[:2], line[3:].strip().strip('"')
        if code == "??" and not path.endswith(".py"):
            continue
        dirty.append(path)
    return {
        "git_available"     : True,
        "git_commit"        : commit,
        "git_branch"        : (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "git_dirty"         : bool(dirty),
        "git_dirty_files"   : dirty[:100],
    }


# ============================================================
# Môi trường chạy: phiên bản Python, thư viện và GPU
def environment_info():
    def version(package):
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

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
# Một lần khởi chạy (ghi nối tiếp vào train_info.json, KHÔNG thuộc chữ ký để vẫn chạy tiếp được sau khi commit)
def launch_record():
    return {
        "started_at"    : datetime.now(timezone.utc).astimezone().isoformat(timespec = "seconds"),
        **git_provenance(),
        "environment"   : environment_info(),
    }


# ============================================================
# Thông số mô hình: số tham số, MACs / FLOPs cho 1 ảnh và tốc độ suy luận.
# FLOPs đếm bằng bộ đếm của PyTorch: tính cả matmul trong attention và Mamba; phép elementwise (gồm phần quét
# của Mamba) không tính, nên tốc độ suy luận đo thực tế là số liệu bổ sung cần thiết khi so sánh hiệu năng.
# Không bọc torch.no_grad(): bộ theo dõi module của FlopCounterMode lỗi với MultiheadAttention khi tắt gradient.
def model_specification(model_factory, classes, device, model_kwargs, batch_size):
    model = model_factory(num_classes = len(classes), **model_kwargs).to(device).eval()
    input_tensor = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)

    with FlopCounterMode(display = False) as counter:
        model(input_tensor)
    flops = int(counter.get_total_flops())

    specification = {
        "parameters"            : sum(p.numel() for p in model.parameters()),
        "trainable_parameters"  : sum(p.numel() for p in model.parameters() if p.requires_grad),
        "macs"                  : flops // 2,
        "flops"                 : flops,
        **inference_speed(model, device, batch_size),
    }
    del model
    return specification


# ============================================================
# Tốc độ suy luận đo thực tế, cùng chế độ với lúc đánh giá (eval, AMP nếu bật):
#   - Độ trễ với batch 1 (ms / ảnh), lấy trung vị để ít bị ảnh hưởng bởi nhiễu
#   - Throughput với batch huấn luyện (ảnh / giây)
@torch.no_grad()
def inference_speed(model, device, batch_size, warmup = None, repeats = None):
    warmup = config.INFERENCE_WARMUP_STEPS if warmup is None else warmup
    repeats = config.INFERENCE_TIMED_STEPS if repeats is None else repeats
    use_amp = bool(config.USE_AMP and device.type == "cuda")

    def median_seconds(batch):
        x = torch.randn(batch, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
        timings = []
        with torch.autocast(device_type = device.type, dtype = torch.float16, enabled = use_amp):
            for i in range(warmup + repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if i >= warmup:
                    timings.append(time.perf_counter() - started)
        return float(np.median(timings))

    latency = median_seconds(1)
    batch_time = median_seconds(int(batch_size))
    return {
        "inference_precision"               : "fp16 AMP" if use_amp else "fp32",
        "inference_latency_ms_batch_1"      : latency * 1000,
        "inference_batch_size"              : int(batch_size),
        "inference_images_per_second"       : int(batch_size) / batch_time,
        "inference_warmup_steps"            : int(warmup),
        "inference_timed_steps"             : int(repeats),
    }
