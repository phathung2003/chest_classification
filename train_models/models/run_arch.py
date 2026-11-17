# Khai báo utilities
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"                                 # cuBLAS tất định (DETERMINISTIC_ALGORITHMS)
os.environ["TRITON_CACHE_AUTOTUNING"] = "1"                                        # mamba-ssm >= 2.3 ở chế độ tất định: lưu autotune kernel Triton thay vì cảnh báo

# Khai báo thư viện
import argparse
import ast
import copy
import subprocess
import time

import pandas as pd

import config
import models.mambavision_cxr as mvx
import models.stitch_cxr as stx
import models.stitch_cxr_s16 as s16
from models.run_clis import GATE_EXIT, PAUSED_EXIT, history_score, seed_summary
from models.run_stitch import stitch_hashes
from models.vmamba_cxr import sha256_file
from training.experiment import start_train
from training.train import TrainingPaused
from utilities.dataset import NIHChestXrayDataset
from utilities.environment import check_environment
from utilities.file import read_json, to_snake_case
from utilities.log import timestamp
from utilities.setting import apply_config, keep_system_awake

# ============================================================
# Hàng đợi hai kiến trúc bổ sung ngày 2026-11-16 (prereg/analysis_plan.md mục 7, dòng 2026-11-16 21:05)
#   mv_t     : MambaVision-T CXR (models/mambavision_cxr.py), đối chứng lai CNN-Mamba tiền huấn luyện trọn khối
#   ccmm_s16 : ConvNeXt-VMamba CXR bước 16 (models/stitch_cxr_s16.py), ccmm với stage 4 quét 1.024 token thay vì 256 ở 512 px
# Mục tiêu: Chạy hai mô hình này với đúng giao thức của ConvNeXt-T, VMamba-T và ccmm (config.CLIS_CONFIG + config.CLIS_PROTOCOL), log riêng clis_arch_queue.log
# ** LR: backbone 1e-4 (--lr), đầu phân loại 1e-4 x --head-lr (mặc định m* của ConvNeXt-T = 40), như models/run_vmamba.py và models/run_stitch.py
# ** ccmm_s16 dùng lớp nối bình phương tối thiểu của ccmm (analysis/fit_stitch.py); SHA-256 của tệp lớp nối, models/vmamba_cxr.py và models/stitch_cxr.py
#    vào model_kwargs (pinned_sha256). mv_t: models/mambavision_cxr.py tự ghim mã nhúng và tệp trọng số ImageNet
# ** Mỗi (mô hình, seed) chạy trong tiến trình con riêng; lỗi --> chờ rồi chạy lại, tiếp tục từ last_checkpoint.pt; việc đã xong được bỏ qua;
#    tạm dừng theo PAUSE_AFTER_EPOCH (mã 75) không phải lỗi, điểm là tốt nhất tới epoch đã chạy
#
# Ví dụ (WSL2, conda activate mamba-t29):
#   python models/run_arch.py mv_t --seeds 42 43 44                                                                --> đối chứng MambaVision-T, fold 1
#   python models/run_arch.py ccmm_s16 --seeds 42                                                                  --> ccmm bước 16, fold 1
#   python models/run_arch.py ccmm_s16 --seeds 42 --fold 2                                                         --> xác nhận trên fold 2 (chỉ khi quy tắc cho phép)
#   python models/run_arch.py mv_t --seeds 42 --init-from-size 512 --lr 5e-5 --set IMAGE_SIZE=1024 BATCH_SIZE=4 GRAD_ACCUM_STEPS=8 EPOCHS=5
QUEUE_LOG = config.MODELS_OUTPUT_DIRECTORY / "clis_arch_queue.log"
DEFAULT_HEAD_LR = float(config.CLIS_LR_MULTIPLIER["convnext_t"])                  # Hệ số LR đầu phân loại: m* đã tinh chỉnh của ConvNeXt-T
S16_LAYOUT = "ccmm"
S16_STITCH = stx.STITCH_INITS[0]                                                  # Lớp nối bình phương tối thiểu, như ccmm

# Tên dùng ở dòng lệnh --> (lớp mô hình, hàm tên mô hình theo độ phân giải)
ARCHITECTURES = {
    "mv_t"      : (mvx.MambaVisionCXR, mvx.model_name),
    "ccmm_s16"  : (s16.StitchS16CXR, lambda size = None: s16.model_name(S16_LAYOUT, size)),
}
NAMES = tuple(ARCHITECTURES)


def log(message):
    line = f"[{timestamp()}] {message}"
    print(line, flush = True)
    QUEUE_LOG.parent.mkdir(parents = True, exist_ok = True)
    with QUEUE_LOG.open("a", encoding = "utf-8") as f:
        f.write(line + "\n")


# Kích thước ảnh và tỉ lệ bệnh nhân sau ghi đè
# Trả về: (size, fraction)
def run_settings(overrides):
    settings = {**config.CLIS_CONFIG, **overrides}
    return int(settings.get("IMAGE_SIZE", config.IMAGE_SIZE)), float(settings.get("TRAIN_PATIENT_FRACTION", config.TRAIN_PATIENT_FRACTION))


# Tên mô hình đầy đủ (lượt trên một phần bệnh nhân: hậu tố tune<phần trăm>; RUN_TAG: hậu tố riêng)
# Trả về: Chuỗi
def full_model_name(name, size, fraction = 1.0, tag = ""):
    return ARCHITECTURES[name][1](size) + (f" tune{round(fraction * 100)}" if fraction < 1 else "") + (f" {tag}" if tag else "")


# Thư mục kết quả của một mô hình ở một độ phân giải
# Trả về: Path
def model_directory(name, size, fraction = 1.0, tag = ""):
    dataset_name = read_json(config.DATASET_INFO_DIRECTORY)["dataset_name"]
    return config.MODELS_OUTPUT_DIRECTORY / to_snake_case(dataset_name) / to_snake_case(full_model_name(name, size, fraction, tag))


# Tham số mô hình (trừ init_from): checkpointing từng khối theo BLOCK_GRAD_CHECKPOINT, drop path chỉ ghi khi > 0;
# ccmm_s16 thêm layout, khởi tạo lớp nối và SHA-256 các tệp mà phép tính phụ thuộc
# Trả về: dict
def base_kwargs(name):
    kwargs = {"grad_checkpoint": bool(config.BLOCK_GRAD_CHECKPOINT)}
    if name == "ccmm_s16":
        kwargs.update({"layout": S16_LAYOUT, "stitch_init": S16_STITCH,
                       "pinned_sha256": {stx.VMAMBA_SOURCE.name: sha256_file(stx.VMAMBA_SOURCE), s16.STITCH_SOURCE.name: sha256_file(s16.STITCH_SOURCE),
                                         **stitch_hashes(S16_LAYOUT, S16_STITCH)}})
    if float(config.DROP_PATH_RATE) > 0:
        kwargs["drop_path_rate"] = float(config.DROP_PATH_RATE)
    return kwargs


# Các lượt chạy cùng (seed, LR backbone, hệ số đầu) trong một thư mục mô hình
# Trả về: Danh sách (train_info path, settings)
def matching_runs(directory, seed, learning_rate, head_lr):
    runs = []
    for info_path in sorted(directory.glob("run_*/train_info.json")):
        settings = read_json(info_path).get("settings", {})
        optimizer = settings.get("training_config", {}).get("optimizer", {})
        if (settings.get("seed") == int(seed) and optimizer.get("new_module_lr_multiplier") == float(head_lr)
                and optimizer.get("learning_rate") == float(learning_rate)):
            runs.append((info_path, settings))
    return runs


# best_model.pt của lượt cùng mô hình / seed ở độ phân giải `size` (LR mặc định của CLIS_PROTOCOL), đã xong fold. Phải có đúng một lượt
# Trả về: Path
def find_checkpoint(name, seed, fold, size, fraction, head_lr, tag = ""):
    learning_rate = float(config.CLIS_PROTOCOL["optimizer_config"]["learning_rate"])
    directory = model_directory(name, size, fraction, tag)
    matches = [info_path.parent / f"fold_{fold}" / "best_model.pt" for info_path, settings in matching_runs(directory, seed, learning_rate, head_lr)
               if settings.get("image_size") == int(size) and (info_path.parent / f"fold_{fold}" / "fold_done.json").is_file()]
    if len(matches) != 1:
        raise SystemExit(f"Cần đúng một lượt {name} | seed {seed} | {size} px đã xong fold {fold} trong {directory}, tìm thấy {len(matches)}")
    return matches[0]


# Thư mục fold của lượt cùng (mô hình, seed, LR, ghi đè); None nếu chưa chạy. Nhiều lượt khớp: ưu tiên lượt đã xong fold, rồi lượt mới nhất
# Trả về: Path hoặc None
def find_fold_directory(name, seed, fold, overrides, learning_rate, head_lr):
    size, fraction = run_settings(overrides)
    tag = str({**config.CLIS_CONFIG, **overrides}.get("RUN_TAG", config.RUN_TAG))
    candidates = []
    for info_path, settings in matching_runs(model_directory(name, size, fraction, tag), seed, learning_rate, head_lr):
        snapshot = settings.get("config", {})
        fold_directory = info_path.parent / f"fold_{fold}"
        if all(snapshot.get(key) == value for key, value in overrides.items() if key in snapshot) and fold_directory.is_dir():
            candidates.append(((fold_directory / "fold_done.json").is_file(), info_path.stat().st_mtime, fold_directory))
    return max(candidates)[2] if candidates else None


# ============================================================
# Tiến trình con: một (mô hình, seed, fold)
def run_child(name, seed, fold, overrides, lr = None, head_lr = DEFAULT_HEAD_LR, init_from_size = None):

    # Cấu hình chung của nghiên cứu + ghi đè, một fold
    apply_config({**config.CLIS_CONFIG, **overrides, "SEED": int(seed), "TRAIN_ALL_FOLD": False, "SELECTED_FOLD": [int(fold)]})

    # Tối ưu giống ConvNeXt-T, VMamba-T và ccmm: LR backbone 1e-4 hoặc --lr, đầu phân loại x --head-lr
    protocol = copy.deepcopy(config.CLIS_PROTOCOL)
    protocol["optimizer_config"]["new_module_lr_multiplier"] = float(head_lr)
    if lr is not None:
        protocol["optimizer_config"]["learning_rate"] = float(lr)

    # Tham số mô hình; nguồn khởi tạo theo --init-from-size
    kwargs = base_kwargs(name)
    if init_from_size is not None:
        kwargs["init_from"] = str(find_checkpoint(name, seed, fold, init_from_size, float(config.TRAIN_PATIENT_FRACTION), head_lr, config.RUN_TAG))
        print(f"Khởi tạo từ {kwargs['init_from']}", flush = True)

    model_name = full_model_name(name, None, float(config.TRAIN_PATIENT_FRACTION), config.RUN_TAG)
    return start_train(model_name = model_name, dataset_class = NIHChestXrayDataset, model_factory = ARCHITECTURES[name][0], model_kwargs = kwargs, **protocol)


# Tốc độ huấn luyện và VRAM đỉnh của lượt chạy (trung bình ảnh / giây qua các epoch, GB lớn nhất); None nếu không đọc được
# Trả về: Chuỗi mô tả hoặc None
def describe_cost(fold_directory):
    try:
        history = pd.read_csv(fold_directory / "history.csv")
        return f"{history['train_images_per_second'].mean():.1f} ảnh/s, VRAM đỉnh {history['peak_gpu_memory_gb'].max():.2f} GB"
    except (OSError, KeyError, ValueError):
        return None


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Hàng đợi kiến trúc bổ sung: MambaVision-T CXR (mv_t) và ConvNeXt-VMamba CXR bước 16 (ccmm_s16)")
    parser.add_argument("names", nargs = "+", choices = NAMES, help = "Mô hình: " + ", ".join(NAMES))
    parser.add_argument("--seeds", nargs = "+", type = int, default = [int(config.SEED)])
    parser.add_argument("--fold", type = int, default = 1, help = "Fold dùng làm Validation")
    parser.add_argument("--lr", type = float, default = None, help = "LR backbone (mặc định learning_rate của CLIS_PROTOCOL)")
    parser.add_argument("--head-lr", type = float, default = DEFAULT_HEAD_LR, help = "LR đầu phân loại = hệ số x LR backbone")
    parser.add_argument("--init-from-size", type = int, default = None, help = "Khởi tạo từ best_model.pt cùng mô hình / seed ở độ phân giải này")
    parser.add_argument("--set", nargs = "+", default = [], metavar = "KEY=VALUE", help = "Ghi đè hằng số config (giá trị Python, ví dụ IMAGE_SIZE=1024)")
    parser.add_argument("--min-score", type = float, default = None, help = "Cổng: mã thoát 3 nếu macro AUROC Validation trung bình qua seed của một mô hình thấp hơn ngưỡng")
    parser.add_argument("--retries", type = int, default = 2)
    parser.add_argument("--retry-wait", type = int, default = 60)
    parser.add_argument("--dry-run", action = "store_true")
    parser.add_argument("--child", nargs = 2, default = None, help = argparse.SUPPRESS)
    parser.add_argument("--check-only", action = "store_true", help = argparse.SUPPRESS)
    args = parser.parse_args()

    # Ghi đè config: giá trị Python (số, tuple, None, True / False), còn lại là chuỗi
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            overrides[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            overrides[key] = value

    # Tiến trình con (tạm dừng có chủ đích: mã PAUSED_EXIT)
    if args.child is not None:
        seed, fold = args.child
        try:
            run_child(args.names[0], int(seed), int(fold), overrides, lr = args.lr, head_lr = args.head_lr, init_from_size = args.init_from_size)
        except TrainingPaused as paused:
            print(paused, flush = True)
            sys.exit(PAUSED_EXIT)
        sys.exit(0)

    # Kiểm tra môi trường trong tiến trình con (tiến trình cha không khởi tạo CUDA)
    settings = {**config.CLIS_CONFIG, **overrides}
    if args.check_only:
        try:
            check_environment(require_kernel = True, amp_dtype = settings.get("AMP_DTYPE", config.AMP_DTYPE),
                              deterministic = settings.get("DETERMINISTIC_ALGORITHMS", config.DETERMINISTIC_ALGORITHMS), adam_fused = settings.get("ADAM_FUSED", config.ADAM_FUSED))
        except RuntimeError as error:
            print(error, flush = True)
            sys.exit(1)
        sys.exit(0)

    # Danh sách việc; tệp lớp nối (ccmm_s16) và tệp trọng số ImageNet (mv_t) phải có trước khi vào hàng đợi
    jobs = [(name, seed) for name in args.names for seed in args.seeds]
    learning_rate = float(args.lr if args.lr is not None else config.CLIS_PROTOCOL["optimizer_config"]["learning_rate"])
    if "ccmm_s16" in args.names:
        stitch_hashes(S16_LAYOUT, S16_STITCH)
    if "mv_t" in args.names and args.init_from_size is None:
        mvx.imagenet_state()
    size = run_settings(overrides)[0]
    if args.dry_run:
        for index, (name, seed) in enumerate(jobs, start = 1):
            start = "" if args.init_from_size is None else f" | khởi tạo từ {args.init_from_size} px"
            print(f"{index:3d}. {ARCHITECTURES[name][1](size)} | seed {seed} | LR {learning_rate:g}, đầu x {args.head_lr:g} | fold {args.fold}{start} | {overrides or ''}")
        sys.exit(0)

    # Kiểm tra môi trường một lần cho cả hàng đợi
    script = str(Path(__file__).resolve())
    check = subprocess.run([sys.executable, script, *args.names, "--check-only"] + (["--set", *args.set] if args.set else []), cwd = ROOT_DIR)
    if check.returncode != 0:
        sys.exit(1)

    keep_system_awake()
    log(f"HÀNG ĐỢI KIẾN TRÚC {len(jobs)} việc | fold {args.fold} | LR {learning_rate:g}, đầu x {args.head_lr:g} | ghi đè {overrides or 'không'}"
        + ("" if args.init_from_size is None else f" | khởi tạo từ {args.init_from_size} px")
        + ("" if args.min_score is None else f" | cổng macro_auroc >= {args.min_score:g}"))
    selection_metric = config.CLIS_PROTOCOL["selection_metric"]
    failed, scores = [], {}
    for index, (name, seed) in enumerate(jobs, start = 1):
        command = ([sys.executable, script, name, "--child", str(seed), str(args.fold), "--head-lr", str(args.head_lr)]
                   + (["--set", *args.set] if args.set else []) + (["--lr", str(args.lr)] if args.lr is not None else [])
                   + (["--init-from-size", str(args.init_from_size)] if args.init_from_size is not None else []))
        label = f"{index}/{len(jobs)} {name} | seed {seed}"

        # Chạy, lỗi thì chờ rồi chạy lại (tạm dừng có chủ đích không phải lỗi)
        for attempt in range(1, args.retries + 2):
            log(f"BẮT ĐẦU {label} | lần {attempt}")
            started = time.time()
            code = subprocess.run(command, cwd = ROOT_DIR).returncode
            log(f"KẾT THÚC {label} | mã {code}{' (tạm dừng)' if code == PAUSED_EXIT else ''} | {(time.time() - started) / 3600:.1f} giờ")
            if code in (0, PAUSED_EXIT):
                break
            if attempt <= args.retries:
                time.sleep(args.retry_wait)
        if code not in (0, PAUSED_EXIT):
            failed.append((name, seed))
            continue

        # Điểm Validation tốt nhất (lượt tạm dừng: tốt nhất tới epoch đã chạy), kèm tốc độ và VRAM
        fold_directory = find_fold_directory(name, seed, args.fold, overrides, learning_rate, args.head_lr)
        score = None if fold_directory is None else history_score(fold_directory, selection_metric)
        if score is None or not (score["finished"] or code == PAUSED_EXIT):
            failed.append((name, seed, "không đọc được điểm"))
            continue
        scores.setdefault(name, {})[seed] = score["best_score"]
        cost = describe_cost(fold_directory)
        paused_at = "" if score["finished"] else f" | tạm dừng sau epoch {int(pd.read_csv(fold_directory / 'history.csv')['epoch'].max())}"
        log(f"ĐIỂM {name} | seed {seed} | Validation {selection_metric} {score['best_score']:.4f} ở epoch {score['best_epoch']}{paused_at} | {fold_directory.parent.name}"
            + ("" if cost is None else f" | {cost}"))

    # Tổng kết theo mô hình; cổng --min-score xét điểm trung bình
    below = []
    for name, (mean, spread) in seed_summary(scores).items():
        log(f"TỔNG KẾT {name} | {len(scores[name])} seed | Validation {selection_metric} {mean:.4f} ± {spread:.4f} | "
            + ", ".join(f"seed {seed}: {value:.4f}" for seed, value in scores[name].items()))
        if args.min_score is not None and mean < args.min_score:
            below.append(name)
    log("XONG HÀNG ĐỢI KIẾN TRÚC" + (f" | lỗi: {failed}" if failed else ""))
    if args.min_score is not None:
        verdict = "ĐẠT" if not (below or failed) else ("KHÔNG ĐẠT: " + (f"thấp hơn ngưỡng {below}" if below else "thiếu điểm của việc lỗi"))
        log(f"CỔNG {selection_metric} trung bình qua seed >= {args.min_score:g}: {verdict}")
    sys.exit(1 if failed else (GATE_EXIT if below else 0))
