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
import models.stitch_cxr as stx
from models.run_clis import GATE_EXIT, PAUSED_EXIT, history_score, seed_summary
from models.vmamba_cxr import sha256_file
from training.experiment import start_train
from training.train import TrainingPaused
from utilities.dataset import NIHChestXrayDataset
from utilities.environment import check_environment
from utilities.file import read_json, to_snake_case
from utilities.log import timestamp
from utilities.setting import apply_config, keep_system_awake

# ============================================================
# Hàng đợi ConvNeXt-VMamba CXR (models/stitch_cxr.py, mô hình lai ghép tầng)
# Mục tiêu: Chạy các layout (ccmm, mmcc, ...) với đúng giao thức của baseline ConvNeXt-T và VMamba-T (config.CLIS_CONFIG + config.CLIS_PROTOCOL), log riêng clis_stitch_queue.log
# ** LR: hai backbone và lớp nối 1e-4 (--lr), đầu phân loại 1e-4 x --head-lr (mặc định m* của ConvNeXt-T = 40), như models/run_vmamba.py
# ** Lớp nối "lstsq" cần tệp của analysis/fit_stitch.py; SHA-256 của tệp lớp nối và của models/vmamba_cxr.py vào model_kwargs (pinned_sha256),
#    nên chữ ký thí nghiệm ghim đúng lớp nối và mã VMamba đã dùng
# ** Mỗi (layout, seed) chạy trong tiến trình con riêng; lỗi --> chờ rồi chạy lại, tiếp tục từ last_checkpoint.pt; việc đã xong được bỏ qua;
#    tạm dừng theo PAUSE_AFTER_EPOCH (mã 75) không phải lỗi, điểm là tốt nhất tới epoch đã chạy
#
# Ví dụ (WSL2, conda activate mamba-t29):
#   python analysis/fit_stitch.py --layout ccmm                                                                    --> lớp nối bình phương tối thiểu (một lần)
#   python models/run_stitch.py ccmm mmcc --seeds 42 --set TRAIN_PATIENT_FRACTION=0.3 PAUSE_AFTER_EPOCH=5          --> pilot 30% bệnh nhân
#   python models/run_stitch.py ccmm --seeds 42 --stitch-init identity                                             --> đối chứng lớp nối đơn vị (thư mục hậu tố " identity")
#   python models/run_stitch.py ccmm --seeds 42 --init-from-size 512 --lr 5e-5 --set IMAGE_SIZE=1024 BATCH_SIZE=4 GRAD_ACCUM_STEPS=8 EPOCHS=5 BLOCK_GRAD_CHECKPOINT=True
QUEUE_LOG = config.MODELS_OUTPUT_DIRECTORY / "clis_stitch_queue.log"
DEFAULT_HEAD_LR = float(config.CLIS_LR_MULTIPLIER["convnext_t"])                  # Hệ số LR đầu phân loại: m* đã tinh chỉnh của ConvNeXt-T
DEFAULT_STITCH = stx.STITCH_INITS[0]


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


# Tên mô hình kèm cách khởi tạo lớp nối (mặc định "lstsq" không thêm hậu tố)
# Trả về: Chuỗi
def variant_name(layout, size = None, stitch_init = DEFAULT_STITCH):
    return stx.model_name(layout, size) + ("" if stitch_init == DEFAULT_STITCH else f" {stitch_init}")


# Thư mục kết quả của một layout ở một độ phân giải (lượt trên một phần bệnh nhân: hậu tố tune<phần trăm>)
# Trả về: Path
def model_directory(layout, size, fraction = 1.0, stitch_init = DEFAULT_STITCH):
    model_name = variant_name(layout, size, stitch_init) + (f" tune{round(fraction * 100)}" if fraction < 1 else "")
    dataset_name = read_json(config.DATASET_INFO_DIRECTORY)["dataset_name"]
    return config.MODELS_OUTPUT_DIRECTORY / to_snake_case(dataset_name) / to_snake_case(model_name)


# SHA-256 các tệp lớp nối mà layout cần (rỗng nếu lớp nối đơn vị hoặc layout một họ); thiếu tệp thì dừng kèm lệnh cần chạy
# Trả về: dict {tên tệp: SHA-256}
def stitch_hashes(layout, stitch_init):
    if stitch_init != "lstsq":
        return {}
    paths = [stx.stitch_path(layout[stage], layout[stage + 1], stage) for stage in stx.boundaries(layout)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Thiếu tệp lớp nối {missing} trong {stx.STITCH_DIRECTORY}: chạy python analysis/fit_stitch.py --layout {layout}")
    return {path.name: sha256_file(path) for path in paths}


# Các lượt chạy cùng (layout, seed, LR backbone, hệ số đầu, khởi tạo lớp nối) trong một thư mục
# Trả về: Danh sách (train_info path, settings)
def matching_runs(directory, layout, seed, learning_rate, head_lr, stitch_init = DEFAULT_STITCH):
    runs = []
    for info_path in sorted(directory.glob("run_*/train_info.json")):
        settings = read_json(info_path).get("settings", {})
        optimizer = settings.get("training_config", {}).get("optimizer", {})
        kwargs = settings.get("model_kwargs", {})
        if (settings.get("seed") == int(seed) and kwargs.get("layout") == layout and kwargs.get("stitch_init") == stitch_init
                and optimizer.get("new_module_lr_multiplier") == float(head_lr) and optimizer.get("learning_rate") == float(learning_rate)):
            runs.append((info_path, settings))
    return runs


# best_model.pt của lượt cùng layout / seed ở độ phân giải `size` (LR mặc định của CLIS_PROTOCOL), đã xong fold. Phải có đúng một lượt
# Trả về: Path
def find_checkpoint(layout, seed, fold, size, fraction, head_lr, stitch_init = DEFAULT_STITCH):
    learning_rate = float(config.CLIS_PROTOCOL["optimizer_config"]["learning_rate"])
    directory = model_directory(layout, size, fraction, stitch_init)
    matches = [info_path.parent / f"fold_{fold}" / "best_model.pt" for info_path, settings in matching_runs(directory, layout, seed, learning_rate, head_lr, stitch_init)
               if settings.get("image_size") == int(size) and (info_path.parent / f"fold_{fold}" / "fold_done.json").is_file()]
    if len(matches) != 1:
        raise SystemExit(f"Cần đúng một lượt {layout} | seed {seed} | {size} px đã xong fold {fold} trong {directory}, tìm thấy {len(matches)}")
    return matches[0]


# Thư mục fold của lượt cùng (layout, seed, LR, ghi đè); None nếu chưa chạy. Nhiều lượt khớp: ưu tiên lượt đã xong fold, rồi lượt mới nhất
# Trả về: Path hoặc None
def find_fold_directory(layout, seed, fold, overrides, learning_rate, head_lr, stitch_init = DEFAULT_STITCH):
    size, fraction = run_settings(overrides)
    candidates = []
    for info_path, settings in matching_runs(model_directory(layout, size, fraction, stitch_init), layout, seed, learning_rate, head_lr, stitch_init):
        snapshot = settings.get("config", {})
        if all(snapshot.get(key) == value for key, value in overrides.items() if key in snapshot):
            fold_directory = info_path.parent / f"fold_{fold}"
            candidates.append(((fold_directory / "fold_done.json").is_file(), info_path.stat().st_mtime, fold_directory))
    return max(candidates)[2] if candidates else None


# ============================================================
# Tiến trình con: một (layout, seed, fold)
def run_child(layout, seed, fold, overrides, lr = None, head_lr = DEFAULT_HEAD_LR, init_from_size = None, stitch_init = DEFAULT_STITCH):

    # Cấu hình chung của nghiên cứu CLIS + ghi đè, một fold
    apply_config({**config.CLIS_CONFIG, **overrides, "SEED": int(seed), "TRAIN_ALL_FOLD": False, "SELECTED_FOLD": [int(fold)]})

    # Tối ưu giống baseline ConvNeXt-T và VMamba-T: LR backbone 1e-4 hoặc --lr, đầu phân loại x --head-lr
    protocol = copy.deepcopy(config.CLIS_PROTOCOL)
    protocol["optimizer_config"]["new_module_lr_multiplier"] = float(head_lr)
    if lr is not None:
        protocol["optimizer_config"]["learning_rate"] = float(lr)

    # Tham số mô hình: layout, khởi tạo lớp nối, SHA-256 của mã VMamba CXR và các tệp lớp nối, checkpointing từng khối VSS theo BLOCK_GRAD_CHECKPOINT; nguồn khởi tạo theo --init-from-size
    kwargs = {"layout": layout, "stitch_init": stitch_init, "grad_checkpoint": bool(config.BLOCK_GRAD_CHECKPOINT),
              "pinned_sha256": {stx.VMAMBA_SOURCE.name: sha256_file(stx.VMAMBA_SOURCE), **stitch_hashes(layout, stitch_init)}}
    if init_from_size is not None:
        kwargs["init_from"] = str(find_checkpoint(layout, seed, fold, init_from_size, float(config.TRAIN_PATIENT_FRACTION), head_lr, stitch_init))
        print(f"Khởi tạo từ {kwargs['init_from']}", flush = True)

    # Tên mô hình (lượt trên một phần bệnh nhân: thư mục riêng)
    model_name = variant_name(layout, stitch_init = stitch_init)
    if float(config.TRAIN_PATIENT_FRACTION) < 1:
        model_name = f"{model_name} tune{round(float(config.TRAIN_PATIENT_FRACTION) * 100)}"
    return start_train(model_name = model_name, dataset_class = NIHChestXrayDataset, model_factory = stx.StitchCXR, model_kwargs = kwargs, **protocol)


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
    parser = argparse.ArgumentParser(description = "Hàng đợi ConvNeXt-VMamba CXR (mô hình lai ghép tầng)")
    parser.add_argument("layouts", nargs = "+", help = "Layout 4 ký tự c / m, ví dụ ccmm mmcc")
    parser.add_argument("--seeds", nargs = "+", type = int, default = [int(config.SEED)])
    parser.add_argument("--fold", type = int, default = 1, help = "Fold dùng làm Validation")
    parser.add_argument("--lr", type = float, default = None, help = "LR backbone (mặc định learning_rate của CLIS_PROTOCOL)")
    parser.add_argument("--head-lr", type = float, default = DEFAULT_HEAD_LR, help = "LR đầu phân loại = hệ số x LR backbone")
    parser.add_argument("--init-from-size", type = int, default = None, help = "Khởi tạo từ best_model.pt cùng layout / seed ở độ phân giải này")
    parser.add_argument("--stitch-init", choices = stx.STITCH_INITS, default = DEFAULT_STITCH, help = "Khởi tạo lớp nối: bình phương tối thiểu (analysis/fit_stitch.py) hoặc đơn vị")
    parser.add_argument("--set", nargs = "+", default = [], metavar = "KEY=VALUE", help = "Ghi đè hằng số config (giá trị Python, ví dụ IMAGE_SIZE=1024)")
    parser.add_argument("--min-score", type = float, default = None, help = "Cổng: mã thoát 3 nếu macro AUROC Validation trung bình qua seed của một layout thấp hơn ngưỡng")
    parser.add_argument("--retries", type = int, default = 2)
    parser.add_argument("--retry-wait", type = int, default = 60)
    parser.add_argument("--dry-run", action = "store_true")
    parser.add_argument("--child", nargs = 2, default = None, help = argparse.SUPPRESS)
    parser.add_argument("--check-only", action = "store_true", help = argparse.SUPPRESS)
    args = parser.parse_args()
    for layout in args.layouts:
        if len(layout) != len(stx.CHANNELS) or any(family not in stx.FAMILIES for family in layout):
            parser.error(f"layout '{layout}' không hợp lệ: cần {len(stx.CHANNELS)} ký tự thuộc {list(stx.FAMILIES)}")

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
            run_child(args.layouts[0], int(seed), int(fold), overrides, lr = args.lr, head_lr = args.head_lr, init_from_size = args.init_from_size, stitch_init = args.stitch_init)
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

    # Danh sách việc; tệp lớp nối phải có trước khi vào hàng đợi
    jobs = [(layout, seed) for layout in args.layouts for seed in args.seeds]
    learning_rate = float(args.lr if args.lr is not None else config.CLIS_PROTOCOL["optimizer_config"]["learning_rate"])
    for layout in args.layouts:
        stitch_hashes(layout, args.stitch_init)
    if args.dry_run:
        for index, (layout, seed) in enumerate(jobs, start = 1):
            start = "" if args.init_from_size is None else f" | khởi tạo từ {args.init_from_size} px"
            print(f"{index:3d}. {variant_name(layout, run_settings(overrides)[0], args.stitch_init)} | seed {seed} | LR {learning_rate:g}, đầu x {args.head_lr:g} | fold {args.fold}{start} | {overrides or ''}")
        sys.exit(0)

    # Kiểm tra môi trường một lần cho cả hàng đợi
    script = str(Path(__file__).resolve())
    check = subprocess.run([sys.executable, script, *args.layouts, "--check-only"] + (["--set", *args.set] if args.set else []), cwd = ROOT_DIR)
    if check.returncode != 0:
        sys.exit(1)

    keep_system_awake()
    log(f"HÀNG ĐỢI LAI {len(jobs)} việc | fold {args.fold} | LR {learning_rate:g}, đầu x {args.head_lr:g} | lớp nối {args.stitch_init} | ghi đè {overrides or 'không'}"
        + ("" if args.init_from_size is None else f" | khởi tạo từ {args.init_from_size} px")
        + ("" if args.min_score is None else f" | cổng macro_auroc >= {args.min_score:g}"))
    selection_metric = config.CLIS_PROTOCOL["selection_metric"]
    failed, scores = [], {}
    for index, (layout, seed) in enumerate(jobs, start = 1):
        command = ([sys.executable, script, layout, "--child", str(seed), str(args.fold), "--head-lr", str(args.head_lr), "--stitch-init", args.stitch_init]
                   + (["--set", *args.set] if args.set else []) + (["--lr", str(args.lr)] if args.lr is not None else [])
                   + (["--init-from-size", str(args.init_from_size)] if args.init_from_size is not None else []))
        label = f"{index}/{len(jobs)} {layout}{'' if args.stitch_init == DEFAULT_STITCH else ' ' + args.stitch_init} | seed {seed}"

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
            failed.append((layout, seed))
            continue

        # Điểm Validation tốt nhất (lượt tạm dừng: tốt nhất tới epoch đã chạy), kèm tốc độ và VRAM
        fold_directory = find_fold_directory(layout, seed, args.fold, overrides, learning_rate, args.head_lr, args.stitch_init)
        score = None if fold_directory is None else history_score(fold_directory, selection_metric)
        if score is None or not (score["finished"] or code == PAUSED_EXIT):
            failed.append((layout, seed, "không đọc được điểm"))
            continue
        scores.setdefault(layout, {})[seed] = score["best_score"]
        cost = describe_cost(fold_directory)
        paused_at = "" if score["finished"] else f" | tạm dừng sau epoch {int(pd.read_csv(fold_directory / 'history.csv')['epoch'].max())}"
        log(f"ĐIỂM {layout} | seed {seed} | Validation {selection_metric} {score['best_score']:.4f} ở epoch {score['best_epoch']}{paused_at} | {fold_directory.parent.name}"
            + ("" if cost is None else f" | {cost}"))

    # Tổng kết theo layout; cổng --min-score xét điểm trung bình
    below = []
    for layout, (mean, spread) in seed_summary(scores).items():
        log(f"TỔNG KẾT {layout} | {len(scores[layout])} seed | Validation {selection_metric} {mean:.4f} ± {spread:.4f} | "
            + ", ".join(f"seed {seed}: {value:.4f}" for seed, value in scores[layout].items()))
        if args.min_score is not None and mean < args.min_score:
            below.append(layout)
    log("XONG HÀNG ĐỢI LAI" + (f" | lỗi: {failed}" if failed else ""))
    if args.min_score is not None:
        verdict = "ĐẠT" if not (below or failed) else ("KHÔNG ĐẠT: " + (f"thấp hơn ngưỡng {below}" if below else "thiếu điểm của việc lỗi"))
        log(f"CỔNG {selection_metric} trung bình qua seed >= {args.min_score:g}: {verdict}")
    sys.exit(1 if failed else (GATE_EXIT if below else 0))
