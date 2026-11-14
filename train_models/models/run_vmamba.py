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
import models.vmamba_cxr as vmx
from models.run_clis import GATE_EXIT, PAUSED_EXIT, history_score, seed_summary
from training.experiment import start_train
from training.train import TrainingPaused
from utilities.dataset import NIHChestXrayDataset
from utilities.environment import check_environment
from utilities.file import load_torch, read_json, to_snake_case
from utilities.log import timestamp
from utilities.setting import apply_config, keep_system_awake

# ============================================================
# Hàng đợi VMamba-T CXR (models/vmamba_cxr.py, phương án A)
# Mục tiêu: Chạy vm_ras / vm_clis / vm_rand với đúng giao thức của baseline ConvNeXt-T (config.CLIS_CONFIG + config.CLIS_PROTOCOL), log riêng clis_vmamba_queue.log
# ** LR: backbone 1e-4 (--lr), đầu phân loại 1e-4 x --head-lr (mặc định m* của ConvNeXt-T = 40)
# ** Mỗi (biến thể, seed) chạy trong tiến trình con riêng; lỗi --> chờ rồi chạy lại, tiếp tục từ last_checkpoint.pt; việc đã xong được bỏ qua;
#    tạm dừng theo PAUSE_AFTER_EPOCH (mã 75) không phải lỗi, điểm là tốt nhất tới epoch đã chạy
#
# Ví dụ (WSL2, conda activate mamba-t29):
#   python models/run_vmamba.py vm_ras vm_clis vm_rand --seeds 42 --set TRAIN_PATIENT_FRACTION=0.3 PAUSE_AFTER_EPOCH=5   --> pilot 30% bệnh nhân
#   python models/run_vmamba.py vm_clis vm_rand --seeds 42 --route-init copy                                    --> tuyến thêm chép đủ từ tuyến hàng (thư mục hậu tố " copy")
#   python models/run_vmamba.py vm_clis vm_rand --seeds 42 --route-init copy --route-sign signed               --> tuyến thêm quét hiệu có dấu hai nửa ảnh (hậu tố " copy signed")
#   python models/run_vmamba.py vm_clis --seeds 42 --init-from-size 512 --lr 5e-5 --set IMAGE_SIZE=1024 BATCH_SIZE=4 GRAD_ACCUM_STEPS=8 EPOCHS=5 BLOCK_GRAD_CHECKPOINT=True
QUEUE_LOG = config.MODELS_OUTPUT_DIRECTORY / "clis_vmamba_queue.log"
DEFAULT_HEAD_LR = float(config.CLIS_LR_MULTIPLIER["convnext_t"])                  # Hệ số LR đầu phân loại: m* đã tinh chỉnh của ConvNeXt-T


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


# Tên mô hình kèm cách khởi tạo và dấu của tuyến thêm (mặc định "zero" / "none" không thêm hậu tố, giữ tên thư mục của các lượt cũ)
# Trả về: Chuỗi
def variant_name(name, size = None, route_init = "zero", route_sign = "none"):
    return vmx.model_name(name, size) + ("" if route_init == "zero" else f" {route_init}") + ("" if route_sign == "none" else f" {route_sign}")


# Thư mục kết quả của một biến thể ở một độ phân giải (lượt trên một phần bệnh nhân: hậu tố tune<phần trăm>)
# Trả về: Path
def model_directory(name, size, fraction = 1.0, route_init = "zero", route_sign = "none", tag = ""):
    model_name = variant_name(name, size, route_init, route_sign) + (f" tune{round(fraction * 100)}" if fraction < 1 else "") + (f" {tag}" if tag else "")
    dataset_name = read_json(config.DATASET_INFO_DIRECTORY)["dataset_name"]
    return config.MODELS_OUTPUT_DIRECTORY / to_snake_case(dataset_name) / to_snake_case(model_name)


# Các lượt chạy cùng (biến thể, seed, LR backbone, hệ số đầu, khởi tạo và dấu tuyến thêm) trong một thư mục
# Trả về: Danh sách (train_info path, settings)
def matching_runs(directory, name, seed, learning_rate, head_lr, route_init = "zero", route_sign = "none"):
    runs = []
    for info_path in sorted(directory.glob("run_*/train_info.json")):
        settings = read_json(info_path).get("settings", {})
        optimizer = settings.get("training_config", {}).get("optimizer", {})
        if (settings.get("seed") == int(seed) and settings.get("model_kwargs", {}).get("mixer") == name
                and optimizer.get("new_module_lr_multiplier") == float(head_lr) and optimizer.get("learning_rate") == float(learning_rate)
                and settings.get("model_kwargs", {}).get("route_init", "zero") == route_init
                and settings.get("model_kwargs", {}).get("route_sign", "none") == route_sign):
            runs.append((info_path, settings))
    return runs


# best_model.pt của lượt cùng biến thể / seed ở độ phân giải `size` (LR mặc định của CLIS_PROTOCOL), đã xong fold. Phải có đúng một lượt
# Trả về: Path
def find_checkpoint(name, seed, fold, size, fraction, head_lr, route_init = "zero", route_sign = "none", tag = ""):
    learning_rate = float(config.CLIS_PROTOCOL["optimizer_config"]["learning_rate"])
    directory = model_directory(name, size, fraction, route_init, route_sign, tag)
    matches = [info_path.parent / f"fold_{fold}" / "best_model.pt" for info_path, settings in matching_runs(directory, name, seed, learning_rate, head_lr, route_init, route_sign)
               if settings.get("image_size") == int(size) and (info_path.parent / f"fold_{fold}" / "fold_done.json").is_file()]
    if len(matches) != 1:
        raise SystemExit(f"Cần đúng một lượt {name} | seed {seed} | {size} px đã xong fold {fold} trong {directory}, tìm thấy {len(matches)}")
    return matches[0]


# Thư mục fold của lượt cùng (biến thể, seed, LR, ghi đè); None nếu chưa chạy. Nhiều lượt khớp: ưu tiên lượt đã xong fold, rồi lượt mới nhất
# Trả về: Path hoặc None
def find_fold_directory(name, seed, fold, overrides, learning_rate, head_lr, route_init = "zero", route_sign = "none"):
    size, fraction = run_settings(overrides)
    tag = str({**config.CLIS_CONFIG, **overrides}.get("RUN_TAG", config.RUN_TAG))
    candidates = []
    for info_path, settings in matching_runs(model_directory(name, size, fraction, route_init, route_sign, tag), name, seed, learning_rate, head_lr, route_init, route_sign):
        snapshot = settings.get("config", {})
        if all(snapshot.get(key) == value for key, value in overrides.items() if key in snapshot):
            fold_directory = info_path.parent / f"fold_{fold}"
            candidates.append(((fold_directory / "fold_done.json").is_file(), info_path.stat().st_mtime, fold_directory))
    return max(candidates)[2] if candidates else None


# ============================================================
# Tiến trình con: một (biến thể, seed, fold)
def run_child(name, seed, fold, overrides, lr = None, head_lr = DEFAULT_HEAD_LR, init_from_size = None, route_init = "zero", route_sign = "none"):

    # Cấu hình chung của nghiên cứu CLIS + ghi đè, một fold
    apply_config({**config.CLIS_CONFIG, **overrides, "SEED": int(seed), "TRAIN_ALL_FOLD": False, "SELECTED_FOLD": [int(fold)]})

    # Tối ưu giống baseline ConvNeXt-T: LR backbone 1e-4 hoặc --lr, đầu phân loại x --head-lr
    protocol = copy.deepcopy(config.CLIS_PROTOCOL)
    protocol["optimizer_config"]["new_module_lr_multiplier"] = float(head_lr)
    if lr is not None:
        protocol["optimizer_config"]["learning_rate"] = float(lr)

    # Tham số mô hình: checkpointing từng khối VSS theo BLOCK_GRAD_CHECKPOINT; khởi tạo và dấu tuyến thêm (chỉ ghi khóa khi khác mặc định
    # "zero" / "none", nên model_kwargs của lượt mặc định giữ như các lượt cũ); nguồn khởi tạo theo --init-from-size
    kwargs = {"mixer": name, "grad_checkpoint": bool(config.BLOCK_GRAD_CHECKPOINT)}
    if route_init != "zero":
        kwargs["route_init"] = route_init
    if route_sign != "none":
        kwargs["route_sign"] = route_sign
    if float(config.DROP_PATH_RATE) > 0:
        kwargs["drop_path_rate"] = float(config.DROP_PATH_RATE)                     # Chỉ ghi khi > 0: model_kwargs của lượt cũ giữ nguyên
    if init_from_size is not None:
        kwargs["init_from"] = str(find_checkpoint(name, seed, fold, init_from_size, float(config.TRAIN_PATIENT_FRACTION), head_lr, route_init, route_sign, config.RUN_TAG))
        print(f"Khởi tạo từ {kwargs['init_from']}", flush = True)

    # Tên mô hình (lượt trên một phần bệnh nhân: thư mục riêng)
    model_name = variant_name(name, route_init = route_init, route_sign = route_sign)
    if float(config.TRAIN_PATIENT_FRACTION) < 1:
        model_name = f"{model_name} tune{round(float(config.TRAIN_PATIENT_FRACTION) * 100)}"
    if config.RUN_TAG:
        model_name = f"{model_name} {config.RUN_TAG}"
    return start_train(model_name = model_name, dataset_class = NIHChestXrayDataset, model_factory = vmx.VMambaCXR, model_kwargs = kwargs, **protocol)


# Mức tham gia của tuyến thêm trong best_model.pt: trung bình |hàng C của x_proj| / |D| của tuyến mới theo stage; None nếu vm_ras hoặc không đọc được
# Trả về: Chuỗi mô tả hoặc None
def describe_engagement(fold_directory):
    try:
        saved = load_torch(fold_directory / "best_model.pt")
        model = vmx.VMambaCXR(14, pretrained = False, **{key: value for key, value in saved["model_kwargs"].items() if key != "init_from"})
        model.load_state_dict(saved["model_state_dict"], strict = True)
        values = model.engagement()
        return " ".join(f"stage {stage + 1} {c:.3g}/{d:.3g}" for stage, c, d in values) if values else None
    except (OSError, KeyError, RuntimeError, ValueError):
        return None


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Hàng đợi VMamba-T CXR")
    parser.add_argument("names", nargs = "+", choices = vmx.NAMES, help = "Biến thể")
    parser.add_argument("--seeds", nargs = "+", type = int, default = [int(config.SEED)])
    parser.add_argument("--fold", type = int, default = 1, help = "Fold dùng làm Validation")
    parser.add_argument("--lr", type = float, default = None, help = "LR backbone (mặc định learning_rate của CLIS_PROTOCOL)")
    parser.add_argument("--head-lr", type = float, default = DEFAULT_HEAD_LR, help = "LR đầu phân loại = hệ số x LR backbone")
    parser.add_argument("--init-from-size", type = int, default = None, help = "Khởi tạo từ best_model.pt cùng biến thể / seed ở độ phân giải này")
    parser.add_argument("--route-init", choices = vmx.ROUTE_INITS, default = "zero", help = "Khởi tạo tuyến thêm của vm_clis / vm_rand (vm_ras không có tuyến thêm)")
    parser.add_argument("--route-sign", choices = vmx.ROUTE_SIGNS, default = "none", help = "Tuyến thêm quét tổng (none) hay hiệu có dấu hai nửa ảnh (signed)")
    parser.add_argument("--set", nargs = "+", default = [], metavar = "KEY=VALUE", help = "Ghi đè hằng số config (giá trị Python, ví dụ IMAGE_SIZE=1024)")
    parser.add_argument("--min-score", type = float, default = None, help = "Cổng: mã thoát 3 nếu macro AUROC Validation trung bình qua seed của một biến thể thấp hơn ngưỡng")
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
            run_child(args.names[0], int(seed), int(fold), overrides, lr = args.lr, head_lr = args.head_lr, init_from_size = args.init_from_size, route_init = args.route_init, route_sign = args.route_sign)
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

    # Danh sách việc
    jobs = [(name, seed) for name in args.names for seed in args.seeds]
    learning_rate = float(args.lr if args.lr is not None else config.CLIS_PROTOCOL["optimizer_config"]["learning_rate"])
    if args.dry_run:
        for index, (name, seed) in enumerate(jobs, start = 1):
            start = "" if args.init_from_size is None else f" | khởi tạo từ {args.init_from_size} px"
            print(f"{index:3d}. {variant_name(name, run_settings(overrides)[0], args.route_init, args.route_sign)} | seed {seed} | LR {learning_rate:g}, đầu x {args.head_lr:g} | fold {args.fold}{start} | {overrides or ''}")
        sys.exit(0)

    # Kiểm tra môi trường một lần cho cả hàng đợi
    script = str(Path(__file__).resolve())
    check = subprocess.run([sys.executable, script, *args.names, "--check-only"] + (["--set", *args.set] if args.set else []), cwd = ROOT_DIR)
    if check.returncode != 0:
        sys.exit(1)

    keep_system_awake()
    log(f"HÀNG ĐỢI VMAMBA {len(jobs)} việc | fold {args.fold} | LR {learning_rate:g}, đầu x {args.head_lr:g} | ghi đè {overrides or 'không'}"
        + ("" if args.init_from_size is None else f" | khởi tạo từ {args.init_from_size} px") + ("" if args.route_init == "zero" else f" | tuyến thêm khởi tạo {args.route_init}")
        + ("" if args.route_sign == "none" else f" | tuyến thêm {args.route_sign}")
        + ("" if args.min_score is None else f" | cổng macro_auroc >= {args.min_score:g}"))
    selection_metric = config.CLIS_PROTOCOL["selection_metric"]
    failed, scores = [], {}
    for index, (name, seed) in enumerate(jobs, start = 1):
        command = ([sys.executable, script, name, "--child", str(seed), str(args.fold), "--head-lr", str(args.head_lr)] + (["--set", *args.set] if args.set else [])
                   + (["--lr", str(args.lr)] if args.lr is not None else []) + (["--init-from-size", str(args.init_from_size)] if args.init_from_size is not None else [])
                   + ["--route-init", args.route_init, "--route-sign", args.route_sign])
        suffix = ("" if args.route_init == "zero" else f" {args.route_init}") + ("" if args.route_sign == "none" else f" {args.route_sign}")
        label = f"{index}/{len(jobs)} {name}{suffix} | seed {seed}"

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

        # Điểm Validation tốt nhất (lượt tạm dừng: tốt nhất tới epoch đã chạy), kèm mức tham gia của tuyến thêm
        fold_directory = find_fold_directory(name, seed, args.fold, overrides, learning_rate, args.head_lr, args.route_init, args.route_sign)
        score = None if fold_directory is None else history_score(fold_directory, selection_metric)
        if score is None or not (score["finished"] or code == PAUSED_EXIT):
            failed.append((name, seed, "không đọc được điểm"))
            continue
        scores.setdefault(name, {})[seed] = score["best_score"]
        engagement = None if name == "vm_ras" else describe_engagement(fold_directory)
        paused_at = "" if score["finished"] else f" | tạm dừng sau epoch {int(pd.read_csv(fold_directory / 'history.csv')['epoch'].max())}"
        log(f"ĐIỂM {name} | seed {seed} | Validation {selection_metric} {score['best_score']:.4f} ở epoch {score['best_epoch']}{paused_at} | {fold_directory.parent.name}"
            + ("" if engagement is None else f" | |C|/|D| tuyến thêm {engagement}"))

    # Tổng kết theo biến thể; cổng --min-score xét điểm trung bình
    below = []
    for name, (mean, spread) in seed_summary(scores).items():
        log(f"TỔNG KẾT {name} | {len(scores[name])} seed | Validation {selection_metric} {mean:.4f} ± {spread:.4f} | "
            + ", ".join(f"seed {seed}: {value:.4f}" for seed, value in scores[name].items()))
        if args.min_score is not None and mean < args.min_score:
            below.append(name)
    log("XONG HÀNG ĐỢI VMAMBA" + (f" | lỗi: {failed}" if failed else ""))
    if args.min_score is not None:
        verdict = "ĐẠT" if not (below or failed) else ("KHÔNG ĐẠT: " + (f"thấp hơn ngưỡng {below}" if below else "thiếu điểm của việc lỗi"))
        log(f"CỔNG {selection_metric} trung bình qua seed >= {args.min_score:g}: {verdict}")
    sys.exit(1 if failed else (GATE_EXIT if below else 0))
