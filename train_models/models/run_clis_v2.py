# Khai báo utilities
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"                                 # cuBLAS tất định (DETERMINISTIC_ALGORITHMS)
os.environ["TRITON_CACHE_AUTOTUNING"] = "1"                                        # mamba-ssm >= 2.3 ở chế độ tất định: lưu autotune kernel Triton (CLIS không dùng) thay vì cảnh báo

# Khai báo thư viện
import argparse
import ast
import copy
import subprocess
import time

import config
import models.clis_mamba_v2 as v2
from models.run_clis import GATE_EXIT, history_score, seed_summary
from training.experiment import start_train
from utilities.dataset import NIHChestXrayDataset
from utilities.environment import check_environment
from utilities.file import load_torch, read_json, to_snake_case
from utilities.log import timestamp
from utilities.setting import apply_config, keep_system_awake

# ============================================================
# Hàng đợi CLIS-Mamba v2 (models/clis_mamba_v2.py)
# Mục tiêu: Chạy các biến thể v2 với đúng giao thức của baseline ConvNeXt-T (config.CLIS_CONFIG + config.CLIS_PROTOCOL), log riêng clis_v2_queue.log
# ** LR: backbone và khối chèn 1e-4 (--lr đổi cả hai), đầu phân loại 1e-4 x m* của ConvNeXt-T (config.CLIS_LR_MULTIPLIER["convnext_t"] = 40): không tinh chỉnh LR mới
# ** Mỗi (biến thể, seed) chạy trong tiến trình con riêng; lỗi --> chờ rồi chạy lại, lượt chạy tự tiếp tục từ last_checkpoint.pt; việc đã xong được bỏ qua
#
# Ví dụ (WSL2, conda activate mamba-t29):
#   python models/run_clis_v2.py m_clis m_ras m_rand gatedconv --seeds 42                --> 512 px, toàn bộ bệnh nhân
#   python models/run_clis_v2.py m_clis --seeds 42 --min-score 0.85                     --> cổng: mã thoát 3 nếu macro AUROC Validation trung bình < 0.85
#   python models/run_clis_v2.py convnext_t m_clis --seeds 42 --init-from-size 512 --lr 5e-5 --set IMAGE_SIZE=1024 BATCH_SIZE=4 GRAD_ACCUM_STEPS=8 EPOCHS=5 BLOCK_GRAD_CHECKPOINT=True
#                                                                                       --> tinh chỉnh 1024 px từ best_model.pt 512 px cùng biến thể, cùng seed (convnext_t: checkpoint E1)
QUEUE_LOG = config.MODELS_OUTPUT_DIRECTORY / "clis_v2_queue.log"
HEAD_LR_MULTIPLIER = float(config.CLIS_LR_MULTIPLIER["convnext_t"])              # Hệ số LR đầu phân loại: m* đã tinh chỉnh của ConvNeXt-T


def log(message):
    line = f"[{timestamp()}] {message}"
    print(line, flush = True)
    QUEUE_LOG.parent.mkdir(parents = True, exist_ok = True)
    with QUEUE_LOG.open("a", encoding = "utf-8") as f:
        f.write(line + "\n")


# Thư mục kết quả của một biến thể ở một độ phân giải (lượt trên một phần bệnh nhân: hậu tố tune<phần trăm>)
# Trả về: Path
def model_directory(name, size, fraction = 1.0):
    model_name = v2.model_name(name, size) + (f" tune{round(fraction * 100)}" if fraction < 1 else "")
    dataset_name = read_json(config.DATASET_INFO_DIRECTORY)["dataset_name"]
    return config.MODELS_OUTPUT_DIRECTORY / to_snake_case(dataset_name) / to_snake_case(model_name)


# Kích thước ảnh và tỉ lệ bệnh nhân sau ghi đè (cùng cách đọc ở tiến trình cha và con)
# Trả về: (size, fraction)
def run_settings(overrides):
    settings = {**config.CLIS_CONFIG, **overrides}
    return int(settings.get("IMAGE_SIZE", config.IMAGE_SIZE)), float(settings.get("TRAIN_PATIENT_FRACTION", config.TRAIN_PATIENT_FRACTION))


# ============================================================
# best_model.pt của lượt chạy cùng biến thể, cùng seed ở độ phân giải `size`, LR mặc định của CLIS_PROTOCOL, đầu phân loại x m* của ConvNeXt-T,
# đã xong fold (điểm khởi đầu khi tinh chỉnh ở độ phân giải cao hơn). convnext_t: lượt E1 của models/run_clis.py (model_kwargs rỗng). Phải tìm thấy đúng một lượt
# Trả về: Path
def find_checkpoint(name, seed, fold, size):
    directory = model_directory(name, size)
    learning_rate = float(config.CLIS_PROTOCOL["optimizer_config"]["learning_rate"])
    matches = []
    for info_path in sorted(directory.glob("run_*/train_info.json")):
        settings = read_json(info_path).get("settings", {})
        optimizer = settings.get("training_config", {}).get("optimizer", {})
        if (settings.get("seed") == int(seed) and settings.get("image_size") == int(size)
                and settings.get("model_kwargs", {}).get("mixer", "convnext_t") == name
                and optimizer.get("new_module_lr_multiplier") == HEAD_LR_MULTIPLIER and optimizer.get("learning_rate") == learning_rate
                and (info_path.parent / f"fold_{fold}" / "fold_done.json").is_file()):
            matches.append(info_path.parent / f"fold_{fold}" / "best_model.pt")
    if len(matches) != 1:
        raise SystemExit(f"Cần đúng một lượt chạy {name} | seed {seed} | {size} px | LR {learning_rate:g}, đầu phân loại x {HEAD_LR_MULTIPLIER:g} đã xong fold {fold} "
                         f"trong {directory}, tìm thấy {len(matches)}: {[str(m) for m in matches]}")
    return matches[0]


# Thư mục fold của lượt chạy cùng (biến thể, seed, LR, ghi đè); None nếu chưa chạy. Nhiều lượt khớp: ưu tiên lượt đã xong fold, rồi lượt mới nhất
# Trả về: Path hoặc None
def find_fold_directory(name, seed, fold, overrides, learning_rate):
    size, fraction = run_settings(overrides)
    candidates = []
    for info_path in sorted(model_directory(name, size, fraction).glob("run_*/train_info.json")):
        settings = read_json(info_path).get("settings", {})
        snapshot = settings.get("config", {})
        optimizer = settings.get("training_config", {}).get("optimizer", {})
        if (settings.get("seed") == int(seed)
                and settings.get("model_kwargs", {}).get("mixer", "convnext_t") == name
                and optimizer.get("new_module_lr_multiplier") == HEAD_LR_MULTIPLIER
                and optimizer.get("learning_rate") == float(learning_rate)
                and all(snapshot.get(key) == value for key, value in overrides.items() if key in snapshot)):
            fold_directory = info_path.parent / f"fold_{fold}"
            candidates.append(((fold_directory / "fold_done.json").is_file(), info_path.stat().st_mtime, fold_directory))
    return max(candidates)[2] if candidates else None


# ============================================================
# Tiến trình con: một (biến thể, seed, fold)
def run_child(name, seed, fold, overrides, lr = None, init_from_size = None):

    # Cấu hình chung của nghiên cứu CLIS + ghi đè, một fold
    apply_config({**config.CLIS_CONFIG, **overrides, "SEED": int(seed), "TRAIN_ALL_FOLD": False, "SELECTED_FOLD": [int(fold)]})

    # Tối ưu giống baseline ConvNeXt-T: LR backbone (và khối chèn) 1e-4 hoặc --lr, đầu phân loại x m* của ConvNeXt-T
    protocol = copy.deepcopy(config.CLIS_PROTOCOL)
    protocol["optimizer_config"]["new_module_lr_multiplier"] = HEAD_LR_MULTIPLIER
    if lr is not None:
        protocol["optimizer_config"]["learning_rate"] = float(lr)

    # Tham số mô hình: checkpointing khối chèn theo BLOCK_GRAD_CHECKPOINT (convnext_t không có khối chèn)
    kwargs = {"mixer": name}
    if name != "convnext_t":
        kwargs["grad_checkpoint"] = bool(config.BLOCK_GRAD_CHECKPOINT)
    if init_from_size is not None:
        kwargs["init_from"] = str(find_checkpoint(name, seed, fold, init_from_size))
        print(f"Khởi tạo từ {kwargs['init_from']}", flush = True)

    # Tên mô hình (lượt trên một phần bệnh nhân: thư mục riêng, find_checkpoint không lấy nhầm)
    model_name = v2.model_name(name)
    if float(config.TRAIN_PATIENT_FRACTION) < 1:
        model_name = f"{model_name} tune{round(float(config.TRAIN_PATIENT_FRACTION) * 100)}"
    return start_train(model_name = model_name, dataset_class = NIHChestXrayDataset, model_factory = v2.CLISMambaV2, model_kwargs = kwargs, **protocol)


# |γ| trung bình của khối chèn trong best_model.pt (khối có được dùng không); None nếu không đọc được
# Trả về: Chuỗi mô tả hoặc None
def describe_layer_scales(fold_directory):
    try:
        state = load_torch(fold_directory / "best_model.pt")["model_state_dict"]
        values = [(state[f"inserted.blocks.{index}.gamma1"].abs().mean().item(), state[f"inserted.blocks.{index}.gamma2"].abs().mean().item())
                  for index in range(len({key.split(".")[2] for key in state if key.startswith("inserted.blocks.")}))]
        return " ".join(f"k{index + 1} {mixer:.3g}/{mlp:.3g}" for index, (mixer, mlp) in enumerate(values)) if values else None
    except (OSError, KeyError, RuntimeError):
        return None


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Hàng đợi CLIS-Mamba v2")
    parser.add_argument("names", nargs = "+", choices = v2.NAMES, help = "Biến thể v2")
    parser.add_argument("--seeds", nargs = "+", type = int, default = [int(config.SEED)])
    parser.add_argument("--fold", type = int, default = 1, help = "Fold dùng làm Validation")
    parser.add_argument("--lr", type = float, default = None, help = "LR backbone và khối chèn (mặc định learning_rate của CLIS_PROTOCOL)")
    parser.add_argument("--init-from-size", type = int, default = None, help = "Khởi tạo từ best_model.pt cùng biến thể / seed ở độ phân giải này")
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

    # Tiến trình con
    if args.child is not None:
        seed, fold = args.child
        run_child(args.names[0], int(seed), int(fold), overrides, lr = args.lr, init_from_size = args.init_from_size)
        sys.exit(0)

    # Kiểm tra môi trường trong tiến trình con (tiến trình cha không khởi tạo CUDA nên không giữ VRAM suốt hàng đợi)
    settings = {**config.CLIS_CONFIG, **overrides}
    if args.check_only:
        try:
            check_environment(require_kernel = any(name in v2.ROUTES for name in args.names), amp_dtype = settings.get("AMP_DTYPE", config.AMP_DTYPE),
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
            print(f"{index:3d}. {v2.model_name(name, run_settings(overrides)[0])} | seed {seed} | LR {learning_rate:g}, đầu x {HEAD_LR_MULTIPLIER:g} | fold {args.fold}{start} | {overrides or ''}")
        sys.exit(0)

    # Kiểm tra môi trường một lần cho cả hàng đợi
    script = str(Path(__file__).resolve())
    check = subprocess.run([sys.executable, script, *args.names, "--check-only"] + (["--set", *args.set] if args.set else []), cwd = ROOT_DIR)
    if check.returncode != 0:
        sys.exit(1)

    keep_system_awake()
    log(f"HÀNG ĐỢI V2 {len(jobs)} việc | fold {args.fold} | LR {learning_rate:g}, đầu x {HEAD_LR_MULTIPLIER:g} | ghi đè {overrides or 'không'}"
        + ("" if args.init_from_size is None else f" | khởi tạo từ {args.init_from_size} px") + ("" if args.min_score is None else f" | cổng macro_auroc >= {args.min_score:g}"))
    selection_metric = config.CLIS_PROTOCOL["selection_metric"]
    failed, scores = [], {}
    for index, (name, seed) in enumerate(jobs, start = 1):
        command = ([sys.executable, script, name, "--child", str(seed), str(args.fold)] + (["--set", *args.set] if args.set else [])
                   + (["--lr", str(args.lr)] if args.lr is not None else []) + (["--init-from-size", str(args.init_from_size)] if args.init_from_size is not None else []))
        label = f"{index}/{len(jobs)} {name} | seed {seed}"

        # Chạy, lỗi thì chờ rồi chạy lại
        for attempt in range(1, args.retries + 2):
            log(f"BẮT ĐẦU {label} | lần {attempt}")
            started = time.time()
            code = subprocess.run(command, cwd = ROOT_DIR).returncode
            log(f"KẾT THÚC {label} | mã {code} | {(time.time() - started) / 3600:.1f} giờ")
            if code == 0:
                break
            if attempt <= args.retries:
                time.sleep(args.retry_wait)
        if code != 0:
            failed.append((name, seed))
            continue

        # Điểm Validation tốt nhất của lượt vừa xong, kèm |γ| của khối chèn
        fold_directory = find_fold_directory(name, seed, args.fold, overrides, learning_rate)
        score = None if fold_directory is None else history_score(fold_directory, selection_metric)
        if score is None or not score["finished"]:
            failed.append((name, seed, "không đọc được điểm"))
            continue
        scores.setdefault(name, {})[seed] = score["best_score"]
        scales = None if name == "convnext_t" else describe_layer_scales(fold_directory)
        log(f"ĐIỂM {name} | seed {seed} | Validation {selection_metric} {score['best_score']:.4f} ở epoch {score['best_epoch']} | {fold_directory.parent.name}"
            + ("" if scales is None else f" | |γ| trộn/MLP {scales}"))

    # Tổng kết theo biến thể: trung bình ± độ lệch chuẩn qua seed; cổng --min-score xét điểm trung bình
    below = []
    for name, (mean, spread) in seed_summary(scores).items():
        log(f"TỔNG KẾT {name} | {len(scores[name])} seed | Validation {selection_metric} {mean:.4f} ± {spread:.4f} | "
            + ", ".join(f"seed {seed}: {value:.4f}" for seed, value in scores[name].items()))
        if args.min_score is not None and mean < args.min_score:
            below.append(name)
    log("XONG HÀNG ĐỢI V2" + (f" | lỗi: {failed}" if failed else ""))
    if args.min_score is not None:
        verdict = "ĐẠT" if not (below or failed) else ("KHÔNG ĐẠT: " + (f"thấp hơn ngưỡng {below}" if below else "thiếu điểm của việc lỗi"))
        log(f"CỔNG {selection_metric} trung bình qua seed >= {args.min_score:g}: {verdict}")
    sys.exit(1 if failed else (GATE_EXIT if below else 0))
