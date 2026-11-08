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

import pandas as pd

import config
import models.clis_mamba_v2 as v2
from models.run_clis import GATE_EXIT, PAUSED_EXIT, history_score, seed_summary
from training.experiment import start_train
from training.train import TrainingPaused
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
#
# Pilot khối chèn 2026-11-08 (thư mục kết quả có hậu tố riêng nên không lẫn với lượt thường):
#   --block-lr 10                     --> LR khối chèn = 10 x LR backbone (tham số hóa lại trong mô hình, hậu tố " blr10")
#   --freeze-backbone --set EPOCHS=2  --> LP-FT giai đoạn 1: đóng băng ConvNeXt-T, chỉ khối chèn và đầu phân loại học (hậu tố " lp")
#   --lpft 10                         --> LP-FT giai đoạn 2: tinh chỉnh toàn bộ từ best_model.pt của giai đoạn 1 cùng biến thể / seed / tỉ lệ bệnh nhân
#                                         đã chạy với --freeze-backbone --block-lr 10 (hậu tố " lpft"); kèm --init-from-size: nguồn là lượt " lpft" ở độ phân giải đó
#   --set PAUSE_AFTER_EPOCH=5         --> dừng sau epoch 5 (mã 75, không phải lỗi, chạy lại để tiếp tục); điểm là tốt nhất tới epoch đó
QUEUE_LOG = config.MODELS_OUTPUT_DIRECTORY / "clis_v2_queue.log"
HEAD_LR_MULTIPLIER = float(config.CLIS_LR_MULTIPLIER["convnext_t"])              # Hệ số LR đầu phân loại: m* đã tinh chỉnh của ConvNeXt-T
RESIDUAL_INIT = "zero"                                                             # Khởi tạo phần dư của khối chèn; lượt LayerScale (thiếu khóa này trong model_kwargs) bị loại khỏi mọi tìm kiếm


# Lượt chạy có đúng cách khởi tạo phần dư hiện hành không (convnext_t không có khối chèn nên luôn đúng)
# Trả về: bool
def current_init(name, model_kwargs):
    return name == "convnext_t" or model_kwargs.get("residual_init") == RESIDUAL_INIT


def log(message):
    line = f"[{timestamp()}] {message}"
    print(line, flush = True)
    QUEUE_LOG.parent.mkdir(parents = True, exist_ok = True)
    with QUEUE_LOG.open("a", encoding = "utf-8") as f:
        f.write(line + "\n")


# Hậu tố tên mô hình theo cách huấn luyện khối chèn: " lp" (LP-FT giai đoạn 1) | " lpft" (giai đoạn 2), " blr<s>" (LR khối chèn = s x LR backbone)
# Trả về: Chuỗi (rỗng với lượt thường)
def variant_suffix(block_lr = 1.0, freeze = False, lpft = None):
    return (" lp" if freeze else "") + (" lpft" if lpft is not None else "") + (f" blr{block_lr:g}" if float(block_lr) != 1.0 else "")


# Tham số model_kwargs của cách huấn luyện khối chèn (chỉ ghi khóa khác mặc định nên chữ ký lượt thường không đổi)
# Trả về: dict
def training_kwargs(block_lr = 1.0, freeze = False):
    return {**({"block_lr_scale": float(block_lr)} if float(block_lr) != 1.0 else {}), **({"freeze_backbone": True} if freeze else {})}


# Lượt chạy có đúng cách huấn luyện khối chèn không (so khóa block_lr_scale, freeze_backbone của model_kwargs)
def same_training(model_kwargs, block_lr = 1.0, freeze = False):
    return float(model_kwargs.get("block_lr_scale", 1.0)) == float(block_lr) and bool(model_kwargs.get("freeze_backbone", False)) == bool(freeze)


# Thư mục kết quả của một biến thể ở một độ phân giải (hậu tố cách huấn luyện khối chèn; lượt trên một phần bệnh nhân: hậu tố tune<phần trăm>)
# Trả về: Path
def model_directory(name, size, fraction = 1.0, suffix = ""):
    model_name = v2.model_name(name, size) + suffix + (f" tune{round(fraction * 100)}" if fraction < 1 else "")
    dataset_name = read_json(config.DATASET_INFO_DIRECTORY)["dataset_name"]
    return config.MODELS_OUTPUT_DIRECTORY / to_snake_case(dataset_name) / to_snake_case(model_name)


# Kích thước ảnh và tỉ lệ bệnh nhân sau ghi đè (cùng cách đọc ở tiến trình cha và con)
# Trả về: (size, fraction)
def run_settings(overrides):
    settings = {**config.CLIS_CONFIG, **overrides}
    return int(settings.get("IMAGE_SIZE", config.IMAGE_SIZE)), float(settings.get("TRAIN_PATIENT_FRACTION", config.TRAIN_PATIENT_FRACTION))


# ============================================================
# best_model.pt của lượt chạy cùng biến thể, cùng seed ở độ phân giải `size`, LR mặc định của CLIS_PROTOCOL, đầu phân loại x m* của ConvNeXt-T,
# cùng cách huấn luyện khối chèn (block_lr, freeze), đã xong fold. Dùng làm điểm khởi đầu khi tinh chỉnh ở độ phân giải cao hơn, hoặc giai đoạn 2 LP-FT
# (nguồn là lượt freeze). convnext_t: lượt E1 của models/run_clis.py (model_kwargs rỗng). Phải tìm thấy đúng một lượt
# Trả về: Path
def find_checkpoint(name, seed, fold, size, fraction = 1.0, suffix = "", block_lr = 1.0, freeze = False):
    directory = model_directory(name, size, fraction, suffix)
    learning_rate = float(config.CLIS_PROTOCOL["optimizer_config"]["learning_rate"])
    matches = []
    for info_path in sorted(directory.glob("run_*/train_info.json")):
        settings = read_json(info_path).get("settings", {})
        optimizer = settings.get("training_config", {}).get("optimizer", {})
        model_kwargs = settings.get("model_kwargs", {})
        if (settings.get("seed") == int(seed) and settings.get("image_size") == int(size)
                and model_kwargs.get("mixer", "convnext_t") == name and current_init(name, model_kwargs) and same_training(model_kwargs, block_lr, freeze)
                and optimizer.get("new_module_lr_multiplier") == HEAD_LR_MULTIPLIER and optimizer.get("learning_rate") == learning_rate
                and (info_path.parent / f"fold_{fold}" / "fold_done.json").is_file()):
            matches.append(info_path.parent / f"fold_{fold}" / "best_model.pt")
    if len(matches) != 1:
        raise SystemExit(f"Cần đúng một lượt chạy {name}{suffix} | seed {seed} | {size} px | LR {learning_rate:g}, đầu phân loại x {HEAD_LR_MULTIPLIER:g} đã xong fold {fold} "
                         f"trong {directory}, tìm thấy {len(matches)}: {[str(m) for m in matches]}")
    return matches[0]


# Nguồn khởi tạo của lượt hiện tại: --init-from-size (cùng cách huấn luyện ở độ phân giải đó) hoặc --lpft (lượt giai đoạn 1 cùng độ phân giải, tỉ lệ bệnh nhân)
# Trả về: Path hoặc None
def init_source(name, seed, fold, overrides, block_lr, lpft, init_from_size):
    size, fraction = run_settings(overrides)
    if init_from_size is not None:
        return find_checkpoint(name, seed, fold, init_from_size, fraction, variant_suffix(block_lr, lpft = lpft), block_lr)
    if lpft is not None:
        return find_checkpoint(name, seed, fold, size, fraction, variant_suffix(lpft, freeze = True), lpft, freeze = True)
    return None


# Thư mục fold của lượt chạy cùng (biến thể, seed, LR, ghi đè); None nếu chưa chạy. Nhiều lượt khớp: ưu tiên lượt đã xong fold, rồi lượt mới nhất
# Trả về: Path hoặc None
def find_fold_directory(name, seed, fold, overrides, learning_rate, block_lr = 1.0, freeze = False, lpft = None):
    size, fraction = run_settings(overrides)
    candidates = []
    for info_path in sorted(model_directory(name, size, fraction, variant_suffix(block_lr, freeze, lpft)).glob("run_*/train_info.json")):
        settings = read_json(info_path).get("settings", {})
        snapshot = settings.get("config", {})
        optimizer = settings.get("training_config", {}).get("optimizer", {})
        model_kwargs = settings.get("model_kwargs", {})
        if (settings.get("seed") == int(seed)
                and model_kwargs.get("mixer", "convnext_t") == name and current_init(name, model_kwargs) and same_training(model_kwargs, block_lr, freeze)
                and optimizer.get("new_module_lr_multiplier") == HEAD_LR_MULTIPLIER
                and optimizer.get("learning_rate") == float(learning_rate)
                and all(snapshot.get(key) == value for key, value in overrides.items() if key in snapshot)):
            fold_directory = info_path.parent / f"fold_{fold}"
            candidates.append(((fold_directory / "fold_done.json").is_file(), info_path.stat().st_mtime, fold_directory))
    return max(candidates)[2] if candidates else None


# ============================================================
# Tiến trình con: một (biến thể, seed, fold)
def run_child(name, seed, fold, overrides, lr = None, init_from_size = None, block_lr = 1.0, freeze = False, lpft = None):

    # Cấu hình chung của nghiên cứu CLIS + ghi đè, một fold
    apply_config({**config.CLIS_CONFIG, **overrides, "SEED": int(seed), "TRAIN_ALL_FOLD": False, "SELECTED_FOLD": [int(fold)]})

    # Tối ưu giống baseline ConvNeXt-T: LR backbone (và khối chèn) 1e-4 hoặc --lr, đầu phân loại x m* của ConvNeXt-T
    protocol = copy.deepcopy(config.CLIS_PROTOCOL)
    protocol["optimizer_config"]["new_module_lr_multiplier"] = HEAD_LR_MULTIPLIER
    if lr is not None:
        protocol["optimizer_config"]["learning_rate"] = float(lr)

    # Tham số mô hình: khởi tạo 0 phép chiếu ra của khối chèn (ghi rõ vào chữ ký), checkpointing khối chèn theo BLOCK_GRAD_CHECKPOINT (convnext_t không có khối chèn)
    # Cách huấn luyện khối chèn (LR khối chèn, đóng băng backbone) chỉ ghi khi khác mặc định; nguồn khởi tạo theo --init-from-size / --lpft
    kwargs = {"mixer": name}
    if name != "convnext_t":
        kwargs["residual_init"] = RESIDUAL_INIT
        kwargs["grad_checkpoint"] = bool(config.BLOCK_GRAD_CHECKPOINT)
    kwargs.update(training_kwargs(block_lr, freeze))
    source = init_source(name, seed, fold, overrides, block_lr, lpft, init_from_size)
    if source is not None:
        kwargs["init_from"] = str(source)
        print(f"Khởi tạo từ {kwargs['init_from']}", flush = True)

    # Tên mô hình (hậu tố cách huấn luyện khối chèn; lượt trên một phần bệnh nhân: thư mục riêng, find_checkpoint không lấy nhầm)
    model_name = v2.model_name(name) + variant_suffix(block_lr, freeze, lpft)
    if float(config.TRAIN_PATIENT_FRACTION) < 1:
        model_name = f"{model_name} tune{round(float(config.TRAIN_PATIENT_FRACTION) * 100)}"
    return start_train(model_name = model_name, dataset_class = NIHChestXrayDataset, model_factory = v2.CLISMambaV2, model_kwargs = kwargs, **protocol)


# Mức tham gia của khối chèn trong best_model.pt: trung bình |trọng số| phép chiếu ra của nhánh trộn / nhánh MLP (0 lúc khởi tạo); None nếu không đọc được
# Trả về: Chuỗi mô tả hoặc None
def describe_engagement(fold_directory):
    try:
        saved = load_torch(fold_directory / "best_model.pt")
        state = v2.effective_state(saved["model_state_dict"], float(saved.get("model_kwargs", {}).get("block_lr_scale", 1.0)))
        values = []
        for index in range(len({key.split(".")[2] for key in state if key.startswith("inserted.blocks.")})):
            mixer = next(state[f"inserted.blocks.{index}.mixer.{name}.weight"] for name in ("out_proj", "fc2", "proj") if f"inserted.blocks.{index}.mixer.{name}.weight" in state)
            values.append((mixer.abs().mean().item(), state[f"inserted.blocks.{index}.mlp.2.weight"].abs().mean().item()))
        return " ".join(f"k{index + 1} {mixer:.3g}/{mlp:.3g}" for index, (mixer, mlp) in enumerate(values)) if values else None
    except (OSError, KeyError, RuntimeError, StopIteration):
        return None


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Hàng đợi CLIS-Mamba v2")
    parser.add_argument("names", nargs = "+", choices = v2.NAMES, help = "Biến thể v2")
    parser.add_argument("--seeds", nargs = "+", type = int, default = [int(config.SEED)])
    parser.add_argument("--fold", type = int, default = 1, help = "Fold dùng làm Validation")
    parser.add_argument("--lr", type = float, default = None, help = "LR backbone và khối chèn (mặc định learning_rate của CLIS_PROTOCOL)")
    parser.add_argument("--init-from-size", type = int, default = None, help = "Khởi tạo từ best_model.pt cùng biến thể / seed ở độ phân giải này")
    parser.add_argument("--block-lr", type = float, default = 1.0, help = "LR khối chèn = hệ số x LR backbone (tham số hóa lại trong mô hình)")
    parser.add_argument("--freeze-backbone", action = "store_true", help = "LP-FT giai đoạn 1: đóng băng ConvNeXt-T")
    parser.add_argument("--lpft", type = float, default = None, metavar = "STAGE1_BLOCK_LR", help = "LP-FT giai đoạn 2: khởi tạo từ lượt --freeze-backbone với --block-lr này")
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

    # Kiểm tra tổ hợp cờ huấn luyện khối chèn
    if args.freeze_backbone and args.lpft is not None:
        parser.error("--freeze-backbone (giai đoạn 1) và --lpft (giai đoạn 2) không dùng cùng lúc")
    if args.block_lr <= 0 or (args.lpft is not None and args.lpft <= 0):
        parser.error("--block-lr và --lpft phải dương")
    if args.block_lr != 1.0 and "convnext_t" in args.names:
        parser.error("convnext_t không có khối chèn: chạy riêng, không kèm --block-lr")
    flags = {"block_lr": args.block_lr, "freeze": args.freeze_backbone, "lpft": args.lpft}
    flag_text = ("" if args.block_lr == 1.0 else f" | LR khối chèn x {args.block_lr:g}") + (" | đóng băng backbone (LP-FT 1)" if args.freeze_backbone else "") \
        + ("" if args.lpft is None else f" | LP-FT 2 từ lượt đóng băng x {args.lpft:g}")

    # Tiến trình con (tạm dừng có chủ đích theo PAUSE_AFTER_EPOCH: mã PAUSED_EXIT, không phải lỗi)
    if args.child is not None:
        seed, fold = args.child
        try:
            run_child(args.names[0], int(seed), int(fold), overrides, lr = args.lr, init_from_size = args.init_from_size, **flags)
        except TrainingPaused as paused:
            print(paused, flush = True)
            sys.exit(PAUSED_EXIT)
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
            print(f"{index:3d}. {v2.model_name(name, run_settings(overrides)[0])}{variant_suffix(**flags)} | seed {seed} | LR {learning_rate:g}, đầu x {HEAD_LR_MULTIPLIER:g}"
                  f" | fold {args.fold}{start}{flag_text} | {overrides or ''}")
        sys.exit(0)

    # Kiểm tra môi trường một lần cho cả hàng đợi
    script = str(Path(__file__).resolve())
    check = subprocess.run([sys.executable, script, *args.names, "--check-only"] + (["--set", *args.set] if args.set else []), cwd = ROOT_DIR)
    if check.returncode != 0:
        sys.exit(1)

    keep_system_awake()
    log(f"HÀNG ĐỢI V2 {len(jobs)} việc | fold {args.fold} | LR {learning_rate:g}, đầu x {HEAD_LR_MULTIPLIER:g} | ghi đè {overrides or 'không'}"
        + ("" if args.init_from_size is None else f" | khởi tạo từ {args.init_from_size} px") + flag_text + ("" if args.min_score is None else f" | cổng macro_auroc >= {args.min_score:g}"))
    selection_metric = config.CLIS_PROTOCOL["selection_metric"]
    failed, scores = [], {}
    for index, (name, seed) in enumerate(jobs, start = 1):
        command = ([sys.executable, script, name, "--child", str(seed), str(args.fold)] + (["--set", *args.set] if args.set else [])
                   + (["--lr", str(args.lr)] if args.lr is not None else []) + (["--init-from-size", str(args.init_from_size)] if args.init_from_size is not None else [])
                   + (["--block-lr", str(args.block_lr)] if args.block_lr != 1.0 else []) + (["--freeze-backbone"] if args.freeze_backbone else [])
                   + (["--lpft", str(args.lpft)] if args.lpft is not None else []))
        label = f"{index}/{len(jobs)} {name}{variant_suffix(**flags)} | seed {seed}"

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

        # Điểm Validation tốt nhất của lượt vừa xong (lượt tạm dừng: tốt nhất tới epoch đã chạy), kèm mức tham gia của khối chèn
        fold_directory = find_fold_directory(name, seed, args.fold, overrides, learning_rate, **flags)
        score = None if fold_directory is None else history_score(fold_directory, selection_metric)
        if score is None or not (score["finished"] or code == PAUSED_EXIT):
            failed.append((name, seed, "không đọc được điểm"))
            continue
        scores.setdefault(name, {})[seed] = score["best_score"]
        engagement = None if name == "convnext_t" else describe_engagement(fold_directory)
        paused_at = "" if score["finished"] else f" | tạm dừng sau epoch {int(pd.read_csv(fold_directory / 'history.csv')['epoch'].max())}"
        log(f"ĐIỂM {name}{variant_suffix(**flags)} | seed {seed} | Validation {selection_metric} {score['best_score']:.4f} ở epoch {score['best_epoch']}{paused_at} | {fold_directory.parent.name}"
            + ("" if engagement is None else f" | |W chiếu ra| trộn/MLP {engagement}"))

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
