# Khai báo utilities
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import argparse
import ast
import copy
import subprocess
import time

import config
import models.clis_mamba as clis
import models.convnext_tiny as convnext
import models.densenet121 as densenet
from training.experiment import run_experiment
from utilities.dataset import NIHChestXrayDataset
from utilities.file import read_json, to_snake_case
from utilities.log import timestamp
from utilities.setting import apply_config, keep_system_awake

# ============================================================
# Hàng đợi của nghiên cứu CLIS-Mamba: mọi mô hình dùng chung config.CLIS_CONFIG + config.CLIS_PROTOCOL, fold 1 làm Validation.
#   densenet121, convnext_t : baseline E1
#   m_ras, m_clis, m_rand, m_clis4, gatedconv, attn, flipdiff : các biến thể CLIS-Mamba (E2 / E3)
# Mỗi (mô hình, seed) chạy trong tiến trình con riêng: lỗi CUDA / hết bộ nhớ chỉ làm hỏng việc đó; lỗi --> chờ rồi chạy lại,
# lượt chạy tự tiếp tục từ last_checkpoint.pt. Biến thể SS2D cần kernel mamba-ssm: chạy trong WSL2 (conda activate mamba-main).
BASELINES = {
    "densenet121"   : (densenet.MODEL_NAME, densenet.DenseNetBaseline, {}),
    "convnext_t"    : (convnext.MODEL_NAME, convnext.ConvNeXtTinyBaseline, {}),
}
NAMES = list(BASELINES) + list(clis.VARIANTS)
QUEUE_LOG = config.MODELS_OUTPUT_DIRECTORY / "clis_queue.log"


def log(message):
    line = f"[{timestamp()}] {message}"
    print(line, flush = True)
    QUEUE_LOG.parent.mkdir(parents = True, exist_ok = True)
    with QUEUE_LOG.open("a", encoding = "utf-8") as f:
        f.write(line + "\n")


# ============================================================
# best_model.pt của lượt chạy cùng biến thể, cùng seed, cùng hệ số LR mặc định của CLIS_PROTOCOL ở độ phân giải `size`,
# đã xong fold (điểm khởi đầu khi tinh chỉnh ở độ phân giải cao hơn). Phải tìm thấy đúng một lượt chạy
def find_checkpoint(name, seed, fold, size):
    dataset_name = read_json(config.DATASET_INFO_DIRECTORY)["dataset_name"]
    directory = config.MODELS_OUTPUT_DIRECTORY / to_snake_case(dataset_name) / to_snake_case(clis.model_name(name, size))
    multiplier = float(config.CLIS_PROTOCOL["optimizer_config"]["new_module_lr_multiplier"])
    matches = []
    for info_path in sorted(directory.glob("run_*/train_info.json")):
        settings = read_json(info_path).get("settings", {})
        if (settings.get("seed") == int(seed) and settings.get("image_size") == int(size)
                and settings.get("model_kwargs", {}).get("mixer") == clis.VARIANTS[name]["mixer"]
                and settings.get("training_config", {}).get("optimizer", {}).get("new_module_lr_multiplier") == multiplier
                and (info_path.parent / f"fold_{fold}" / "fold_done.json").is_file()):
            matches.append(info_path.parent / f"fold_{fold}" / "best_model.pt")
    if len(matches) != 1:
        raise SystemExit(f"Cần đúng một lượt chạy {name} | seed {seed} | {size} px | LR lớp mới x {multiplier} đã xong fold {fold} "
                         f"trong {directory}, tìm thấy {len(matches)}: {[str(m) for m in matches]}")
    return matches[0]


# ============================================================
# Tiến trình con: một (mô hình, seed, fold) với giao thức CLIS
def run_child(name, seed, fold, lr_multiplier, overrides, lr = None, init_from_size = None):
    apply_config({**config.CLIS_CONFIG, **overrides, "SEED": int(seed), "TRAIN_ALL_FOLD": False, "SELECTED_FOLD": [int(fold)]})
    protocol = copy.deepcopy(config.CLIS_PROTOCOL)
    if lr_multiplier is not None:
        protocol["optimizer_config"]["new_module_lr_multiplier"] = float(lr_multiplier)
    if lr is not None:
        protocol["optimizer_config"]["lr"] = float(lr)
    if name in BASELINES:
        if init_from_size is not None:
            raise SystemExit("--init-from-size chỉ dùng cho các biến thể CLIS-Mamba")
        base_name, factory, kwargs = BASELINES[name]
        model_name = f"{base_name} CLIS {config.IMAGE_SIZE}px"
    else:
        model_name, factory, kwargs = clis.model_name(name), clis.CLISMamba, dict(clis.VARIANTS[name])
        if init_from_size is not None:
            kwargs["init_from"] = str(find_checkpoint(name, seed, fold, init_from_size))
            print(f"Khởi tạo từ {kwargs['init_from']}")
    run_experiment(model_name = model_name, dataset_class = NIHChestXrayDataset, model_factory = factory, model_kwargs = kwargs, **protocol)


# ============================================================
# Chạy thí nghiệm. Ví dụ:
#   python models/run_clis.py densenet121 convnext_t --seeds 42 43 44        --> E1: baseline, 3 seed
#   python models/run_clis.py m_ras m_clis m_rand gatedconv attn flipdiff    --> E2 / E3 ở 512 px, seed 42
#   python models/run_clis.py m_clis --lr-multiplier 2 4 10 20               --> tinh chỉnh LR lớp mới (1e-4 x hệ số)
#   python models/run_clis.py m_clis m_ras --seeds 42 43 44 --init-from-size 512 --lr 5e-5 --set IMAGE_SIZE=1024 BATCH_SIZE=4 GRAD_ACCUM_STEPS=8 EPOCHS=5
#                                                                            --> E3: tinh chỉnh ở 1024 px từ best_model.pt 512 px của cùng biến thể và seed
# Chạy lại đúng lệnh sau khi bị ngắt: việc đã xong được bỏ qua, việc dở chạy tiếp từ checkpoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Hàng đợi huấn luyện của nghiên cứu CLIS-Mamba (giao thức config.CLIS_CONFIG / CLIS_PROTOCOL)")
    parser.add_argument("names", nargs = "+", help = f"Một hoặc nhiều mô hình trong {NAMES}")
    parser.add_argument("--seeds", nargs = "+", type = int, default = [int(config.SEED)])
    parser.add_argument("--fold", type = int, default = 1, help = "Fold dùng làm Validation")
    parser.add_argument("--lr-multiplier", nargs = "+", type = float, default = [None], help = "Hệ số LR lớp mới (mặc định theo CLIS_PROTOCOL)")
    parser.add_argument("--lr", type = float, default = None, help = "LR backbone (mặc định theo CLIS_PROTOCOL); lớp mới = LR x hệ số")
    parser.add_argument("--init-from-size", type = int, default = None, help = "Khởi tạo từ best_model.pt cùng biến thể / seed ở độ phân giải này")
    parser.add_argument("--set", nargs = "+", default = [], metavar = "KEY=VALUE", help = "Ghi đè thêm hằng số config (giá trị Python, ví dụ IMAGE_SIZE=1024)")
    parser.add_argument("--retries", type = int, default = 2)
    parser.add_argument("--retry-wait", type = int, default = 60)
    parser.add_argument("--dry-run", action = "store_true")
    parser.add_argument("--child", nargs = 4, default = None, help = argparse.SUPPRESS)
    args = parser.parse_args()

    unknown = [name for name in args.names if name not in NAMES]
    if unknown:
        parser.error(f"Mô hình không hợp lệ: {unknown}. Chọn trong {NAMES}")
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            overrides[key] = ast.literal_eval(value)                       # Giá trị Python: số, tuple, None, True / False
        except (ValueError, SyntaxError):
            overrides[key] = value                                         # Còn lại là chuỗi (ví dụ IMAGE_VARIANT=aligned)

    if args.child is not None:
        name, seed, fold, multiplier = args.child
        run_child(name, int(seed), int(fold), None if multiplier == "None" else float(multiplier), overrides, lr = args.lr, init_from_size = args.init_from_size)
        sys.exit(0)

    if clis.selective_scan_fn is None and any(clis.VARIANTS.get(name, {}).get("mixer") in clis.ROUTES for name in args.names):
        parser.error("Biến thể SS2D cần kernel selective_scan_cuda của mamba-ssm: chạy trong WSL2 (conda activate mamba-main)")

    jobs = [(name, seed, multiplier) for name in args.names for multiplier in args.lr_multiplier for seed in args.seeds]
    if args.dry_run:
        for index, job in enumerate(jobs, start = 1):
            print(f"{index:3d}. {job[0]} | seed {job[1]} | lr x {job[2] or config.CLIS_PROTOCOL['optimizer_config']['new_module_lr_multiplier']} | fold {args.fold} | {overrides or ''}")
        sys.exit(0)

    keep_system_awake()
    log(f"HÀNG ĐỢI CLIS {len(jobs)} việc | fold {args.fold} | ghi đè {overrides or 'không'}")
    failed = []
    for index, (name, seed, multiplier) in enumerate(jobs, start = 1):
        command = ([sys.executable, str(Path(__file__).resolve()), name, "--child", name, str(seed), str(args.fold), str(multiplier)]
                   + (["--set", *args.set] if args.set else []) + (["--lr", str(args.lr)] if args.lr is not None else [])
                   + (["--init-from-size", str(args.init_from_size)] if args.init_from_size is not None else []))
        for attempt in range(1, args.retries + 2):
            log(f"BẮT ĐẦU {index}/{len(jobs)} {name} | seed {seed} | lr x {multiplier} | lần {attempt}")
            started = time.time()
            code = subprocess.run(command, cwd = ROOT_DIR).returncode
            log(f"KẾT THÚC {index}/{len(jobs)} {name} | mã {code} | {(time.time() - started) / 3600:.1f} giờ")
            if code == 0:
                break
            if attempt <= args.retries:
                time.sleep(args.retry_wait)
        else:
            failed.append((name, seed, multiplier))
    log(f"XONG HÀNG ĐỢI CLIS" + (f" | lỗi: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)
