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
import torch

import config
import models.clis_mamba as clis
import models.convnext_tiny as convnext
import models.densenet121 as densenet
from training.experiment import start_train
from training.train import TrainingPaused
from utilities.dataset import NIHChestXrayDataset
from utilities.environment import check_environment
from utilities.file import read_json, save_json, to_snake_case
from utilities.log import timestamp
from utilities.setting import apply_config, keep_system_awake

# ============================================================
# Hàng đợi của nghiên cứu CLIS-Mamba: mọi mô hình dùng chung config.CLIS_CONFIG + config.CLIS_PROTOCOL, fold 1 làm Validation.
#   densenet121, convnext_t : baseline E1
#   m_ras, m_clis, m_rand, m_clis4, gatedconv, attn, flipdiff : các biến thể CLIS-Mamba (E2 / E3)
# Mỗi (mô hình, seed) chạy trong tiến trình con riêng: lỗi CUDA / hết bộ nhớ chỉ làm hỏng việc đó; lỗi --> chờ rồi chạy lại,
# lượt chạy tự tiếp tục từ last_checkpoint.pt. Biến thể SS2D cần kernel mamba-ssm: chạy trong WSL2 (conda activate mamba-t29: torch 2.9.1 + mamba-ssm 2.3.2.post1;
# env mamba-main: torch 2.4.0 + mamba-ssm 2.2.2 của các lượt tinh chỉnh đầu, cần --set ADAM_FUSED=False).
BASELINES = {
    "densenet121"   : (densenet.MODEL_NAME, densenet.DenseNetBaseline, {}),
    "convnext_t"    : (convnext.MODEL_NAME, convnext.ConvNeXtTinyBaseline, {}),
}
NAMES = list(BASELINES) + list(clis.VARIANTS)
QUEUE_LOG = config.MODELS_OUTPUT_DIRECTORY / "clis_queue.log"
TUNING_DIRECTORY = config.MODELS_OUTPUT_DIRECTORY / "clis_tuning"                 # Điểm từng mức LR và kết quả loại dần (--halving)
PAUSED_EXIT = 75                                                                   # Mã thoát của tiến trình con tạm dừng có chủ đích (PAUSE_AFTER_EPOCH)


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
        protocol["optimizer_config"]["learning_rate"] = float(lr)
    if name in BASELINES:
        if init_from_size is not None:
            raise SystemExit("--init-from-size chỉ dùng cho các biến thể CLIS-Mamba")
        base_name, factory, kwargs = BASELINES[name]
        model_name = f"{base_name} CLIS {config.IMAGE_SIZE}px"
    else:
        model_name, factory, kwargs = clis.model_name(name), clis.CLISMamba, dict(clis.VARIANTS[name])
        kwargs["grad_checkpoint"] = bool(config.BLOCK_GRAD_CHECKPOINT)            # Checkpointing 4 khối MetaFormer: cùng phép tính, chỉ đổi VRAM / tốc độ
        if init_from_size is not None:
            kwargs["init_from"] = str(find_checkpoint(name, seed, fold, init_from_size))
            print(f"Khởi tạo từ {kwargs['init_from']}")

    # Lượt tinh chỉnh trên một phần bệnh nhân: thư mục kết quả riêng (find_checkpoint không lấy nhầm làm điểm khởi đầu)
    if float(config.TRAIN_PATIENT_FRACTION) < 1:
        model_name = f"{model_name} tune{round(float(config.TRAIN_PATIENT_FRACTION) * 100)}"
    return start_train(model_name = model_name, dataset_class = NIHChestXrayDataset, model_factory = factory, model_kwargs = kwargs, **protocol)


# ============================================================
# Loại dần mức LR (--halving): giữ `keep` hệ số có điểm Validation tốt nhất tới lúc tạm dừng (hòa điểm thì giữ thứ tự đã khai báo)
def select_survivors(results, keep):
    higher_is_better = next(iter(results.values()))["higher_is_better"]
    return sorted(results, key = lambda multiplier: results[multiplier]["best_score"], reverse = higher_is_better)[:keep]


# Thư mục kết quả của một mô hình, cùng cách đặt tên với run_child (IMAGE_SIZE và TRAIN_PATIENT_FRACTION sau ghi đè)
def model_directory(name, overrides):
    size = int(overrides.get("IMAGE_SIZE", config.CLIS_CONFIG.get("IMAGE_SIZE", config.IMAGE_SIZE)))
    model_name = f"{BASELINES[name][0]} CLIS {size}px" if name in BASELINES else clis.model_name(name, size)
    fraction = float(overrides.get("TRAIN_PATIENT_FRACTION", config.CLIS_CONFIG.get("TRAIN_PATIENT_FRACTION", config.TRAIN_PATIENT_FRACTION)))
    if fraction < 1:
        model_name = f"{model_name} tune{round(fraction * 100)}"
    dataset_name = read_json(config.DATASET_INFO_DIRECTORY)["dataset_name"]
    return config.MODELS_OUTPUT_DIRECTORY / to_snake_case(dataset_name) / to_snake_case(model_name)


# Thư mục fold của lượt chạy cùng (mô hình, seed, hệ số LR, ghi đè); None nếu chưa chạy
# - torch_versions: Chỉ xét lượt chạy của các bản torch này (None: mọi bản). Bản torch nằm trong chữ ký, nên lượt dở của bản khác không chạy tiếp được
# - Nhiều lượt khớp (ví dụ lượt dở của bản torch cũ và lượt mới): Ưu tiên lượt đã xong fold, rồi lượt của bản torch hiện tại, rồi lượt mới nhất
def find_fold_directory(name, seed, multiplier, fold, overrides, torch_versions = None):
    candidates = []
    for info_path in sorted(model_directory(name, overrides).glob("run_*/train_info.json")):
        settings = read_json(info_path).get("settings", {})
        snapshot = settings.get("config", {})
        if (settings.get("seed") == int(seed)
                and settings.get("training_config", {}).get("optimizer", {}).get("new_module_lr_multiplier") == float(multiplier)
                and all(snapshot.get(key) == value for key, value in overrides.items() if key in snapshot)
                and (torch_versions is None or settings.get("torch_version") in torch_versions)):
            fold_directory = info_path.parent / f"fold_{fold}"
            candidates.append(((fold_directory / "fold_done.json").is_file(), settings.get("torch_version") == torch.__version__, info_path.stat().st_mtime, fold_directory))
    return max(candidates)[3] if candidates else None


# Bản torch của lượt chạy chứa thư mục fold (ghi trong train_info.json)
def run_torch_version(fold_directory):
    return read_json(fold_directory.parent / "train_info.json").get("settings", {}).get("torch_version")


# Điểm Validation tốt nhất trong các epoch <= last_epoch (None: mọi epoch) đọc từ history.csv của fold.
# Trả về None khi chưa đủ last_epoch epoch mà fold chưa xong (cần chạy tiếp)
def history_score(fold_directory, selection_metric, last_epoch = None):
    path = fold_directory / "history.csv"
    if not path.is_file():
        return None
    history = pd.read_csv(path)
    finished = (fold_directory / "fold_done.json").is_file()
    if last_epoch is not None:
        if history["epoch"].max() < last_epoch and not finished:
            return None
        history = history[history["epoch"] <= last_epoch]
    higher_is_better = selection_metric == "macro_auroc"
    best = history[selection_metric].idxmax() if higher_is_better else history[selection_metric].idxmin()
    return {"best_score": float(history.loc[best, selection_metric]), "best_epoch": int(history.loc[best, "epoch"]),
            "higher_is_better": higher_is_better, "finished": finished}


# Tiến trình con ghi điểm tốt nhất tới lúc tạm dừng (hoặc khi xong fold); giữ cho tương thích, tiến trình cha đọc history.csv
def save_rung_result(path, **values):
    path.parent.mkdir(parents = True, exist_ok = True)
    save_json(path, values)


# ============================================================
# Chạy thí nghiệm. Ví dụ:
#   python models/run_clis.py densenet121 convnext_t --seeds 42 43 44        --> E1: baseline, 3 seed
#   python models/run_clis.py m_ras m_clis m_rand gatedconv attn flipdiff    --> E2 / E3 ở 512 px, seed 42
#   python models/run_clis.py m_clis --lr-multiplier 5 2 10 20 --halving 4 2 --set TRAIN_PATIENT_FRACTION=0.3
#                                                                            --> tinh chỉnh LR lớp mới (1e-4 x hệ số) trên 30% bệnh nhân: cả 4 hệ số
#                                                                                chạy 4 epoch, giữ 2 hệ số có Validation tốt nhất chạy tiếp tới hết.
#                                                                                Thêm hệ số vào lệnh rồi chạy lại: hệ số đã có điểm không chạy lại
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
    parser.add_argument("--halving", nargs = 2, type = int, default = None, metavar = ("EPOCHS", "KEEP"),
                        help = "Loại dần mức LR: mọi hệ số chạy EPOCHS epoch rồi tạm dừng, KEEP hệ số có điểm Validation tốt nhất chạy tiếp tới hết")
    parser.add_argument("--reuse-other-torch", action = "store_true",
                        help = "--halving: Dùng lại điểm của lượt chạy trên bản torch khác (mặc định chỉ dùng lượt của bản torch hiện tại; lượt dở của bản khác luôn chạy lại từ đầu)")
    parser.add_argument("--init-from-size", type = int, default = None, help = "Khởi tạo từ best_model.pt cùng biến thể / seed ở độ phân giải này")
    parser.add_argument("--set", nargs = "+", default = [], metavar = "KEY=VALUE", help = "Ghi đè thêm hằng số config (giá trị Python, ví dụ IMAGE_SIZE=1024)")
    parser.add_argument("--retries", type = int, default = 2)
    parser.add_argument("--retry-wait", type = int, default = 60)
    parser.add_argument("--dry-run", action = "store_true")
    parser.add_argument("--child", nargs = 4, default = None, help = argparse.SUPPRESS)
    parser.add_argument("--rung-result", default = None, help = argparse.SUPPRESS)
    parser.add_argument("--check-only", action = "store_true", help = argparse.SUPPRESS)
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
        try:
            frame = run_child(name, int(seed), int(fold), None if multiplier == "None" else float(multiplier), overrides, lr = args.lr, init_from_size = args.init_from_size)
        except TrainingPaused as paused:
            if args.rung_result:
                save_rung_result(Path(args.rung_result), paused = True, epoch = paused.epoch, best_score = paused.best_score, best_epoch = paused.best_epoch,
                                 selection_metric = paused.selection_metric, higher_is_better = paused.higher_is_better)
            sys.exit(PAUSED_EXIT)
        if args.rung_result:
            row = frame.loc[frame["fold"].astype(str) == str(fold)].iloc[0]
            save_rung_result(Path(args.rung_result), paused = False, epoch = int(row["epochs_trained"]), best_score = float(row["best_selection_score"]),
                             best_epoch = int(row["best_epoch"]), selection_metric = str(row["best_selection_metric"]),
                             higher_is_better = str(row["best_selection_metric"]) == "macro_auroc")
        sys.exit(0)

    # Tiến trình con kiểm tra môi trường: phiên bản thư viện, GPU, kernel selective scan cho biến thể SS2D
    if args.check_only:
        settings = {**config.CLIS_CONFIG, **overrides}
        try:
            check_environment(require_kernel = any(clis.VARIANTS.get(name, {}).get("mixer") in clis.ROUTES for name in args.names),
                              amp_dtype = settings.get("AMP_DTYPE", config.AMP_DTYPE), deterministic = settings.get("DETERMINISTIC_ALGORITHMS", config.DETERMINISTIC_ALGORITHMS),
                              adam_fused = settings.get("ADAM_FUSED", config.ADAM_FUSED))
        except RuntimeError as error:
            print(error, flush = True)
            sys.exit(1)
        sys.exit(0)

    # Kiểm tra môi trường một lần trước hàng đợi, trong tiến trình con (tiến trình cha không khởi tạo CUDA nên không giữ VRAM suốt hàng đợi);
    # kết quả ghi vào log hàng đợi. Trả về True nếu đạt
    def environment_ok():
        command = [sys.executable, str(Path(__file__).resolve()), *args.names, "--check-only"] + (["--set", *args.set] if args.set else [])
        result = subprocess.run(command, cwd = ROOT_DIR, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                                env = {**os.environ, "PYTHONIOENCODING": "utf-8"})
        for line in result.stdout.splitlines():
            if line.strip():
                log(line)
        if result.returncode != 0 and result.stderr.strip():
            print(result.stderr.strip()[-3000:], flush = True)
        return result.returncode == 0

    if not args.dry_run and not environment_ok():
        sys.exit(1)
    if args.halving is not None:
        rung_epochs, keep = args.halving
        if None in args.lr_multiplier or len(args.lr_multiplier) <= keep or keep < 1 or rung_epochs < 1:
            parser.error("--halving EPOCHS KEEP cần --lr-multiplier với nhiều hơn KEEP hệ số, KEEP >= 1, EPOCHS >= 1")
        if "PAUSE_AFTER_EPOCH" in overrides:
            parser.error("--halving tự đặt PAUSE_AFTER_EPOCH, không ghi đè bằng --set")

    # Lệnh của một tiến trình con (extra_set: ghi đè thêm, gộp vào cùng một --set vì --set chỉ nhận một lần)
    def child_command(name, seed, multiplier, extra_set = ()):
        settings = [*args.set, *extra_set]
        return ([sys.executable, str(Path(__file__).resolve()), name, "--child", name, str(seed), str(args.fold), str(multiplier)]
                + (["--set", *settings] if settings else []) + (["--lr", str(args.lr)] if args.lr is not None else [])
                + (["--init-from-size", str(args.init_from_size)] if args.init_from_size is not None else []))

    # Chạy một việc, lỗi thì chờ rồi chạy lại (ok_codes: mã thoát được coi là xong). Trả về mã thoát cuối cùng
    def run_job(label, command, ok_codes = (0,)):
        for attempt in range(1, args.retries + 2):
            log(f"BẮT ĐẦU {label} | lần {attempt}")
            started = time.time()
            code = subprocess.run(command, cwd = ROOT_DIR).returncode
            log(f"KẾT THÚC {label} | mã {code} | {(time.time() - started) / 3600:.1f} giờ")
            if code in ok_codes:
                return code
            if attempt <= args.retries:
                time.sleep(args.retry_wait)
        return code

    # Hàng đợi thường: mọi (mô hình, hệ số LR, seed) chạy tới hết
    if args.halving is None:
        jobs = [(name, seed, multiplier) for name in args.names for multiplier in args.lr_multiplier for seed in args.seeds]
        if args.dry_run:
            for index, job in enumerate(jobs, start = 1):
                print(f"{index:3d}. {job[0]} | seed {job[1]} | lr x {job[2] or config.CLIS_PROTOCOL['optimizer_config']['new_module_lr_multiplier']} | fold {args.fold} | {overrides or ''}")
            sys.exit(0)

        keep_system_awake()
        log(f"HÀNG ĐỢI CLIS {len(jobs)} việc | fold {args.fold} | ghi đè {overrides or 'không'}")
        failed = []
        for index, (name, seed, multiplier) in enumerate(jobs, start = 1):
            if run_job(f"{index}/{len(jobs)} {name} | seed {seed} | lr x {multiplier}", child_command(name, seed, multiplier)) != 0:
                failed.append((name, seed, multiplier))
        log("XONG HÀNG ĐỢI CLIS" + (f" | lỗi: {failed}" if failed else ""))
        sys.exit(1 if failed else 0)

    # Hàng đợi loại dần: với mỗi (mô hình, seed), mọi hệ số chạy rung_epochs epoch rồi tạm dừng (PAUSE_AFTER_EPOCH, không vào chữ ký);
    # keep hệ số có điểm Validation tốt nhất tới epoch rung_epochs chạy tiếp từ checkpoint tới hết (cùng chữ ký, cùng lịch LR theo EPOCHS).
    # Điểm đọc từ history.csv: vòng 1 chỉ xét epoch <= rung_epochs (cả khi lượt chạy đã đi xa hơn), hệ số đã có đủ epoch thì không chạy lại
    selection_metric = config.CLIS_PROTOCOL.get("selection_metric", config.SELECTION_METRIC)
    groups = [(name, seed) for name in args.names for seed in args.seeds]
    if args.dry_run:
        for name, seed in groups:
            print(f"{name} | seed {seed} | vòng 1: lr x {[f'{m:g}' for m in args.lr_multiplier]} tới epoch {rung_epochs} | vòng 2: {keep} hệ số tốt nhất chạy tới hết | fold {args.fold} | {overrides or ''}")
        sys.exit(0)

    keep_system_awake()
    torch_versions = None if args.reuse_other_torch else {torch.__version__}       # Lượt chạy được dùng điểm
    log(f"HÀNG ĐỢI CLIS LOẠI DẦN {len(groups)} nhóm x {len(args.lr_multiplier)} hệ số | {rung_epochs} epoch, giữ {keep} | fold {args.fold} | ghi đè {overrides or 'không'} | "
        f"torch {torch.__version__}" + (" (dùng lại điểm của bản torch khác)" if args.reuse_other_torch else ""))

    # Điểm kèm bản torch nếu khác bản hiện tại (để log ghi rõ nguồn điểm)
    def describe(multiplier, scores):
        version = scores[multiplier]["torch_version"]
        return f"lr x {multiplier:g}: {scores[multiplier]['best_score']:.4f}" + ("" if version == torch.__version__ else f" (torch {version})")

    failed = []
    for group_index, (name, seed) in enumerate(groups, start = 1):
        results = {}

        # Vòng 1: mọi hệ số tới epoch rung_epochs (đã có đủ epoch thì dùng luôn điểm trong history.csv)
        for multiplier in args.lr_multiplier:
            label = f"nhóm {group_index}/{len(groups)} vòng 1 {name} | seed {seed} | lr x {multiplier:g}"
            fold_directory = find_fold_directory(name, seed, multiplier, args.fold, overrides, torch_versions)
            score = None if fold_directory is None else history_score(fold_directory, selection_metric, rung_epochs)
            if score is None:
                if run_job(label, child_command(name, seed, multiplier, [f"PAUSE_AFTER_EPOCH={rung_epochs}"]), ok_codes = (0, PAUSED_EXIT)) not in (0, PAUSED_EXIT):
                    failed.append((name, seed, multiplier, "vòng 1"))
                    continue
                fold_directory = find_fold_directory(name, seed, multiplier, args.fold, overrides, torch_versions)
                score = None if fold_directory is None else history_score(fold_directory, selection_metric, rung_epochs)
                if score is None:
                    failed.append((name, seed, multiplier, "vòng 1: không đọc được history.csv"))
                    continue
            results[multiplier] = {**score, "torch_version": run_torch_version(fold_directory)}
        if not results:
            continue
        survivors = select_survivors(results, keep)
        log(f"XẾP HẠNG {name} | seed {seed} | tới epoch {rung_epochs}: " + ", ".join(describe(m, results) for m in results) + f" | giữ {[f'{m:g}' for m in survivors]}")

        # Vòng 2: hệ số được giữ chạy tiếp từ checkpoint tới hết (đã xong fold thì không chạy lại).
        # Tìm lại thư mục sau khi chạy: lượt dở của bản torch khác không chạy tiếp được, tiến trình con tạo lượt mới
        final = {}
        for multiplier in survivors:
            label = f"nhóm {group_index}/{len(groups)} vòng 2 {name} | seed {seed} | lr x {multiplier:g}"
            fold_directory = find_fold_directory(name, seed, multiplier, args.fold, overrides, torch_versions)
            if fold_directory is None or not (fold_directory / "fold_done.json").is_file():
                if run_job(label, child_command(name, seed, multiplier)) != 0:
                    failed.append((name, seed, multiplier, "vòng 2"))
                    continue
                fold_directory = find_fold_directory(name, seed, multiplier, args.fold, overrides, torch_versions)
            if fold_directory is None or not (fold_directory / "fold_done.json").is_file():
                failed.append((name, seed, multiplier, "vòng 2: chưa xong fold"))
                continue
            final[multiplier] = {**history_score(fold_directory, selection_metric), "torch_version": run_torch_version(fold_directory)}

        # Tổng kết của nhóm
        best = select_survivors(final, 1)[0] if final else None
        summary_path = TUNING_DIRECTORY / f"{to_snake_case(' '.join([name, f'seed {seed}', 'halving', *args.set]))}.json"
        summary_path.parent.mkdir(parents = True, exist_ok = True)
        save_json(summary_path, {"model": name, "seed": seed, "fold": args.fold, "overrides": args.set, "rung_epochs": rung_epochs, "keep": keep,
                                 "multipliers": [f"{m:g}" for m in args.lr_multiplier],
                                 "rung_scores": {f"{m:g}": results[m]["best_score"] for m in results}, "survivors": [f"{m:g}" for m in survivors],
                                 "final_scores": {f"{m:g}": final[m]["best_score"] for m in final}, "best_multiplier": None if best is None else f"{best:g}",
                                 "torch_versions": {f"{m:g}": (final.get(m) or results[m])["torch_version"] for m in results}})
        if best is None:
            log(f"CHỌN {name} | seed {seed} | không có hệ số nào chạy xong vòng 2")
        else:
            log(f"CHỌN {name} | seed {seed} | lr x {best:g} | điểm cuối " + ", ".join(describe(m, final) for m in final))
    log("XONG HÀNG ĐỢI CLIS LOẠI DẦN" + (f" | lỗi: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)
