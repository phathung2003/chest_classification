"""Tự chưng cất từ ensemble 3 seed của chính kiến trúc (phương án P2, prereg/analysis_plan.md mục 7, dòng 2026-11-20 01:05).

Vì sao file riêng: chữ ký thí nghiệm băm tệp mã nguồn của lớp mô hình và mọi hằng số config.py (xem models/run_in22k.py). Sửa training/loss.py
hay config.py sẽ đổi chữ ký mọi lượt cũ. File này định nghĩa lớp con (chỉ nhận thêm SHA-256 tệp teacher và λ để ghi vào chữ ký) và hàm Loss
bọc ngoài, rồi gọi đúng run_child của models/run_stitch.py (ccmm) hay models/run_clis.py (ConvNeXt-T) trong cùng tiến trình:
công thức huấn luyện giữ nguyên, chỉ đổi nhãn huấn luyện.

Cách làm:
  1. teacher : trung bình xác suất của các lượt (3 seed, bảng 8.1) trên ảnh huấn luyện của fold, phép biến đổi Validation
               --> training_results/explore/kd/teacher_<tên>_fold<k>.csv
  2. train   : nhãn huấn luyện y' = (1 - λ) y + λ p_teacher, λ = 0,5; Loss Validation dùng nhãn cứng; hậu tố thư mục _kd
  3. train --dry-run <lượt gốc>: không huấn luyện, so cấu hình của học trò với lượt gốc cùng seed (chỉ được khác ở phần chưng cất)

Ví dụ (WSL2, conda activate mamba-t29):
    python models/run_kd.py teacher ccmm --runs "training_results/nih_chest_xray_14/convnextvmamba_cxr_ccmm_512px/run_[123]"
    python models/run_kd.py train ccmm --seed 42 --dry-run training_results/nih_chest_xray_14/convnextvmamba_cxr_ccmm_512px/run_1
    python models/run_kd.py train ccmm --seed 42
"""
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
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
import models.convnext_tiny as convnext
import models.run_clis as clis_runner
import models.run_stitch as stitch_runner
import models.stitch_cxr as stx
import training.train as train_module
from analysis.evaluate_locked import build_model, expand_runs, sha256_file
from analysis.weight_soup import load_run, predict, prepare_data
from training.experiment import start_train
from training.loss import validate_loss_config
from training.optimizer import build_training_config
from utilities.file import json_clean, read_json, save_csv
from utilities.setting import device_info
from utilities.tracking import config_snapshot

RUN_TAG = "kd"                                                                     # Hậu tố tên mô hình / thư mục kết quả
DISTILL_LAMBDA = 0.5                                                               # λ cố định trong prereg, không dò
TEACHER_DIRECTORY = config.MODELS_OUTPUT_DIRECTORY / "explore" / "kd"
NAMES = ("ccmm", "convnext_t")


# Tệp xác suất teacher của một kiến trúc trên ảnh huấn luyện của fold
# Trả về: Path
def teacher_path(name, fold):
    return TEACHER_DIRECTORY / f"teacher_{name}_fold{fold}.csv"


# ============================================================
# Lớp con: kiến trúc, tham số và trọng số khởi tạo như lớp gốc; chỉ nhận thêm SHA-256 tệp teacher và λ để chúng (và file này) vào chữ ký
class StitchCXRKD(stx.StitchCXR):
    def __init__(self, num_classes, teacher_sha256 = None, distill_lambda = None, **kwargs):
        super().__init__(num_classes, **kwargs)


class ConvNeXtTinyKD(convnext.ConvNeXtTinyBaseline):
    def __init__(self, num_classes, teacher_sha256 = None, distill_lambda = None, **kwargs):
        super().__init__(num_classes, **kwargs)


STUDENTS = {stx.StitchCXR: StitchCXRKD, convnext.ConvNeXtTinyBaseline: ConvNeXtTinyKD}   # Lớp gốc mà run_child truyền --> lớp học trò


# ============================================================
# Hàm Loss chưng cất: Loss gốc của lượt cũ với nhãn mềm y' = (1 - λ) y + λ p_teacher
# - Gọi kèm chỉ số ảnh (huấn luyện; train_one_epoch truyền như với NVUMLoss): nhãn mềm
# - Gọi không kèm (Validation, Test): nhãn cứng --> Loss Validation so được với lượt không chưng cất
class DistillLoss(nn.Module):

    needs_indices = True                                                           # train_one_epoch truyền chỉ số ảnh của batch

    def __init__(self, base, teacher, distill_lambda):
        super().__init__()
        self.base = base
        self.distill_lambda = float(distill_lambda)
        self.register_buffer("teacher", torch.as_tensor(teacher, dtype = torch.float32), persistent = False)   # Không lưu vào checkpoint: nạp lại từ tệp đã ghim SHA-256

    def forward(self, logits, targets, indices = None):
        if indices is None:
            return self.base(logits, targets)
        return self.base(logits, (1.0 - self.distill_lambda) * targets + self.distill_lambda * self.teacher[indices])


# Thay build_loss của training/train.py trong tiến trình này: Loss gốc như lượt cũ, bọc nhãn mềm của teacher theo đúng thứ tự train_df của fold
def distill_build_loss(path, distill_lambda):
    original = train_module.build_loss

    def build(loss_config, train_df, classes, device):
        criterion, metadata = original(loss_config, train_df, classes, device)
        teacher = pd.read_csv(path).set_index(config.IMAGE_ID_COLUMN)
        image_ids = train_df[config.IMAGE_ID_COLUMN]
        missing = int((~image_ids.isin(teacher.index)).sum())
        if missing:
            raise SystemExit(f"{path.name} thiếu {missing} ảnh huấn luyện của fold")
        probabilities = teacher.loc[image_ids, [f"prob_{name}" for name in classes]].to_numpy(dtype = np.float32)
        return DistillLoss(criterion, probabilities, distill_lambda).to(device), {**metadata, "distill_lambda": float(distill_lambda), "teacher_file": path.name}

    return build


# ============================================================
# Kiểm tra không huấn luyện: cấu hình học trò phải trùng lượt gốc cùng seed, trừ phần chưng cất (lớp mô hình, 2 khóa model_kwargs, RUN_TAG)
# Trả về: Danh sách mục khác nhau (rỗng = KHỚP); hằng số config khác nhau ghi kèm giá trị hiện tại / của lượt gốc
def differences_from_reference(reference, model_kwargs, protocol):
    settings = read_json(Path(reference) / "train_info.json")["settings"]
    normalize = lambda value: json.loads(json.dumps(json_clean(value)))
    current = {"model_kwargs": model_kwargs, "loss_config": validate_loss_config(protocol["loss_config"]),
               "training_config": build_training_config(protocol.get("optimizer_config"), protocol.get("scheduler_config"), protocol.get("selection_metric")),
               "seed": int(config.SEED), "image_size": int(config.IMAGE_SIZE)}
    differences = [key for key, value in current.items() if normalize(value) != settings[key]]
    snapshot, saved = normalize(config_snapshot()), settings["config"]
    differences += [f"config.{key} (hiện tại {snapshot.get(key, '<không có>')!r}, lượt gốc {saved.get(key, '<không có>')!r})"
                    for key in sorted(set(snapshot) | set(saved)) if key != "RUN_TAG" and snapshot.get(key, "<không có>") != saved.get(key, "<không có>")]
    return differences


# Thay start_train mà run_child của run_stitch / run_clis gọi: lớp gốc --> lớp học trò, SHA-256 tệp teacher và λ vào model_kwargs
# reference khác None: chỉ so cấu hình với lượt gốc rồi thoát (mã 0 khi KHỚP)
def student_start_train(path, distill_lambda, reference = None):

    def start(model_name, dataset_class, model_factory, model_kwargs = None, **protocol):
        if reference is not None:
            differences = differences_from_reference(reference, dict(model_kwargs or {}), protocol)
            print(f"Dry run {model_name} so với {reference}: {'KHỚP' if not differences else 'LỆCH ở ' + ', '.join(differences)}", flush = True)
            sys.exit(1 if differences else 0)
        kwargs = {**(model_kwargs or {}), "teacher_sha256": sha256_file(path), "distill_lambda": float(distill_lambda)}
        return start_train(model_name = model_name, dataset_class = dataset_class, model_factory = STUDENTS[model_factory], model_kwargs = kwargs, **protocol)

    return start


# ============================================================
# Teacher: trung bình xác suất của các lượt cùng kiến trúc trên ảnh huấn luyện của fold (không có ảnh Validation của fold)
def make_teacher(name, patterns, fold, workers):
    runs = expand_runs(patterns)
    if not runs:
        raise SystemExit("--runs: không tìm thấy lượt chạy nào")
    loaded = [load_run(run, fold) for run in runs]
    settings = loaded[0][0]
    if any(other["model_name"] != settings["model_name"] or other["model_kwargs"] != settings["model_kwargs"] for other, _ in loaded):
        raise SystemExit("Các lượt teacher khác mô hình / tham số mô hình")
    expected = stx.model_name("ccmm", settings["image_size"]) if name == "ccmm" else f"{convnext.MODEL_NAME} CLIS {settings['image_size']}px"
    if settings["model_name"] != expected:
        raise SystemExit(f"Teacher của {name} phải là lượt '{expected}', nhận '{settings['model_name']}'")

    # Suy luận từng lượt trên ảnh huấn luyện của fold, cộng dồn xác suất
    train_df, classes = prepare_data(settings, workers)
    frame = train_df[train_df["Fold"] != int(fold)].reset_index(drop = True)
    columns = [f"prob_{label}" for label in classes]
    device = device_info()
    total = np.zeros((len(frame), len(classes)), dtype = np.float64)
    for run, (run_settings, state) in zip(runs, loaded):
        model = build_model(run_settings, len(classes))
        model.load_state_dict(state, strict = True)
        total += predict(model.to(device), run_settings, frame, classes, device)[columns].to_numpy(dtype = np.float64)
        print(f"Teacher {name}: xong {run}", flush = True)
        del model
        torch.cuda.empty_cache()

    # Lưu trung bình xác suất
    teacher = frame[[config.IMAGE_ID_COLUMN]].copy()
    teacher[columns] = total / len(runs)
    path = teacher_path(name, fold)
    path.parent.mkdir(parents = True, exist_ok = True)
    save_csv(path, teacher)
    print(f"Teacher {name}: {len(runs)} lượt, {len(teacher):,} ảnh huấn luyện fold {fold} | {path} | SHA-256 {sha256_file(path)}", flush = True)


# Học trò: đúng run_child của lượt gốc, chỉ thay start_train (lớp, model_kwargs) và build_loss (nhãn mềm)
def train(name, seed, fold, reference = None):
    path = teacher_path(name, fold)
    if reference is None:                                                          # Dry run chỉ so cấu hình: chưa cần tệp teacher
        if not path.is_file():
            raise SystemExit(f"Thiếu {path}: chạy python models/run_kd.py teacher {name} --runs ... trước")
        train_module.build_loss = distill_build_loss(path, DISTILL_LAMBDA)
    start = student_start_train(path, DISTILL_LAMBDA, reference)
    overrides = {"RUN_TAG": RUN_TAG}
    if name == "ccmm":
        stitch_runner.start_train = start
        return stitch_runner.run_child("ccmm", seed, fold, overrides)
    clis_runner.start_train = start
    return clis_runner.run_child("convnext_t", seed, fold, None, overrides)


def main():
    parser = argparse.ArgumentParser(description = "Tự chưng cất từ ensemble 3 seed (prereg mục 7, P2)")
    commands = parser.add_subparsers(dest = "command", required = True)
    teacher = commands.add_parser("teacher", help = "Tính xác suất teacher trên ảnh huấn luyện của fold")
    teacher.add_argument("name", choices = NAMES)
    teacher.add_argument("--runs", nargs = "+", required = True, help = "Thư mục lượt chạy teacher (3 seed, bảng 8.1); nhận mẫu *")
    teacher.add_argument("--fold", type = int, default = 1)
    teacher.add_argument("--workers", type = int, default = None)
    student = commands.add_parser("train", help = "Huấn luyện học trò")
    student.add_argument("name", choices = NAMES)
    student.add_argument("--seed", type = int, required = True)
    student.add_argument("--fold", type = int, default = 1)
    student.add_argument("--dry-run", metavar = "REFERENCE_RUN", default = None, help = "Không huấn luyện: so cấu hình với lượt gốc cùng seed")
    args = parser.parse_args()
    if args.command == "teacher":
        make_teacher(args.name, args.runs, args.fold, args.workers)
    else:
        train(args.name, args.seed, args.fold, args.dry_run)


if __name__ == "__main__":
    main()
