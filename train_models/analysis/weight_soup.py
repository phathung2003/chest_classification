"""Gộp trọng số (model soup) các seed của một mô hình đã khóa rồi chấm Validation (phương án P1, không mở Test).

Vì sao cần: ensemble 3 seed tăng macro AUROC Validation fold 1 khoảng +0,005 nhưng tốn 3 lần suy luận. Các lượt fine-tune từ cùng trọng số
tiền huấn luyện thường nằm trong cùng một vùng của hàm loss (Neyshabur et al., NeurIPS 2020), nên trung bình trọng số (model soup;
Wortsman et al., ICML 2022) có thể giữ một phần lợi ích đó với chi phí suy luận của một mô hình.

Cách làm (quy tắc ghi trước khi chạy: prereg/analysis_plan.md mục 7, dòng 2026-11-20 01:05, P1):
  - Soup = trung bình cộng đều tham số và buffer số thực của các best_model.pt (trọng số EMA); buffer số nguyên lấy của lượt đầu
  - Mô hình có BatchNorm: tính lại thống kê trên 8.192 ảnh huấn luyện của fold (seed 0), chỉ lớp BatchNorm ở chế độ train
  - Một lượt duy nhất: không gộp, không tính lại BatchNorm, so dự đoán với oof_predictions.csv (kiểm tra đường nạp trọng số)

Ví dụ (WSL2, conda activate mamba-t29):
    python analysis/weight_soup.py --runs "training_results/nih_chest_xray_14/convnextvmamba_cxr_ccmm_512px/run_1" --name check --limit 256
    python analysis/weight_soup.py --runs "training_results/nih_chest_xray_14/convnextvmamba_cxr_ccmm_512px/run_[123]" --name ccmm
Kết quả: training_results/explore/soup/<tên>_soup.csv (cùng định dạng oof_predictions.csv) và <tên>_soup.json
"""
# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"                           # cuBLAS tất định (DETERMINISTIC_ALGORITHMS)
os.environ["TRITON_CACHE_AUTOTUNING"] = "1"                                  # mamba-ssm >= 2.3 ở chế độ tất định: lưu autotune kernel Triton thay vì cảnh báo

# Khai báo thư viện
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from analysis.evaluate_locked import MAX_PROBABILITY_DIFFERENCE, build_model, expand_runs, probability_difference, sha256_file
from training.evaluate import evaluate
from utilities.data import create_dataloader, load_dataset, transform_setting
from utilities.dataset import NIHChestXrayDataset
from utilities.file import load_torch, read_json, save_csv, save_json
from utilities.metric import with_probabilities
from utilities.setting import apply_config, device_info
from utilities.tracking import data_hash

OUTPUT_DIRECTORY = config.MODELS_OUTPUT_DIRECTORY / "explore" / "soup"
BATCHNORM_IMAGES = 8192                                                     # Số ảnh huấn luyện dùng để tính lại thống kê BatchNorm (prereg P1)
BATCHNORM_SEED = 0                                                          # Seed chọn các ảnh đó


# ============================================================
# Nạp thông tin và trọng số tốt nhất (EMA) của một lượt chạy, kiểm tra đúng chữ ký và Fold
# Trả về: (settings, model_state_dict trên CPU)
def load_run(run_directory, fold):
    info = read_json(run_directory / "train_info.json")
    best = load_torch(run_directory / f"fold_{fold}" / "best_model.pt", "cpu")
    if best.get("signature") != info["signature"] or int(best.get("fold")) != int(fold):
        raise SystemExit(f"{run_directory}: best_model.pt không khớp chữ ký / Fold của lượt chạy.")
    return info["settings"], best["model_state_dict"]


# Cấu hình của lượt chạy (độ phân giải, bộ ảnh) và dữ liệu, kiểm tra đúng dữ liệu lúc huấn luyện
# Trả về: (train_df, classes)
def prepare_data(settings, workers = None):
    snapshot = settings.get("config", {})
    apply_config({**config.CLIS_CONFIG, "IMAGE_SIZE": int(settings["image_size"]), "IMAGE_VARIANT": snapshot.get("IMAGE_VARIANT", config.CLIS_CONFIG["IMAGE_VARIANT"]),
                  **({} if workers is None else {"NUMBER_OF_WORKERS": int(workers)})})
    train_df, test_df, _, classes, _ = load_dataset()
    if data_hash(test_df) != settings["test_df_hash"] or data_hash(train_df) != settings["train_df_hash"]:
        raise SystemExit("Dữ liệu khác lúc huấn luyện (hash của train_df / test_df không khớp).")
    return train_df, classes


# ============================================================
# Trung bình cộng đều các state_dict cùng kiến trúc: tensor số thực lấy trung bình, tensor số nguyên (num_batches_tracked) lấy của bản đầu
# Trả về: state_dict
def average_state_dicts(state_dicts):
    first = state_dicts[0]
    if any(state.keys() != first.keys() for state in state_dicts):
        raise ValueError("Các checkpoint khác tập tham số: không gộp được")
    return {key: torch.stack([state[key].float() for state in state_dicts]).mean(0).to(value.dtype) if value.is_floating_point() else value.clone()
            for key, value in first.items()}


# Loader không xáo trộn của một khung dữ liệu, phép biến đổi Validation (cùng cách chấm với evaluate_locked.py)
def eval_loader(settings, frame, classes, device):
    _, eval_transform = transform_setting()
    _, loader = create_dataloader(NIHChestXrayDataset, classes, device, frame, eval_transform, False, batch_size = int(settings["training_config"]["batch_size"]), seed = int(settings["seed"]) + 900_000)
    return loader


# Tính lại thống kê BatchNorm sau khi gộp: chỉ lớp BatchNorm ở chế độ train (trung bình tích lũy, momentum None),
# mọi lớp khác ở chế độ eval nên dropout / drop path không tác động; cùng AMP với lúc chấm
# Trả về: Số lớp BatchNorm đã tính lại (0: mô hình không có BatchNorm, không làm gì)
def recalibrate_batchnorm(model, loader, device):
    layers = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not layers:
        return 0
    model.eval()
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()
    amp_dtype = torch.bfloat16 if config.AMP_DTYPE == "bfloat16" else torch.float16
    with torch.no_grad():
        for images, _ in loader:
            with torch.autocast(device_type = device.type, dtype = amp_dtype, enabled = bool(config.USE_AMP and device.type == "cuda")):
                model(images.to(device, non_blocking = True))
    model.eval()
    return len(layers)


# Xác suất của mô hình trên một khung dữ liệu (cùng hàm chấm với lúc huấn luyện)
# Trả về: Khung dữ liệu kèm cột prob_<lớp>
def predict(model, settings, frame, classes, device):
    _, _, targets, probabilities = evaluate(device, len(classes), model, eval_loader(settings, frame, classes, device))
    if not (targets == frame[classes].to_numpy(dtype = np.float32)).all():
        raise SystemExit("Nhãn của loader khác khung dữ liệu.")
    return with_probabilities(frame, probabilities, classes)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs = "+", required = True, help = "Thư mục lượt chạy của cùng một mô hình (mỗi seed một lượt); nhận mẫu *")
    parser.add_argument("--name", required = True, help = "Tên ngắn của mô hình trong tên tệp kết quả")
    parser.add_argument("--fold", type = int, default = 1)
    parser.add_argument("--limit", type = int, default = None, help = "Chỉ chấm N ảnh Validation đầu (kiểm tra nhanh)")
    parser.add_argument("--workers", type = int, default = None, help = "Số worker của DataLoader (mặc định theo config)")
    args = parser.parse_args()

    # Kiểm tra: các lượt cùng mô hình, cùng tham số mô hình, cùng độ phân giải
    runs = expand_runs(args.runs)
    if not runs:
        parser.error("--runs: không tìm thấy lượt chạy nào")
    loaded = [load_run(run, args.fold) for run in runs]
    settings = loaded[0][0]
    if any(other[key] != settings[key] for other, _ in loaded for key in ("model_name", "model_kwargs", "image_size")):
        raise SystemExit("Các lượt khác mô hình / tham số mô hình / độ phân giải: không gộp được")

    # Dữ liệu và mô hình với trọng số đã gộp
    train_df, classes = prepare_data(settings, args.workers)
    device = device_info()
    model = build_model(settings, len(classes))
    model.load_state_dict(average_state_dicts([state for _, state in loaded]), strict = True)
    model = model.to(device)

    # Tính lại BatchNorm trên ảnh huấn luyện của fold (chỉ khi có gộp)
    recalibrated = 0
    if len(runs) > 1:
        training_frame = train_df[train_df["Fold"] != int(args.fold)]
        sample = training_frame.sample(n = min(BATCHNORM_IMAGES, len(training_frame)), random_state = BATCHNORM_SEED).reset_index(drop = True)
        recalibrated = recalibrate_batchnorm(model, eval_loader(settings, sample, classes, device), device)

    # Chấm Validation
    frame = train_df[train_df["Fold"] == int(args.fold)].reset_index(drop = True)
    frame = frame if args.limit is None else frame.iloc[:int(args.limit)].reset_index(drop = True)
    scored = predict(model, settings, frame, classes, device)

    # Lưu dự đoán và nguồn gốc
    OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok = True)
    output_path = OUTPUT_DIRECTORY / f"{args.name}_soup.csv"
    save_csv(output_path, scored)
    save_json(OUTPUT_DIRECTORY / f"{args.name}_soup.json", {
        "runs": [run.resolve().relative_to(ROOT_DIR).as_posix() for run in runs], "fold": int(args.fold),
        "best_model_sha256": [sha256_file(run / f"fold_{args.fold}" / "best_model.pt") for run in runs],
        "batchnorm_layers_recalibrated": recalibrated, "batchnorm_images": BATCHNORM_IMAGES if recalibrated else 0, "images": len(scored), "limit": args.limit})
    print(f"{args.name}: soup {len(runs)} lượt | BatchNorm tính lại {recalibrated} lớp | {len(scored):,} ảnh Validation | {output_path}", flush = True)

    # Một lượt: so với oof_predictions.csv đã lưu (kiểm tra đường nạp trọng số và chấm)
    if len(runs) == 1:
        difference = probability_difference(pd.read_csv(runs[0] / f"fold_{args.fold}" / "oof_predictions.csv"), scored, classes, config.IMAGE_ID_COLUMN)
        passed = difference <= MAX_PROBABILITY_DIFFERENCE
        print(f"Kiểm tra một lượt: độ lệch xác suất lớn nhất {difference:.2e} | {'KHỚP' if passed else 'LỆCH'}", flush = True)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
