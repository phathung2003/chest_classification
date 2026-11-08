"""Đo mức tham gia của khối chèn CLIS-Mamba v2 trên Validation (không mở Test).

Vì sao cần: lượt m_clis v2 seed 42 (2026-11-08) trùng ConvNeXt-T seed 42 từng epoch, nên điểm AUROC một mình không cho biết khối chèn có được dùng hay không.
Script chạy best_model.pt hai lần trên cùng ảnh Validation, chung đoạn stem --> stage 3:
  - bật : stage 3 --> khối chèn --> stage 4 --> đầu phân loại (đúng mô hình đã huấn luyện)
  - tắt : stage 3 --> stage 4 --> đầu phân loại (bỏ khối chèn, mọi trọng số khác giữ nguyên)
và báo:
  - ratio       : ||khối(x) - x|| / ||x|| trên bản đồ đặc trưng sau stage 3, trung bình theo ảnh (0 = khối là phép đồng nhất)
  - auroc_on/off: macro AUROC 14 nhãn khi bật / tắt khối chèn; drop = on - off (khối được dùng --> bỏ đi làm giảm AUROC)
  - prob_change : trung bình |xác suất bật - xác suất tắt|
  - saved_diff  : độ lệch lớn nhất giữa xác suất "bật" và oof_predictions.csv đã lưu (kiểm tra nạp đúng trọng số; chỉ còn sai số làm tròn BF16)

Ví dụ (WSL2, conda activate mamba-t29):
    python analysis/block_engagement.py --runs "training_results/nih_chest_xray_14/clismamba_v2_m_clis_512px*/run_*"
Kết quả ghi thêm vào training_results/block_engagement.csv (một dòng mỗi lượt chạy, đo lại thì thay dòng cũ).
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
import glob

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import config
import models.clis_mamba_v2 as v2
from utilities.data import create_dataloader, load_dataset, transform_setting
from utilities.dataset import NIHChestXrayDataset
from utilities.file import load_torch, read_json
from utilities.log import timestamp
from utilities.setting import apply_config

OUTPUT_PATH = config.MODELS_OUTPUT_DIRECTORY / "block_engagement.csv"


# ============================================================
# Mở rộng danh sách thư mục lượt chạy (mẫu * do script tự mở rộng)
# Trả về: Danh sách đường dẫn
def expand_runs(patterns):
    return [Path(path).resolve() for pattern in patterns for path in (sorted(glob.glob(str(pattern))) if glob.has_magic(str(pattern)) else [pattern])]


# ============================================================
# Macro AUROC trên các nhãn có cả mẫu dương và âm
# Trả về: float
def macro_auroc(targets, probabilities):
    values = [roc_auc_score(targets[:, k], probabilities[:, k]) for k in range(targets.shape[1]) if 0 < targets[:, k].sum() < len(targets)]
    return float(np.mean(values))


# ============================================================
# Đo một lượt chạy v2 trên Validation của fold
# Trả về: dict một dòng kết quả; None nếu lượt chạy không có khối chèn
def measure(run_directory, fold, limit, workers, device):
    info = read_json(run_directory / "train_info.json")
    settings = info["settings"]
    kwargs = {key: value for key, value in settings["model_kwargs"].items() if key not in ("init_from", "freeze_backbone")}
    if kwargs.get("mixer", "convnext_t") == "convnext_t":
        return None
    kwargs.setdefault("residual_init", "layerscale")                         # Lượt LayerScale cũ (trước 2026-11-07 21:30) chưa ghi khóa này
    fold_directory = run_directory / f"fold_{fold}"

    # Cấu hình của lượt chạy: độ phân giải và bộ ảnh
    snapshot = settings.get("config", {})
    apply_config({**config.CLIS_CONFIG, "IMAGE_SIZE": int(settings["image_size"]), "IMAGE_VARIANT": snapshot.get("IMAGE_VARIANT", config.CLIS_CONFIG["IMAGE_VARIANT"]),
                  **({} if workers is None else {"NUMBER_OF_WORKERS": int(workers)})})

    # Ảnh Validation của fold
    train_df, _, _, classes, _ = load_dataset()
    frame = train_df[train_df["Fold"] == int(fold)].reset_index(drop = True)
    frame = frame if limit is None else frame.iloc[:int(limit)].reset_index(drop = True)

    # Nạp trọng số tốt nhất (đúng chữ ký, đúng Fold)
    best = load_torch(fold_directory / "best_model.pt", device)
    if best.get("signature") != info["signature"] or int(best.get("fold")) != int(fold):
        raise SystemExit(f"{run_directory}: best_model.pt không khớp chữ ký / Fold của lượt chạy.")
    model = v2.CLISMambaV2(num_classes = len(classes), pretrained = False, **{**kwargs, "grad_checkpoint": False}).to(device).eval()
    model.load_state_dict(best["model_state_dict"], strict = True)
    cnn = model.cnn

    # Suy luận bật / tắt khối chèn trên cùng đặc trưng stage 3
    _, eval_transform = transform_setting()
    _, loader = create_dataloader(NIHChestXrayDataset, classes, device, frame, eval_transform, False, batch_size = int(settings["training_config"]["batch_size"]), seed = int(settings["seed"]) + 900_000)
    on, off, ratios, targets = [], [], [], []
    with torch.no_grad(), torch.autocast(device_type = device.type, dtype = torch.bfloat16 if config.AMP_DTYPE == "bfloat16" else torch.float16, enabled = bool(config.USE_AMP and device.type == "cuda")):
        for images, labels in loader:
            x = cnn.stages[2](cnn.stages[1](cnn.stages[0](cnn.stem(images.to(device, non_blocking = True)))))
            y = model.inserted(x)
            ratios.append(((y.float() - x.float()).flatten(1).norm(dim = 1) / x.float().flatten(1).norm(dim = 1)).cpu())
            on.append(torch.sigmoid(model.classifier(cnn.forward_head(cnn.norm_pre(cnn.stages[3](y)))).float()).cpu())
            off.append(torch.sigmoid(model.classifier(cnn.forward_head(cnn.norm_pre(cnn.stages[3](x)))).float()).cpu())
            targets.append(labels)
    on, off, ratios, targets = torch.cat(on).numpy(), torch.cat(off).numpy(), torch.cat(ratios).numpy(), torch.cat(targets).numpy()

    # Kiểm tra nạp đúng: xác suất "bật" so với oof_predictions.csv đã lưu lúc huấn luyện (ghép theo mã ảnh)
    # (lượt tạm dừng theo PAUSE_AFTER_EPOCH chưa ghi oof_predictions.csv: bỏ kiểm tra, saved_diff = NaN)
    saved_path = fold_directory / "oof_predictions.csv"
    saved_diff = float("nan")
    if saved_path.is_file():
        saved = pd.read_csv(saved_path).set_index("Image Index").loc[frame["Image Index"], [f"prob_{name}" for name in classes]].to_numpy(dtype = np.float64)
        saved_diff = float(np.abs(on - saved).max())
    auroc_on, auroc_off = macro_auroc(targets, on), macro_auroc(targets, off)

    del model, best
    return {"run": str(run_directory.relative_to(config.MODELS_OUTPUT_DIRECTORY)).replace("\\", "/"), "mixer": kwargs["mixer"],
            "block_lr_scale": float(kwargs.get("block_lr_scale", 1.0)), "freeze_backbone": bool(settings["model_kwargs"].get("freeze_backbone", False)),
            "init_from": settings["model_kwargs"].get("init_from"), "images": len(frame), "ratio_mean": float(ratios.mean()), "ratio_median": float(np.median(ratios)),
            "auroc_on": auroc_on, "auroc_off": auroc_off, "auroc_drop": auroc_on - auroc_off, "prob_change": float(np.abs(on - off).mean()),
            "saved_diff": saved_diff, "measured_at": timestamp()}


# ============================================================
def main():
    parser = argparse.ArgumentParser(description = "Mức tham gia của khối chèn CLIS-Mamba v2 (Validation)")
    parser.add_argument("--runs", nargs = "+", required = True, help = "Thư mục lượt chạy (run_N), cho phép mẫu *")
    parser.add_argument("--fold", type = int, default = 1)
    parser.add_argument("--limit", type = int, default = None, help = "Chỉ dùng N ảnh Validation đầu tiên (kiểm tra nhanh)")
    parser.add_argument("--workers", type = int, default = None)
    parser.add_argument("--output", type = Path, default = OUTPUT_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for run_directory in expand_runs(args.runs):
        if not (run_directory / f"fold_{args.fold}" / "best_model.pt").is_file():
            print(f"Bỏ qua {run_directory}: chưa có best_model.pt", flush = True)
            continue
        try:
            row = measure(run_directory, args.fold, args.limit, args.workers, device)
        except (SystemExit, RuntimeError, KeyError, ValueError, OSError) as error:
            print(f"Lỗi khi đo {run_directory}: {error}", flush = True)
            continue
        if row is None:
            continue
        rows.append(row)
        print(f"{row['run']} | tỉ lệ ||khối(x) - x|| / ||x|| {row['ratio_mean']:.4f} | macro AUROC bật {row['auroc_on']:.4f}, tắt {row['auroc_off']:.4f} "
              f"(giảm {row['auroc_drop']:+.4f}) | |Δ xác suất| {row['prob_change']:.4f} | lệch so với tệp đã lưu {row['saved_diff']:.1e}", flush = True)

    # Ghi kết quả: thay dòng cũ của cùng lượt chạy (cùng số ảnh), giữ các dòng khác
    if rows and args.limit is None:
        new = pd.DataFrame(rows)
        if args.output.is_file():
            old = pd.read_csv(args.output)
            new = pd.concat([old[~old["run"].isin(new["run"])], new], ignore_index = True)
        new.to_csv(args.output, index = False)
        print(f"Đã ghi {args.output}", flush = True)


if __name__ == "__main__":
    main()
