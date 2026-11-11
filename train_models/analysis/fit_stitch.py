"""Tính lớp nối (stitch) bình phương tối thiểu giữa ConvNeXt-T và VMamba-T tiền huấn luyện ImageNet cho mô hình lai models/stitch_cxr.py.

Vì sao cần: ở ranh giới đổi họ (ví dụ "ccmm": sau stage 2), đơn vị stage kế tiếp của VMamba được tiền huấn luyện trên đặc trưng stage 2 của VMamba,
không phải của ConvNeXt. Lớp nối conv 1 x 1 đưa đặc trưng của họ nguồn về gần đặc trưng của họ đích (cùng ảnh, cùng vị trí), nên mô hình lai xuất phát
từ hai nửa tiền huấn luyện còn khớp nhau (Lenc & Vedaldi, 2015; Bansal et al., 2021; SN-Net, Pan et al., CVPR 2023).

Cách tính, với mỗi ranh giới của layout:
  - X: đặc trưng sau stage i của họ nguồn (backbone ImageNet nguyên bản), Y: đặc trưng sau stage i của họ đích, cùng ảnh, cùng token
  - Nghiệm ridge của Y = X·W + b trên phần ảnh dùng để khớp; R² trên phần ảnh để riêng (không dùng khi khớp), kèm R² của lớp nối đơn vị (Y = X)
Ảnh lấy từ các fold huấn luyện (không lấy fold Validation, không mở Test), mỗi bệnh nhân một ảnh, không dùng nhãn.

Ví dụ (WSL2, conda activate mamba-t29):
    python analysis/fit_stitch.py --layout ccmm
Kết quả: pretrained/stitch/stitch_<nguồn>2<đích>_stage<k>.pt (weight, bias, R², danh sách ảnh). models/run_stitch.py ghi SHA-256 của tệp vào model_kwargs.
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

import torch

import config
import models.stitch_cxr as stx
import models.vmamba_cxr as vmx
from utilities.data import create_dataloader, load_dataset, transform_setting
from utilities.dataset import NIHChestXrayDataset
from utilities.log import timestamp
from utilities.setting import apply_config

TOKENS_PER_IMAGE = 1024                                                     # Số token lấy ngẫu nhiên ở mỗi ảnh (lưới lớn hơn thì lấy mẫu, cùng vị trí cho X và Y)
RIDGE = 1e-4                                                                # Hệ số ridge so với trung bình đường chéo của XᵀX


# ============================================================
# Nghiệm ridge của y = x·W + b (float64, có tâm hóa)
# x: (n, c) đặc trưng nguồn, y: (n, c) đặc trưng đích
# Trả về: (weight (c, c) dạng [kênh ra, kênh vào] của conv 1 x 1, bias (c,))
def solve_stitch(x, y, ridge = RIDGE):
    x, y = x.double(), y.double()
    mean_x, mean_y = x.mean(dim = 0), y.mean(dim = 0)
    centered_x, centered_y = x - mean_x, y - mean_y
    gram = centered_x.T @ centered_x
    gram = gram + ridge * gram.diagonal().mean() * torch.eye(gram.size(0), dtype = gram.dtype)
    solution = torch.linalg.solve(gram, centered_x.T @ centered_y)                              # (c vào, c ra)
    return solution.T.contiguous(), mean_y - mean_x @ solution


# R² của dự đoán x·Wᵀ + b so với y (1 - phần phương sai còn lại / phương sai của y quanh trung bình từng kênh)
# Trả về: float
def r_squared(x, y, weight, bias):
    x, y = x.double(), y.double()
    residual = y - (x @ weight.T + bias)
    return float(1 - residual.pow(2).sum() / (y - y.mean(dim = 0)).pow(2).sum())


# ============================================================
# Đặc trưng sau stage `stage` của hai backbone nguyên bản trên cùng ảnh, lấy cùng các token ngẫu nhiên ở mỗi ảnh
# Trả về: {họ: (số ảnh, token, c)}
def collect(models, loader, stage, device, generator):
    features = {family: [] for family in models}
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            maps = {family: model.stage_features(images, stage, stitched = False).float().flatten(2).transpose(1, 2).cpu() for family, model in models.items()}     # (b, l, c)
            length = next(iter(maps.values())).size(1)
            index = torch.randperm(length, generator = generator)[:TOKENS_PER_IMAGE] if length > TOKENS_PER_IMAGE else torch.arange(length)
            for family, value in maps.items():
                features[family].append(value[:, index])
    return {family: torch.cat(values) for family, values in features.items()}


# ============================================================
def main():
    parser = argparse.ArgumentParser(description = "Lớp nối bình phương tối thiểu ConvNeXt-T <--> VMamba-T (ảnh huấn luyện, không nhãn)")
    parser.add_argument("--layout", default = "ccmm", help = "Layout của mô hình lai (models/stitch_cxr.py); tính lớp nối cho mọi ranh giới đổi họ")
    parser.add_argument("--images", type = int, default = 96, help = "Số ảnh (mỗi bệnh nhân một ảnh); 3 / 4 để khớp, 1 / 4 để đo R²")
    parser.add_argument("--fold", type = int, default = 1, help = "Fold Validation (không lấy ảnh từ fold này)")
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--batch-size", type = int, default = 4)
    parser.add_argument("--workers", type = int, default = 2)
    parser.add_argument("--device", default = "cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not stx.boundaries(args.layout):
        raise SystemExit(f"Layout {args.layout} không có ranh giới đổi họ")
    device = torch.device(args.device)
    if device.type == "cpu":
        vmx.vm.WITH_SELECTIVESCAN_MAMBA = False                                                 # Selective scan bản PyTorch của VMamba trên CPU

    # Ảnh huấn luyện: mỗi bệnh nhân một ảnh, chọn ngẫu nhiên có seed, không lấy fold Validation
    apply_config({**config.CLIS_CONFIG, "NUMBER_OF_WORKERS": int(args.workers)})
    train_df, _, _, classes, _ = load_dataset()
    frame = train_df[train_df["Fold"] != int(args.fold)]
    frame = frame.groupby(config.PATIENT_ID_COLUMN).sample(n = 1, random_state = args.seed).sample(n = args.images, random_state = args.seed).reset_index(drop = True)
    _, eval_transform = transform_setting()
    _, loader = create_dataloader(NIHChestXrayDataset, classes, device, frame, eval_transform, False, batch_size = int(args.batch_size), seed = args.seed)
    fit_images = (3 * args.images) // 4

    # Hai backbone ImageNet nguyên bản (layout một họ không có lớp nối)
    models = {family: stx.StitchCXR(len(classes), layout = family * len(args.layout), pretrained = True).to(device).eval() for family in stx.FAMILIES}
    stx.STITCH_DIRECTORY.mkdir(parents = True, exist_ok = True)
    for stage in stx.boundaries(args.layout):
        source, target = args.layout[stage], args.layout[stage + 1]
        features = collect(models, loader, stage, device, torch.Generator().manual_seed(args.seed))
        x, y = features[source], features[target]
        channels = x.size(-1)
        weight, bias = solve_stitch(x[:fit_images].reshape(-1, channels), y[:fit_images].reshape(-1, channels))
        held_x, held_y = x[fit_images:].reshape(-1, channels), y[fit_images:].reshape(-1, channels)
        r2 = r_squared(held_x, held_y, weight, bias)
        r2_identity = r_squared(held_x, held_y, torch.eye(channels, dtype = torch.float64), torch.zeros(channels, dtype = torch.float64))
        path = stx.stitch_path(source, target, stage)
        torch.save({"weight": weight.float(), "bias": bias.float(), "source": source, "target": target, "stage": int(stage), "r2": r2, "r2_identity": r2_identity,
                    "images": frame[config.IMAGE_ID_COLUMN].tolist(), "fit_images": int(fit_images), "tokens_per_image": int(x.size(1)), "image_size": int(config.IMAGE_SIZE),
                    "image_variant": str(config.IMAGE_VARIANT), "ridge": RIDGE, "seed": int(args.seed), "created_at": timestamp()}, path)
        print(f"Ranh giới sau stage {stage + 1} ({source} --> {target}, {channels} kênh): khớp trên {fit_images} ảnh x {x.size(1)} token | "
              f"R² ảnh để riêng {r2:.3f} (lớp nối đơn vị {r2_identity:.3f}) | {path.name} SHA-256 {vmx.sha256_file(path)}", flush = True)


if __name__ == "__main__":
    main()
