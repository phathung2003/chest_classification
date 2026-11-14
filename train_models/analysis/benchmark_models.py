# Đo VRAM đỉnh và tốc độ một bước huấn luyện của các mô hình đang dùng (ccmm, vm_ras, ConvNeXt-T, DenseNet-121) theo độ phân giải,
# micro-batch và checkpointing: kiểm tra 1024 px có khả thi trên GPU 8 GB không, trước khi đăng ký thiết kế xác nhận.
# Cùng cách đo với analysis/benchmark_clis.py: train(), autocast BF16, AdamW fused, chế độ tất định như config.py, dữ liệu tổng hợp trên GPU
# (không tính DataLoader). Không chạy khi GPU đang huấn luyện: số đo sai và có thể làm lượt kia hết bộ nhớ.
#   conda activate mamba-t29
#   python analysis/benchmark_models.py
#   python analysis/benchmark_models.py --models ccmm --sizes 1024 --batches 2 4 --checkpoint off on
# Giờ / epoch quy đổi từ ảnh / giây huấn luyện và suy luận (fold 1: 69.235 ảnh huấn luyện, 17.289 ảnh Validation).

# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"                         # cuBLAS tất định (DETERMINISTIC_ALGORITHMS)
os.environ["TRITON_CACHE_AUTOTUNING"] = "1"                                # mamba-ssm >= 2.3 ở chế độ tất định: lưu autotune kernel Triton

# Khai báo thư viện
import argparse
import gc
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
import models.stitch_cxr as stx
import models.vmamba_cxr as vmx
from models.convnext_tiny import ConvNeXtTinyBaseline
from models.densenet121 import DenseNetBaseline
from utilities.environment import check_environment
from utilities.log import GPUMonitor, timestamp
from utilities.setting import set_seed

TRAIN_IMAGES = 69_235                                                       # Ảnh huấn luyện của fold 1
VALIDATION_IMAGES = 17_289                                                  # Ảnh Validation của fold 1
MEMORY_LIMIT_GB = 7.0                                                       # Chừa khoảng 1 GB của 8 GB cho driver / WDDM (như benchmark_clis.py)
MODELS = ["ccmm", "vm_ras", "convnext_t", "densenet121"]


# Dựng mô hình (không tải trọng số ImageNet: tốc độ / VRAM không phụ thuộc trọng số). checkpoint: checkpointing từng khối
# (ccmm, vm_ras: tham số grad_checkpoint của mô hình; ConvNeXt-T: set_grad_checkpointing của timm; DenseNet-121: không hỗ trợ)
# Trả về: nn.Module hoặc None nếu mô hình không hỗ trợ checkpointing
def build(name, checkpoint, drop_path_rate):
    if name == "ccmm":
        return stx.StitchCXR(14, layout = "ccmm", stitch_init = "identity", pretrained = False, grad_checkpoint = checkpoint, drop_path_rate = drop_path_rate)
    if name == "vm_ras":
        return vmx.VMambaCXR(14, mixer = "vm_ras", pretrained = False, grad_checkpoint = checkpoint, drop_path_rate = drop_path_rate)
    if name == "convnext_t":
        model = ConvNeXtTinyBaseline(14, pretrained = False, drop_path_rate = drop_path_rate)
        model.cnn.set_grad_checkpointing(checkpoint)
        return model
    return None if checkpoint else DenseNetBaseline(14)                     # Luôn tải trọng số ImageNet (torchvision, đã có trong cache)


# ============================================================
# Một cấu hình: trả về dict kết quả, "oom" nếu hết bộ nhớ, None nếu không hỗ trợ
def measure(name, size, batch, checkpoint, drop_path_rate, warmup, steps, monitor):
    model = optimizer = x = y = None
    try:
        model = build(name, checkpoint, drop_path_rate)
        if model is None:
            return None
        model = model.cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, fused = bool(config.CLIS_CONFIG["ADAM_FUSED"]))
        x = torch.randn(batch, 3, size, size, device = "cuda")
        y = (torch.rand(batch, 14, device = "cuda") > 0.9).float()

        def train_step():
            optimizer.zero_grad(set_to_none = True)
            with torch.autocast(device_type = "cuda", dtype = torch.bfloat16):
                logits = model(x)
            F.binary_cross_entropy_with_logits(logits.float(), y).backward()
            optimizer.step()

        torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            train_step()
        torch.cuda.synchronize()
        if monitor is not None:
            monitor.reset_statistics("epoch")
        timings = []
        for _ in range(steps):
            started = time.perf_counter()
            train_step()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
        gpu = monitor.epoch_columns() if monitor is not None else {}
        allocated, reserved = torch.cuda.max_memory_allocated() / 1024 ** 3, torch.cuda.max_memory_reserved() / 1024 ** 3

        # Suy luận cùng batch (eval, không gradient, BF16)
        model.eval()
        with torch.no_grad(), torch.autocast(device_type = "cuda", dtype = torch.bfloat16):
            for _ in range(warmup):
                model(x)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(steps):
                model(x)
            torch.cuda.synchronize()
        eval_seconds = (time.perf_counter() - started) / steps

        step_seconds = float(np.median(timings))
        train_rate, eval_rate = batch / step_seconds, batch / eval_seconds
        return {
            "peak_allocated_gb"      : round(allocated, 2),
            "peak_reserved_gb"       : round(reserved, 2),
            "step_seconds"           : round(step_seconds, 4),
            "step_spread"            : round((max(timings) - min(timings)) / step_seconds, 3),
            "train_images_per_second": round(train_rate, 1),
            "eval_images_per_second" : round(eval_rate, 1),
            "hours_per_epoch"        : round((TRAIN_IMAGES / train_rate + VALIDATION_IMAGES / eval_rate) / 3600, 2),
            "within_memory_limit"    : reserved <= MEMORY_LIMIT_GB,
            "sm_clock_mean_mhz"      : gpu.get("gpu_sm_clock_mhz_mean"),
            "temperature_max_c"      : gpu.get("gpu_temperature_c_max"),
            "thermal_slowdown_s"     : gpu.get("gpu_slowdown_thermal_seconds"),
        }
    except torch.cuda.OutOfMemoryError:
        return "oom"
    finally:
        del model, optimizer, x, y
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Đo VRAM / tốc độ huấn luyện của ccmm, vm_ras, ConvNeXt-T, DenseNet-121 theo độ phân giải")
    parser.add_argument("--models", nargs = "+", default = MODELS, choices = MODELS)
    parser.add_argument("--sizes", nargs = "+", type = int, default = [512, 1024])
    parser.add_argument("--batches", nargs = "+", type = int, default = [2, 4, 8])
    parser.add_argument("--checkpoint", nargs = "+", choices = ["off", "on"], default = ["off", "on"])
    parser.add_argument("--drop-path", type = float, default = 0.0, help = "drop_path_rate (R2: 0.1); không đổi đáng kể VRAM / tốc độ")
    parser.add_argument("--warmup", type = int, default = 3)
    parser.add_argument("--steps", type = int, default = 10)
    parser.add_argument("--output", type = Path, default = config.MODELS_OUTPUT_DIRECTORY / "benchmarks" / f"models_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    args = parser.parse_args()
    check_environment(require_kernel = any(m in ("ccmm", "vm_ras") for m in args.models), amp_dtype = "bfloat16",
                      deterministic = config.DETERMINISTIC_ALGORITHMS, adam_fused = config.CLIS_CONFIG["ADAM_FUSED"])

    set_seed()
    monitor = GPUMonitor.get()
    print(f"[{timestamp()}] {torch.cuda.get_device_name(0)} | torch {torch.__version__} | tất định {torch.are_deterministic_algorithms_enabled()} | BF16 | "
          f"fused AdamW {bool(config.CLIS_CONFIG['ADAM_FUSED'])} | drop_path {args.drop_path:g}")

    rows = []
    for name in args.models:
        for size in args.sizes:
            for checkpoint in [value == "on" for value in args.checkpoint]:
                for batch in sorted(args.batches):
                    result = measure(name, size, batch, checkpoint, args.drop_path, args.warmup, args.steps, monitor)
                    if result is None:
                        break                                               # Mô hình không hỗ trợ checkpointing
                    setting = f"{name:11s} {size:5d} px | batch {batch} | checkpoint {checkpoint!s:5s}"
                    row = {"model": name, "size": size, "batch": batch, "checkpoint": checkpoint, "oom": result == "oom",
                           **(result if isinstance(result, dict) else {}), "torch_version": torch.__version__}
                    rows.append(row)
                    if result == "oom":
                        print(f"{setting} | HẾT BỘ NHỚ", flush = True)
                        break
                    print(f"{setting} | VRAM {result['peak_allocated_gb']:.2f} / {result['peak_reserved_gb']:.2f} GB (allocated / reserved) | "
                          f"{result['train_images_per_second']:6.1f} ảnh/s huấn luyện, {result['eval_images_per_second']:6.1f} ảnh/s suy luận | "
                          f"{result['hours_per_epoch']:.2f} giờ/epoch | {result['sm_clock_mean_mhz']} MHz, {result['temperature_max_c']}°C", flush = True)
                    if not result["within_memory_limit"]:
                        break                                               # Vượt giới hạn: WSL / WDDM tràn sang RAM dùng chung

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents = True, exist_ok = True)
    frame.to_csv(args.output, index = False)
    print(f"\nĐã lưu: {args.output}")

    # Cấu hình nhanh nhất trong giới hạn VRAM của mỗi (mô hình, độ phân giải)
    feasible = frame[~frame["oom"] & frame["within_memory_limit"].fillna(False).astype(bool)]
    if not feasible.empty:
        best = feasible.sort_values("train_images_per_second").groupby(["model", "size"]).tail(1).sort_values(["model", "size"])
        print(f"\nCấu hình nhanh nhất có VRAM reserved <= {MEMORY_LIMIT_GB} GB:")
        print(best[["model", "size", "batch", "checkpoint", "peak_reserved_gb", "train_images_per_second", "eval_images_per_second", "hours_per_epoch"]].to_string(index = False))
