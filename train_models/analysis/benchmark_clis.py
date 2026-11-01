# Đo VRAM đỉnh và thời gian một bước huấn luyện của CLIS-Mamba theo độ phân giải và micro-batch (cổng G0, thí nghiệm E5).
# Cùng chế độ với lúc huấn luyện: train(), autocast BF16, checkpointing từng khối, AdamW như config.CLIS_CONFIG, chế độ tất định như config.py.
# Dữ liệu tổng hợp trên GPU (không tính DataLoader). Nên cắm sạc, đặt chế độ hiệu năng, không chạy việc khác trên GPU.
#   conda activate mamba-main
#   python analysis/benchmark_clis.py
#   python analysis/benchmark_clis.py --mixers m_clis --sizes 1024 --batches 1 2 --backbone-checkpoint off on
# Quy tắc chốt độ phân giải (tài liệu G0): chọn 1024 nếu micro-batch >= 2 có VRAM đỉnh (reserved) <= 7.0 GB và
# thời gian ước tính của E3 + E4 ở 1024 nằm trong ngân sách GPU còn lại; ngược lại chọn 768.

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
import gc
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
import models.clis_mamba as clis
from utilities.log import GPUMonitor, timestamp
from utilities.setting import set_seed

TRAIN_IMAGES = 69_235                                                       # Số ảnh huấn luyện của fold 1 (để quy đổi giờ / epoch)
MEMORY_LIMIT_GB = 7.0                                                       # Chừa khoảng 1 GB của 8 GB cho driver / WDDM


# ============================================================
# Một cấu hình: trả về dict kết quả, hoặc None nếu hết bộ nhớ
def measure(mixer, size, batch, backbone_checkpoint, warmup, steps, monitor):
    model = optimizer = x = y = None
    try:
        model = clis.CLISMamba(14, mixer = mixer, pretrained = False, backbone_checkpoint = backbone_checkpoint).cuda().train()
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
        return {
            "peak_allocated_gb"     : round(allocated, 2),
            "peak_reserved_gb"      : round(reserved, 2),
            "step_seconds"          : round(step_seconds, 4),
            "step_spread"           : round((max(timings) - min(timings)) / step_seconds, 3),
            "train_images_per_second": round(batch / step_seconds, 1),
            "eval_images_per_second": round(batch / eval_seconds, 1),
            "hours_per_epoch"       : round(TRAIN_IMAGES / (batch / step_seconds) / 3600, 2),
            "within_memory_limit"   : reserved <= MEMORY_LIMIT_GB,
            "sm_clock_mean_mhz"     : gpu.get("gpu_sm_clock_mhz_mean"),
            "sm_clock_min_mhz"      : gpu.get("gpu_sm_clock_mhz_min"),
            "temperature_max_c"     : gpu.get("gpu_temperature_c_max"),
            "power_mean_w"          : gpu.get("gpu_power_w_mean"),
            "thermal_slowdown_s"    : gpu.get("gpu_slowdown_thermal_seconds"),
        }
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        del model, optimizer, x, y
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Đo VRAM / thời gian huấn luyện CLIS-Mamba theo độ phân giải")
    parser.add_argument("--mixers", nargs = "+", default = ["m_clis", "attn", "gatedconv"], choices = sorted({v["mixer"] for v in clis.VARIANTS.values()}))
    parser.add_argument("--sizes", nargs = "+", type = int, default = [512, 768, 1024])
    parser.add_argument("--batches", nargs = "+", type = int, default = [1, 2, 4, 8])
    parser.add_argument("--backbone-checkpoint", nargs = "+", choices = ["off", "on"], default = ["off", "on"])
    parser.add_argument("--warmup", type = int, default = 3)
    parser.add_argument("--steps", type = int, default = 10)
    parser.add_argument("--output", type = Path, default = config.MODELS_OUTPUT_DIRECTORY / "benchmarks" / f"clis_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Cần GPU CUDA")
    if clis.selective_scan_fn is None and any(m in clis.ROUTES for m in args.mixers):
        parser.error("Biến thể SS2D cần kernel selective_scan_cuda của mamba-ssm: chạy trong WSL2 (conda activate mamba-main)")

    set_seed()                                                              # Chế độ tất định như lúc huấn luyện (config.py)
    monitor = GPUMonitor.get()
    print(f"[{timestamp()}] {torch.cuda.get_device_name(0)} | torch {torch.__version__} | tất định {torch.are_deterministic_algorithms_enabled()} | BF16 | checkpointing từng khối")

    rows = []
    for mixer in args.mixers:
        for size in args.sizes:
            for backbone_checkpoint in [value == "on" for value in args.backbone_checkpoint]:
                for batch in sorted(args.batches):
                    result = measure(mixer, size, batch, backbone_checkpoint, args.warmup, args.steps, monitor)
                    row = {"mixer": mixer, "size": size, "tokens": (size // 16) ** 2, "batch": batch, "backbone_checkpoint": backbone_checkpoint, "oom": result is None, **(result or {})}
                    rows.append(row)
                    if result is None:
                        print(f"{mixer:9s} {size:5d} px | batch {batch} | backbone ckpt {backbone_checkpoint!s:5s} | HẾT BỘ NHỚ", flush = True)
                        break                                               # Batch lớn hơn cũng sẽ hết bộ nhớ
                    print(f"{mixer:9s} {size:5d} px | batch {batch} | backbone ckpt {backbone_checkpoint!s:5s} | "
                          f"VRAM {result['peak_allocated_gb']:.2f} / {result['peak_reserved_gb']:.2f} GB (allocated / reserved) | "
                          f"{result['step_seconds'] * 1000:6.0f} ms/bước | {result['train_images_per_second']:6.1f} ảnh/s huấn luyện, "
                          f"{result['eval_images_per_second']:6.1f} ảnh/s suy luận | {result['hours_per_epoch']:.2f} giờ/epoch | "
                          f"xung SM {result['sm_clock_mean_mhz']} MHz, {result['temperature_max_c']}°C", flush = True)
                    if not result["within_memory_limit"]:
                        break                                               # Vượt giới hạn: WSL / WDDM tràn sang RAM dùng chung, batch lớn hơn chỉ chậm hơn

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents = True, exist_ok = True)
    frame.to_csv(args.output, index = False)
    print(f"\nĐã lưu: {args.output}")

    # Tóm tắt cho quy tắc chốt độ phân giải: micro-batch lớn nhất nằm trong giới hạn VRAM của mỗi (bộ trộn, độ phân giải)
    feasible = frame[~frame["oom"] & frame["within_memory_limit"].fillna(False).astype(bool)]
    if not feasible.empty:
        best = feasible.sort_values(["batch", "train_images_per_second"]).groupby(["mixer", "size"]).tail(1).sort_values(["mixer", "size"])
        print(f"\nMicro-batch lớn nhất có VRAM reserved <= {MEMORY_LIMIT_GB} GB:")
        print(best[["mixer", "size", "batch", "backbone_checkpoint", "peak_reserved_gb", "train_images_per_second", "hours_per_epoch"]].to_string(index = False))
