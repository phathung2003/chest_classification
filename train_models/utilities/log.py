import os
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

import config


# ============================================================
# Định dạng thông báo console (notebook tiền xử lý cũng dùng các hàm này)
def print_heading(title):
    print("=" * config.CONSOLE_WIDTH)
    print(title)
    print("=" * config.CONSOLE_WIDTH)


def print_title(title):
    print(title)
    print("-" * config.CONSOLE_WIDTH)


def print_result(RESULT, TITLE, MESSAGE):
    print(f"{RESULT:<{config.MIN_RESULT_WIDTH}} {TITLE:<{config.MIN_TITLE_WIDTH}}: {MESSAGE}")


def print_message(TITLE, MESSAGE):
    print(f"{TITLE:<{config.MIN_TITLE_WIDTH}}: {MESSAGE}")


# ============================================================
# Thanh tiến trình có dấu phân cách hàng nghìn (1,234/5,678)
class TqdmComma(tqdm):
    @staticmethod
    def format_meter(n, total, elapsed, **kwargs):
        text = tqdm.format_meter(n = n, total = total, elapsed = elapsed, **kwargs)
        if total is not None:
            text = text.replace(f"{int(n)}/{int(total)}", f"{int(n):,}/{int(total):,}")
        # Đơn vị " batch" có khoảng trắng để hiện "8.32 batch/s"; khi chậm hơn 1 batch/s tqdm đảo thành "2.39 s/batch"
        return text.replace("s/ batch", " s/batch")


# ============================================================
# Theo dõi GPU bằng một luồng nền (lấy mẫu mỗi giây, không làm chậm vòng huấn luyện).
# Nguồn: pynvml nếu đã cài (pip install nvidia-ml-py), nếu không thì nvidia-smi, cuối cùng là bộ nhớ từ PyTorch.
class GPUMonitor:
    _instance = None

    def __init__(self, device_index = 0, interval = None):
        self.device_index = device_index
        self.interval = config.GPU_MONITOR_INTERVAL if interval is None else interval
        self.snapshot = {"utilization": None, "memory_used_mb": None, "memory_total_mb": None}
        self._reader = self._make_reader()
        threading.Thread(target = self._loop, daemon = True, name = "gpu-monitor").start()

    # Một monitor dùng chung cho cả tiến trình
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # Chọn nguồn đọc số liệu GPU
    def _make_reader(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

            def read_nvml():
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return (pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                        memory.used / 1024 ** 2, memory.total / 1024 ** 2)
            return read_nvml
        except Exception:
            pass

        def read_nvidia_smi():
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits", "-i", str(self.device_index)],
                capture_output = True, text = True, timeout = 5,
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            utilization, used, total = (float(v) for v in output.strip().split(","))
            return utilization, used, total

        try:
            read_nvidia_smi()
            return read_nvidia_smi
        except Exception:
            pass

        def read_torch():
            import torch
            free, total = torch.cuda.mem_get_info(self.device_index)
            return None, (total - free) / 1024 ** 2, total / 1024 ** 2
        return read_torch

    def _loop(self):
        while True:
            try:
                utilization, used, total = self._reader()
                self.snapshot = {"utilization": utilization, "memory_used_mb": used, "memory_total_mb": total}
            except Exception:
                pass
            time.sleep(self.interval)

    # Chuỗi hiển thị: "GPU 97% | VRAM 5.00/8.00 GB"
    def text(self):
        s = self.snapshot
        parts = []
        if s["utilization"] is not None:
            parts.append(f"GPU {s['utilization']:.0f}%")
        if s["memory_used_mb"] is not None:
            parts.append(f"VRAM {s['memory_used_mb'] / 1024:,.2f}/{s['memory_total_mb'] / 1024:,.2f} GB")
        return " | ".join(parts)


# ============================================================
# Thanh tiến trình cho huấn luyện / đánh giá: số batch có dấu phẩy hàng nghìn, kèm GPU usage và VRAM.
# KHÔNG dùng tqdm.auto: lớp tqdm_asyncio gọi iter(loader) ngay khi khởi tạo, tạo thêm một iterator DataLoader
# thừa mỗi epoch (thêm một bộ worker nạp ảnh rồi bỏ, chậm gần gấp đôi lúc đầu epoch, và lệch seed của loader).
def progress_bar(iterable, description, device):
    bar = TqdmComma(iterable, desc = description, unit = " batch", leave = False, dynamic_ncols = True)
    bar.gpu_monitor = GPUMonitor.get() if device.type == "cuda" else None
    return bar


# Cập nhật phần đuôi thanh tiến trình (không ép vẽ lại; tqdm tự giới hạn tần suất vẽ)
def update_progress(bar, **values):
    parts = [f"{key} {value:,.4f}" if isinstance(value, float) else f"{key} {value:,}" for key, value in values.items()]
    if bar.gpu_monitor is not None:
        gpu_text = bar.gpu_monitor.text()
        if gpu_text:
            parts.append(gpu_text)
    bar.set_postfix_str(" | ".join(parts), refresh = False)


# ============================================================
# Thời điểm hiện tại theo giờ địa phương, có múi giờ (dùng làm mốc trong log)
def timestamp():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec = "seconds")


# ============================================================
# Luồng ghi kép: vừa in ra console, vừa ghi vào run.log (bỏ qua các lần vẽ lại thanh tiến trình)
class _RunLogStream:
    def __init__(self, console, log_file, lock):
        self.console = console
        self.log_file = log_file
        self.lock = lock

    def write(self, value):
        if not isinstance(value, str):
            value = str(value)
        with self.lock:
            self.console.write(value)
            # tqdm vẽ lại bằng ký tự '\r'; ghi mọi lần vẽ lại sẽ làm log phình to
            if value and "\r" not in value:
                self.log_file.write(value)
                self.log_file.flush()
                # Đồng bộ xuống đĩa ở cuối mỗi dòng để log còn nguyên khi chương trình dừng đột ngột
                if "\n" in value:
                    os.fsync(self.log_file.fileno())
        return len(value)

    def flush(self):
        with self.lock:
            self.console.flush()
            self.log_file.flush()

    def isatty(self):
        return self.console.isatty()

    def fileno(self):
        return self.console.fileno()

    @property
    def encoding(self):
        return getattr(self.console, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self.console, "errors", "replace")

    def writable(self):
        return True


# ============================================================
# Ghi toàn bộ output của một lần khởi chạy vào run.log (nối tiếp khi chạy lại) và in cấu hình ở đầu
@contextmanager
def capture_run_log(run_directory, *, model_name, dataset_name, train_df, test_df, folds, settings, signature, launch):
    run_directory = Path(run_directory)
    run_directory.mkdir(parents = True, exist_ok = True)
    log_path = run_directory / "run.log"
    resumed_log = log_path.is_file() and log_path.stat().st_size > 0
    old_stdout, old_stderr = sys.stdout, sys.stderr

    with log_path.open("a", encoding = "utf-8", buffering = 1) as file:
        mutex = threading.RLock()
        stdout_mirror = _RunLogStream(old_stdout, file, mutex)
        stderr_mirror = _RunLogStream(old_stderr, file, mutex)
        sys.stdout, sys.stderr = stdout_mirror, stderr_mirror

        try:
            _print_run_header(run_directory, resumed_log, model_name, dataset_name, train_df, test_df, folds, settings, signature, launch)
            yield log_path

        # Lỗi hoặc Ctrl+C: ghi traceback vào log rồi ném lỗi tiếp
        except BaseException as exc:
            print(f"[{timestamp()}] LẦN CHẠY BỊ LỖI / TẠM DỪNG: {type(exc).__name__}: {exc}", file = sys.stderr)
            with mutex:
                file.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
                file.flush()
                os.fsync(file.fileno())
            raise
        else:
            print(f"[{timestamp()}] KẾT THÚC LẦN CHẠY | {run_directory.name}")

        # Trả lại stdout / stderr trước khi flush, để lỗi I/O không làm output sau đó trỏ vào tệp đã đóng
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            stdout_mirror.flush()
            stderr_mirror.flush()
            with mutex:
                file.flush()
                os.fsync(file.fileno())


# ============================================================
# Đầu log: lượt chạy, dữ liệu, cấu hình (lấy từ settings, tức đúng những gì nằm trong chữ ký) và trạng thái Git
def _print_run_header(run_directory, resumed_log, model_name, dataset_name, train_df, test_df, folds,
                      settings, signature, launch):
    tc = settings["training_config"]

    print("\n" + "=" * config.CONSOLE_WIDTH)
    print(f"[{timestamp()}] {'TIẾP TỤC' if resumed_log else 'BẮT ĐẦU'} HUẤN LUYỆN | {model_name} | {run_directory.name}")
    print("=" * config.CONSOLE_WIDTH)
    print_message("Thư mục lượt chạy", run_directory)
    print_message("Chữ ký cấu hình", signature[:16])
    print_message("Git commit", f"{launch['git_commit']} ({launch['git_branch']})")
    if launch["git_dirty"]:
        print_message("CẢNH BÁO", f"{len(launch['git_dirty_files'])} tệp chưa commit (danh sách trong train_info.json)")
    print()

    print_heading("Tập dữ liệu")
    print_message("Tên", dataset_name)
    print_message("Số ảnh", f"Huấn luyện / Validation: {len(train_df):,} | Test: {len(test_df):,}")
    print_message("Số fold / fold chạy lần này", f"{settings['number_of_folds']} / {folds}")
    print_message("Số lớp", len(settings["classes"]))
    print_message("Kích cỡ ảnh", f"{settings['image_size']}\n")

    print_heading("Cấu hình huấn luyện")
    print_message("Hàm Loss", settings["loss_config"])
    print_message("Optimizer", tc["optimizer"])
    print_message("Scheduler", tc["scheduler"])
    print_message("Tiêu chí chọn mô hình", tc["selection_metric"])
    print_message("Batch size", tc["batch_size"])
    print_message("Số epoch tối đa", tc["epochs"])
    print_message("Early stopping", f"patience={tc['early_stop_patience']}, min_delta={tc['min_delta']}")
    print_message("Gradient", f"accum_steps={tc['grad_accum_steps']}, clip_norm={tc['grad_clip_norm']}")
    print_message("Mixed precision (AMP)", settings["use_amp"])
    print_message("Ngưỡng mặc định", settings["threshold"])
    print_message("Số worker", f"{settings['number_of_workers']}\n")

    print_heading("Làm giàu ảnh")
    for key, value in settings["augmentation"].items():
        print_message(key, value)
    print()

    print_heading("Tái lập")
    print_message("Seed", settings["seed"])
    print_message("cuDNN deterministic", settings["cudnn_deterministic"])
    print_message("cuDNN benchmark", settings["cudnn_benchmark"])
    print_message("Deterministic algorithms", f"{settings['deterministic_algorithms']}\n")


# ============================================================
# Thông số mô hình
def print_model_specification(specification):
    print_heading("Thông số mô hình")
    print_message("Parameters", f"{specification['parameters']:,}")
    print_message("Trainable parameters", f"{specification['trainable_parameters']:,}")
    print_message("MACs (1 ảnh)", f"{specification['macs']:,}")
    print_message("FLOPs (1 ảnh)", f"{specification['flops']:,}")
    print_message("Độ trễ suy luận (batch 1)", f"{specification['inference_latency_ms_batch_1']:.2f} ms "
                                               f"({specification['inference_precision']})")
    print_message(f"Throughput (batch {specification['inference_batch_size']})", f"{specification['inference_images_per_second']:,.0f} ảnh/s")
# ============================================================
# Tóm tắt chỉ số Test trên một dòng (chi tiết nằm trong test_summary.json)
def print_test_summary(title, summary):
    ci = summary.get("macro_auroc_ci95")
    ci_text = f" [95% CI {ci[0]:.4f}, {ci[1]:.4f}]" if ci and None not in ci else ""
    loss = summary.get("test_loss")
    line = (f"{title} | Macro AUROC {summary['macro_auroc']:.4f}{ci_text} | Macro AP {summary['macro_ap']:.4f} | "
            f"Macro F1@{summary['threshold']:g} {summary['macro_f1']:.4f}")
    if loss is not None:
        line += f" | Loss {loss:.4f}"
    if "postprocessed" in summary:
        line += f" | Macro F1 (ngưỡng theo lớp) {summary['postprocessed']['macro_f1']:.4f}"
    print(line)


# ============================================================
# Tổng kết cuối lượt chạy: Validation từng fold, mean ± std trên Validation và Test, ensemble
def print_run_summary(cv_frame, test_report = None):
    def mean_std(values):
        values = values.dropna()
        if len(values) == 0:
            return "không có"
        std = f" ± {values.std(ddof = 1):.4f}" if len(values) > 1 else ""
        return f"{values.mean():.4f}{std} ({len(values)} fold)"

    print()
    print_heading(f"[{timestamp()}] TỔNG KẾT LƯỢT CHẠY")
    columns = [c for c in ("fold", "best_epoch", "epochs_trained", "stop_reason", "training_minutes",
                           "train_images_per_second", "peak_gpu_memory_gb", "amp_skipped_steps",
                           "macro_auroc", "macro_ap") if c in cv_frame]
    print(cv_frame[columns].round(4).to_string(index = False))
    print()
    print_message("Validation macro AUROC", mean_std(cv_frame["macro_auroc"]))
    print_message("Validation macro AP", mean_std(cv_frame["macro_ap"]))
    if test_report is not None:
        auroc = test_report["metric_mean_std"].get("macro_auroc", {})
        if auroc.get("mean") is not None:
            std = f" ± {auroc['std']:.4f}" if auroc.get("std") is not None else ""
            print_message("Test macro AUROC (từng fold)", f"{auroc['mean']:.4f}{std} ({auroc['number_of_folds']} fold)")
        ensemble = test_report.get("ensemble")
        if ensemble is not None:
            ci = ensemble.get("macro_auroc_ci95")
            ci_text = f" [95% CI {ci[0]:.4f}, {ci[1]:.4f}]" if ci and None not in ci else ""
            print_message("Test macro AUROC (ensemble)", f"{ensemble['macro_auroc']:.4f}{ci_text}")
    print()
