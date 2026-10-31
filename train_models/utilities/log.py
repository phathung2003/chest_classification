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
from tqdm.utils import disp_trim

import config


# ============================================================
# Định dạng thông báo console
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
# Thanh tiến trình (Có dấu phân cách hàng nghìn)
class TqdmComma(tqdm):
    @staticmethod
    def format_meter(n, total, elapsed, **kwargs):
        ncols = kwargs.get("ncols")
        plain, grouped = (f"{int(n)}/{int(total)}", f"{int(n):,}/{int(total):,}") if total is not None else ("", "")
        if ncols:
            kwargs["ncols"] = max(ncols - (len(grouped) - len(plain)) - 1, 10)
        text = tqdm.format_meter(n = n, total = total, elapsed = elapsed, **kwargs)
        if total is not None:
            text = text.replace(plain, grouped, 1)
        text = text.replace("s/ batch", " s/batch")
        return disp_trim(text, ncols - 1) if ncols else text


# ============================================================
# Theo dõi GPU
class GPUMonitor:
    _instance = None

    def __init__(self, device_index = 0, interval = None):
        self.device_index = device_index
        self.interval = config.GPU_MONITOR_INTERVAL if interval is None else interval
        self.snapshot = {"utilization": None, "memory_used_mb": None, "memory_total_mb": None, "power_w": None, "temperature_c": None,
                         "sm_clock_mhz": None, "slowdown_us": None}
        self._lock = threading.Lock()
        self._scopes = {}
        for scope in self._SCOPES:
            self.reset_statistics(scope)
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

            # Điện năng / nhiệt độ có thể không hỗ trợ trên một số GPU --> None
            def optional(read):
                try:
                    return read()
                except pynvml.NVMLError:
                    return None

            def read_nvml():
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                power_mw = optional(lambda: pynvml.nvmlDeviceGetPowerUsage(handle))
                return {"utilization"   : pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                        "memory_used_mb": memory.used / 1024 ** 2, "memory_total_mb": memory.total / 1024 ** 2,
                        "power_w"       : None if power_mw is None else power_mw / 1000,
                        "temperature_c" : optional(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
                        "sm_clock_mhz"  : optional(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
                        "slowdown_us"   : {name: optional(lambda: pynvml.nvmlDeviceGetViolationStatus(handle, policy).violationTime / 1000)
                                           for name, policy in (("power_cap", pynvml.NVML_PERF_POLICY_POWER),
                                                                ("thermal", pynvml.NVML_PERF_POLICY_THERMAL))}}
            read_nvml()
            return read_nvml
        except Exception:
            pass

        def read_nvidia_smi():
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,"
                 "clocks_event_reasons_counters.sw_power_cap,clocks_event_reasons_counters.sw_thermal_slowdown,"
                 "clocks_event_reasons_counters.hw_thermal_slowdown,clocks_event_reasons_counters.hw_power_brake_slowdown",
                 "--format=csv,noheader,nounits", "-i", str(self.device_index)],
                capture_output = True, text = True, timeout = 5,
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            # Trường không hỗ trợ trả về "[N/A]" --> None
            values = []
            for v in output.strip().split(","):
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(None)
            utilization, used, total, power, temperature, clock, power_cap, sw_thermal, hw_thermal, power_brake = values
            thermal = None if sw_thermal is None or hw_thermal is None else sw_thermal + hw_thermal
            return {"utilization": utilization, "memory_used_mb": used, "memory_total_mb": total,
                    "power_w": power, "temperature_c": temperature, "sm_clock_mhz": clock,
                    "slowdown_us": {"power_cap": power_cap, "thermal": thermal, "power_brake": power_brake}}

        try:
            read_nvidia_smi()
            return read_nvidia_smi
        except Exception:
            pass

        def read_torch():
            import torch
            free, total = torch.cuda.mem_get_info(self.device_index)
            return {"utilization": None, "memory_used_mb": (total - free) / 1024 ** 2, "memory_total_mb": total / 1024 ** 2,
                    "power_w": None, "temperature_c": None, "sm_clock_mhz": None, "slowdown_us": None}
        return read_torch

    def _loop(self):
        while True:
            try:
                snapshot = self._reader()
                self.snapshot = snapshot
                self._accumulate(snapshot)
            except Exception:
                pass
            time.sleep(self.interval)

    # ------------------------------------------------------------
    # Thống kê theo thời gian, tính riêng cho từng phạm vi ("run": train_info.json, "epoch": history.csv):
    #   - utilization / power / temperature / xung SM: trung bình và cao nhất (xung lấy thêm thấp nhất) trên mọi mẫu đọc
    #   - Thời gian giảm xung: hiệu bộ đếm tích lũy của driver giữa mẫu đầu và mẫu cuối kể từ lần reset
    _STATISTIC_KEYS = ("utilization", "power_w", "temperature_c", "sm_clock_mhz")
    _SCOPES = ("run", "epoch")

    def reset_statistics(self, scope = "run"):
        with self._lock:
            self._scopes[scope] = {"started": time.time(), "samples": 0, "values": {key: [] for key in self._STATISTIC_KEYS},
                                   "slowdown_first": None, "slowdown_last": None}

    def _accumulate(self, snapshot):
        with self._lock:
            for state in self._scopes.values():
                state["samples"] += 1
                for key in self._STATISTIC_KEYS:
                    if snapshot.get(key) is not None:
                        state["values"][key].append(float(snapshot[key]))
                if snapshot.get("slowdown_us") is not None:
                    if state["slowdown_first"] is None:
                        state["slowdown_first"] = snapshot["slowdown_us"]
                    state["slowdown_last"] = snapshot["slowdown_us"]

    def statistics(self, scope = "run"):
        with self._lock:
            state = self._scopes[scope]
            result = {"monitored_seconds": round(time.time() - state["started"], 1), "samples": state["samples"],
                      "interval_seconds": self.interval}
            for key, values in state["values"].items():
                result[key] = {"mean": round(sum(values) / len(values), 2), "max": round(max(values), 2)} if values else None
            if result["sm_clock_mhz"] is not None:
                result["sm_clock_mhz"]["min"] = round(min(state["values"]["sm_clock_mhz"]), 2)

            slowdown = None
            if state["slowdown_first"] is not None:
                slowdown = {}
                for reason, last in state["slowdown_last"].items():
                    first = state["slowdown_first"].get(reason)
                    slowdown[reason] = None if first is None or last is None else round(max(last - first, 0) / 1e6, 3)
            result["clock_slowdown_seconds"] = slowdown
            return result

    # Thống kê phạm vi epoch dạng phẳng cho history.csv: gpu_<chỉ số>_<mean|max|min>, gpu_slowdown_<lý do>_seconds
    def epoch_columns(self):
        statistics = self.statistics("epoch")
        columns = {}
        for key in self._STATISTIC_KEYS:
            for name in ("mean", "max", "min") if key == "sm_clock_mhz" else ("mean", "max"):
                columns[f"gpu_{key}_{name}"] = None if statistics[key] is None else statistics[key][name]
        for reason in ("power_cap", "thermal", "power_brake"):
            columns[f"gpu_slowdown_{reason}_seconds"] = (statistics["clock_slowdown_seconds"] or {}).get(reason)
        return columns

    # Chuỗi hiển thị: "GPU 97% | VRAM 5.00/8.00 GB | 85 W | 74°C"
    def text(self):
        s = self.snapshot
        parts = []
        if s["utilization"] is not None:
            parts.append(f"GPU {s['utilization']:.0f}%")
        if s["memory_used_mb"] is not None:
            parts.append(f"VRAM {s['memory_used_mb'] / 1024:,.2f}/{s['memory_total_mb'] / 1024:,.2f} GB")
        if s["power_w"] is not None:
            parts.append(f"{s['power_w']:.0f} W")
        if s["temperature_c"] is not None:
            parts.append(f"{s['temperature_c']:.0f}°C")
        return " | ".join(parts)

# ============================================================
# Thanh tiến trình cho huấn luyện / đánh giá
def progress_bar(iterable, description, device):
    sys.stdout.flush()
    sys.stderr.write("\r\n")
    bar = TqdmComma(iterable, desc = description, unit = " batch", leave = False, dynamic_ncols = True)
    bar.gpu_monitor = GPUMonitor.get() if device.type == "cuda" else None
    return bar

# ============================================================
# Cập nhật phần đuôi thanh tiến trình
def update_progress(bar, **values):
    parts = [f"{key} {value:,.4f}" if isinstance(value, float) else f"{key} {value:,}" for key, value in values.items()]
    if bar.gpu_monitor is not None:
        gpu_text = bar.gpu_monitor.text()
        if gpu_text:
            parts.append(gpu_text)
    bar.set_postfix_str(" | ".join(parts), refresh = False)

# ============================================================
# Thời gian hiện tại
def timestamp():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec = "seconds")

# ============================================================
# Ghi Log
class RunLogStream:
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
# In thông số lần chạy
@contextmanager
def capture_run_log(run_directory, *, model_name, dataset_name, train_df, test_df, folds, settings, signature, launch):
    run_directory = Path(run_directory)
    run_directory.mkdir(parents = True, exist_ok = True)
    log_path = run_directory / "run.log"
    resumed_log = log_path.is_file() and log_path.stat().st_size > 0
    old_stdout, old_stderr = sys.stdout, sys.stderr

    with log_path.open("a", encoding = "utf-8", buffering = 1) as file:
        mutex = threading.RLock()
        stdout_mirror = RunLogStream(old_stdout, file, mutex)
        stderr_mirror = RunLogStream(old_stderr, file, mutex)
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
def _print_run_header(run_directory, resumed_log, model_name, dataset_name, train_df, test_df, folds, settings, signature, launch):
    tc = settings["training_config"]

    print("\n" + "=" * config.CONSOLE_WIDTH)
    print(f"[{timestamp()}] {'TIẾP TỤC' if resumed_log else 'BẮT ĐẦU'} HUẤN LUYỆN | Lần chạy thứ: {run_directory.name.removeprefix('run_')}")
    print_message("Model", model_name)
    print_message("Tên tập dữ liệu", dataset_name)
    print_message("Số ảnh", f"Huấn luyện/Kiểm thử: {len(train_df):,} | Kiểm tra: {len(test_df):,}\n")
    
    # Thông tin tập dữ liệu
    print_heading("Cấu hình tập dữ liệu (Ứng với cài đặt trong file config.py)")
    print_message("Tổng số Fold được chia", settings['number_of_folds'])
    print_message("Fold được chọn", folds)
    print_message("Số lớp", len(settings["classes"]))
    print_message("Kích cỡ ảnh", f"{settings['image_size']}\n")

    # Cấu hình huấn luyện
    print_heading("Cấu hình huấn luyện (Ứng với cài đặt trong file config.py)")
    print_message("Batch size", tc["batch_size"])
    print_message("Gradient accum step", tc['grad_accum_steps'])
    print_message("Gradient clip norm", tc['grad_clip_norm'])
    print_message("EMA decay", tc.get('ema_decay'))
    print_message("Number of worker", config.NUMBER_OF_WORKERS)
    print_message("Prefetch factor", config.PREFETCH_FACTOR)
    print_message("Số epoch tối đa", tc["epochs"])
    print_message("Patience", tc['early_stop_patience'])
    print_message("Min delta", tc['min_delta'])
    print_message("Threshold", f"{settings['threshold']}\n")
    
    # Cấu hình hàm huấn luyện
    print_heading("Cấu hình hàm huấn luyện (Ứng với cài đặt trong file config.py)")
    print_message("Hàm Loss", settings["loss_config"])
    print_message("Hàm Optimizer", tc["optimizer"])
    print_message("Hàm Scheduler", tc["scheduler"])
    print_message("Tiêu chí chọn mô hình", f"{tc['selection_metric']}\n")
    
    # Hàm làm giàu ảnh
    print_heading("Cấu hình hàm làm giàu ảnh (Ứng với cài đặt trong file config.py)")
    for key, value in settings["augmentation"].items():
        print_message(key, value)
    print()
    
    # Cấu hình biến ngẫu nhiên
    print_heading("Cấu hình biến ngẫu nhiên (Ứng với cài đặt trong file config.py)")
    print_message("Seed", settings["seed"])
    print_message("Use AMP", settings["use_amp"])
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
    print_message("Độ trễ suy luận (batch 1)", f"{specification['inference_latency_ms_batch_1']:.2f} ms ({specification['inference_precision']})")
    print_message(f"Throughput (batch {specification['inference_batch_size']})", f"{specification['inference_images_per_second']:,.0f} ảnh/s")

# ============================================================
# Tóm tắt chỉ số Test trên một dòng (chi tiết nằm trong test_summary.json)
def print_test_summary(title, summary):
    ci = summary.get("macro_auroc_ci95")
    ci_text = f" [95% CI {ci[0]:.4f}, {ci[1]:.4f}]" if ci and None not in ci else ""
    loss = summary.get("test_loss")
    
    
    print(f"\n === Kết quả {title} ===")
    
    print_message("Threshold", f"{summary['threshold']:.4f}")
    
    if loss is not None:
        print_message("Loss", f"{loss}")
            
    print_message("Macro AUROC", f"{summary['macro_auroc']}")       
    print_message("Macro AP", f"{summary['macro_ap']}")       
    print_message("Macro Precision", f"{summary['macro_precision']}")        
    print_message("Macro Recall", f"{summary['macro_recall']}")     
    print_message("Macro F1", f"{summary['macro_f1']}")
    print_message("Micro F1", f"{summary['micro_f1']}")
    print_message("Weighted F1", f"{summary['weighted_f1']}")


# ============================================================
# Tổng kết cuối lượt chạy
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
