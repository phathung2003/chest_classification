# Khai báo utilities
import sys
from pathlib import Path
from tqdm import tqdm   
from config import MIN_RESULT_WIDTH, MIN_TITLE_WIDTH
import torch
from thop import profile
import config
def print_heading(title):
    print("=" * 100)
    print(title)
    print("=" * 100)

def print_title(title):
    print(title)
    print("-" * 100)
    
def print_result(RESULT, TITLE, MESSAGE):
    print(f"{RESULT:<{MIN_RESULT_WIDTH}} {TITLE:<{MIN_TITLE_WIDTH}}: {MESSAGE}")

def print_message(TITLE, MESSAGE):
    print(f"{TITLE:<{MIN_TITLE_WIDTH}}: {MESSAGE}")

class TqdmComma(tqdm):

    @staticmethod
    def format_meter(n, total, elapsed, **kwargs):

        text = tqdm.format_meter(
            n=n,
            total=total,
            elapsed=elapsed,
            **kwargs
        )

        if total is not None:

            old_value = f"{int(n)}/{int(total)}"
            new_value = f"{int(n):,}/{int(total):,}"
            text = text.replace(old_value, new_value)

        return text

# ============================================================
# One durable, append-only text log per experiment run.
# Keep the existing print_heading / print_message functions unchanged.
import os
import sys
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone


def _log_timestamp():
    # Local timestamp with timezone offset, convenient for audit/research logs.
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


class _RunLogStream:
    """Mirror stdout/stderr to a run log without recording tqdm redraws."""

    def __init__(self, console, log_file, lock):
        self.console = console
        self.log_file = log_file
        self.lock = lock

    def write(self, value):
        if not isinstance(value, str):
            value = str(value)
        with self.lock:
            # Always keep the original terminal/notebook output visible.
            self.console.write(value)
            # tqdm repeatedly redraws progress bars using carriage returns;
            # recording each redraw would create huge, unreadable log files.
            if value and '\r' not in value:
                self.log_file.write(value)
                self.log_file.flush()
                # Durability at completed line boundaries (epoch summaries,
                # warnings and exception lines). Also survives abrupt exit much
                # better than a buffered-only stream.
                if '\n' in value:
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
        return getattr(self.console, 'encoding', 'utf-8')

    @property
    def errors(self):
        return getattr(self.console, 'errors', 'replace')

    def writable(self):
        return True


@contextmanager
def capture_run_log(run_directory, *, model_name, dataset_name, train_df, test_df, folds, include_final, settings):
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    log_path = run_directory / 'run.log'
    resumed_log = log_path.is_file() and log_path.stat().st_size > 0
    old_stdout, old_stderr = sys.stdout, sys.stderr
    
    with log_path.open('a', encoding='utf-8', buffering=1) as file:
        mutex = threading.RLock()
        stdout_mirror = _RunLogStream(old_stdout, file, mutex)
        stderr_mirror = _RunLogStream(old_stderr, file, mutex)
        sys.stdout, sys.stderr = stdout_mirror, stderr_mirror
        
        
        try:
            status = 'TIẾP TỤC HUẤN LUYỆN' if resumed_log else 'BẮT ĐẦU HUẤN LUYỆN'
            run_no = int(run_directory.name.split("_")[-1])
            
            print('\n' + '=' * 100)
            print(f'[{_log_timestamp()}] {status} | Lần chạy thứ: {run_no}')
            print(f'Model: {model_name}')
            print(f"Hàm mất mát sử dụng: {settings['loss_name']}")
            print(f"Tên tập dữ liệu: {dataset_name}")
            print(f"Số lượng dữ liệu: Huấn luyện/Kiểm thử: {len(train_df):,} ảnh | Kiểm tra: {len(test_df):,} ảnh")
            print('=' * 100 + '\n')
            
            print_heading("Cấu hình tập dữ liệu (Ứng với cài đặt trong file config.py)")
            print(f"Tổng số fold được chia          : {settings['number_of_folds']}")
            print(f"Fold được chọn                  : {folds}")
            print(f"Các lớp trong tập dữ liệu       : {settings['classes']}")
            print(f"Kích cỡ ảnh                     : {settings['image_size']}\n")
            
            print_heading("Cấu hình huấn luyện (Ứng với cài đặt trong file config.py)")
            print(f"Huấn luyện mô hình cuối cùng    : {'Có' if include_final else 'Không'}")
            print(f"Batch size                      : {settings['batch_size']}")
            print(f"Patience                        : {settings['patience']}")
            print(f"Number of worker                : {settings['number_of_workers']}")
            
            print(f"Learning rate                   : {settings['learning_rate']}")
            print(f"Weight decay                    : {settings['weight_decay']}")
            print(f"Min delta                       : {settings['min_delta']}")
            print(f"Scheduler patience              : {settings['scheduler_patience']}")
            print(f"Scheduler factor                : {settings['scheduler_factor']}")
            print(f"Threshold                       : {settings['threshold']}\n")
            
            print_heading("Cấu hình làm giàu ảnh (Ứng với cài đặt trong file config.py)")
            print(f"Degrees                         : {settings['degrees']}")
            print(f"Scale                           : {settings['scale']}")
            print(f"Translate                       : {settings['translate']}")
            print(f"Normalize mean                  : {settings['normalize_mean']}")
            print(f"Normalize standard deviation    : {settings['normalize_std']}\n")
            
            print_heading("Cấu hình biến ngẫu nhiên (Ứng với cài đặt trong file config.py)")
            print(f"Seed                            : {settings['seed']}")
            print(f"Use AMP                         : {settings['use_amp']}")
            print(f"Cudnn benchmark                 : {settings['cudnn_benchmark']}")
            print(f"Cudnn deterministic             : {settings['cudnn_deterministic']}")
            print(f"Deterministic algorithms        : {settings['deterministic_algorithms']}\n")
            
            yield log_path
        except BaseException as exc:
            # Important: KeyboardInterrupt also enters this branch.
            print(f'[{_log_timestamp()}] LƯỢT HUẤN LUYỆN BỊ LỖI/TẠM DỪNG: '
                  f'{type(exc).__name__}: {exc}', file=sys.stderr)
            with mutex:
                file.write(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
                file.flush()
                os.fsync(file.fileno())
            raise
        else:
            print(f'[{_log_timestamp()}] HOÀN THÀNH HUẤN LUYỆN | {run_directory.name}')
        finally:
            # Restore streams *before* flushing so even an I/O error cannot
            # leave later console output redirected to a closed file.
            sys.stdout, sys.stderr = old_stdout, old_stderr
            stdout_mirror.flush()
            stderr_mirror.flush()
            with mutex:
                file.flush()
                os.fsync(file.fileno())

def model_specification(model_factory, classes, device, model_kwargs):
    model = model_factory(num_classes=len(classes), **model_kwargs).to(device)
    # Batch size = 1, RGB, 224x224
    input_tensor = torch.randn(config.BATCH_SIZE, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    
    # Tính Parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Tính MAC
    with torch.no_grad():
        macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
        
    # Tính FLOPS
    flops = 2 * macs


    # Thông báo

    print_heading("Thông số mô hình")
    print(f"Parameters                      : {total_params:,}")
    print(f"Trainable Params                : {trainable_params}")
    print(f"MACs                            : {macs:,.0f}")
    print(f"FLOPs                           : {flops:,.0f}")
    
    del model
    