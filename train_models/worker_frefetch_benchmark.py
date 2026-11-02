# ============================================================
# Đo tốc độ DataLoader (DataLoader Benchmark)
# Mục tiêu: Tìm số worker (NUMBER_OF_WORKERS) và số batch tải trước (PREFETCH_FACTOR) cho DataLoader nhanh nhất trên máy hiện tại, bằng cách đo thời gian đọc batch của tập huấn luyện thật với các tổ hợp (worker, prefetch).
# ** Chỉ đo tốc độ đọc dữ liệu, không phải tốc độ huấn luyện đầy đủ của mô hình.

# Quy trình (ví dụ máy 8 CPU logic, prefetch thử [2, 4]):
# - Tổ hợp: (0, None), (1, 2), (1, 4), ..., (8, 2), (8, 4); mỗi tổ hợp chạy --repeats lần, lấy trung vị.
# - Khuyến nghị: Tổ hợp ít worker nhất trong phạm vi --tolerance (mặc định 5%) so với tổ hợp nhanh nhất.
# ============================================================

# Chế độ tất định của cuBLAS: Phải đặt trước khi CUDA khởi tạo
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from utilities.data import _seed_worker, transform_setting
from utilities.dataset import NIHChestXrayDataset

# ============================================================
# Số CPU logic tiến trình được phép dùng (ước lượng thận trọng)
# - os.sched_getaffinity() (Linux) và psutil (nếu có cài) cho biết số CPU thực sự được gán cho tiến trình.
# - Windows Python 3.10 không có os.sched_getaffinity(); psutil là tùy chọn.
# Trả về: Số CPU (ít nhất 1)
def detected_worker_limit():

    # Số CPU logic của máy
    count = os.cpu_count() or 1

    # Linux: Số CPU được gán cho tiến trình
    if hasattr(os, "sched_getaffinity"):
        try:
            count = min(count, len(os.sched_getaffinity(0)))
        except (OSError, ValueError):
            pass

    # psutil (nếu có): Số CPU được gán cho tiến trình trên mọi hệ điều hành
    try:
        import psutil
        affinity = psutil.Process().cpu_affinity()
        if affinity:
            count = min(count, len(affinity))
    except (ImportError, AttributeError, OSError, ValueError):
        pass

    return max(1, int(count))

# ============================================================
# Tạo DataLoader chỉ để đo tốc độ đọc dữ liệu
# - Huấn luyện thật dùng persistent workers (khởi động worker một lần mỗi Fold) và EpochSampler.
# - Ở đây mỗi lần thử tạo worker mới (persistent_workers = False) để mọi lần thử đều tính cả thời gian khởi động.
# Trả về: DataLoader
def create_loader(dataset, device, workers, prefetch_factor, seed):

    # Bộ sinh ngẫu nhiên của DataLoader (cùng seed --> cùng thứ tự lấy mẫu)
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size = int(config.BATCH_SIZE),
        shuffle = True,
        num_workers = int(workers),
        prefetch_factor = int(prefetch_factor) if workers > 0 else None,
        pin_memory = (device.type == "cuda"),
        drop_last = False,
        worker_init_fn = _seed_worker,
        generator = generator,
        persistent_workers = False,  # Worker mới cho mỗi lần thử, để mọi lần thử đều tính thời gian khởi động
        timeout = 120 if workers > 0 else 0,
    )

# ============================================================
# Đo một lần thử: Đọc `batches` batch đầu tiên (kèm chép lên GPU nếu có), tách thời gian batch đầu (khởi động worker)
# Trả về: dict số batch đã đo, thời gian batch đầu, thời gian trung bình mỗi batch sau đó, tổng thời gian, ước lượng một epoch
def measure_one(dataset, device, workers, prefetch_factor, batches, seed):

    # Mọi tổ hợp dùng cùng seed và cùng thứ tự lấy mẫu
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    # Tạo DataLoader và chọn số batch đo
    loader = create_loader(dataset, device, workers, prefetch_factor, seed)
    total_batches = len(loader)
    n = min(int(batches), total_batches)

    # Kiểm tra dữ liệu đầu vào
    ## Cần ít nhất 2 batch (tách batch đầu khỏi các batch sau)
    if n < 2:
        raise ValueError(
            "[measure_one]: Cần ít nhất 2 batch để đo.\n"
            f"Nhận được: Số batch = {n}"
        )

    # Khởi tạo biến
    iterator = None
    first_seconds = None

    try:

        # Bắt đầu đo (đồng bộ GPU trước khi bấm giờ)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        iterator = iter(loader)

        # Đọc từng batch (chép lên GPU như lúc huấn luyện), ghi thời gian batch đầu
        for idx in range(n):
            images, targets = next(iterator)
            if device.type == "cuda":
                images = images.to(device, non_blocking = True)
                targets = targets.to(device, dtype = torch.float32,
                                     non_blocking = True)
                if idx == 0:
                    torch.cuda.synchronize()
            if idx == 0:
                first_seconds = time.perf_counter() - start

        # Kết thúc đo (đồng bộ GPU trước khi bấm giờ)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    # Dừng sớm giữa epoch nên phải đóng các worker đã spawn
    finally:
        del iterator
        del loader
        gc.collect()

    # Thời gian mỗi batch ở trạng thái ổn định và ước lượng thời gian một epoch
    steady_sec = max(0.0, (elapsed - first_seconds) / (n - 1))
    epoch_estimate = first_seconds + (total_batches - 1) * steady_sec

    return {
        "batches_measured": n,
        "first_batch_sec": first_seconds,
        "steady_sec_per_batch": steady_sec,
        "total_sec": elapsed,
        "batches_per_sec": n / elapsed,
        "estimated_epoch_sec": epoch_estimate,
    }

# ============================================================
# Hiển thị prefetch_factor (0 worker không có prefetch --> "None")
# Trả về: Chuỗi hiển thị
def _display_prefetch(value):
    return "None" if pd.isna(value) else str(int(value))

# ============================================================
# Chạy toàn bộ phép đo: Mọi tổ hợp (worker, prefetch) x số lần lặp, lưu kết quả thô và bảng tổng hợp, in khuyến nghị
# Trả về: Bảng tổng hợp (trung vị theo tổ hợp, sắp xếp theo thời gian ước lượng một epoch)
def benchmark(args):

    # 1. Kiểm tra dữ liệu đầu vào
    ## Fold >= 1, batches >= 2, repeats >= 1
    if args.fold < 1 or args.batches < 2 or args.repeats < 1:
        raise ValueError(
            "[benchmark]: Cần Fold >= 1, batches >= 2 và repeats >= 1.\n"
            f"Nhận được: Fold = {args.fold}, batches = {args.batches}, repeats = {args.repeats}"
        )

    ## tolerance nằm trong khoảng [0, 1)
    if not 0 <= args.tolerance < 1:
        raise ValueError(
            "[benchmark]: tolerance phải nằm trong khoảng [0, 1).\n"
            f"Nhận được: tolerance = {args.tolerance}"
        )

    ## prefetch là các số nguyên dương
    if not args.prefetch or any(value < 1 for value in args.prefetch):
        raise ValueError(
            "[benchmark]: prefetch phải là các số nguyên dương.\n"
            f"Nhận được: prefetch = {args.prefetch}"
        )

    ## max-workers không âm
    system_limit = detected_worker_limit()
    if args.max_workers is not None and args.max_workers < 0:
        raise ValueError(
            "[benchmark]: max-workers phải lớn hơn hoặc bằng 0.\n"
            f"Nhận được: max-workers = {args.max_workers}"
        )

    # 2. Các tổ hợp cần đo: (0 worker, không prefetch) + mọi (worker, prefetch)
    max_workers = min(system_limit, args.max_workers) if args.max_workers is not None else system_limit
    prefetch_values = sorted(set(int(value) for value in args.prefetch))
    combinations = [(0, None)] + [
        (w, p)
        for w in range(1, max_workers + 1)
        for p in prefetch_values
    ]

    # 3. Đọc tập dữ liệu
    with Path(config.DATASET_INFO_DIRECTORY).open("r", encoding = "utf-8") as f:
        info = json.load(f)
    classes = list(info["classes"])
    df = pd.read_csv(config.TRAIN_FOLDS_DIRECTORY)

    ## train_folds.csv phải có đủ cột
    missing = {"image_path", "Fold", *classes} - set(df.columns)
    if missing:
        raise ValueError(
            "[benchmark]: train_folds.csv thiếu cột bắt buộc.\n"
            f"Thiếu cột: {sorted(missing)}"
        )

    ## Fold phải có trong train_folds.csv
    if not df["Fold"].eq(args.fold).any():
        raise ValueError(
            f"[benchmark]: Không tìm thấy Fold {args.fold} trong train_folds.csv.\n"
            f"Nhận được: Các Fold = {sorted(int(fold) for fold in df['Fold'].unique())}"
        )

    ## Tập huấn luyện của Fold (bỏ Fold Validation)
    train_df = df.loc[df["Fold"].ne(args.fold)].reset_index(drop = True)

    ## Tập huấn luyện của Fold không được rỗng
    if train_df.empty:
        raise ValueError(
            "[benchmark]: Không còn ảnh huấn luyện sau khi bỏ Fold Validation.\n"
            f"Nhận được: Fold = {args.fold}"
        )

    # 4. Chuẩn bị môi trường đo
    ## Cùng đường đi dữ liệu như train_one_fold(): Cùng biến đổi ảnh và lớp Dataset
    ## Đặt seed như utilities.setting.set_seed(), không import các công cụ ghi log không cần thiết
    random.seed(int(config.SEED))
    np.random.seed(int(config.SEED))
    torch.manual_seed(int(config.SEED))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.SEED))
    torch.backends.cudnn.deterministic = bool(config.CUDNN_DETERMINISTIC)
    torch.backends.cudnn.benchmark = bool(config.CUDNN_BENCHMARK)
    torch.use_deterministic_algorithms(bool(config.DETERMINISTIC_ALGORITHMS))
    transform, _ = transform_setting()
    dataset = NIHChestXrayDataset(dataframe = train_df, classes = classes,
                                 transform = transform)

    ## Khởi tạo CUDA trước khi đo (không tính thời gian khởi tạo vào lần thử đầu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.empty(1, device = device)
        torch.cuda.synchronize()

    ## Đường dẫn tệp kết quả (tương đối theo thư mục của script)
    root = Path(__file__).resolve().parent
    raw_path = root / args.csv
    summary_path = root / args.summary_csv
    raw_path.parent.mkdir(parents = True, exist_ok = True)
    summary_path.parent.mkdir(parents = True, exist_ok = True)

    ## In cấu hình phép đo
    print(f"Thiết bị                  : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"Số CPU logic (hệ điều hành): {os.cpu_count()}")
    print(f"Số CPU được dùng          : {system_limit}")
    print(f"Số worker                 : 0..{max_workers}")
    print(f"Prefetch (worker > 0)     : {prefetch_values}")
    print(f"Số tổ hợp                 : {len(combinations)}")
    print(f"Tổng số lần thử           : {len(combinations) * args.repeats}")
    print(f"Fold / số ảnh             : {args.fold} / {len(dataset):,}")
    print(f"Batch size                : {config.BATCH_SIZE}")
    print(f"Số batch mỗi lần thử      : {min(args.batches, (len(dataset) + config.BATCH_SIZE - 1) // config.BATCH_SIZE)}")
    print(f"Số lần lặp                : {args.repeats}\n", flush = True)

    # 5. Đo mọi tổ hợp
    rows = []
    base_seed = int(config.SEED) + args.fold * 100_000 + 1

    try:
        for trial in range(1, args.repeats + 1):

            ## Đảo thứ tự tổ hợp ở mỗi lượt chẵn để giảm thiên lệch do bộ đệm đĩa / thứ tự chạy
            order = combinations if trial % 2 else list(reversed(combinations))
            for idx, (workers, prefetch) in enumerate(order, start = 1):
                print(f"[Lượt {trial}/{args.repeats} | {idx}/{len(order)}] "
                      f"workers={workers}, prefetch={prefetch}", flush = True)

                ### Kết quả của lần thử
                record = {
                    "trial": trial,
                    "workers": workers,
                    "prefetch_factor": prefetch,
                    "status": "ok",
                    "error": "",
                }

                ### Đo (lỗi --> ghi lại lỗi, chuyển sang tổ hợp kế tiếp)
                try:
                    record.update(measure_one(
                        dataset, device, workers, prefetch,
                        args.batches, base_seed,
                    ))
                    print(f"    {record['total_sec']:.2f}s | "
                          f"{record['batches_per_sec']:.2f} batch/s | "
                          f"ước lượng epoch = {record['estimated_epoch_sec']:.1f}s", flush = True)
                except (OSError, RuntimeError, ValueError, MemoryError) as exc:
                    record.update(status = "error", error = f"{type(exc).__name__}: {exc}")
                    print(f"    LỖI: {record['error']}", flush = True)

                    # Giải phóng bộ nhớ đệm của GPU sau một lần chép lên GPU bị lỗi
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                ### Lưu kết quả sau mỗi lần thử: Bị ngắt giữa chừng vẫn giữ các lần thử trước đó
                rows.append(record)
                pd.DataFrame(rows).to_csv(raw_path, index = False)

    ## Ctrl+C: Giữ kết quả một phần đã lưu
    except KeyboardInterrupt:
        print(f"\nĐã dừng. Kết quả một phần đã lưu: {raw_path}")
        raise

    # 6. Bảng tổng hợp: Trung vị theo tổ hợp (chỉ các lần thử thành công)
    raw = pd.DataFrame(rows)
    passed = raw[raw["status"] == "ok"].copy()

    ## Phải có ít nhất một lần thử thành công
    if passed.empty:
        raise RuntimeError(
            "[benchmark]: Mọi tổ hợp đều lỗi.\n"
            f"Xem chi tiết: {raw_path}"
        )

    ## dropna = False: Giữ tổ hợp workers = 0 / prefetch_factor = None
    summary = passed.groupby(
        ["workers", "prefetch_factor"], dropna = False, as_index = False,
    ).agg(
        successful_runs = ("total_sec", "size"),
        median_total_sec = ("total_sec", "median"),
        median_batch_per_sec = ("batches_per_sec", "median"),
        median_first_batch_sec = ("first_batch_sec", "median"),
        median_epoch_sec = ("estimated_epoch_sec", "median"),
    ).sort_values("median_epoch_sec", ascending = True)
    summary.to_csv(summary_path, index = False)

    ## In bảng tổng hợp
    print("\n--- KẾT QUẢ (median_epoch_sec nhỏ nhất là nhanh nhất) ---")
    print(summary.to_string(index = False, float_format = lambda x: f"{x:.2f}"))

    # 7. Khuyến nghị: Trong các tổ hợp chạy đủ mọi lần lặp, chọn tổ hợp ít worker nhất trong phạm vi tolerance so với nhanh nhất
    complete = summary[summary["successful_runs"] == args.repeats]
    if complete.empty:
        print("\nKhông có tổ hợp nào chạy đủ mọi lần lặp; không đưa ra khuyến nghị.")
    else:
        fastest = float(complete["median_epoch_sec"].min())
        near_best = complete[
            complete["median_epoch_sec"] <= fastest * (1 + args.tolerance)
        ].sort_values(["workers", "prefetch_factor"], na_position = "first")
        choice = near_best.iloc[0]
        print(f"\nKHUYẾN NGHỊ NUMBER_OF_WORKERS : {int(choice['workers'])}")
        print(f"KHUYẾN NGHỊ PREFETCH_FACTOR   : {_display_prefetch(choice['prefetch_factor'])}")
        print(f"Trong phạm vi {args.tolerance:.0%} so với ước lượng epoch nhanh nhất.")

    # In đường dẫn kết quả và lưu ý
    print(f"\nKết quả thô  : {raw_path}")
    print(f"Bảng tổng hợp: {summary_path}")
    print("Chỉ ước lượng tốc độ đọc dữ liệu, không phải tốc độ huấn luyện đầy đủ của mô hình.")
    print("Áp dụng khuyến nghị: Đổi NUMBER_OF_WORKERS / PREFETCH_FACTOR trong config.py.")
    return summary

# ============================================================
# Dòng lệnh: Đọc tham số và chạy phép đo
# Trả về: Bảng tổng hợp của benchmark
def main():

    # Đọc tham số dòng lệnh
    parser = argparse.ArgumentParser(description = "Đo tốc độ DataLoader theo số worker / prefetch")
    parser.add_argument("--max-workers", type = int, default = None,
                        help = "Số worker tối đa cần thử. Mặc định: mọi CPU logic được dùng.")
    parser.add_argument("--prefetch", type = int, nargs = "+", default = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--batches", type = int, default = 200)
    parser.add_argument("--repeats", type = int, default = 5)
    parser.add_argument("--fold", type = int, default = 1)
    parser.add_argument("--tolerance", type = float, default = 0.05)
    parser.add_argument("--csv", type = str, default = "worker_benchmark_raw.csv")
    parser.add_argument("--summary-csv", type = str, default = "worker_benchmark_summary.csv")

    # Chạy phép đo
    return benchmark(parser.parse_args())

if __name__ == "__main__":

    # Cần cho multiprocessing khi đóng gói thành tệp chạy (.exe) trên Windows
    from multiprocessing import freeze_support
    freeze_support()
    main()
