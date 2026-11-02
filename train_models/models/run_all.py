# ============================================================
# Hàng đợi huấn luyện các mô hình so sánh (Training Queue)
# Mục tiêu: Chạy lần lượt các mô hình so sánh với cùng giao thức (EXPERIMENT_PROTOCOL).
# ** Mỗi (mô hình, Fold) chạy trong một tiến trình con riêng: Lỗi (CUDA, hết bộ nhớ, driver) chỉ làm hỏng việc đó và được chạy lại từ checkpoint.

# Quy trình (ví dụ python models/run_all.py densenet121 sym4 --folds 1 2):
# - Hàng đợi 4 việc: (densenet121, Fold 1), (densenet121, Fold 2), (sym4, Fold 1), (sym4, Fold 2).
# - Việc lỗi: Chờ --retry-wait giây rồi chạy lại, tối đa --retries lần; 3 việc liên tiếp đều lỗi --> dừng hàng đợi.
# ============================================================

# Khai báo utilities: Thêm thư mục train_models vào sys.path để chạy trực tiếp bằng python models/run_all.py
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Chế độ tất định của cuBLAS: Phải đặt trước khi CUDA khởi tạo
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import argparse
import json
import subprocess
import time

import config
import models.cnn_symmamba as symmamba
import models.densenet121 as densenet
from training.experiment import start_train
from utilities.dataset import NIHChestXrayDataset
from utilities.log import timestamp
from utilities.setting import keep_system_awake
from utilities.tracking import git_provenance

# ============================================================
# Các mô hình so sánh: Tên --> (tên mô hình / thư mục kết quả, lớp mô hình, tham số mô hình)
# Thứ tự khai báo là thứ tự chạy mặc định, quan trọng nhất trước:
#   densenet121 : baseline
#   sym4        : mô hình đề xuất (quét đối xứng)
#   none        : không trộn token          --> khối Mamba có đóng góp không
#   cross4      : quét kiểu VMamba          --> quét đối xứng có hơn quét thông thường không
#   random4     : 4 hoán vị ngẫu nhiên      --> thứ tự quét có quan trọng không
#   transformer : self-attention cùng số tham số --> vì sao chọn Mamba
#   <tên>_noskip: cùng biến thể nhưng bỏ đường tắt GAP --> khối trộn token đóng góp bao nhiêu khi không có nhánh DenseNet121
VARIANT_ORDER = ("sym4", "none", "cross4", "random4", "transformer")
EXPERIMENTS = {
    "densenet121"   : (densenet.MODEL_NAME, densenet.DenseNetBaseline, {}),
    **{variant: (symmamba.model_name(variant), symmamba.CNNSymMamba, symmamba.VARIANTS[variant])
       for variant in VARIANT_ORDER + tuple(f"{name}_noskip" for name in VARIANT_ORDER)},
}

QUEUE_LOG = config.MODELS_OUTPUT_DIRECTORY / "queue.log"                    # Nhật ký hàng đợi (ghi nối tiếp qua mọi lần chạy)
MAX_CONSECUTIVE_FAILURES = 3                                                # Số việc lỗi liên tiếp tối đa trước khi dừng hàng đợi

# ============================================================
# Ghi một dòng ra console và queue.log (nhật ký hàng đợi, ghi nối tiếp qua mọi lần chạy)
def log(message):

    # Thêm thời gian vào đầu dòng và in ra console
    line = f"[{timestamp()}] {message}"
    print(line, flush = True)

    # Ghi nối tiếp vào queue.log
    QUEUE_LOG.parent.mkdir(parents = True, exist_ok = True)
    with QUEUE_LOG.open("a", encoding = "utf-8") as file:
        file.write(line + "\n")

# ============================================================
# Chọn danh sách Fold: --folds, --official (mọi Fold), hoặc theo config.py (TRAIN_ALL_FOLD / SELECTED_FOLD)
# Trả về: Danh sách Fold cần chạy
def queue_folds(args):

    # Lấy số Fold của tập dữ liệu
    number_of_folds = int(json.loads(config.DATASET_INFO_DIRECTORY.read_text(encoding = "utf-8"))["number_of_folds"])

    # Chọn Fold theo thứ tự ưu tiên: --folds --> --official / TRAIN_ALL_FOLD --> SELECTED_FOLD
    if args.folds:
        folds = args.folds
    elif args.official or config.TRAIN_ALL_FOLD:
        folds = list(range(1, number_of_folds + 1))
    else:
        folds = [int(fold) for fold in config.SELECTED_FOLD] if isinstance(config.SELECTED_FOLD, (list, tuple)) else [int(config.SELECTED_FOLD)]

    # Kiểm tra dữ liệu đầu vào
    ## Fold phải nằm trong khoảng [1, number_of_folds] và không trùng lặp
    if any(fold not in range(1, number_of_folds + 1) for fold in folds) or len(set(folds)) != len(folds):
        raise SystemExit(
            f"[queue_folds]: Fold phải nằm trong khoảng [1, {number_of_folds}] và không trùng lặp.\n"
            f"Nhận được: folds = {folds}"
        )

    return folds

# ============================================================
# Chạy một (mô hình, Fold) trong tiến trình con riêng (lỗi CUDA / hết bộ nhớ / treo driver chỉ làm hỏng việc đó)
# - Lỗi --> chờ rồi chạy lại, lượt chạy tự tiếp tục từ last_checkpoint.pt; hết số lần thử --> chuyển sang việc kế tiếp.
# Trả về: (trạng thái "ok" | "failed", số lần đã chạy, số phút của lần chạy cuối)
def run_job(name, fold, args):

    # Lệnh chạy tiến trình con
    command = [sys.executable, str(Path(__file__).resolve()), "--child", name, "--fold", str(fold)] + (["--official"] if args.official else [])

    # Chạy, lỗi thì chạy lại (tối đa retries lần)
    for attempt in range(1, args.retries + 2):

        # Chạy tiến trình con và đo thời gian
        log(f"BẮT ĐẦU {name} | Fold {fold} | lần {attempt}")
        started = time.time()
        code = subprocess.run(command, cwd = ROOT_DIR).returncode
        minutes = (time.time() - started) / 60

        # Thành công
        if code == 0:
            log(f"XONG    {name} | Fold {fold} | {minutes:.1f} phút")
            return "ok", attempt, minutes

        # Lỗi: Chờ rồi chạy lại (nếu còn lượt)
        log(f"LỖI     {name} | Fold {fold} | lần {attempt} | mã thoát {code} | {minutes:.1f} phút (chi tiết trong run.log của lượt chạy)")
        if attempt <= args.retries:
            time.sleep(args.retry_wait)

    return "failed", args.retries + 1, minutes

# ============================================================
# Tiến trình con: Huấn luyện một Fold của một mô hình (cùng giao thức EXPERIMENT_PROTOCOL)
def run_child(name, fold, official):

    # Chỉ chạy đúng Fold được giao; lượt chạy chính thức bắt buộc git sạch
    config.TRAIN_ALL_FOLD = False
    config.SELECTED_FOLD = [int(fold)]
    if official:
        config.REQUIRE_CLEAN_GIT = True

    # Huấn luyện mô hình
    model_name, model_factory, model_kwargs = EXPERIMENTS[name]
    start_train(model_name = model_name, dataset_class = NIHChestXrayDataset, model_factory = model_factory, model_kwargs = model_kwargs, **config.EXPERIMENT_PROTOCOL)

# ============================================================
# Chạy thí nghiệm. Ví dụ:
#   python models/run_all.py                                              --> toàn bộ, theo thứ tự khai báo, Fold theo config.py
#   python models/run_all.py densenet121 none cross4 random4 transformer  --> các mô hình được chọn, theo thứ tự nhập
#   python models/run_all.py sym4_noskip none_noskip cross4_noskip        --> ablation bỏ đường tắt GAP
#   python models/run_all.py --skip sym4                                  --> toàn bộ trừ các mô hình đã chạy
#   python models/run_all.py --official                                   --> kết quả chính thức: đủ mọi Fold, bắt buộc git sạch
#   python models/run_all.py --folds 2 3 --order fold                     --> Fold 2 của mọi mô hình, rồi Fold 3
#   python models/run_all.py --official --dry-run                         --> chỉ in danh sách việc
# Chạy lại đúng lệnh sau khi bị ngắt: Việc đã xong được bỏ qua, việc dở chạy tiếp từ checkpoint
if __name__ == "__main__":

    # Đọc tham số dòng lệnh
    parser = argparse.ArgumentParser(description = "Hàng đợi huấn luyện các mô hình so sánh với cùng giao thức")
    parser.add_argument("names", nargs = "*", default = list(EXPERIMENTS), help = f"Một hoặc nhiều mô hình trong {list(EXPERIMENTS)}")
    parser.add_argument("--skip", nargs = "+", default = [], help = "Bỏ qua các mô hình")
    parser.add_argument("--folds", nargs = "+", type = int, default = None, help = "Các Fold cần chạy (mặc định theo config.py)")
    parser.add_argument("--official", action = "store_true", help = "Lượt chạy chính thức: mọi Fold, từ chối chạy khi còn thay đổi chưa commit")
    parser.add_argument("--order", choices = ("model", "fold"), default = "model", help = "model: xong mọi Fold của một mô hình rồi mới sang mô hình sau | fold: chạy từng Fold cho mọi mô hình")
    parser.add_argument("--retries", type = int, default = 2, help = "Số lần chạy lại một việc bị lỗi")
    parser.add_argument("--retry-wait", type = int, default = 60, help = "Số giây chờ trước khi chạy lại")
    parser.add_argument("--dry-run", action = "store_true", help = "Chỉ in danh sách việc")
    parser.add_argument("--child", default = None, help = argparse.SUPPRESS)
    parser.add_argument("--fold", type = int, default = None, help = argparse.SUPPRESS)
    args = parser.parse_args()

    # Tiến trình con: Chạy đúng một (mô hình, Fold) rồi thoát
    if args.child is not None:
        run_child(args.child, args.fold, args.official)
        sys.exit(0)

    # Kiểm tra dữ liệu đầu vào
    ## Tên mô hình phải hợp lệ
    unknown = [name for name in args.names + args.skip if name not in EXPERIMENTS]
    if unknown:
        parser.error(f"Mô hình không hợp lệ: {unknown}. Chọn trong {list(EXPERIMENTS)}")

    ## Số lần chạy lại và thời gian chờ không âm
    if args.retries < 0 or args.retry_wait < 0:
        parser.error(f"--retries và --retry-wait phải lớn hơn hoặc bằng 0. Nhận được: retries = {args.retries}, retry-wait = {args.retry_wait}")

    # Danh sách việc: (mô hình, Fold) theo thứ tự --order
    names = [name for name in args.names if name not in args.skip]
    folds = queue_folds(args)
    jobs = ([(name, fold) for name in names for fold in folds] if args.order == "model"
            else [(name, fold) for fold in folds for name in names])

    # Chỉ in danh sách việc
    if args.dry_run:
        for index, (name, fold) in enumerate(jobs, start = 1):
            print(f"{index:3d}. {name} | Fold {fold}")
        sys.exit(0)

    # Kết quả chính thức chỉ chạy từ mã đã commit: Kiểm tra một lần trước khi bắt đầu hàng đợi
    if args.official:
        git = git_provenance()
        if git["git_dirty"] is not False:
            raise SystemExit(f"--official cần git sạch. Tệp chưa commit: {git['git_dirty_files'][:10]}")

    # Giữ máy chạy liên tục
    keep_system_awake()

    # Chạy hàng đợi
    log(f"HÀNG ĐỢI {len(jobs)} việc | {'chính thức' if args.official else 'thử nghiệm'} | thứ tự theo {args.order} | mô hình {names} | Fold {folds}")
    summary = []

    try:

        # Chạy từng việc
        for name, fold in jobs:
            summary.append((name, fold, *run_job(name, fold, args)))

            # Nhiều việc liên tiếp đều lỗi: Lỗi hệ thống (driver, dữ liệu, mã nguồn) --> dừng thay vì chạy thử hết hàng đợi
            if len(summary) >= MAX_CONSECUTIVE_FAILURES and all(record[2] != "ok" for record in summary[-MAX_CONSECUTIVE_FAILURES:]):
                log(f"DỪNG: {MAX_CONSECUTIVE_FAILURES} việc liên tiếp bị lỗi, cần kiểm tra trước khi chạy tiếp")
                break

    # Ctrl+C: Ghi nhật ký rồi thoát (chạy lại cùng lệnh để tiếp tục)
    except KeyboardInterrupt:
        log("HÀNG ĐỢI BỊ DỪNG (Ctrl+C). Chạy lại cùng lệnh để tiếp tục")
        raise SystemExit(130)

    # Tổng kết hàng đợi: Mã thoát 0 khi mọi việc đều xong
    failed = [(name, fold) for name, fold, status, *_ in summary if status != "ok"]
    log(f"KẾT THÚC HÀNG ĐỢI | xong {len(summary) - len(failed)}/{len(jobs)} | tổng {sum(record[-1] for record in summary) / 60:.1f} giờ"
        + (f" | lỗi: {failed}" if failed else ""))
    sys.exit(0 if len(summary) == len(jobs) and not failed else 1)
