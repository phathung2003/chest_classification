# ============================================================
# Xử lý lưu/lấy mô hình
# Mục tiêu: Lưu / khôi phục đầy đủ trạng thái huấn luyện để chạy tiếp sau khi bị ngắt cho kết quả trùng khớp với chạy liền một mạch, và chọn thư mục lượt chạy theo chữ ký cấu hình.

# Sinh ra hai file lưu mô hình:
# - last_checkpoint.pt: Trạng thái đầy đủ sau mỗi epoch (mô hình, optimizer, scheduler, GradScaler, EMA, lịch sử, trạng thái ngẫu nhiên).
# - best_model.pt: Chỉ trọng số của epoch tốt nhất và thông tin mô tả, dùng cho đánh giá / suy luận.
# ============================================================

import random
from pathlib import Path

import numpy as np
import torch

from utilities.file import load_torch, read_json, save_json, save_torch

# ============================================================
# Lưu trạng thái các bộ sinh ngẫu nhiên
def capture_rng():
    name, keys, pos, has_gauss, cached = np.random.get_state()

    return {
        "python"    : random.getstate(),
        "numpy"     : (name, keys.tolist(), int(pos), int(has_gauss), float(cached)),
        "torch"     : torch.get_rng_state(),
        "cuda"      : torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

# ============================================================
# Khôi phục trạng thái các bộ sinh ngẫu nhiên
def restore_rng(state):
    
    # Kiểm tra dữ liệu đầu vào
    ## Kiểm tra dữ liệu trạng thái sinh ngẫu nhiên
    if not state:
        raise ValueError(
            "[restore_rng]: Checkpoint phải chứa trạng thái sinh ngẫu nhiên.\n"
            f"Nhận được: rng_state = {state!r}"
        )

    # Cài đặt lại các bộ sinh ngẫu nhiên
    random.setstate(state["python"])                                                            # Python
    name, keys, pos, has_gauss, cached = state["numpy"]                                         # NumPy (tách trạng thái)
    np.random.set_state((name, np.array(keys, dtype = np.uint32), pos, has_gauss, cached))      # NumPy
    torch.set_rng_state(state["torch"].cpu())                                                   # PyTorch CPU

    # Kiểm tra có chạy GPU CUDA không
    if torch.cuda.is_available():
        
        # Lấy thông tin trạng thái CUDA
        cuda_state = state.get("cuda")

        # Kiểm tra thông tin về CUDA trong state khớp với CUDA trong GPU hiện tại
        if cuda_state is None or len(cuda_state) != torch.cuda.device_count():
            raise ValueError(
                "[restore_rng]: Trạng thái ngẫu nhiên CUDA trong checkpoint phải khớp cấu hình GPU hiện tại.\n"
                f"Nhận được: Số GPU trong checkpoint = {None if cuda_state is None else len(cuda_state)}. Số GPU hiện tại = {torch.cuda.device_count()}"
            )

        # Cài đặt lại bộ sinh ngẫu nhiên của CUDA
        torch.cuda.set_rng_state_all([device_state.cpu() for device_state in cuda_state])

# ============================================================
# Kiểm tra checkpoint
def check_checkpoint(saved, *, signature, fold, status):
    
    # Kiểm tra chữ ký
    if saved.get("signature") != signature:
        raise ValueError(
            "[check_checkpoint]: Checkpoint thuộc cấu hình khác (mô hình, dữ liệu, hàm Loss hoặc cấu hình huấn luyện đã thay đổi).\n"
            f"Nhận được: Chữ ký checkpoint = {saved.get('signature')}. Chữ ký hiện tại = {signature}"
        )

    # Kiểm tra Fold và thông tin checkpoint
    if saved.get("fold") != int(fold) or saved.get("status") != status:
        raise ValueError(
            "[check_checkpoint]: Checkpoint phải khớp Fold và loại checkpoint hiện tại.\n"
            f"Nhận được: Fold = {saved.get('fold')}, status = {saved.get('status')!r}; cần Fold = {int(fold)}, status = {status!r}"
        )

# ============================================================
# Lưu checkpoint
def save_training_state(path, *, fold, epoch, model, criterion, optimizer, scheduler, scaler, history, best_score, best_epoch, no_improvement, signature, training_finished, selection_metric, ema = None):
    save_torch(path, {
        # Thông tin về lần chạy
        "status"                : "fold",
        "fold"                  : int(fold),
        "epoch"                 : int(epoch),
        "signature"             : signature,
        "training_finished"     : bool(training_finished),
        
        # Thông tin về cấu hình huấn luyện
        "model_state_dict"      : model.state_dict(),
        "criterion_state_dict"  : criterion.state_dict(),
        "optimizer_state_dict"  : optimizer.state_dict(),
        "scheduler_state_dict"  : scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict"     : scaler.state_dict(),
        "ema_state_dict"        : ema.state_dict() if ema is not None else None,
        "selection_metric"      : str(selection_metric),
        "rng_state"             : capture_rng(),
        
        # Thông tin về kết quả chạy
        "history"               : history,
        "best_score"            : float(best_score),                            # Điểm tốt nhất theo selection_metric
        "best_epoch"            : int(best_epoch),
        "no_improvement"        : int(no_improvement),
    })

# ============================================================
# Khôi phục huấn luyện
# Trả về: Nội dung checkpoint (epoch, lịch sử, điểm tốt nhất, ...)
def resume_training(path, *, model, criterion, optimizer, scheduler, scaler, device, signature, fold, ema = None):
    
    # Đọc checkpoint
    saved = load_torch(path, device)
    
    # Kiểm tra checkpoint
    check_checkpoint(saved, signature = signature, fold = fold, status = "fold")

    # Kiểm tra cấu hình
    ## Scheduler có khớp với checkpoint
    if (scheduler is None) != (saved["scheduler_state_dict"] is None):
        raise ValueError(
            "[resume_training]: Cấu hình scheduler phải khớp checkpoint.\n"
            f"Nhận được: Scheduler hiện tại = {scheduler is not None}, scheduler trong checkpoint = {saved['scheduler_state_dict'] is not None}"
        )

    ## EMA có khớp với checkpoint
    if (ema is None) != (saved.get("ema_state_dict") is None):
        raise ValueError(
            "[resume_training]: Cấu hình EMA phải khớp checkpoint.\n"
            f"Nhận được: EMA hiện tại = {ema is not None}, EMA trong checkpoint = {saved.get('ema_state_dict') is not None}"
        )

    # Khôi phục trạng thái
    model.load_state_dict(saved["model_state_dict"], strict = True)             # Trọng số mô hình
    criterion.load_state_dict(saved["criterion_state_dict"], strict = True)     # Hàm tính Loss (Criterion)
    optimizer.load_state_dict(saved["optimizer_state_dict"])                    # Optimizer
    scaler.load_state_dict(saved["scaler_state_dict"])                          # GradScaler (Scaler)
    restore_rng(saved["rng_state"])                                             # Trạng thái sinh ngẫu nhiên

    if scheduler is not None:
        scheduler.load_state_dict(saved["scheduler_state_dict"])                # Scheduler
    
    if ema is not None:
        ema.load_state_dict(saved["ema_state_dict"])                            # EMA
    
    return saved

# ============================================================
# Lưu mô hình tốt nhất
def save_best_model(path, *, model_state_dict, fold, epoch, score, validation_macro_auroc, selection_metric, signature, classes, model_name, model_kwargs):
    save_torch(path, {
        # Thông tin về lần chạy
        "status"                    : "best",
        "model_name"                : model_name,
        "fold"                      : int(fold),
        "epoch"                     : int(epoch),
        "signature"                 : signature,
        "classes"                   : list(classes),
        # Thông tin về cấu hình huấn luyện
        "selection_metric"          : str(selection_metric),
        "selection_score"           : float(score),                             # Điểm tốt nhất theo selection_metric 
        "model_kwargs"              : model_kwargs,
        
        # Thông tin về kết quả chạy
        "model_state_dict"          : model_state_dict,
        "validation_macro_auroc"    : float(validation_macro_auroc),
    })

# ============================================================
# Đồng bộ mô hình tốt nhất với checkpoint
def reconcile_best(last, best_path, *, signature, fold, classes, model_name, model_kwargs):
    best_epoch = int(last["best_epoch"])

    # Kiểm tra dữ liệu đầu vào
    ## Checkpoint phải đã ghi nhận epoch tốt nhất
    if best_epoch == 0:
        raise ValueError(
            "[reconcile_best]: Checkpoint chưa ghi nhận epoch tốt nhất nào.\n"
            f"Nhận được: best_epoch = {best_epoch}"
        )

    # best_model.pt đã đúng epoch --> không cần làm gì
    if Path(best_path).exists():
        best = load_torch(best_path, "cpu")
        check_checkpoint(best, signature = signature, fold = fold, status = "best")

        if int(best["epoch"]) == best_epoch:
            return

        ## best_model.pt không được mới hơn checkpoint cuối
        if int(best["epoch"]) > best_epoch:
            raise ValueError(
                "[reconcile_best]: best_model.pt không được mới hơn checkpoint cuối.\n"
                f"Nhận được: Epoch của best_model.pt = {int(best['epoch'])}. Epoch tốt nhất trong checkpoint = {best_epoch}"
            )

    ## Chỉ khôi phục được khi epoch tốt nhất chính là epoch của checkpoint cuối
    if best_epoch != int(last["epoch"]):
        raise ValueError(
            "[reconcile_best]: Không thể khôi phục trọng số tốt nhất từ checkpoint cuối (epoch tốt nhất khác epoch của checkpoint).\n"
            f"Nhận được: Epoch tốt nhất = {best_epoch}. Epoch của checkpoint = {int(last['epoch'])}"
        )

    # Khôi phục mô hình
    validation_auroc = next((row["macro_auroc"] for row in last["history"] if int(row["epoch"]) == best_epoch), last["best_score"])
    ema_state = last.get("ema_state_dict")
    model_state_dict = ema_state["module_state_dict"] if ema_state is not None else last["model_state_dict"]
    save_best_model(best_path, model_state_dict = model_state_dict, fold = fold, epoch = best_epoch, score = last["best_score"], validation_macro_auroc = validation_auroc, selection_metric = last["selection_metric"], signature = signature, classes = classes, model_name = model_name, model_kwargs = model_kwargs)

# ============================================================
# Chọn thư mục lượt chạy theo chữ ký
# - Có lượt chạy cùng chữ ký (mới nhất trước): Đã hoàn thành --> dùng lại
# - Thư mục lượt chạy mới nhất còn rỗng --> dùng lại.
# - Còn lại: Tạo run_<n+1>.

# Trả về: Đường dẫn thư mục lượt chạy
def choose_run(result_directory, signature, resume, settings):
    
    # Tạo thư mục kết quả (nếu chưa có)
    result_directory.mkdir(parents = True, exist_ok = True)

    # Danh sách các lượt chạy trước đó (Sắp xếp theo số thứ tự tăng dần)
    runs = sorted((path for path in result_directory.glob("run_*") if path.is_dir() and path.name[4:].isdigit()), key = lambda path: int(path.name[4:]))

    # Đã có lần chạy trước đó --> Xử lý chạy tiếp
    if resume and runs:
        
        for run in reversed(runs):
            manifest_path = run / "train_info.json"
            
            # Khớp chữ ký --> Đưa đường dẫn thư mục
            if manifest_path.exists() and read_json(manifest_path).get("signature") == signature:
                completed = read_json(manifest_path).get("train_completed", False)
                print(f"{'Lượt chạy đã hoàn thành' if completed else 'Huấn luyện tiếp lượt chạy'}: {run}")
                return run

        # Lấy lượt chạy mới nhất
        latest = runs[-1]
        
        # Kiểm tra xem không có "train_info.json" --> Folder trắng
        if not (latest / "train_info.json").exists():
            
            # Kiểm tra thư mục có chứa gì không --> Không chứa thì lưu file thông tin lần chạy
            if not any(latest.iterdir()):
                
                # Lưu file thông tin lần chạy
                save_json(latest / "train_info.json", {"signature": signature, "settings": settings, "train_completed": False})
                
                return latest
            
            # Có thư mục khác --> Bị thiếu file --> Chạy lượt mới
            print(f"Không tìm thấy train_info.json tại {latest}")
        
        # Có "train_info.json" --> Không khớp cấu hình --> Chạy lượt mới
        else:
            print("Không có lượt chạy nào cùng cấu hình")

    # Tạo lượt chạy mới
    next_number = int(runs[-1].name[4:]) + 1 if runs else 1         # Tạo số tiếp theo hoặc trả về 1 nếu chưa chạy lần nào
    
    # Tạo thư mục
    run_path = result_directory / f"run_{next_number}"
    run_path.mkdir(parents = True, exist_ok = False)
    
    # Lưu file thông tin lần chạy
    save_json(run_path / "train_info.json", {"signature": signature, "settings": settings, "train_completed": False})
    print("Bắt đầu lượt huấn luyện mới")

    return run_path
