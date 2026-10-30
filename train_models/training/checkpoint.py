import random
from pathlib import Path

import numpy as np
import torch

from utilities.file import load_torch, read_json, save_json, save_torch

# ============================================================
# Lưu trạng thái sinh ngẫu nhiên
def capture_rng():
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "python"    : random.getstate(),
        "numpy"     : (name, keys.tolist(), int(pos), int(has_gauss), float(cached)),
        "torch"     : torch.get_rng_state(),
        "cuda"      : torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

# ============================================================
# Khôi phục trạng thái sinh ngẫu nhiên
def restore_rng(state):
    if not state:
        raise ValueError("Checkpoint không chứa trạng thái sinh ngẫu nhiên")

    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype = np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())

    # GPU phải khớp với lần chạy trước
    if torch.cuda.is_available():
        cuda = state.get("cuda")
        if cuda is None or len(cuda) != torch.cuda.device_count():
            raise ValueError("Trạng thái ngẫu nhiên CUDA trong checkpoint không khớp cấu hình GPU hiện tại")
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda])

# ============================================================
# Kiểm tra checkpoint
def check_checkpoint(saved, *, signature, fold, status):
    if saved.get("signature") != signature:
        raise ValueError("Checkpoint thuộc cấu hình khác (mô hình, dữ liệu, hàm Loss hoặc cấu hình huấn luyện đã thay đổi)")
    if saved.get("fold") != int(fold) or saved.get("status") != status:
        raise ValueError("Checkpoint không khớp fold hoặc loại checkpoint hiện tại")

# ============================================================
# Lưu checkpoint
def save_training_state(path, *, fold, epoch, model, criterion, optimizer, scheduler, scaler, history, best_score, best_epoch, no_improvement, signature, training_finished, selection_metric, ema = None):
    save_torch(path, {
        "status"                : "fold",
        "fold"                  : int(fold),
        "epoch"                 : int(epoch),
        "signature"             : signature,
        "model_state_dict"      : model.state_dict(),
        "criterion_state_dict"  : criterion.state_dict(),
        "optimizer_state_dict"  : optimizer.state_dict(),
        "scheduler_state_dict"  : scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict"     : scaler.state_dict(),
        "ema_state_dict"        : ema.state_dict() if ema is not None else None,
        "history"               : history,
        "selection_metric"      : str(selection_metric),
        "best_score"            : float(best_score),        # Điểm tốt nhất theo selection_metric
        "best_epoch"            : int(best_epoch),
        "no_improvement"        : int(no_improvement),
        "training_finished"     : bool(training_finished),
        "rng_state"             : capture_rng(),
    })

# ============================================================
# Tải lại mô hình
def resume_training(path, *, model, criterion, optimizer, scheduler, scaler, device, signature, fold, ema = None):
    saved = load_torch(path, device)
    check_checkpoint(saved, signature = signature, fold = fold, status = "fold")

    model.load_state_dict(saved["model_state_dict"], strict = True)
    criterion.load_state_dict(saved["criterion_state_dict"], strict = True)
    optimizer.load_state_dict(saved["optimizer_state_dict"])

    if (scheduler is None) != (saved["scheduler_state_dict"] is None):
        raise ValueError("Cấu hình scheduler không khớp checkpoint")
    if (ema is None) != (saved.get("ema_state_dict") is None):
        raise ValueError("Cấu hình EMA không khớp checkpoint")

    if scheduler is not None:
        scheduler.load_state_dict(saved["scheduler_state_dict"])
    if ema is not None:
        ema.load_state_dict(saved["ema_state_dict"])
    scaler.load_state_dict(saved["scaler_state_dict"])
    restore_rng(saved["rng_state"])

    return saved

# ============================================================
# Lưu mô hình tốt nhất (best_model.pt): chỉ trọng số và thông tin mô tả, dùng cho đánh giá / suy luận
def save_best_model(path, *, model_state_dict, fold, epoch, score, validation_macro_auroc, selection_metric,
                    signature, classes, model_name, model_kwargs):
    save_torch(path, {
        "status"                    : "best",
        "fold"                      : int(fold),
        "epoch"                     : int(epoch),
        "signature"                 : signature,
        "selection_metric"          : str(selection_metric),
        "selection_score"           : float(score),
        "validation_macro_auroc"    : float(validation_macro_auroc),
        "model_name"                : model_name,
        "model_kwargs"              : model_kwargs,
        "classes"                   : list(classes),
        "model_state_dict"          : model_state_dict,
    })

# ============================================================
# Cập nhật mô hình tốt nhất
def reconcile_best(last, best_path, *, signature, fold, classes, model_name, model_kwargs):
    best_epoch = int(last["best_epoch"])
    if best_epoch == 0:
        raise ValueError("Checkpoint chưa ghi nhận epoch tốt nhất nào")

    # best_model.pt đã đúng epoch --> không cần làm gì
    if Path(best_path).exists():
        best = load_torch(best_path, "cpu")
        check_checkpoint(best, signature = signature, fold = fold, status = "best")
        if int(best["epoch"]) == best_epoch:
            return
        if int(best["epoch"]) > best_epoch:
            raise ValueError("best_model.pt mới hơn checkpoint cuối")

    # Chỉ khôi phục được khi epoch tốt nhất chính là epoch của checkpoint cuối
    if best_epoch != int(last["epoch"]):
        raise ValueError("Không thể khôi phục trọng số tốt nhất từ checkpoint cuối")

    # Có EMA thì mô hình được đánh giá (và cần lưu) là trọng số EMA
    validation_auroc = next((row["macro_auroc"] for row in last["history"] if int(row["epoch"]) == best_epoch), last["best_score"])
    ema_state = last.get("ema_state_dict")
    model_state_dict = ema_state["module_state_dict"] if ema_state is not None else last["model_state_dict"]
    save_best_model(best_path, model_state_dict = model_state_dict, fold = fold, epoch = best_epoch, score = last["best_score"], validation_macro_auroc = validation_auroc, selection_metric = last["selection_metric"], signature = signature, classes = classes, model_name = model_name, model_kwargs = model_kwargs)

# ============================================================
# Chọn lượt chạy
def choose_run(result_directory, signature, resume, settings):
    result_directory.mkdir(parents = True, exist_ok = True)
    runs = sorted((p for p in result_directory.glob("run_*") if p.is_dir() and p.name[4:].isdigit()), key = lambda p: int(p.name[4:]))

    if resume and runs:
        latest = runs[-1]
        manifest_path = latest / "train_info.json"

        # Lượt chạy dở cùng chữ ký --> chạy tiếp
        if manifest_path.exists():
            info = read_json(manifest_path)
            if not info.get("train_completed", False):
                if info.get("signature") == signature:
                    print(f"Huấn luyện tiếp lượt chạy: {latest}")
                    return latest
                print("Phát hiện cấu hình thay đổi so với lượt chạy dở gần nhất")

        # Thư mục rỗng --> dùng lại
        elif not any(latest.iterdir()):
            save_json(manifest_path, {"signature": signature, "settings": settings, "train_completed": False})
            return latest
        else:
            print(f"Không tìm thấy train_info.json tại {latest}")

    # Tạo lượt chạy mới
    next_number = int(runs[-1].name[4:]) + 1 if runs else 1
    run_path = result_directory / f"run_{next_number}"
    run_path.mkdir(parents = True, exist_ok = False)
    save_json(run_path / "train_info.json", {"signature": signature, "settings": settings, "train_completed": False})
    print("Bắt đầu lượt huấn luyện mới")
    return run_path
