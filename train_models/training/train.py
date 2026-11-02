# ============================================================
# Huấn luyện mô hình (Training)
# Mục tiêu: Huấn luyện từng Fold, đánh giá kiểm thử, lưu checkpoint / mô hình tốt nhất sau mỗi epoch và lưu dự đoán OOF khi kết thúc Fold.
# ============================================================

import gc
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import config
from training.checkpoint import check_checkpoint, reconcile_best, resume_training, save_best_model, save_training_state
from training.ema import ModelEMA
from training.evaluate import evaluate
from training.loss import build_loss
from training.optimizer import make_optimizer, make_scheduler
from utilities.data import create_dataloader
from utilities.file import load_torch, read_json, save_csv, save_json
from utilities.log import GPUMonitor, progress_bar, update_progress
from utilities.metric import class_metrics_dataframe, with_probabilities
from utilities.setting import set_seed

# ============================================================
# Giải phóng bộ nhớ CPU / GPU giữa các Fold
def cleanup():
    gc.collect()                                                                # CPU: Thu gom rác
    if torch.cuda.is_available():
        torch.cuda.empty_cache()                                                # GPU: Trả bộ nhớ đệm của PyTorch

# ============================================================
# Chia dữ liệu cho một Fold (Fold được chọn là Validation, các Fold còn lại là tập huấn luyện)
# Trả về: (train_df, validation_df)
def split_fold(train_folds_df, fold):

    # Chia dữ liệu theo cột Fold
    train_df = train_folds_df.loc[train_folds_df["Fold"] != fold].reset_index(drop = True)
    validation_df = train_folds_df.loc[train_folds_df["Fold"] == fold].reset_index(drop = True)

    # Kiểm tra dữ liệu đầu vào
    ## Tập huấn luyện và kiểm thử của Fold không được rỗng
    if train_df.empty or validation_df.empty:
        raise ValueError(
            f"[split_fold]: Không lấy được dữ liệu huấn luyện / kiểm thử cho Fold {fold}.\n"
            f"Nhận được: Huấn luyện = {len(train_df)} ảnh. Kiểm thử = {len(validation_df)} ảnh"
        )

    return train_df, validation_df

# ============================================================
# Khởi tạo mô hình và hàm Loss
# Trả về: (model, criterion, loss_metadata)
def build_model_and_loss(experiment_context, train_df):

    # Khởi tạo mô hình trên thiết bị huấn luyện
    model = experiment_context.model_factory(num_classes = experiment_context.number_of_classes, **experiment_context.model_kwargs).to(experiment_context.device)

    # Khởi tạo hàm Loss (pos_weight tính trên tập huấn luyện của Fold)
    criterion, loss_metadata = build_loss(experiment_context.loss_config, train_df, experiment_context.classes, experiment_context.device)

    return model, criterion, loss_metadata

# ============================================================
# Khởi tạo optimizer, scheduler và GradScaler (AMP)
# Trả về: (optimizer, scheduler hoặc None, scaler)
def build_optimization(experiment_context, model, criterion):

    # Tham số được tối ưu: Tham số mô hình và tham số học được của hàm Loss (nếu có)
    parameters = list(model.parameters()) + [parameter for parameter in criterion.parameters() if parameter.requires_grad]

    # Optimizer và scheduler
    optimizer = make_optimizer(parameters, experiment_context.training_config["optimizer"])
    scheduler = make_scheduler(optimizer, experiment_context.training_config["scheduler"])

    # GradScaler (chỉ bật khi dùng AMP trên GPU)
    scaler = torch.amp.GradScaler("cuda", enabled = bool(config.USE_AMP and experiment_context.device.type == "cuda"))

    return optimizer, scheduler, scaler

# ============================================================
# Nạp trọng số tốt nhất của Fold (best_model.pt) vào mô hình
# Trả về: Nội dung best_model.pt (epoch, điểm, ...)
def load_best_weights(experiment_context, best_path, model, fold):

    # Đọc best_model.pt
    best = load_torch(best_path, experiment_context.device)

    # Kiểm tra checkpoint (đúng cấu hình, đúng Fold, đúng loại)
    check_checkpoint(best, signature = experiment_context.signature, fold = fold, status = "best")

    # Nạp trọng số
    model.load_state_dict(best["model_state_dict"], strict = True)

    return best

# ============================================================
# Lưu lịch sử epoch
def save_history(fold_directory, history, classes):
    # Thứ tự cột của history.csv
    HISTORY_COLUMNS = [
        "fold", "epoch", "no_improvement",
        "elapsed_seconds", "data_wait_seconds", "peak_gpu_memory_gb", "train_seconds", "train_images_per_second",
        "learning_rate", "grad_norm_mean", "grad_norm_max", "amp_skipped_steps",
        "train_loss", "train_accuracy", "train_auroc",
        "validation_seconds", "validation_loss", "macro_auroc", "macro_ap", "macro_precision", "macro_recall", "macro_f1", "micro_f1", "weighted_f1",
    ]
    
    # Chuyển lịch sử sang DataFrame
    frame = pd.DataFrame(history)

    # Tách các cột AUROC từng lớp (tiền tố "auroc_") ra tệp riêng
    class_columns = [f"auroc_{name}" for name in classes if f"auroc_{name}" in frame]
    epoch_frame = frame.drop(columns = class_columns)

    # Sắp xếp cột
    gpu_columns = [column for column in epoch_frame if column.startswith("gpu_")]
    ordered = ([column for column in HISTORY_COLUMNS if column in epoch_frame]
               + [column for column in epoch_frame if column not in HISTORY_COLUMNS and column not in gpu_columns] + gpu_columns)

    # Lưu history.csv
    save_csv(fold_directory / "history.csv", epoch_frame[ordered])

    # Lưu history_class_auroc.csv
    if class_columns:
        class_frame = frame[["epoch", *class_columns]].rename(columns = lambda column: column.removeprefix("auroc_"))  # Bỏ tiền tố "auroc_" khỏi tên cột
        save_csv(fold_directory / "history_class_auroc.csv", class_frame)

# ============================================================
# Seed của Dataloader cho huấn luyện của một Fold
# Trả về: Seed (số nguyên)
def train_loader_seed(fold):
    return int(config.SEED) + int(fold) * 100_000

# ============================================================
# Kiểm tra độ cải thiện của mô hình so với điểm tốt nhất
# - higher_is_better = True (macro AUROC): Cải thiện khi score > best_score + min_delta.
# - higher_is_better = False (validation loss): Cải thiện khi score < best_score - min_delta.
# Trả về: True nếu cải thiện
def is_improvement(score, best_score, has_best, higher_is_better, min_delta):

    # Chưa có điểm tốt nhất --> Luôn là cải thiện
    if not has_best:
        return True

    # So sánh với điểm tốt nhất
    return score > best_score + min_delta if higher_is_better else score < best_score - min_delta

# ============================================================
# Huấn luyện một epoch
# Trả về: dict thống kê của epoch (loss, accuracy, AUROC, thời gian, chuẩn gradient, số bước AMP bỏ qua)
def train_one_epoch(epoch, device, model, loader, criterion, optimizer, scaler, training_config, ema = None):

    # Lấy dữ liệu
    grad_accum_steps = int(training_config["grad_accum_steps"])                 # Số batch cộng dồn gradient trước mỗi lần cập nhật
    grad_clip_norm = training_config["grad_clip_norm"]                          # Ngưỡng clip chuẩn gradient (None: Không clip)
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]

    # Chuyển mô hình và hàm Loss sang chế độ huấn luyện
    model.train()
    criterion.train()

    # Khởi tạo biến lưu kết quả
    total_loss = torch.zeros((), dtype = torch.float64, device = device)        # Tổng Loss (giữ trên GPU)
    n_samples, accumulated_samples, amp_skipped_steps, data_wait_seconds = 0, 0, 0, 0.0
    truths, logit_batches, grad_norms = [], [], []

    # Xóa gradient cũ
    optimizer.zero_grad(set_to_none = True)

    # Bắt đầu đo thời gian (thời gian chờ dữ liệu = khoảng từ lúc xử lý xong batch trước tới lúc nhận batch sau)
    epoch_started = batch_finished = time.perf_counter()

    # Khởi động thanh tiến trình
    progress = progress_bar(loader, f"Epoch {epoch:02d} [Huấn luyện]", device)

    # Duyệt từng batch
    for step, (images, targets) in enumerate(progress, start = 1):

        # Cộng dồn thời gian chờ dữ liệu
        data_wait_seconds += time.perf_counter() - batch_finished

        # Chuyển dữ liệu lên thiết bị huấn luyện
        images = images.to(device, non_blocking = True)
        targets = targets.to(device, dtype = torch.float32, non_blocking = True)
        batch_size = images.size(0)

        # Forward ở FP16 (AMP), loss tính ở FP32
        with torch.autocast(device_type = device.type, dtype = torch.float16, enabled = bool(config.USE_AMP and device.type == "cuda")):
            logits = model(images)
        loss = criterion(logits.float(), targets)

        # Backward: Cộng dồn gradient theo số mẫu (loss * batch_size), chia lại khi cập nhật --> trung bình theo mẫu
        scaler.scale(loss * batch_size).backward()
        accumulated_samples += batch_size

        # Cập nhật trọng số sau mỗi grad_accum_steps batch (hoặc ở batch cuối)
        if step % grad_accum_steps == 0 or step == len(loader):

            # Bỏ hệ số scale của AMP, chia gradient cho số mẫu đã cộng dồn
            scaler.unscale_(optimizer)
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            torch._foreach_div_(gradients, accumulated_samples)

            # Chuẩn gradient trước khi clip (đo sau unscale nên là giá trị thật)
            grad_norm = torch.nn.utils.get_total_norm(gradients)
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grads_with_norm_(parameters, float(grad_clip_norm), grad_norm)

            # Chuẩn không hữu hạn <=> có gradient tràn số FP16 <=> GradScaler bỏ qua bước cập nhật này
            stepped = bool(torch.isfinite(grad_norm).item())

            # Cập nhật trọng số và hệ số scale của AMP
            scaler.step(optimizer)
            scaler.update()

            # Bước cập nhật thật sự --> ghi chuẩn gradient và cập nhật EMA; bước bị bỏ qua --> đếm
            if stepped:
                grad_norms.append(grad_norm)
                if ema is not None:
                    ema.update(model)
            else:
                amp_skipped_steps += 1

            # Xóa gradient, bắt đầu lượt cộng dồn mới
            optimizer.zero_grad(set_to_none = True)
            accumulated_samples = 0

        # Thống kê huấn luyện (giữ trên GPU)
        total_loss.add_(loss.detach(), alpha = batch_size)
        n_samples += batch_size
        truths.append(targets)
        logit_batches.append(logits.detach())

        # Thanh tiến trình
        if step % config.PROGRESS_SYNC_STEPS == 0 or step == len(loader):
            running_loss = total_loss.item() / n_samples

            # Kiểm tra Loss hữu hạn
            if not math.isfinite(running_loss):
                raise FloatingPointError(
                    f"[train_one_epoch]: Giá trị Loss phải hữu hạn (epoch {epoch}, trước batch {step}).\n"
                    f"Nhận được: Loss trung bình = {running_loss}"
                )

            update_progress(progress, loss = running_loss)

        # Đánh dấu thời điểm xử lý xong batch
        batch_finished = time.perf_counter()
        
    # Kiểm tra tập huấn luyện phải có dữ liệu
    if n_samples == 0:
        raise ValueError(
            "[train_one_epoch]: Tập huấn luyện không có dữ liệu.\n"
            f"Nhận được: Số ảnh = {n_samples}"
        )

    # Thời gian huấn luyện của epoch
    train_seconds = time.perf_counter() - epoch_started

    # Chuyển nhãn và xác suất về CPU (một lần ở cuối epoch)
    y_true = torch.cat(truths).cpu().numpy()
    y_prob = torch.sigmoid(torch.cat(logit_batches).float()).cpu().numpy()

    # AUROC từng lớp (chỉ các lớp có đủ nhãn dương và âm)
    aucs = [roc_auc_score(y_true[:, index], y_prob[:, index]) for index in range(y_true.shape[1]) if np.unique(y_true[:, index]).size == 2]

    # Chuẩn gradient của các bước cập nhật thật sự
    grad_norms = torch.stack(grad_norms).cpu().numpy() if grad_norms else np.array([])

    return {
        # Loss và chỉ số trên tập huấn luyện (accuracy theo nhãn tại ngưỡng mặc định)
        "train_loss"                : total_loss.item() / n_samples,
        "train_accuracy"            : float(((y_prob >= config.THRESHOLD) == y_true.astype(bool)).mean()),
        "train_auroc"               : float(np.mean(aucs)) if aucs else float("nan"),

        # Thời gian
        "train_seconds"             : train_seconds,
        "data_wait_seconds"         : data_wait_seconds,
        "train_images_per_second"   : n_samples / train_seconds,

        # Gradient và AMP
        "grad_norm_mean"            : float(np.mean(grad_norms)) if grad_norms.size else float("nan"),
        "grad_norm_max"             : float(np.max(grad_norms)) if grad_norms.size else float("nan"),
        "amp_skipped_steps"         : amp_skipped_steps,
    }

# ============================================================
# Huấn luyện một Fold
# Trả về: (model đã nạp trọng số tốt nhất, criterion, kết quả của Fold)
def train_one_fold(experiment_context, fold, train_folds_df):

    # Lấy dữ liệu
    fold = int(fold)
    training_config = experiment_context.training_config

    # Thư mục và tệp kết quả của Fold
    fold_directory = experiment_context.fold_directory(fold)
    fold_directory.mkdir(parents = True, exist_ok = True)
    done_path = fold_directory / "fold_done.json"                               # Đánh dấu Fold đã hoàn thành
    last_path = fold_directory / "last_checkpoint.pt"                           # Checkpoint đầy đủ sau mỗi epoch
    best_path = fold_directory / "best_model.pt"                                # Trọng số của epoch tốt nhất

    # Chia dữ liệu huấn luyện / kiểm thử của Fold
    train_df, validation_df = split_fold(train_folds_df, fold)

    # 1. Fold đã hoàn thành
    if done_path.exists():
        done = read_json(done_path)

        ## Fold phải hoàn thành với cùng cấu hình
        if done.get("signature") != experiment_context.signature:
            raise ValueError(
                f"[train_one_fold]: Fold {fold} đã hoàn thành với cấu hình khác.\n"
                f"Nhận được: Chữ ký của Fold = {done.get('signature')}, chữ ký hiện tại = {experiment_context.signature}"
            )

        ## Phải có mô hình tốt nhất của Fold
        if not best_path.is_file():
            raise FileNotFoundError(
                "[train_one_fold]: Thiếu mô hình tốt nhất của Fold đã hoàn thành.\n"
                f"Thiếu: {best_path}"
            )

        ## Nạp mô hình tốt nhất, trả về kết quả cũ
        model, criterion, _ = build_model_and_loss(experiment_context, train_df)
        load_best_weights(experiment_context, best_path, model, fold)
        print(f"Fold {fold} đã hoàn thành trước đó (epoch tốt nhất: {done['result']['best_epoch']})")
        return model, criterion, done["result"]

    print(f"\n=== Fold {fold} | Huấn luyện: {len(train_df):,} ảnh | Kiểm thử: {len(validation_df):,} ảnh ===")

    # 2. Khởi tạo (seed riêng cho từng Fold: Khởi tạo không phụ thuộc vào các Fold chạy trước)
    set_seed(int(config.SEED) + fold)

    ## Mô hình, hàm Loss, optimizer, scheduler và GradScaler
    model, criterion, loss_metadata = build_model_and_loss(experiment_context, train_df)
    optimizer, scheduler, scaler = build_optimization(experiment_context, model, criterion)

    ## EMA (Kiểm thử, early stopping, scheduler và best_model.pt đều dùng trọng số EMA)
    ema = ModelEMA(model, training_config["ema_decay"]) if training_config["ema_decay"] is not None else None
    evaluated_model = model if ema is None else ema.module

    ## In pos_weight của Fold (hàm Loss BCE)
    if "pos_weight" in loss_metadata:
        print(f"pos_weight của Fold: {[round(weight, 3) for weight in loss_metadata['pos_weight']]}")

    ## Trạng thái dừng sớm (early stopping)
    selection_metric = training_config["selection_metric"]
    higher_is_better = selection_metric == "macro_auroc"
    epoch_start, history, best_epoch, no_improvement, finished = 1, [], 0, 0, False
    best_score = -math.inf if higher_is_better else math.inf

    ## Chạy tiếp từ checkpoint cuối
    if last_path.exists():
        saved = resume_training(last_path, model = model, criterion = criterion, optimizer = optimizer, scheduler = scheduler, scaler = scaler, device = experiment_context.device, signature = experiment_context.signature, fold = fold, ema = ema)
        epoch_start, history = int(saved["epoch"]) + 1, saved["history"]
        best_score, best_epoch = float(saved["best_score"]), int(saved["best_epoch"])
        no_improvement, finished = int(saved["no_improvement"]), bool(saved["training_finished"])
        reconcile_best(saved, best_path, signature = experiment_context.signature, fold = fold, classes = experiment_context.classes, model_name = experiment_context.model_name, model_kwargs = experiment_context.model_kwargs)
        print(f"Chạy tiếp Fold {fold} [Epoch đã hoàn thành: {epoch_start - 1} | Epoch tốt nhất: {best_epoch}]")

    ## Có best_model.pt thì phải có last_checkpoint.pt (không thì không chạy tiếp đúng được)
    elif best_path.exists():
        raise RuntimeError(
            "[train_one_fold]: Có best_model.pt nhưng thiếu last_checkpoint.pt.\n"
            f"Nhận được: best_model.pt = {best_path}"
        )

    ## DataLoader: Worker giữ nguyên suốt Fold.
    _, train_loader = create_dataloader(experiment_context.dataset_class, experiment_context.classes, experiment_context.device, train_df, experiment_context.train_transform, True, batch_size = experiment_context.batch_size, seed = train_loader_seed(fold))
    _, validation_loader = create_dataloader(experiment_context.dataset_class, experiment_context.classes, experiment_context.device, validation_df, experiment_context.eval_transform, False, batch_size = experiment_context.batch_size, seed = int(config.SEED))

    # 3. Vòng lặp epoch
    gpu_monitor = GPUMonitor.get() if experiment_context.device.type == "cuda" else None

    for epoch in range(epoch_start, training_config["epochs"] + 1):
        if finished:
            break

        ## Bắt đầu đo thời gian, bộ nhớ GPU đỉnh và thống kê GPU của epoch
        started = time.perf_counter()
        if experiment_context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(experiment_context.device)
        if gpu_monitor is not None:
            gpu_monitor.reset_statistics("epoch")

        ## Huấn luyện
        train_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(epoch, experiment_context.device, model, train_loader, criterion, optimizer, scaler, training_config, ema = ema)

        ## Đánh giá kiểm thử
        validation_started = time.perf_counter()
        validation_loss, metrics, _, _ = evaluate(experiment_context.device, experiment_context.number_of_classes, evaluated_model, validation_loader, criterion, epoch = epoch)
        validation_seconds = time.perf_counter() - validation_started
        auc = float(metrics["macro_auroc"])
        score = auc if higher_is_better else float(validation_loss)

        ### Kiểm tra tiêu chí chọn mô hình phải tính được (hữu hạn)
        if not math.isfinite(score):
            raise ValueError(
                f"[train_one_fold]: Epoch {epoch}: không tính được {selection_metric}.\n"
                f"Nhận được: {selection_metric} = {score}"
            )

        ## Kiểm tra có dừng sớm không (early stopping)
        improved = is_improvement(score, best_score, best_epoch > 0, higher_is_better, training_config["min_delta"])
        if improved:
            best_score, best_epoch, no_improvement = score, epoch, 0
        else:
            no_improvement += 1
        finished = no_improvement >= training_config["early_stop_patience"] or epoch >= training_config["epochs"]

        ## Scheduler theo dõi cùng chỉ số với tiêu chí chọn mô hình ("lr": Tên khóa của torch trong param_groups)
        learning_rate = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(float(validation_loss) if training_config["scheduler"]["mode"] == "min" else auc)
        next_learning_rate = optimizer.param_groups[0]["lr"]

        ## Lịch sử epoch
        history.append({
            "fold"                  : fold,
            "epoch"                 : epoch,
            "elapsed_seconds"       : time.perf_counter() - started,
            **train_stats,
            "validation_seconds"    : validation_seconds,
            "validation_loss"       : validation_loss,
            **{key: value for key, value in metrics.items() if not key.startswith("class_")},
            "learning_rate"         : next_learning_rate,
            "no_improvement"        : no_improvement,
            "peak_gpu_memory_gb"    : (torch.cuda.max_memory_allocated(experiment_context.device) / 1024 ** 3
                                       if experiment_context.device.type == "cuda" else None),
            **{f"auroc_{name}": value for name, value in zip(experiment_context.classes, metrics["class_aurocs"])},
            **(gpu_monitor.epoch_columns() if gpu_monitor is not None else {}),
        })

        ## Lưu checkpoint
        save_training_state(last_path, fold = fold, epoch = epoch, model = model, criterion = criterion, optimizer = optimizer, scheduler = scheduler, scaler = scaler, history = history, best_score = best_score, best_epoch = best_epoch, no_improvement = no_improvement, signature = experiment_context.signature, training_finished = finished, selection_metric = selection_metric, ema = ema)
        
        ## Có cải thiện --> Lưu mô hình tốt nhất
        if improved:
            save_best_model(best_path, model_state_dict = evaluated_model.state_dict(), fold = fold, epoch = epoch, score = score, validation_macro_auroc = auc, selection_metric = selection_metric, signature = experiment_context.signature, classes = experiment_context.classes, model_name = experiment_context.model_name, model_kwargs = experiment_context.model_kwargs)
        
        # Cập nhật lịch sử
        save_history(fold_directory, history, experiment_context.classes)

        ## In kết quả huấn luyện của Epoch
        data_share = train_stats["data_wait_seconds"] / max(train_stats["train_seconds"], 1e-9)
        print(f"Epoch: {epoch:02d} | Huấn luyện: [Loss: {train_stats['train_loss']:.4f}, Macro AUROC: {train_stats['train_auroc']:.4f}] | "
              f"Kiểm thử: [Loss: {validation_loss:.4f}, Macro AUROC: {auc:.4f}, AP: {metrics['macro_ap']:.4f}] | "
              f"\nTốc độ học: {learning_rate:.1e} | Tổng thời gian: {time.perf_counter() - started:.0f}s ({train_stats['train_images_per_second']:.0f} ảnh/s, chờ dữ liệu {data_share:.0%}) | "
              + ("[Mô hình tốt nhất]" if improved
                 else f"Không cải thiện {no_improvement}/{training_config['early_stop_patience']} "
                      f"(Tốt nhất: {selection_metric}: {best_score:.4f} ở Epoch {best_epoch:02d})"))

        ## Thông báo scheduler giảm learning rate (bỏ qua ở epoch cuối)
        if next_learning_rate < learning_rate and not finished:
            print(f"Scheduler giảm learning rate: {learning_rate:.1e} -> {next_learning_rate:.1e}")

        ## Cảnh báo khi mọi bước cập nhật của epoch đều bị AMP bỏ qua
        if not math.isfinite(train_stats["grad_norm_max"]):
            print("CẢNH BÁO: mọi bước cập nhật trong epoch đều bị AMP bỏ qua (gradient tràn số FP16)")

    # Giải phóng DataLoader
    del train_loader, validation_loader

    # Kiểm tra kết quả huấn luyện
    ## Phải lưu được mô hình tốt nhất
    if not best_path.exists():
        raise RuntimeError(
            f"[train_one_fold]: Không lưu được mô hình tốt nhất cho Fold {fold}.\n"
            f"Thiếu: {best_path}"
        )

    # 4. Kết thúc Fold
    result = finalize_fold(experiment_context, fold, model, criterion, validation_df, best_path, history)

    # Giải phóng bộ nhớ
    del optimizer, scheduler, scaler, ema, evaluated_model
    cleanup()

    return model, criterion, result

# ============================================================
# Kết thúc Fold
# Trả về: Kết quả của Fold (dict)
def finalize_fold(experiment_context, fold, model, criterion, validation_df, best_path, history):

    # Thư mục kết quả của Fold
    fold_directory = experiment_context.fold_directory(fold)

    # Nạp trọng số tốt nhất
    best = load_best_weights(experiment_context, best_path, model, fold)

    # Đánh giá kiểm thử bằng trọng số tốt nhất (dự đoán OOF của Fold)
    _, loader = create_dataloader(experiment_context.dataset_class, experiment_context.classes, experiment_context.device, validation_df, experiment_context.eval_transform, False, batch_size = experiment_context.batch_size, seed = int(config.SEED) + fold * 100_000 + 90_000)
    loss, metrics, targets, probabilities = evaluate(experiment_context.device, experiment_context.number_of_classes, model, loader, criterion)

    # Lý do dừng (lấy từ lịch sử epoch)
    stopped_early = history[-1]["no_improvement"] >= experiment_context.training_config["early_stop_patience"]

    # Kết quả của Fold
    result = {
        # Quá trình huấn luyện
        "fold"                  : fold,
        "best_epoch"            : int(best["epoch"]),
        "epochs_trained"        : len(history),
        "stop_reason"           : "early_stopping" if stopped_early else "max_epochs",
        "training_minutes"      : sum(row["elapsed_seconds"] for row in history) / 60,
        "train_images_per_second" : float(np.mean([row["train_images_per_second"] for row in history])),
        "peak_gpu_memory_gb"    : max((row["peak_gpu_memory_gb"] or 0.0) for row in history) or None,
        "amp_skipped_steps"     : int(sum(row["amp_skipped_steps"] for row in history)),

        # Kết quả kiểm thử bằng trọng số tốt nhất
        "loss"                  : loss,
        "best_macro_auroc"      : float(metrics["macro_auroc"]),
        "best_selection_metric" : best["selection_metric"],
        "best_selection_score"  : float(best["selection_score"]),
        **{key: value for key, value in metrics.items() if not key.startswith("class_")},
    }

    # Lưu kết quả
    save_csv(fold_directory / "oof_predictions.csv", with_probabilities(validation_df, probabilities, experiment_context.classes))
    save_csv(fold_directory / "validation_class_metrics.csv", class_metrics_dataframe(targets, probabilities, experiment_context.classes, config.THRESHOLD))
    save_history(fold_directory, history, experiment_context.classes)
    save_json(fold_directory / "fold_done.json", {"signature": experiment_context.signature, "result": result})

    # In kết quả của Fold
    print(f"\nKết thúc Fold {fold} | {'Early stopping' if stopped_early else 'Đủ số epoch tối đa'} sau {len(history)} epoch, {result['training_minutes']:.0f} phút | "
          f"Epoch tốt nhất {result['best_epoch']} | Validation [Loss: {result['loss']:.4f}, Macro AUROC {result['macro_auroc']:.4f}, Macro AP {result['macro_ap']:.4f}]")

    return result
