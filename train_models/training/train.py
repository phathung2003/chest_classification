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
from utilities.log import progress_bar, timestamp, update_progress
from utilities.metric import class_metrics_dataframe, with_probabilities
from utilities.setting import set_seed

# ============================================================
# Giải phóng bộ nhớ CPU / GPU giữa các fold
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Chia dữ liệu
def split_fold(train_folds_df, fold):
    train_df = train_folds_df.loc[train_folds_df["Fold"] != fold].reset_index(drop = True)
    validation_df = train_folds_df.loc[train_folds_df["Fold"] == fold].reset_index(drop = True)
    if train_df.empty or validation_df.empty:
        raise ValueError(f"Không lấy được dữ liệu huấn luyện / kiểm thử cho fold {fold}")
    return train_df, validation_df

# ============================================================
# Khởi tạo mô hình và hàm Loss
def build_model_and_loss(ctx, train_df):
    model = ctx.model_factory(num_classes = ctx.number_of_classes, **ctx.model_kwargs).to(ctx.device)
    criterion, loss_metadata = build_loss(ctx.loss_config, train_df, ctx.classes, ctx.device)
    return model, criterion, loss_metadata

# ============================================================
# Khởi tạo optimizer, scheduler và GradScaler (AMP)
def build_optimization(ctx, model, criterion):
    parameters = list(model.parameters()) + [p for p in criterion.parameters() if p.requires_grad]
    optimizer = make_optimizer(parameters, ctx.training_config["optimizer"])
    scheduler = make_scheduler(optimizer, ctx.training_config["scheduler"])
    scaler = torch.amp.GradScaler("cuda", enabled = bool(config.USE_AMP and ctx.device.type == "cuda"))
    return optimizer, scheduler, scaler

# ============================================================
# Lấy trọng số tốt nhất của fold
def load_best_weights(ctx, best_path, model, fold):
    best = load_torch(best_path, ctx.device)
    check_checkpoint(best, signature = ctx.signature, fold = fold, status = "best")
    model.load_state_dict(best["model_state_dict"], strict = True)
    return best

# ============================================================
# Lưu lịch sử epoch
HISTORY_COLUMNS = [
    "fold", "epoch", "no_improvement",
    "elapsed_seconds", "data_wait_seconds", "peak_gpu_memory_gb", "train_seconds", "train_images_per_second",
    "learning_rate", "grad_norm_mean", "grad_norm_max", "amp_skipped_steps",
    "train_loss", "train_accuracy", "train_auroc",
    "validation_seconds", "validation_loss", "macro_auroc", "macro_ap", "macro_precision", "macro_recall", "macro_f1", "micro_f1", "weighted_f1",
]

def save_history(fold_directory, history, classes):
    frame = pd.DataFrame(history)
    class_columns = [f"auroc_{name}" for name in classes if f"auroc_{name}" in frame]
    epoch_frame = frame.drop(columns = class_columns)
    ordered = [c for c in HISTORY_COLUMNS if c in epoch_frame] + [c for c in epoch_frame if c not in HISTORY_COLUMNS]
    save_csv(fold_directory / "history.csv", epoch_frame[ordered])
    if class_columns:
        class_frame = frame[["epoch", *class_columns]].rename(columns = lambda c: c.removeprefix("auroc_"))
        save_csv(fold_directory / "history_class_auroc.csv", class_frame)

# ============================================================
# Load seed
def train_loader_seed(fold, epoch):
    return int(config.SEED) + int(fold) * 100_000 + int(epoch)

# ============================================================
# Kiểm tra độ cải thiện của mô hình
def is_improvement(score, best_score, has_best, higher_is_better, min_delta):
    if not has_best:
        return True
    return score > best_score + min_delta if higher_is_better else score < best_score - min_delta

# ============================================================
# Chuẩn L2 của toàn bộ gradient
def _total_grad_norm(optimizer):
    norms = [torch.linalg.vector_norm(p.grad.detach()) for g in optimizer.param_groups for p in g["params"]
             if p.grad is not None]
    return torch.linalg.vector_norm(torch.stack(norms)) if norms else torch.tensor(0.0)

# ============================================================
# Huấn luyện một epoch
def train_one_epoch(epoch, device, model, loader, criterion, optimizer, scaler, training_config, ema = None):
    grad_accum_steps = int(training_config["grad_accum_steps"])
    grad_clip_norm = training_config["grad_clip_norm"]

    model.train()
    criterion.train()
    total_loss, correct, labels, n_samples, accumulated_samples = 0.0, 0, 0, 0, 0
    truths, scores = [], []
    grad_norms, amp_skipped_steps, data_wait_seconds = [], 0, 0.0
    optimizer.zero_grad(set_to_none = True)

    # Thời gian chờ dữ liệu = khoảng từ lúc xử lý xong batch trước tới lúc nhận batch sau (gồm cả khởi động worker)
    epoch_started = batch_finished = time.perf_counter()
    progress = progress_bar(loader, f"Epoch {epoch:02d} [Huấn luyện]", device)
    for step, (images, targets) in enumerate(progress, start = 1):
        data_wait_seconds += time.perf_counter() - batch_finished
        images = images.to(device, non_blocking = True)
        targets = targets.to(device, dtype = torch.float32, non_blocking = True)
        batch_size = images.size(0)

        # Forward ở FP16 (AMP), loss tính ở FP32
        with torch.autocast(device_type = device.type, dtype = torch.float16, enabled = bool(config.USE_AMP and device.type == "cuda")):
            logits = model(images)
        loss = criterion(logits.float(), targets)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Giá trị Loss không hợp lệ tại epoch {epoch}")

        # Cộng dồn gradient theo số mẫu (loss * batch_size), chia lại khi cập nhật --> trung bình theo mẫu
        scaler.scale(loss * batch_size).backward()
        accumulated_samples += batch_size

        # Cập nhật trọng số sau mỗi grad_accum_steps batch (hoặc ở batch cuối)
        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.div_(accumulated_samples)
            # Chuẩn gradient trước khi clip (đo sau unscale nên là giá trị thật)
            if grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], max_norm = float(grad_clip_norm))
            else:
                grad_norm = _total_grad_norm(optimizer)

            # GradScaler giảm scale nghĩa là gradient tràn số FP16 và bước cập nhật này bị bỏ qua
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < scale_before:
                amp_skipped_steps += 1
            else:
                grad_norms.append(float(grad_norm))
                if ema is not None:
                    ema.update(model)
            optimizer.zero_grad(set_to_none = True)
            accumulated_samples = 0

        # Thống kê huấn luyện
        total_loss += float(loss.detach()) * batch_size
        n_samples += batch_size
        probabilities = torch.sigmoid(logits.detach().float())
        correct += int(((probabilities >= config.THRESHOLD) == targets.bool()).sum().item())
        labels += targets.numel()
        truths.append(targets.cpu().numpy())
        scores.append(probabilities.cpu().numpy())
        update_progress(progress, loss = total_loss / n_samples)
        batch_finished = time.perf_counter()

    if n_samples == 0:
        raise ValueError("Tập huấn luyện không có dữ liệu")
    train_seconds = time.perf_counter() - epoch_started

    # Macro AUROC trên các lớp có đủ nhãn dương và âm
    y_true, y_prob = np.concatenate(truths), np.concatenate(scores)
    aucs = [roc_auc_score(y_true[:, i], y_prob[:, i]) for i in range(y_true.shape[1])
            if np.unique(y_true[:, i]).size == 2]

    return {
        "train_loss"                : total_loss / n_samples,
        "train_accuracy"            : correct / labels,
        "train_auroc"               : float(np.mean(aucs)) if aucs else float("nan"),
        "train_seconds"             : train_seconds,
        "data_wait_seconds"         : data_wait_seconds,
        "train_images_per_second"   : n_samples / train_seconds,
        "grad_norm_mean"            : float(np.mean(grad_norms)) if grad_norms else float("nan"),
        "grad_norm_max"             : float(np.max(grad_norms)) if grad_norms else float("nan"),
        "amp_skipped_steps"         : amp_skipped_steps,
    }

# ============================================================
# Huấn luyện một fold:
def train_one_fold(ctx, fold, train_folds_df):
    fold = int(fold)
    tc = ctx.training_config
    fold_directory = ctx.fold_directory(fold)
    fold_directory.mkdir(parents = True, exist_ok = True)
    done_path = fold_directory / "fold_done.json"
    last_path = fold_directory / "last_checkpoint.pt"
    best_path = fold_directory / "best_model.pt"
    train_df, validation_df = split_fold(train_folds_df, fold)

    # 1. Fold đã hoàn thành
    if done_path.exists():
        done = read_json(done_path)
        if done.get("signature") != ctx.signature:
            raise ValueError(f"Fold {fold} đã hoàn thành với cấu hình khác")
        if not best_path.is_file():
            raise FileNotFoundError(f"Thiếu mô hình tốt nhất của fold đã hoàn thành: {best_path}")
        model, criterion, _ = build_model_and_loss(ctx, train_df)
        load_best_weights(ctx, best_path, model, fold)
        print(f"Fold {fold} đã hoàn thành trước đó (epoch tốt nhất: {done['result']['best_epoch']})")
        return model, criterion, done["result"]

    print(f"\n=== FOLD {fold} | Huấn luyện: {len(train_df):,} ảnh | Validation: {len(validation_df):,} ảnh ===")

    # 2. Khởi tạo (seed riêng cho từng fold: khởi tạo không phụ thuộc vào các fold chạy trước)
    set_seed(int(config.SEED) + fold)
    model, criterion, loss_metadata = build_model_and_loss(ctx, train_df)
    optimizer, scheduler, scaler = build_optimization(ctx, model, criterion)
    # EMA: Validation, early stopping, scheduler và best_model.pt đều dùng trọng số EMA
    ema = ModelEMA(model, tc["ema_decay"]) if tc["ema_decay"] is not None else None
    evaluated_model = model if ema is None else ema.module
    if "pos_weight" in loss_metadata:
        print(f"pos_weight của fold: {[round(w, 3) for w in loss_metadata['pos_weight']]}")

    selection_metric = tc["selection_metric"]
    higher_is_better = selection_metric == "macro_auroc"
    epoch_start, history, best_epoch, no_improvement, finished = 1, [], 0, 0, False
    best_score = -math.inf if higher_is_better else math.inf

    # Chạy tiếp từ checkpoint cuối
    if last_path.exists():
        saved = resume_training(last_path, model = model, criterion = criterion, optimizer = optimizer, scheduler = scheduler, scaler = scaler, device = ctx.device, signature = ctx.signature, fold = fold, ema = ema)
        epoch_start, history = int(saved["epoch"]) + 1, saved["history"]
        best_score, best_epoch = float(saved["best_score"]), int(saved["best_epoch"])
        no_improvement, finished = int(saved["no_improvement"]), bool(saved["training_finished"])
        reconcile_best(saved, best_path, signature = ctx.signature, fold = fold, classes = ctx.classes, model_name = ctx.model_name, model_kwargs = ctx.model_kwargs)
        print(f"Chạy tiếp Fold {fold} [Epoch đã hoàn thành: {epoch_start - 1} | Epoch tốt nhất: {best_epoch}]")
    elif best_path.exists():
        raise RuntimeError(f"Có best_model.pt nhưng thiếu last_checkpoint.pt: {best_path}")

    # DataLoader Validation: không xáo trộn, giữ worker giữa các epoch
    _, validation_loader = create_dataloader(ctx.dataset_class, ctx.classes, ctx.device, validation_df, ctx.eval_transform, False, batch_size = ctx.batch_size, seed = int(config.SEED))

    # 3. Vòng lặp epoch
    for epoch in range(epoch_start, tc["epochs"] + 1):
        if finished:
            break
        started = time.perf_counter()
        if ctx.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(ctx.device)

        # Huấn luyện (train loader tạo lại mỗi epoch với seed theo fold và epoch)
        _, train_loader = create_dataloader(ctx.dataset_class, ctx.classes, ctx.device, train_df, ctx.train_transform, True, batch_size = ctx.batch_size, seed = train_loader_seed(fold, epoch), persistent_workers = False)
        train_stats = train_one_epoch(epoch, ctx.device, model, train_loader, criterion, optimizer, scaler, tc, ema = ema)
        del train_loader

        # Đánh giá Validation
        validation_started = time.perf_counter()
        validation_loss, metrics, _, _ = evaluate(ctx.device, ctx.number_of_classes, evaluated_model, validation_loader, criterion, epoch = epoch)
        validation_seconds = time.perf_counter() - validation_started
        auc = float(metrics["macro_auroc"])
        score = auc if higher_is_better else float(validation_loss)
        if not math.isfinite(score):
            raise ValueError(f"Epoch {epoch}: không tính được {selection_metric}")

        # Early stopping
        improved = is_improvement(score, best_score, best_epoch > 0, higher_is_better, tc["min_delta"])
        if improved:
            best_score, best_epoch, no_improvement = score, epoch, 0
        else:
            no_improvement += 1
        finished = no_improvement >= tc["early_stop_patience"] or epoch >= tc["epochs"]

        # Scheduler theo dõi cùng chỉ số với tiêu chí chọn mô hình
        learning_rate = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(float(validation_loss) if tc["scheduler"]["mode"] == "min" else auc)
        next_learning_rate = optimizer.param_groups[0]["lr"]

        # Lịch sử epoch (learning_rate: giá trị cho epoch kế tiếp, giữ như các lượt chạy trước).
        # AUROC Validation từng lớp lưu với tiền tố "auroc_", được tách ra history_class_auroc.csv
        history.append({
            "fold"                  : fold,
            "epoch"                 : epoch,
            "elapsed_seconds"       : time.perf_counter() - started,
            **train_stats,
            "validation_seconds"    : validation_seconds,
            "validation_loss"       : validation_loss,
            **{k: v for k, v in metrics.items() if not k.startswith("class_")},
            "learning_rate"         : next_learning_rate,
            "no_improvement"        : no_improvement,
            "peak_gpu_memory_gb"    : (torch.cuda.max_memory_allocated(ctx.device) / 1024 ** 3
                                       if ctx.device.type == "cuda" else None),
            **{f"auroc_{name}": value for name, value in zip(ctx.classes, metrics["class_aurocs"])},
        })

        # Lưu checkpoint cuối trước, rồi mới lưu mô hình tốt nhất (reconcile_best xử lý trường hợp dừng giữa hai bước)
        save_training_state(last_path, fold = fold, epoch = epoch, model = model, criterion = criterion, optimizer = optimizer, scheduler = scheduler, scaler = scaler, history = history, best_score = best_score, best_epoch = best_epoch, no_improvement = no_improvement, signature = ctx.signature, training_finished = finished, selection_metric = selection_metric, ema = ema)
        if improved:
            save_best_model(best_path, model_state_dict = evaluated_model.state_dict(), fold = fold, epoch = epoch, score = score, validation_macro_auroc = auc, selection_metric = selection_metric, signature = ctx.signature, classes = ctx.classes, model_name = ctx.model_name, model_kwargs = ctx.model_kwargs)
        save_history(fold_directory, history, ctx.classes)

        # Một dòng mỗi epoch (accuracy theo nhãn, chuẩn gradient và các chỉ số khác nằm trong history.csv)
        data_share = train_stats["data_wait_seconds"] / max(train_stats["train_seconds"], 1e-9)
        print(f"Epoch: {epoch:02d} | Train: [Loss: {train_stats['train_loss']:.4f}, Macro AUROC: {train_stats['train_auroc']:.4f}] | "
              f"Validation: [Loss: {validation_loss:.4f}, Macro AUROC: {auc:.4f}, AP: {metrics['macro_ap']:.4f}] | "
              f"Learning rate: {learning_rate:.1e} | Time: {time.perf_counter() - started:.0f}s ({train_stats['train_images_per_second']:.0f} ảnh/s, chờ dữ liệu {data_share:.0%}) | "
              + ("[Mô hình tốt nhất]" if improved
                 else f"Không cải thiện {no_improvement}/{tc['early_stop_patience']} "
                      f"(Tốt nhất {selection_metric}: {best_score:.4f} ở epoch {best_epoch})"))
        
        if next_learning_rate < learning_rate and finished != True:
            print(f"Scheduler giảm learning rate: {learning_rate:.1e} -> {next_learning_rate:.1e}")
        
        if not math.isfinite(train_stats["grad_norm_max"]):
            print("CẢNH BÁO: mọi bước cập nhật trong epoch đều bị AMP bỏ qua (gradient tràn số FP16)")

    if not best_path.exists():
        raise RuntimeError(f"Không lưu được mô hình tốt nhất cho fold {fold}")

    # 4. Kết thúc fold
    result = finalize_fold(ctx, fold, model, criterion, validation_df, best_path, history)
    del optimizer, scheduler, scaler, ema, evaluated_model
    cleanup()
    return model, criterion, result


# ============================================================
# Kết thúc Fold
def finalize_fold(ctx, fold, model, criterion, validation_df, best_path, history):
    fold_directory = ctx.fold_directory(fold)
    best = load_best_weights(ctx, best_path, model, fold)

    _, loader = create_dataloader(ctx.dataset_class, ctx.classes, ctx.device, validation_df, ctx.eval_transform, False, batch_size = ctx.batch_size, seed = int(config.SEED) + fold * 100_000 + 90_000)
    loss, metrics, targets, probabilities = evaluate(ctx.device, ctx.number_of_classes, model, loader, criterion)

    # Lý do dừng và tổng thời gian huấn luyện (lấy từ lịch sử epoch)
    stopped_early = history[-1]["no_improvement"] >= ctx.training_config["early_stop_patience"]
    result = {
        "fold"                  : fold,
        "best_epoch"            : int(best["epoch"]),
        "epochs_trained"        : len(history),
        "stop_reason"           : "early_stopping" if stopped_early else "max_epochs",
        "training_minutes"      : sum(row["elapsed_seconds"] for row in history) / 60,
        "train_images_per_second" : float(np.mean([row["train_images_per_second"] for row in history])),
        "peak_gpu_memory_gb"    : max((row["peak_gpu_memory_gb"] or 0.0) for row in history) or None,
        "amp_skipped_steps"     : int(sum(row["amp_skipped_steps"] for row in history)),
        "loss"                  : loss,
        "best_macro_auroc"      : float(metrics["macro_auroc"]),
        "best_selection_metric" : best["selection_metric"],
        "best_selection_score"  : float(best["selection_score"]),
        **{k: v for k, v in metrics.items() if not k.startswith("class_")},
    }
    save_csv(fold_directory / "oof_predictions.csv", with_probabilities(validation_df, probabilities, ctx.classes))
    save_csv(fold_directory / "validation_class_metrics.csv", class_metrics_dataframe(targets, probabilities, ctx.classes, config.THRESHOLD))
    save_history(fold_directory, history, ctx.classes)
    save_json(fold_directory / "fold_done.json", {"signature": ctx.signature, "result": result})

    print(f"\nKết thúc Fold {fold} | {'Early stopping' if stopped_early else 'Đủ số epoch tối đa'} sau {len(history)} epoch, {result['training_minutes']:.0f} phút | "
          f"Epoch tốt nhất {result['best_epoch']} | Validation [Loss: {result['loss']:.4f}, Macro AUROC {result['macro_auroc']:.4f}, Macro AP {result['macro_ap']:.4f}]")
    
    return result
