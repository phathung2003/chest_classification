# Khai báo utilities
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from itertools import count
import config as config
import time
import gc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from config import LEARNING_RATE, WEIGHT_DECAY, IMAGE_SIZE
from utilities.data import create_dataloader, pos_weight
from utilities.metric import calculate_metrics, class_metrics_dataframe, with_probabilities 
import json

# ============================================================
# Thông số mô hình
def count_parameters(model):
    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    return (total, trainable)

# ============================================================
# Huấn luyện 1 epoch
def train_one_epoch(device, model, loader, criterion, optimizer, scaler, use_amp = config.USE_AMP, threshold = config.THRESHOLD):
    
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_labels = 0
    total_samples = 0

    all_targets = []
    all_probabilities = []

    progress = tqdm(loader, desc="Huấn luyện [Train]", leave=False,)

    for images, targets in progress:
        images  = images.to(device, non_blocking=True)
        targets = targets.to( device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Forward
        with torch.autocast(
            device_type = device.type,
            dtype = torch.float16,
            enabled = (use_amp and device.type == "cuda"),
        ):

            logits = model(images)
            loss = criterion(logits, targets)

        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Tính Loss
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        # Xác suất dự đoán
        with torch.no_grad():
            probabilities = torch.sigmoid(logits.detach().float())
            predictions = probabilities >= threshold

            # Label-wise Accuracy
            total_correct += (predictions == targets.bool()).sum().item()
            total_labels += targets.numel()

            # Lưu để tính AUROC toàn epoch
            all_targets.append(targets.detach().cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    # Epoch Loss
    epoch_loss = (running_loss / total_samples)

    # Label-wise Accuracy
    epoch_accuracy = (total_correct / total_labels)

    # AUROC theo từng lớp
    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probabilities, axis=0)
    class_aurocs = []

    for class_index in range(y_true.shape[1]):
        labels = y_true[:, class_index]
        probabilities = y_prob[:, class_index]
        
        # AUROC yêu cầu cả hai nhãn 0 và 1
        if np.unique(labels).size < 2:
            class_aurocs.append(np.nan)
            continue

        auc = roc_auc_score(labels, probabilities)
        class_aurocs.append(auc)

    # Macro AUROC
    epoch_auroc = (
        float(np.nanmean(class_aurocs))
        if np.any(np.isfinite(class_aurocs))
        else float("nan")
    )

    return epoch_loss, epoch_accuracy, epoch_auroc

# ============================================================
# Huấn luyện 1 Fold
def train_one_fold(OUTPUT_DIRECTORY, fold, NUMBER_OF_FOLDS, train_transform, validation_transform, dataset_class, train_folds_df, CLASSES, model_factory, model_kwargs, device):
    print(f"\n=== FOLD {fold}/{NUMBER_OF_FOLDS} ===")

    # Chia dữ liệu huấn luyện / kiểm thử
    train_df        = train_folds_df.loc[train_folds_df["Fold"] != fold].reset_index(drop=True)
    validation_df   = train_folds_df.loc[train_folds_df["Fold"] == fold].reset_index(drop=True)
    
    # Tạo Dataloader
    _, train_loader        = create_dataloader(dataset_class, CLASSES, device, train_df       , train_transform       , True)
    _, validation_loader   = create_dataloader(dataset_class, CLASSES, device, validation_df  , validation_transform  , True)
    print(f"Số ảnh dữ liệu: Huấn luyện {len(train_df):,} | Kiểm thử: {len(validation_df):,}")

    # Khai báo mô hình
    model = model_factory(num_classes=len(CLASSES), **model_kwargs).to(device)
    
    # Khởi tạo hàm tính Loss, bộ tối ưu hóa, hỗ trợ huấn luyện
    criterion, optimizer, scaler = build_training_objects(fold, model, device)
    
    # Thiết lập Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "max", factor = config.THRESHOLD, patience = config.PATIENCE)
    
    # Đầu ra lưu mô hình
    if(NUMBER_OF_FOLDS > 1):
        fold_directory = (OUTPUT_DIRECTORY / f"fold_{fold}")
        fold_directory.mkdir(parents=True, exist_ok=True)
    else:
        fold_directory = OUTPUT_DIRECTORY
    
    # Lưu checkpoint
    checkpoint_path = fold_directory / "best_model.pt"
    
    history = []
    best_auc, best_epoch, no_improvement = -np.inf, 0, 0

    # Bắt đầu huấn luyện
    for epoch in count(start=1):
        
        # Huấn luyện
        started = time.perf_counter()
        epoch_loss, epoch_accuracy, epoch_auroc = train_one_epoch(device, model, train_loader, criterion, optimizer, scaler, config.USE_AMP)
        
        # Kiểm thử
        val_loss, metrics, _, _ = evaluate(device, len(CLASSES), model, validation_loader, criterion, config.THRESHOLD)
        auc = metrics["macro_auroc"]
        
        if not np.isfinite(auc):
            raise RuntimeError(f"Fold {fold}: AUROC không xác định ở epoch {epoch}")
        scheduler.step(auc)
        
        statistics = {
            "fold"              : fold, 
            "epoch"             : epoch,
            "elapsed_minutes"   : (time.perf_counter() - started),
            "learning_rate"     : optimizer.param_groups[0]["lr"],
             
            "train_loss"        : epoch_loss,
            "train_accuracy"    : epoch_accuracy,
            "train_auroc"       : epoch_auroc,
            
            "validation_loss"   : val_loss, 
            
            **{key: value for key, value in metrics.items() if key != "class_aurocs"},
        }
        
        # Lưu thông tin
        history.append(statistics)
        pd.DataFrame(history).to_csv(fold_directory / "history.csv", index=False)

        if auc > best_auc:
            best_auc, best_epoch, no_improvement = auc, epoch, 0
            _save_checkpoint(checkpoint_path, model, CLASSES, model_kwargs, fold=fold, epoch=epoch, validation_auroc=auc,)

        else:
            no_improvement += 1
            if no_improvement >= config.PATIENCE:
                print(f"Tạm dừng huấn luyện tại epoch {epoch}")
                break
        
        print(  f"Epoch: {epoch:02d} | "
                f"Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy:.4f} | Train AUROC: {epoch_auroc:.4f} "
                f"Validation Loss: {val_loss:.4f} | Validation AUROC: {auc:.4f} "
                f"F1-Scorce {metrics['macro_f1']:.4f} | Learning Rate {optimizer.param_groups[0]['lr']:.2e}"
                f"[MODEL UPDATE]" if no_improvement == 0 else f"No improvemnt ({no_improvement}/{config.PATIENCE})")


    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, metrics, targets, probabilities = evaluate(device, len(CLASSES), model, validation_loader, criterion, config.THRESHOLD)
    with_probabilities(validation_df, probabilities, CLASSES).to_csv(fold_directory / "oof_predictions.csv", index=False)
    class_metrics_dataframe(targets, probabilities, CLASSES, config.THRESHOLD).to_csv(fold_directory / "validation_class_metrics.csv", index=False)
    result = {"fold": fold, "best_epoch": best_epoch, "best_macro_auroc": metrics["macro_auroc"]}
    result.update({k: v for k, v in metrics.items() if k not in {"macro_auroc", "class_aurocs"}})
    
    # Giải phóng bộ nhớ
    del model, optimizer, scaler, train_loader, validation_loader, criterion
    cleanup()
    
    # Trả kết quả
    return result

# ============================================================
# Huấn luyện mô hình cuối cùng
def train_final_model(OUTPUT_DIRECTORY, dataset_class, train_df, CLASSES, train_transform, model_factory, model_kwargs,  device, epochs):
    print(f"\n=== FINAL TRAINING ({epochs} epochs; {len(train_df):,} images) ===")
    
    # Tạo Dataloader
    _, loader = create_dataloader(dataset_class, CLASSES, device, train_df, train_transform, True)
    
    model = model_factory(num_classes=len(CLASSES), **model_kwargs).to(device)
    criterion, optimizer, scaler = build_training_objects(0, model, device)
    history = []
    
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        loss, epoch_accuracy, epoch_auroc = train_one_epoch(device, model, loader, criterion, optimizer, scaler, config.USE_AMP)
        history.append({
            "epoch": epoch, 
            "train_loss": loss,
            "train_accuracy": epoch_accuracy, 
            "train_auroc": epoch_auroc,
            "elapsed_minutes": (time.perf_counter() - started) / 60
            })
        pd.DataFrame(history).to_csv(OUTPUT_DIRECTORY / "final_training_history.csv", index=False)
        print(f"Final Epoch {epoch:02d}/{epochs} | Loss {loss:.4f}")
    _save_checkpoint(OUTPUT_DIRECTORY / "final_model.pt", model, CLASSES, model_kwargs, epochs=epochs)
    return model, criterion

# ============================================================
# Đánh giá mô hình
@torch.no_grad()
def evaluate(device, number_of_classes, model, loader, criterion = None, threshold = config.THRESHOLD):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probabilities = []

    progress = tqdm(loader, desc="Đánh giá mô hình", leave = False)
    
    for images, targets in progress:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True,)
        
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(config.USE_AMP and device.type == "cuda")
        ):

            logits = model(images)

            if criterion is not None:
                loss = criterion(logits, targets)

        probabilities = torch.sigmoid(logits)

        if criterion is not None:
            running_loss += (loss.item() * images.size(0))

        all_targets.append(targets.detach().cpu().numpy())
        all_probabilities.append(probabilities.detach().float().cpu().numpy())

    targets = np.concatenate(all_targets, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)
    
    if criterion is not None:
        loss = (running_loss / len(loader.dataset))

    else:
        loss = None

    metrics = calculate_metrics(number_of_classes, targets, probabilities, threshold=threshold,)

    return loss, metrics, targets, probabilities

# ============================================================
# Kiểm tra
def evaluate_test(OUTPUT_DIRECTORY, dataset_class, model, criterion, test_df, CLASSES, test_transform, device):
    print("\n=== FINAL TEST ===")

    _, test_loader   = create_dataloader(dataset_class, CLASSES, device, test_df  , test_transform  , False)
    test_loss, metrics, targets, probabilities = evaluate(device, len(CLASSES), model, test_loader, criterion, config.THRESHOLD)
    class_metrics_dataframe(targets, probabilities, CLASSES, config.THRESHOLD).to_csv(OUTPUT_DIRECTORY / "test_class_metrics.csv", index=False)
    with_probabilities(test_df, probabilities, CLASSES).to_csv(OUTPUT_DIRECTORY / "test_predictions.csv", index=False)
    
    summary = {
        "test_loss": test_loss,
        "threshold": config.THRESHOLD,
        **{key: value for key, value in metrics.items() if key != "class_aurocs"},
    }
    
    with (OUTPUT_DIRECTORY / "test_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    
    print(pd.Series(summary).to_string())
    return summary

# ============================================================
# Trả tài nguyên máy sau khi chạy hoàn tất
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# Lưu điểm lưu
def _save_checkpoint(path, model, classes, model_kwargs, **extra):
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "image_size": IMAGE_SIZE,
        "model_kwargs": model_kwargs,
        "threshold": config.THRESHOLD,
        **extra,
    }, path)

# ============================================================
# Thiết lập tính Loss, bộ tối ưu hóa, hỗ trợ huấn luyện
def build_training_objects(fold_no, model, device):
    
    # Lấy Pos Weight
    class_weights = pos_weight(fold_no, device)
    
    # Hàm tính Loss
    criterion = nn.BCEWithLogitsLoss(pos_weight = class_weights)
    
    # Hàm tối ưu hóa tham số mô hình
    optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    
    # Bộ hỗ trợ huấn luyện Mixed Precision (AMP).
    scaler = torch.amp.GradScaler("cuda", enabled = config.USE_AMP and device.type == "cuda")
    
    return criterion, optimizer, scaler




