# Khai báo utilities
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from itertools import count
import time
import gc
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from utilities.notice import print_heading, print_message
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from utilities.config import TRAIN_FOLDS_DIRECTORY, TEST_DIRECTORY, DERGEES, TRANSLATE, SCALE, TRANSFORM_NORMALIZE_MEAN, TRANSFORM_NORMALIZE_STD, BATCH_SIZE, NUMBBER_OF_WORKERS, FOLDS_DIRECTORY, USE_AMP, TRAIN_ALL_FOLD, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, THRESHOLD, IMAGE_SIZE, DATASET_DIRECTORY
import json

def dataset_info(dataset_config):
    DATASET_NAME        = dataset_config["dataset_name"]
    CLASSES             = dataset_config["classes"]
    NUMBER_OF_CLASSES   = len(CLASSES)
    NUMBER_OF_FOLDS     = dataset_config["number_of_folds"]
    
    train_folds_df = pd.read_csv(TRAIN_FOLDS_DIRECTORY)
    test_df = pd.read_csv(TEST_DIRECTORY)
    
    print_heading("Thông tin tập dữ liệu")
    print_message("Tên tập dữ liệu"             , DATASET_NAME)
    print_message("Số ảnh tập huấn luyện"       , f"{len(train_folds_df):,} ảnh")
    print_message("Số ảnh tập kiểm tra"         , f"{len(test_df):,} ảnh")
    print_message("Kích cỡ ảnh"                 , f"{dataset_config['image_size']} x {dataset_config['image_size']}")
    print_message("Số phần được chia"           , NUMBER_OF_FOLDS)
    print_message("Số lớp"                      , NUMBER_OF_CLASSES)
    
    print("Các lớp trong tập dữ liệu")
    for index, class_name in enumerate(CLASSES, start=1):
        print( f"{index:>2}. "f"{class_name}")
    
    return train_folds_df, test_df, DATASET_NAME, CLASSES, NUMBER_OF_CLASSES, NUMBER_OF_FOLDS

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
# Cấu hình làm giàu ảnh
def transform_setting():
    train_transform = transforms.Compose([

    transforms.RandomAffine(degrees = DERGEES, translate = TRANSLATE, scale = SCALE),
        transforms.ToTensor(),
        transforms.Normalize(mean = TRANSFORM_NORMALIZE_MEAN, std = TRANSFORM_NORMALIZE_STD),
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = TRANSFORM_NORMALIZE_MEAN, std = TRANSFORM_NORMALIZE_STD),
    ])
    
    return train_transform, eval_transform

# ============================================================
# Dataloader
def create_data_loader(dataset_class, CLASSES, DEVICE, dataframe, transform, shuffle):
    dataset = dataset_class(dataframe=dataframe, classes=CLASSES, transform=transform,)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUMBBER_OF_WORKERS,
        prefetch_factor=2,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )

    return (dataset, loader)

# ============================================================
# Pos Weight
def pos_weight(fold, DEVICE):
    
    if fold == 0:
        path = (DATASET_DIRECTORY / "pos_weight.npy").resolve()
    else:
        path = (FOLDS_DIRECTORY / f"fold_{fold}" / "pos_weight.npy").resolve()
    
    # Kiểm tra tệp tin 
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    
    weights = np.load(path, allow_pickle=False)
    
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

# ============================================================
# Đánh giá
def calculate_metrics(NUMBER_OF_CLASSES, targets, probabilities, threshold=0.5,):
    predictions = (probabilities >= threshold).astype(np.uint8)
    
    # Per-class AUROC
    class_aurocs = []

    for index in range(NUMBER_OF_CLASSES):
        y_true = targets[:,index]
        y_score = probabilities[:,index]

        if np.unique(y_true).size < 2:
            auc = np.nan

        else:
            auc = roc_auc_score(y_true, y_score)

        class_aurocs.append(auc)

    # Macro AUROC
    macro_auroc = float(np.nanmean(class_aurocs))
    
    # Macro Precision
    macro_precision = float(precision_score(targets, predictions, average="macro", zero_division=0))

    # Macro Recall
    macro_recall = float(recall_score(targets, predictions, average="macro", zero_division=0))

    # Macro F1-Score
    macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))

    # Micro F1-Score
    micro_f1 = float(f1_score(targets, predictions, average="micro", zero_division=0))
    
    # Weight F1-Score
    weighted_f1 = float(f1_score(targets, predictions, average="weighted", zero_division=0))
    
    return {
        "macro_auroc"       : macro_auroc,
        "macro_precision"   : macro_precision,
        "macro_recall"      : macro_recall,
        "macro_f1"          : macro_f1,
        "micro_f1"          : micro_f1,
        "weighted_f1"       : weighted_f1,
        "class_aurocs"      : class_aurocs
    }

# ============================================================
# Huấn luyện 1 epoch
def train_one_epoch(DEVICE, model, loader, criterion, optimizer, scaler, USE_AMP=True, threshold=0.5):
    
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_labels = 0
    total_samples = 0

    all_targets = []
    all_probabilities = []

    progress = tqdm(loader, desc="Huấn luyện [Train]", leave=False,)

    for images, targets in progress:
        images  = images.to(DEVICE, non_blocking=True)
        targets = targets.to( DEVICE, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Forward
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(USE_AMP and DEVICE.type == "cuda"),
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
# Đánh giá mô hình

@torch.no_grad()
def evaluate(DEVICE, NUMBER_OF_CLASSES, model, loader, criterion=None, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probabilities = []

    progress = tqdm(loader, desc="Evaluate", leave=False,)
    for images, targets in progress:
        images  = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True,)
        
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(USE_AMP and DEVICE.type == "cuda")
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

    metrics = calculate_metrics(NUMBER_OF_CLASSES, targets, probabilities, threshold=threshold,)

    return loss, metrics, targets, probabilities

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _save_checkpoint(path, model, classes, model_kwargs, **extra):
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "image_size": IMAGE_SIZE,
        "model_kwargs": model_kwargs,
        "threshold": THRESHOLD,
        **extra,
    }, path)
    
# ============================================================
def with_probabilities(frame, probabilities, classes):
    result = frame.reset_index(drop=True).copy()
    for i, class_name in enumerate(classes):
        result[f"prob_{class_name}"] = probabilities[:, i]
    return result

# ============================================================
def class_metrics_dataframe(targets, probabilities, classes, threshold=0.5):
    predicted = (probabilities >= threshold).astype(np.uint8)
    rows = []
    for i, name in enumerate(classes):
        y = targets[:, i]
        p = predicted[:, i]
        rows.append({
            "class": name,
            "auroc": roc_auc_score(y, probabilities[:, i]) if np.unique(y).size == 2 else np.nan,
            "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0),
            "positive_support": int(y.sum()),
        })
    return pd.DataFrame(rows)

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
    scaler = torch.amp.GradScaler("cuda", enabled = USE_AMP and device.type == "cuda")
    
    return criterion, optimizer, scaler

# ============================================================
# Huấn luyện 1 Fold
def train_one_fold(OUTPUT_DIRECTORY, fold, NUMBER_OF_FOLDS, train_transform, validation_transform, dataset_class, train_folds_df, CLASSES, model_factory, model_kwargs, DEVICE):
    print(f"\n=== FOLD {fold}/{NUMBER_OF_FOLDS} ===")

    # Chia dữ liệu huấn luyện / kiểm thử
    train_df        = train_folds_df.loc[train_folds_df["Fold"] != fold].reset_index(drop=True)
    validation_df   = train_folds_df.loc[train_folds_df["Fold"] == fold].reset_index(drop=True)
    
    # Tạo Dataloader
    _, train_loader        = create_data_loader(dataset_class, CLASSES, DEVICE, train_df       , train_transform       , True)
    _, validation_loader   = create_data_loader(dataset_class, CLASSES, DEVICE, validation_df  , validation_transform  , True)
    print(f"Số ảnh dữ liệu: Huấn luyện {len(train_df):,} | Kiểm thử: {len(validation_df):,}")

    # Khai báo mô hình
    model = model_factory(num_classes=len(CLASSES), **model_kwargs).to(DEVICE)
    
    # Khởi tạo hàm tính Loss, bộ tối ưu hóa, hỗ trợ huấn luyện
    criterion, optimizer, scaler = build_training_objects(fold, model, DEVICE)
    
    # Thiết lập Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "max", factor = THRESHOLD, patience = PATIENCE)
    
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
        started = time.perf_counter()
        train_loss, epoch_accuracy, epoch_auroc = train_one_epoch(DEVICE, model, train_loader, criterion, optimizer, scaler, USE_AMP)
        val_loss, metrics, _, _ = evaluate(DEVICE, len(CLASSES), model, validation_loader, criterion, THRESHOLD)
        auc = metrics["macro_auroc"]
        
        if not np.isfinite(auc):
            raise RuntimeError(f"Fold {fold}: AUROC không xác định ở epoch {epoch}")
        scheduler.step(auc)
        
        row = {
            "fold": fold, "epoch": epoch, "train_loss": train_loss,
            "validation_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.perf_counter() - started) / 60,
            **{key: value for key, value in metrics.items() if key != "class_aurocs"},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(fold_directory / "history.csv", index=False)
        print(f"Epoch {epoch:02d} | "
              f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"AUROC {auc:.4f} | F1 {metrics['macro_f1']:.4f} | "
              f"LR {optimizer.param_groups[0]['lr']:.2e}")

        if auc > best_auc:
            best_auc, best_epoch, no_improvement = auc, epoch, 0
            _save_checkpoint(checkpoint_path, model, CLASSES, model_kwargs, fold=fold, epoch=epoch, validation_auroc=auc,)
            print("  [BEST] checkpoint saved")
        else:
            no_improvement += 1
            if no_improvement >= PATIENCE:
                print(f"Tạm dừng huấn luyện tại epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, metrics, targets, probabilities = evaluate(DEVICE, len(CLASSES), model, validation_loader, criterion, THRESHOLD)
    with_probabilities(validation_df, probabilities, CLASSES).to_csv(fold_directory / "oof_predictions.csv", index=False)
    class_metrics_dataframe(targets, probabilities, CLASSES, THRESHOLD).to_csv(fold_directory / "validation_class_metrics.csv", index=False)
    result = {"fold": fold, "best_epoch": best_epoch, "best_macro_auroc": metrics["macro_auroc"]}
    result.update({k: v for k, v in metrics.items() if k not in {"macro_auroc", "class_aurocs"}})
    
    # Giải phóng bộ nhớ
    del model, optimizer, scaler, train_loader, validation_loader, criterion
    cleanup()
    
    # Trả kết quả
    return result

# ============================================================
# Huấn luyện mô hình cuối cùng
def train_final_model(OUTPUT_DIRECTORY, dataset_class, train_df, CLASSES, train_transform, model_factory, model_kwargs,  DEVICE, epochs):
    print(f"\n=== FINAL TRAINING ({epochs} epochs; {len(train_df):,} images) ===")
    
    # Tạo Dataloader
    _, loader = create_data_loader(dataset_class, CLASSES, DEVICE, train_df, train_transform, True)
    
    model = model_factory(num_classes=len(CLASSES), **model_kwargs).to(DEVICE)
    criterion, optimizer, scaler = build_training_objects(0, model, DEVICE)
    history = []
    
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        loss, epoch_accuracy, epoch_auroc = train_one_epoch(DEVICE, model, loader, criterion, optimizer, scaler, USE_AMP)
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
# Kiểm tra
def evaluate_test(OUTPUT_DIRECTORY, dataset_class, model, criterion, test_df, CLASSES, test_transform, DEVICE):
    print("\n=== FINAL TEST ===")

    _, test_loader   = create_data_loader(dataset_class, CLASSES, DEVICE, test_df  , test_transform  , True)
    test_loss, metrics, targets, probabilities = evaluate(DEVICE, len(CLASSES), model, test_loader, criterion, THRESHOLD)
    class_metrics_dataframe(targets, probabilities, CLASSES, THRESHOLD).to_csv(OUTPUT_DIRECTORY / "test_class_metrics.csv", index=False)
    with_probabilities(test_df, probabilities, CLASSES).to_csv(OUTPUT_DIRECTORY / "test_predictions.csv", index=False)
    
    summary = {
        "test_loss": test_loss,
        "threshold": THRESHOLD,
        **{key: value for key, value in metrics.items() if key != "class_aurocs"},
    }
    
    with (OUTPUT_DIRECTORY / "test_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    
    print(pd.Series(summary).to_string())
    return summary

# ============================================================
# Chạy mô hình
def run_experiment(DEVICE, CLASSES, dataset_class, FOLD_TRAIN, train_transform, test_transform, OUTPUT_DIRECTORY, train_folds_df, test_df, model_factory, model_kwargs):
    results = []
    oof_frames = []
    
    # Huấn luyện mô hình
    for fold in [FOLD_TRAIN]:
        results.append(train_one_fold(OUTPUT_DIRECTORY, fold, len([FOLD_TRAIN]), train_transform, test_transform, dataset_class, train_folds_df, CLASSES, model_factory, model_kwargs, DEVICE))
        
        if len([FOLD_TRAIN]) > 1:
            oof_frames.append(pd.read_csv(OUTPUT_DIRECTORY / f"fold_{fold}" / "oof_predictions.csv"))
        else:
            pd.DataFrame(results).to_csv(OUTPUT_DIRECTORY / "cross_validation_results.csv", index=False)

    # Kết quả
    result_df = pd.DataFrame(results)
    
    print("\n=== CROSS VALIDATION RESULTS ===")
    print(result_df.to_string(index=False))
    
    if len([FOLD_TRAIN]) > 1:
        pd.concat(oof_frames, ignore_index=True).to_csv(OUTPUT_DIRECTORY / "oof_all.csv", index=False)
        print(f"Mean AUROC: {result_df['best_macro_auroc'].mean():.6f} ± {result_df['best_macro_auroc'].std(ddof=1):.6f}")

    # Chạy huấn luyện lượt cuối (Nếu chạy trên 1 Fold)

    # Chỉ dùng best epoch từ cross-validation ĐẦY ĐỦ vừa chạy.
    final_epochs = max(1, int(np.median(result_df["best_epoch"].to_numpy())))
    final_model, final_criterion = train_final_model(OUTPUT_DIRECTORY, dataset_class, train_folds_df, CLASSES, train_transform, model_factory, model_kwargs, DEVICE, final_epochs)

    # Đánh giá mô hình    
    evaluate_test(OUTPUT_DIRECTORY, dataset_class, final_model, final_criterion, test_df, CLASSES, test_transform, DEVICE)
    del final_model, final_criterion
    cleanup()
    
    return result_df
