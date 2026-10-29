from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from training.threshold import apply_postprocess, load_postprocess
from utilities.data import create_dataloader
from utilities.file import read_json, save_csv, save_json
from utilities.log import print_test_summary, progress_bar, update_progress
from utilities.metric import calculate_metrics, class_metrics_dataframe, with_probabilities
from utilities.statistics import bootstrap_auroc_ci

# ============================================================
# Đánh giá mô hình
@torch.no_grad()
def evaluate(device, number_of_classes, model, loader, criterion = None, epoch = None):
    model.eval()
    running_loss = 0.0
    target_batches, probability_batches = [], []

    description = f"Epoch {epoch:02d} [Kiểm thử]" if epoch is not None else "Kiểm tra"
    progress = progress_bar(loader, description, device)
    seen = 0
    for images, targets in progress:
        images  = images.to(device, non_blocking = True)
        targets = targets.to(device, dtype = torch.float32, non_blocking = True)

        # Forward ở FP16 (nếu bật AMP), loss tính ở FP32 như lúc huấn luyện
        with torch.autocast(device_type = device.type, dtype = torch.float16, enabled = bool(config.USE_AMP and device.type == "cuda")):
            logits = model(images)

        logits = logits.float()

        if criterion is not None:
            running_loss += criterion(logits, targets).item() * images.size(0)

        target_batches.append(targets.cpu().numpy())
        probability_batches.append(torch.sigmoid(logits).cpu().numpy())
        seen += images.size(0)

        if criterion is not None:
            update_progress(progress, loss = running_loss / seen)
        else:
            update_progress(progress)

    if not target_batches:
        raise ValueError("Tập đánh giá rỗng")

    targets = np.concatenate(target_batches, axis = 0)
    probabilities = np.concatenate(probability_batches, axis = 0)
    metrics = calculate_metrics(number_of_classes, targets, probabilities, threshold = float(config.THRESHOLD))
    epoch_loss = running_loss / len(loader.dataset) if criterion is not None else None

    return epoch_loss, metrics, targets, probabilities

# ============================================================
# Báo cáo kết quả
def report_test(output_directory, test_df, classes, targets, probabilities, *, title = "Kiểm tra", test_loss = None, postprocess_path = None, extra = None):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents = True, exist_ok = True)

    # Chỉ số tại ngưỡng mặc định
    metrics = calculate_metrics(len(classes), targets, probabilities, threshold = float(config.THRESHOLD))
    save_csv(output_directory / "test_class_metrics.csv", class_metrics_dataframe(targets, probabilities, classes, float(config.THRESHOLD)))
    save_csv(output_directory / "test_predictions.csv", with_probabilities(test_df, probabilities, classes))
    summary = {
        "test_loss" : test_loss,
        "threshold" : float(config.THRESHOLD),
        **{k: v for k, v in metrics.items() if not k.startswith("class_")},
        **(extra or {}),
    }

    # Khoảng tin cậy 95% của AUROC: bootstrap theo bệnh nhân
    if int(config.BOOTSTRAP_ITERATIONS) > 0:
        if config.PATIENT_ID_COLUMN not in test_df:
            raise ValueError(f"Thiếu cột {config.PATIENT_ID_COLUMN} để bootstrap theo bệnh nhân")
        ci = bootstrap_auroc_ci(targets, probabilities, test_df[config.PATIENT_ID_COLUMN].to_numpy(), classes, n_boot = int(config.BOOTSTRAP_ITERATIONS), seed = int(config.BOOTSTRAP_SEED))
        save_csv(output_directory / "test_class_auroc_ci.csv", pd.DataFrame(ci["classes"]))
        summary["macro_auroc_ci95"] = [ci["macro_auroc"]["ci_low"], ci["macro_auroc"]["ci_high"]]
        summary["bootstrap"] = {k: ci[k] for k in ("method", "n_boot", "seed", "confidence")}

    # Hậu xử lý: ngưỡng (và calibration) đã chọn trên dự đoán OOF, áp lên Test
    if postprocess_path is not None and Path(postprocess_path).is_file():
        post = load_postprocess(postprocess_path, classes)
        calibrated, thresholds = apply_postprocess(probabilities, post, classes)
        tuned = calculate_metrics(len(classes), targets, calibrated, thresholds)
        save_csv(output_directory / "test_class_metrics_postprocessed.csv", class_metrics_dataframe(targets, calibrated, classes, thresholds))
        summary["postprocess_file"] = str(postprocess_path)
        summary["postprocessed"] = {k: v for k, v in tuned.items() if not k.startswith("class_")}
        summary["postprocess_thresholds"] = {name: float(t) for name, t in zip(classes, thresholds)}

    save_json(output_directory / "test_summary.json", summary)
    print_test_summary(title, summary)
    return summary

# ============================================================
# Đánh giá tập kiểm thử
def evaluate_test(ctx, output_directory, model, criterion, test_df, title = "Kiểm tra", postprocess_path = None):
    _, loader = create_dataloader(ctx.dataset_class, ctx.classes, ctx.device, test_df, ctx.eval_transform, False, batch_size = ctx.batch_size, seed = int(config.SEED) + 900_000)
    test_loss, _, targets, probabilities = evaluate(ctx.device, ctx.number_of_classes, model, loader, criterion)
    return report_test(output_directory, test_df, ctx.classes, targets, probabilities, title = title, test_loss = test_loss, postprocess_path = postprocess_path)

# ============================================================
# Trung bình / độ lệch chuẩn
def mean_std(frame, exclude = ("fold",)):
    result = {}
    for column in frame.select_dtypes(include = "number").columns:
        if column in exclude:
            continue
        series = frame[column].dropna()
        result[column] = {
            "mean"              : float(series.mean()) if len(series) else None,
            "std"               : float(series.std(ddof = 1)) if len(series) > 1 else None,
            "number_of_folds"   : int(len(series)),
        }
    return result

# ============================================================
# Tổng hợp Test của các mô hình
def aggregate_fold_tests(output_directory, test_df, classes, number_of_folds, postprocess_path = None):
    output_directory = Path(output_directory)
    rows, probabilities = [], []

    # Đọc kết quả Test của từng fold đã có
    for fold in range(1, int(number_of_folds) + 1):
        directory = output_directory / f"fold_{fold}" / "test"\

        # Không có test summary
        if not (directory / "test_summary.json").is_file():
            continue

        summary = read_json(directory / "test_summary.json")
        rows.append({"fold": fold, **{k: v for k, v in summary.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}})
        frame = pd.read_csv(directory / "test_predictions.csv")

        if len(frame) != len(test_df) or not (frame["image_path"].astype(str).to_numpy() == test_df["image_path"].astype(str).to_numpy()).all():
            raise ValueError(f"Thứ tự ảnh Test của fold {fold} không khớp test.csv")

        probabilities.append(frame[[f"prob_{c}" for c in classes]].to_numpy(dtype = np.float64))

    if not rows:
        return None

    # Mean ± std giữa các fold
    frame = pd.DataFrame(rows)
    save_csv(output_directory / "test_fold_results.csv", frame)
    report = {
        "folds"                 : frame["fold"].astype(int).tolist(),
        "all_folds_evaluated"   : len(rows) == int(number_of_folds),
        "metric_mean_std"        : mean_std(frame),
    }
    save_json(output_directory / "test_fold_summary.json", report)

    # Ensemble chỉ khi đủ mọi fold, để kết quả không phụ thuộc vào việc chạy dở
    if report["all_folds_evaluated"]:
        targets = test_df[list(classes)].to_numpy(dtype = np.float32)
        ensemble = np.mean(np.stack(probabilities, axis = 0), axis = 0)
        report["ensemble"] = report_test(output_directory / "test_ensemble", test_df, classes, targets, ensemble, title = f"Test ensemble {len(probabilities)} fold", postprocess_path = postprocess_path, extra = {"ensemble": f"mean probability of {len(probabilities)} fold models"})
    return report

# ============================================================
# Tổng hợp Cross-Validation
def cross_validation(output_directory, results, number_of_folds, signature):

    # Đọc kết quả các fold đã hoàn thành, kiểm tra chữ ký
    recorded = {}
    for fold in range(1, int(number_of_folds) + 1):
        done_path = output_directory / f"fold_{fold}" / "fold_done.json"
        if not done_path.exists():
            continue
        done = read_json(done_path)
        if done.get("signature") != signature:
            raise ValueError(f"Fold {fold} thuộc cấu hình khác")
        if int(done["result"]["fold"]) != fold:
            raise ValueError(f"Kết quả fold không khớp trong {done_path}")
        recorded[fold] = done["result"]
    for result in results:
        if int(result["fold"]) not in recorded:
            raise ValueError(f"Fold {result['fold']} chưa có fold_done.json")
    if not recorded:
        raise RuntimeError("Chưa có fold nào hoàn thành")

    # Bảng kết quả và mean ± std
    frame = pd.DataFrame([recorded[f] for f in sorted(recorded)])
    save_csv(output_directory / "cross_validation_results.csv", frame)
    complete = len(recorded) == int(number_of_folds)
    save_json(output_directory / "cross_validation_summary.json", {
        "completed_folds"       : sorted(recorded),
        "all_folds_completed"   : complete,
        "number_of_folds"       : len(recorded),
        "metric_mean_std"       : mean_std(frame, exclude = ("fold", "best_epoch")),
    })

    # Gộp dự đoán OOF của mọi fold (dùng để chọn ngưỡng cho ensemble)
    if complete:
        paths = [output_directory / f"fold_{fold}" / "oof_predictions.csv" for fold in range(1, int(number_of_folds) + 1)]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError("Thiếu tệp OOF của một fold đã hoàn thành")
        save_csv(output_directory / "oof_all.csv", pd.concat([pd.read_csv(path) for path in paths], ignore_index = True))
        print("Hoàn thành toàn bộ Cross-Validation")
    else:
        print("Chưa hoàn thành tất cả fold: chưa gộp OOF và chưa tạo ensemble")
    return frame, complete
