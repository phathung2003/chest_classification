# ============================================================
# Đánh giá mô hình (Evaluation)
# Mục tiêu: Đánh giá mô hình, báo cáo chỉ số, tổng hợp Cross-Validation và tính trung bình các mô hình của từng Fold trên tập Test.
# ============================================================

from pathlib import Path

import torch
import numpy as np
import pandas as pd

import config
from utilities.data import create_dataloader
from utilities.statistics import bootstrap_auroc_confidence_interval
from utilities.file import read_json, save_csv, save_json
from training.threshold import apply_postprocess, load_postprocess
from utilities.log import print_test_summary, progress_bar, update_progress
from utilities.metric import calculate_metrics, class_metrics_dataframe, with_probabilities
from utilities.setting import amp_dtype

# ============================================================
# Đánh giá mô hình
# Trả về: (loss trung bình theo ảnh hoặc None, dict chỉ số, nhãn [num_samples, num_classes], xác suất [num_samples, num_classes])
@torch.no_grad()                                                                                    # Tắt Gradient (Tiết kiệm bộ nhớ)
def evaluate(device, number_of_classes, model, loader, criterion = None, epoch = None):
    
    # Chuyển mô hình sang chế độ đánh giá
    model.eval()
    
    # Khởi tạo biến lưu kết quả
    running_loss = torch.zeros((), dtype = torch.float64, device = device)                          # Lưu tổng Loss
    target_batches, probability_batches = [], []                                                    # Lưu kết quả từng batch

    # Khởi động thanh tiến trình
    description = f"Epoch {epoch:02d} [Kiểm thử]" if epoch is not None else "[Kiểm tra]"            # Nội dung đầu thanh tiến trình
    progress = progress_bar(loader, description, device)                                            # Hiển thị thanh tiến trình
    image_processed = 0                                                                             # Ảnh được xử lý

    # Duyệt từng batch
    for step, (images, targets) in enumerate(progress, start = 1):
        
        # Chuyển dữ liệu tính toán
        images  = images.to(device, non_blocking = True)                                            # Tensor chứa ảnh
        targets = targets.to(device, dtype = torch.float32, non_blocking = True)                    # Tensor chứa nhãn thực tế

        # Tạo Logits
        with torch.autocast(device_type = device.type, dtype = amp_dtype(), enabled = bool(config.USE_AMP and device.type == "cuda")):  # Forward Pass với AMP (nếu bật AMP) ở AMP_DTYPE, loss tính ở FP32
            logits = model(images)

        # Chuyển Logits về FP32
        logits = logits.float()

        # Tính và cộng dồn loss theo số ảnh
        if criterion is not None:
            running_loss += criterion(logits, targets).double() * images.size(0)                    # Kết quả = Loss trung bình của batch * số ảnh

        # Lưu nhãn thực tế
        target_batches.append(targets)
        
        # Chuyển Logits thành xác suất bằng Sigmoid:
        probability_batches.append(torch.sigmoid(logits))
        
        # Cập nhật thanh tiến trình
        image_processed += images.size(0)
        
        if step % config.PROGRESS_SYNC_STEPS == 0 or step == len(loader):                           # Cập nhật Loss sau n vòng (n = PROGRESS_SYNC_STEPS được đặt trong config)
            if criterion is not None:
                update_progress(progress, loss = running_loss.item() / image_processed)             
            else:
                update_progress(progress)

    # Kiểm tra dữ liệu
    ## Tập đánh giá không được rỗng
    if not target_batches:
        raise ValueError(
            "[evaluate]: Tập đánh giá không được rỗng.\n"
            f"Nhận được: Số ảnh = {len(loader.dataset)}"
        )

    # Gộp dữ liệu và chuyển về CPU
    targets = torch.cat(target_batches).cpu().numpy()
    probabilities = torch.cat(probability_batches).cpu().numpy()
    
    # Tính các thông số
    metrics = calculate_metrics(number_of_classes, targets, probabilities, threshold = float(config.THRESHOLD))
    
    # Tính Epoch Loss
    running_loss = float(running_loss.item())
    epoch_loss = running_loss / len(loader.dataset) if criterion is not None else None

    return epoch_loss, metrics, targets, probabilities

# ============================================================
# Báo cáo kết quả Kiểm tra
# Trả về: summary (nội dung test_summary.json)
def report_test(output_directory, test_df, classes, targets, probabilities, *, title = "Kiểm tra", test_loss = None, postprocess_path = None, extra = None):
    
    # Khởi tạo thư mục lưu kết quả
    output_directory = Path(output_directory)
    output_directory.mkdir(parents = True, exist_ok = True)

    # Tính chỉ số tại Threshold mặc định
    metrics = calculate_metrics(len(classes), targets, probabilities, threshold = float(config.THRESHOLD))
    
    # Lưu chỉ số riêng từng lớp ra file CSV
    save_csv(output_directory / "test_class_metrics.csv", class_metrics_dataframe(targets, probabilities, classes, float(config.THRESHOLD)))
    
    # Lưu xác suất dự đoán ra file CSV
    save_csv(output_directory / "test_predictions.csv", with_probabilities(test_df, probabilities, classes))
    
    # Khởi tạo báo cáo kết quả kiểm tra
    summary = {
        "test_loss" : test_loss,
        "threshold" : float(config.THRESHOLD),
        **{key: value for key, value in metrics.items() if not key.startswith("class_")},
        **(extra or {}),
    }

    # Bootstrap để tính khoảng tin cậy AUROC
    if int(config.BOOTSTRAP_ITERATION) > 0:
        
        # Kiểm tra cột mã bệnh nhân
        if config.PATIENT_ID_COLUMN not in test_df:
            raise ValueError(
                f"[report_test]: Cần cột {config.PATIENT_ID_COLUMN} để bootstrap theo bệnh nhân.\n"
                f"Nhận được: Các cột = {list(test_df.columns)}"
            )

        # Tính khoảng tin cậy (Thể hiện mức độ bất định của một chỉ số)
        confidence_interval = bootstrap_auroc_confidence_interval(targets, probabilities, test_df[config.PATIENT_ID_COLUMN].to_numpy(), classes, bootstrap_iteration = int(config.BOOTSTRAP_ITERATION), seed = int(config.BOOTSTRAP_SEED))

        # Lưu khoảng tin cậy AUROC của từng lớp ra file CSV
        save_csv(output_directory / "test_class_auroc_confidence_interval.csv", pd.DataFrame(confidence_interval["classes"]))

        # Lưu khoảng tin cậy 95% của Macro AUROC
        summary["macro_auroc_confidence_interval95"] = [confidence_interval["macro_auroc"]["ci_low"], confidence_interval["macro_auroc"]["ci_high"]]

        # Lưu thông tin Bootstrap (phương pháp, số lần lặp, seed, mức tin cậy)
        summary["bootstrap"] = {key: confidence_interval[key] for key in ("method", "bootstrap_iteration", "seed", "confidence")}

    # Hậu xử lý: Điều chỉnh ngưỡng (và calibration) đã chọn trên dự đoán OOF
    if postprocess_path is not None and Path(postprocess_path).is_file():                   # Thực hiện hậu xử lý nếu đường dẫn và file tồn tại
        
        # Đọc cấu hình hậu xử lý
        postprocess = load_postprocess(postprocess_path, classes)
        
        # Áp dụng hậu xử lý
        calibrated, thresholds = apply_postprocess(probabilities, postprocess, classes)
        
        # Tính lại chỉ số
        tuned = calculate_metrics(len(classes), targets, calibrated, thresholds)
        
        # Lưu chỉ số từng lớp ra file CSV
        save_csv(output_directory / "test_class_metrics_postprocessed.csv", class_metrics_dataframe(targets, calibrated, classes, thresholds))
        
        # Lưu kết quả hậu xử lý
        summary["postprocess_file"] = str(postprocess_path)
        summary["postprocessed"] = {key: value for key, value in tuned.items() if not key.startswith("class_")}
        summary["postprocess_thresholds"] = {name: float(threshold) for name, threshold in zip(classes, thresholds)}

    # Lưu báo cáo kiểm tra ra file JSON
    save_json(output_directory / "test_summary.json", summary)
    
    # In kết quả kiểm tra
    print_test_summary(title, summary)

    return summary

# ============================================================
# Đánh giá mô hình trên tập kiểm tra (loader không xáo trộn, seed cố định)
# Trả về: summary (nội dung test_summary.json)
def evaluate_test(experiment_context, output_directory, model, criterion, test_df, title = "Kiểm tra", postprocess_path = None):
    
    # Tạo DataLoader
    _, loader = create_dataloader(experiment_context.dataset_class, experiment_context.classes, experiment_context.device, test_df, experiment_context.eval_transform, False, batch_size = experiment_context.batch_size, seed = int(config.SEED) + 900_000)
    
    # Đánh giá mô hình
    test_loss, _, targets, probabilities = evaluate(experiment_context.device, experiment_context.number_of_classes, model, loader, criterion)

    # Tạo báo cáo
    return report_test(output_directory, test_df, experiment_context.classes, targets, probabilities, title = title, test_loss = test_loss, postprocess_path = postprocess_path)

# ============================================================
# Tính trung bình / độ lệch chuẩn
# Trả về: dict {cột: {"mean", "std", "number_of_folds"}}
def mean_std(frame, exclude = ("fold",)):
    # Khởi tạo biến lưu kết quả
    result = {}

    # Chỉ chọn các cột số
    for column in frame.select_dtypes(include = "number").columns:
        
        # Bỏ qua cột không cần thống kê (exclude)
        if column in exclude:
            continue
        
        # Loại bỏ giá trị thiếu
        series = frame[column].dropna()
        
        # Tính thông số
        result[column] = {
            "mean"              : float(series.mean()) if len(series) else None,                # Trung bình (Mean)
            "std"               : float(series.std(ddof = 1)) if len(series) > 1 else None,     # Độ lệch chuẩn (Standard deviation)
            "number_of_folds"   : int(len(series)),                                             # Tổng số Fold
        }

    return result

# ============================================================
# Tổng hợp kết quả đánh giá trên tập kiểm tra của nhiều Fold
# Trả về: report (nội dung test_fold_summary.json, kèm "ensemble" nếu đủ Fold)
def aggregate_fold_tests(output_directory, test_df, classes, number_of_folds, postprocess_path = None):
    
    # Đường dẫn lưu kết quả
    output_directory = Path(output_directory)
    
    # Khởi tạo biến lưu kết quả
    rows            = []                                                                        # Chỉ số đánh giá của mỗi Fold
    probabilities   = []                                                                        # Ma trận xác suất dự đoán của mỗi Fold

    # Đọc kết quả kiểm tra của từng Fold
    for fold in range(1, int(number_of_folds) + 1):
        
        # Đường dẫn chứa kết quả kiểm tra của Fold
        directory = output_directory / f"fold_{fold}" / "test"

        # Kiểm tra Fold chạy đánh giá chưa (Nếu chưa --> Bỏ qua)
        if not (directory / "test_summary.json").is_file():
            continue
        
        # Đọc kết quả
        summary = read_json(directory / "test_summary.json")
        
        # Lấy các chỉ số dạng số của Fold
        rows.append({"fold": fold, **{key: value for key, value in summary.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}})
        
        # Đọc xác suất dự đoán của Fold
        frame = pd.read_csv(directory / "test_predictions.csv")
        
        # Kiểm tra thứ tự ảnh trong tập kiểm tra của Fold khớp với tập kiểm tra (test.csv)
        if len(frame) != len(test_df) or not (frame["image_path"].astype(str).to_numpy() == test_df["image_path"].astype(str).to_numpy()).all():
            raise ValueError(
                f"[aggregate_fold_tests]: Thứ tự ảnh Test của Fold {fold} phải khớp test.csv.\n"
                f"Nhận được: Số ảnh của Fold = {len(frame)}, số ảnh của test.csv = {len(test_df)}"
            )

        # Lấy cột xác suất dự đoán
        probabilities.append(frame[[f"prob_{name}" for name in classes]].to_numpy(dtype = np.float64))

    # Không có Fold đánh giá --> Trả về None
    if not rows:
        return None

    # Có Fold đánh giá --> Xuất báo cáo đánh giá giữa các Fold
    frame = pd.DataFrame(rows)
    save_csv(output_directory / "test_fold_results.csv", frame)
    report = {
        "folds"                 : frame["fold"].astype(int).tolist(),
        "all_folds_evaluated"   : len(rows) == int(number_of_folds),
        "metric_mean_std"       : mean_std(frame),
    }
    save_json(output_directory / "test_fold_summary.json", report)

    # Tạo Ensemble khi đủ mọi Fold
    if report["all_folds_evaluated"]:
        
        # Lấy nhãn thực tế
        targets = test_df[list(classes)].to_numpy(dtype = np.float32)
        
        # Tính xác suất Ensemble
        ensemble = np.mean(np.stack(probabilities, axis = 0), axis = 0)
        
        # Tạo báo cáo Ensemble
        report["ensemble"] = report_test(output_directory / "test_ensemble", test_df, classes, targets, ensemble, title = f"Test ensemble {len(probabilities)} Fold", postprocess_path = postprocess_path, extra = {"ensemble": f"mean probability of {len(probabilities)} fold models"})

    return report

# ============================================================
# Tổng hợp kết quả K-Fold Cross-Validation
# Trả về: (bảng kết quả từng Fold, đã đủ mọi Fold hay chưa)
def cross_validation(output_directory, results, number_of_folds, signature):
    
    # Khởi tạo biến lưu kết quả
    recorded = {}                                                           # Chứa kết quả các Fold đã hoàn thành
    
    # Đọc kết quả các Fold đã hoàn thành, kiểm tra chữ ký
    for fold in range(1, int(number_of_folds) + 1):
        
        # Đường dẫn chứa kết quả đánh giá của 1 Fold
        done_path = output_directory / f"fold_{fold}" / "fold_done.json"
        
        # Kiểm tra có file kết quả không (Không có --> Bỏ qua)
        if not done_path.exists():
            continue
        
        # Lấy kết quả
        done = read_json(done_path)

        # Kiểm tra cấu hình có khớp nhau không (Bằng cách kiểm tra chữ ký)
        if done.get("signature") != signature:
            raise ValueError(
                f"[cross_validation]: Fold {fold} thuộc cấu hình khác.\n"
                f"Nhận được: Chữ ký của Fold = {done.get('signature')}, chữ ký hiện tại = {signature}"
            )

        # Kiểm tra thứ tự Fold có khớp với dữ liệu trong file fold_done.json
        if int(done["result"]["fold"]) != fold:
            raise ValueError(
                f"[cross_validation]: Kết quả trong {done_path} phải thuộc Fold {fold}.\n"
                f"Nhận được: Fold = {done['result']['fold']}"
            )

        # Lưu kết quả
        recorded[fold] = done["result"]

    ## Mọi Fold vừa chạy phải có fold_done.json
    for result in results:
        if int(result["fold"]) not in recorded:
            raise ValueError(
                f"[cross_validation]: Fold {result['fold']} chưa có fold_done.json.\n"
                f"Nhận được: Fold đã hoàn thành = {sorted(recorded)}"
            )

    # Kiểm tra có ít nhất 1 Fold hoàn thành
    if not recorded:
        raise RuntimeError(
            "[cross_validation]: Chưa có Fold nào hoàn thành.\n"
            f"Nhận được: output_directory = {output_directory}"
        )

    # Sắp xếp dữ liệu
    frame = pd.DataFrame([recorded[fold] for fold in sorted(recorded)])
    
    # Lưu dữ liệu ra file CSV
    save_csv(output_directory / "cross_validation_results.csv", frame)
    
    # Kiểm tra trạng thái hoàn thành đánh giá Fold
    complete = len(recorded) == int(number_of_folds)
    
    # Tính thông số và lưu kết quả ra file JSON
    save_json(output_directory / "cross_validation_summary.json", {
        "completed_folds"       : sorted(recorded),
        "all_folds_completed"   : complete,
        "number_of_folds"       : len(recorded),
        "metric_mean_std"       : mean_std(frame, exclude = ("fold", "best_epoch")),
    })

    # Gộp dự đoán Out-of-Fold (OOF) của mọi Fold (dùng để chọn ngưỡng cho ensemble)
    if complete:
        
        # Đường dẫn tệp OOF của từng Fold
        paths = [output_directory / f"fold_{fold}" / "oof_predictions.csv" for fold in range(1, int(number_of_folds) + 1)]

        ## Kiểm tra các file OOF có đủ không
        missing_paths = [str(path) for path in paths if not path.is_file()]

        if missing_paths:
            raise FileNotFoundError(
                "[cross_validation]: Thiếu tệp OOF của một Fold đã hoàn thành.\n"
                f"Thiếu: {missing_paths}"
            )

        # Gộp dự đoán ra file CSV
        save_csv(output_directory / "oof_all.csv", pd.concat([pd.read_csv(path) for path in paths], ignore_index = True))
        print("Hoàn thành toàn bộ Cross-Validation")
    else:
        print("Chưa hoàn thành tất cả Fold: chưa gộp OOF và chưa tạo ensemble")

    return frame, complete
