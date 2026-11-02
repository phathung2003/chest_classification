# ============================================================
# Chỉ số đánh giá phân loại đa nhãn (Multi-label Classification Metrics)
# Mục tiêu: Tính AUROC, AP, Precision, Recall, F1 theo từng lớp và tổng hợp (macro / micro / weighted) từ nhãn và xác suất dự đoán.
# ============================================================

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

import config

# ============================================================
# Kiểm tra kết quả dự đoán
# Trả về: (actual_label (int8), model_prediction (float64))
def validate_predictions(targets, probabilities):

    # Chuyển đổi dữ liệu
    actual_label        = np.asarray(targets)                                # Nhãn thực tế: Giá trị dữ liệu là [0, 1]
    model_prediction    = np.asarray(probabilities, dtype = np.float64)      # Xác suất dự đoán từ mô hình

    # Kiểm tra dữ liệu đầu vào
    ## Hai biến phải là hai ma trận [num_samples, num_classes] cùng kích thước
    if actual_label.ndim != 2 or actual_label.shape != model_prediction.shape:
        raise ValueError(
            "[validate_predictions]: Dữ liệu targets và probabilities phải là hai ma trận [num_samples, num_classes] cùng kích thước.\n"
            f"Nhận được: targets = {actual_label.shape}, probabilities = {model_prediction.shape}"
        )

    ## Ma trận không được rỗng
    if actual_label.shape[0] == 0 or actual_label.shape[1] == 0:
        raise ValueError(
            "[validate_predictions]: Ma trận phân loại không được rỗng.\n"
            f"Nhận được: targets = {actual_label.shape}"
        )

    ## actual_label phải chứa nhãn nhị phân (0 hoặc 1)
    invalid_label = ~np.isin(actual_label, [0, 1])

    if np.any(invalid_label):
        row, col = np.argwhere(invalid_label)[0]

        raise ValueError(
            "[validate_predictions]: Dữ liệu targets phải chứa nhãn nhị phân (0 hoặc 1).\n"
            f"Số phần tử không hợp lệ: {np.count_nonzero(invalid_label)}.\n"
            f"Lỗi đầu tiên tại [{row}, {col}]: Giá trị = {actual_label[row, col]!r}"
        )

    ## model_prediction phải chứa xác suất hữu hạn trong khoảng [0, 1]
    invalid_prediction = (
        ~np.isfinite(model_prediction)
        | (model_prediction < 0)
        | (model_prediction > 1)
    )

    if np.any(invalid_prediction):
        row, col = np.argwhere(invalid_prediction)[0]

        raise ValueError(
            "[validate_predictions]: Dữ liệu probabilities phải chứa xác suất hữu hạn trong khoảng [0, 1].\n"
            f"Số phần tử không hợp lệ: {np.count_nonzero(invalid_prediction)}.\n"
            f"Lỗi đầu tiên tại [{row}, {col}]: Giá trị = {model_prediction[row, col]!r}"
        )

    # Nhãn chuyển sang int8 (không sao chép nếu đã đúng kiểu)
    return actual_label.astype(np.int8, copy = False), model_prediction

# ============================================================
# Chuẩn hóa ngưỡng dự đoán (Threshold)
# Trả về: Vector ngưỡng [num_classes] (float64)
def normalize_thresholds(threshold, classes):

    # Chuyển đổi dữ liệu
    classes = list(classes)

    # dict {lớp: ngưỡng} --> vector theo thứ tự lớp
    if isinstance(threshold, Mapping):

        ## Khóa của dict phải khớp danh sách lớp
        if set(threshold) != set(classes):
            raise ValueError(
                "[normalize_thresholds]: Khóa của dict ngưỡng phải khớp danh sách lớp.\n"
                f"Thiếu: {sorted(set(classes) - set(threshold), key = str)}, thừa: {sorted(set(threshold) - set(classes), key = str)}"
            )

        thresholds = np.array([threshold[name] for name in classes], dtype = np.float64)

    # Một số --> dùng chung cho mọi lớp
    elif np.isscalar(threshold):
        thresholds = np.full(len(classes), float(threshold), dtype = np.float64)

    # Vector
    else:
        thresholds = np.asarray(threshold, dtype = np.float64)
        
    ## Kiểm tra mỗi lớp một ngưỡng hữu hạn trong khoảng [0, 1]
    if thresholds.shape != (len(classes),) or not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError(
            f"[normalize_thresholds]: Cần {len(classes)} ngưỡng hữu hạn trong khoảng [0, 1].\n"
            f"Nhận được: threshold = {thresholds.tolist()}"
        )

    return thresholds

# ============================================================
# Tính chỉ số theo từng lớp tại ngưỡng cho trước
# Trả về: DataFrame, mỗi dòng là một lớp (AUROC, AP, Precision, Recall, F1 và ma trận nhầm lẫn TP, FP, FN, TN)
def class_metrics_dataframe(targets, probabilities, classes, threshold = None):

    # Lấy dữ liệu
    decision_threshold = config.THRESHOLD if threshold is None else threshold

    # Kiểm tra dữ liệu đầu vào
    actual_label, model_prediction = validate_predictions(targets, probabilities)

    # Kiểm tra số tên lớp phải bằng số cột dự đoán
    if len(classes) != actual_label.shape[1]:
        raise ValueError(
            "[class_metrics_dataframe]: Số tên lớp phải bằng số cột dự đoán.\n"
            f"Nhận được: Số tên lớp = {len(classes)}, số cột = {actual_label.shape[1]}"
        )

    # Dự đoán nhị phân theo ngưỡng của từng lớp
    thresholds = normalize_thresholds(decision_threshold, classes)
    binary_prediction = model_prediction >= thresholds[None, :]

    # Khởi tạo biến lưu kết quả
    rows = []

    # Tính chỉ số cho từng lớp
    for index, name in enumerate(classes):

        # Nhãn, xác suất và dự đoán nhị phân của lớp
        class_label, class_score, class_prediction = actual_label[:, index], model_prediction[:, index], binary_prediction[:, index]

        # AUROC / AP cần cả mẫu dương và mẫu âm (lớp chỉ có một loại nhãn --> NaN)
        evaluable = np.unique(class_label).size == 2

        # Lưu chỉ số của lớp
        rows.append({
            # Thông tin lớp
            "class"             : name,
            "positive_support"  : int(class_label.sum()),
            "negative_support"  : int(len(class_label) - class_label.sum()),
            "prevalence"        : float(class_label.mean()),
            "threshold"         : float(thresholds[index]),

            # Chỉ số không phụ thuộc ngưỡng
            "auroc"             : float(roc_auc_score(class_label, class_score)) if evaluable else np.nan,
            "average_precision" : float(average_precision_score(class_label, class_score)) if evaluable else np.nan,

            # Chỉ số tại ngưỡng
            "precision"         : float(precision_score(class_label, class_prediction, zero_division = 0)),
            "recall"            : float(recall_score(class_label, class_prediction, zero_division = 0)),
            "f1"                : float(f1_score(class_label, class_prediction, zero_division = 0)),

            # Ma trận nhầm lẫn
            "tp"                : int(((class_label == 1) & class_prediction).sum()),
            "fp"                : int(((class_label == 0) & class_prediction).sum()),
            "fn"                : int(((class_label == 1) & ~class_prediction).sum()),
            "tn"                : int(((class_label == 0) & ~class_prediction).sum()),
        })

    return pd.DataFrame(rows)

# ============================================================
# Tính chỉ số
# Trả về: dict chỉ số, kèm class_aurocs / class_aps (danh sách theo lớp)
def calculate_metrics(number_of_classes, targets, probabilities, threshold = None):

    # Lấy dữ liệu
    decision_threshold = config.THRESHOLD if threshold is None else threshold

    # Kiểm tra dữ liệu đầu vào
    actual_label, model_prediction = validate_predictions(targets, probabilities)

    ## number_of_classes phải bằng số cột nhãn
    if number_of_classes != actual_label.shape[1]:
        raise ValueError(
            "[calculate_metrics]: number_of_classes phải bằng số cột nhãn.\n"
            f"Nhận được: number_of_classes = {number_of_classes}, số cột = {actual_label.shape[1]}"
        )

    ## Ngưỡng phải là một số hoặc vector (dict ngưỡng cần tên lớp, dùng class_metrics_dataframe)
    if isinstance(decision_threshold, Mapping):
        raise ValueError(
            "[calculate_metrics]: threshold phải là một số hoặc vector; dict ngưỡng cần tên lớp.\n"
            f"Nhận được: threshold = dict có {len(decision_threshold)} khóa"
        )

    # Dự đoán nhị phân theo ngưỡng của từng lớp
    thresholds = normalize_thresholds(decision_threshold, range(actual_label.shape[1]))
    binary_prediction = (model_prediction >= thresholds[None, :]).astype(np.uint8)

    # Khởi tạo biến lưu kết quả
    class_aurocs, class_aps = [], []

    # Tính AUROC và AP từng lớp (lớp chỉ có một loại nhãn --> NaN)
    for index in range(actual_label.shape[1]):
        evaluable = np.unique(actual_label[:, index]).size == 2
        class_aurocs.append(float(roc_auc_score(actual_label[:, index], model_prediction[:, index])) if evaluable else float("nan"))
        class_aps.append(float(average_precision_score(actual_label[:, index], model_prediction[:, index])) if evaluable else float("nan"))

    # Hàm tính trung bình trên các giá trị hữu hạn (bỏ qua NaN); không có giá trị nào --> NaN
    def mean_finite(values):
        values = np.asarray(values, dtype = np.float64)
        return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float("nan")

    return {
        # Chỉ số không phụ thuộc ngưỡng
        "macro_auroc"       : mean_finite(class_aurocs),
        "macro_ap"          : mean_finite(class_aps),

        # Chỉ số tại ngưỡng
        "macro_precision"   : float(precision_score(actual_label, binary_prediction, average = "macro", zero_division = 0)),
        "macro_recall"      : float(recall_score(actual_label, binary_prediction, average = "macro", zero_division = 0)),
        "macro_f1"          : float(f1_score(actual_label, binary_prediction, average = "macro", zero_division = 0)),
        "micro_f1"          : float(f1_score(actual_label, binary_prediction, average = "micro", zero_division = 0)),
        "weighted_f1"       : float(f1_score(actual_label, binary_prediction, average = "weighted", zero_division = 0)),

        # Chỉ số từng lớp
        "class_aurocs"      : class_aurocs,
        "class_aps"         : class_aps,
    }

# ============================================================
# Ghép xác suất dự đoán vào dataframe gốc
# Trả về: Bản sao dataframe có thêm các cột prob_<lớp>
def with_probabilities(frame, probabilities, classes):

    # Chuyển đổi dữ liệu
    result = frame.reset_index(drop = True).copy()
    model_prediction = np.asarray(probabilities)
    
    ## Kiểm tra ma trận dự đoán phải có kích thước [số dòng của dataframe, số lớp]
    if model_prediction.ndim != 2 or model_prediction.shape != (len(result), len(classes)):
        raise ValueError(
            "[with_probabilities]: Dự đoán phải có kích thước [số dòng của dataframe, số lớp]; loader Validation / Test không được xáo trộn.\n"
            f"Nhận được: probabilities = {model_prediction.shape}, cần = {(len(result), len(classes))}"
        )

    # Thêm cột prob_<lớp>
    for index, name in enumerate(classes):
        result[f"prob_{name}"] = model_prediction[:, index]

    return result
