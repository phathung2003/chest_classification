# ============================================================
# Chọn ngưỡng theo lớp (Per-class Threshold Selection)
# Mục tiêu: Thay ngưỡng mặc định 0.5 bằng ngưỡng riêng cho từng lớp (xác suất >= ngưỡng --> dương tính).
# ** Ngưỡng được chọn trên dự đoán OOF (Out-Of-Fold) rồi áp lên tập Test, không bao giờ chọn trên Test.

# Quy trình:
# 1. extract_oof: Đọc tệp OOF và kiểm tra dữ liệu.
# 2. fit_temperatures, apply_temperature (calibration.py): Hiệu chuẩn xác suất (tùy chọn, chỉ dùng cho ensemble).
# 3. choose_threshold: Chọn ngưỡng tối đa F-beta cho từng lớp.
# 4. fit_fold_thresholds (từng Fold) và fit_postprocessing (ensemble): Lưu nhiệt độ và ngưỡng vào postprocess.json.
# 5. load_postprocess, apply_postprocess: Đọc postprocess.json và áp lên xác suất dự đoán của tập Test.
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit

import config
from training.calibration import apply_temperature, fit_temperatures
from utilities.metric import calculate_metrics, class_metrics_dataframe

PROBABILITY_PREFIX  = "prob_"                                               # Tiền tố cột xác suất trong tệp OOF: prob_<lớp>
CALIBRATION_METHODS = ("none", "temperature")                               # Các kiểu hiệu chuẩn được hỗ trợ
EPSILON             = 1e-12                                                 # Tránh chia cho 0 khi tính F-beta

# ============================================================
# Đọc tệp dự đoán OOF
# Trả về: frame, classes, actual_label [num_samples, num_classes] (int8), model_prediction [num_samples, num_classes]
def extract_oof(oof_path, patient_col = None, fold_col = "Fold", image_col = None):

    # Lấy dữ liệu
    patient_column  = config.PATIENT_ID_COLUMN if patient_col is None else patient_col
    image_column    = config.IMAGE_ID_COLUMN if image_col is None else image_col
    fold_column     = fold_col

    # Đọc tệp OOF, lấy danh sách lớp từ các cột prob_<lớp>
    frame               = pd.read_csv(oof_path)
    classes             = [column[len(PROBABILITY_PREFIX):] for column in frame.columns if column.startswith(PROBABILITY_PREFIX)]
    probability_columns = [PROBABILITY_PREFIX + name for name in classes]

    # Kiểm tra cột và dòng của tệp OOF
    ## Tệp phải có ít nhất một cột prob_<lớp>
    if not classes:
        raise ValueError(
            "[extract_oof]: Tệp OOF phải có ít nhất một cột xác suất prob_<lớp>.\n"
            f"Nhận được: Các cột = {list(frame.columns)}"
        )

    ## Tên lớp không được trùng lặp
    duplicated_classes = sorted({name for name in classes if classes.count(name) > 1})

    if duplicated_classes:
        raise ValueError(
            "[extract_oof]: Tên lớp trong các cột prob_<lớp> không được trùng lặp.\n"
            f"Lớp bị trùng: {duplicated_classes}"
        )

    ## Mỗi cột prob_<lớp> phải có cột nhãn <lớp> tương ứng
    missing_label_columns = [name for name in classes if name not in frame.columns]

    if missing_label_columns:
        raise ValueError(
            "[extract_oof]: Mỗi cột prob_<lớp> phải có cột nhãn <lớp> tương ứng.\n"
            f"Thiếu cột nhãn: {missing_label_columns}"
        )

    ## Tệp phải có dữ liệu
    if len(frame) == 0:
        raise ValueError(
            "[extract_oof]: Tệp OOF không có dòng dữ liệu nào.\n"
            f"Nhận được: oof_path = {oof_path}"
        )

    ## Không có ảnh trùng lặp (chỉ kiểm tra khi có cột mã ảnh)
    if image_column in frame.columns:
        duplicated_images = frame.loc[frame[image_column].duplicated(), image_column]

        if len(duplicated_images) > 0:
            raise ValueError(
                "[extract_oof]: Tệp OOF không được có ảnh trùng lặp.\n"
                f"Số ảnh bị trùng: {duplicated_images.nunique()}.\n"
                f"Ảnh bị trùng đầu tiên: {duplicated_images.iloc[0]}"
            )

    ## Mỗi bệnh nhân chỉ nằm ở một Fold (chỉ kiểm tra khi có cột mã bệnh nhân và cột Fold)
    if patient_column in frame.columns and fold_column in frame.columns:
        folds_per_patient   = frame.groupby(patient_column)[fold_column].nunique()
        leaked_patients     = folds_per_patient.index[folds_per_patient > 1]

        if len(leaked_patients) > 0:
            raise ValueError(
                "[extract_oof]: Một bệnh nhân không được nằm ở nhiều Fold (rò rỉ bệnh nhân).\n"
                f"Số bệnh nhân bị rò rỉ: {len(leaked_patients)}.\n"
                f"Bệnh nhân đầu tiên: {leaked_patients[0]}"
            )

    ## Cột nhãn và cột xác suất phải có kiểu số (cột nhãn có thể là boolean)
    non_numeric_columns = {column: str(frame[column].dtype) for column in classes + probability_columns
                           if not pd.api.types.is_numeric_dtype(frame[column])}

    if non_numeric_columns:
        raise TypeError(
            "[extract_oof]: Cột nhãn và cột xác suất phải có kiểu số (cột nhãn có thể là boolean).\n"
            f"Cột không hợp lệ: {non_numeric_columns}"
        )

    # Chuyển đổi dữ liệu
    actual_label        = frame[classes].to_numpy()                                 # Nhãn thực tế: Giá trị dữ liệu là [0, 1]
    model_prediction    = frame[probability_columns].to_numpy(dtype = np.float64)   # Xác suất dự đoán từ mô hình

    # Kiểm tra giá trị nhãn và xác suất
    ## actual_label phải chứa nhãn nhị phân (0 hoặc 1)
    invalid_label = ~np.isin(actual_label, [0, 1])

    if np.any(invalid_label):
        row, col = np.argwhere(invalid_label)[0]

        raise ValueError(
            "[extract_oof]: Cột nhãn phải chứa nhãn nhị phân (0 hoặc 1).\n"
            f"Số phần tử không hợp lệ: {np.count_nonzero(invalid_label)}.\n"
            f"Lỗi đầu tiên tại [{row}, {col}] (cột {classes[col]}): Giá trị = {actual_label[row, col]!r}"
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
            "[extract_oof]: Cột xác suất phải chứa xác suất hữu hạn trong khoảng [0, 1].\n"
            f"Số phần tử không hợp lệ: {np.count_nonzero(invalid_prediction)}.\n"
            f"Lỗi đầu tiên tại [{row}, {col}] (cột {probability_columns[col]}): Giá trị = {model_prediction[row, col]!r}"
        )

    # Nhãn chuyển về int8 (giống validate_predictions trong utilities/metric.py)
    return frame, classes, actual_label.astype(np.int8, copy = False), model_prediction

# ============================================================
# Chọn ngưỡng cho một lớp
# Trả về: (ngưỡng, trạng thái), trạng thái là một trong:
# - "optimized": Chọn được ngưỡng.
# - "insufficient_label_variation": Lớp chỉ có một loại nhãn --> dùng ngưỡng mặc định.
# - "min_recall_unmet": Không ngưỡng nào đạt Recall tối thiểu --> dùng ngưỡng mặc định.
def choose_threshold(labels, probabilities, beta = None, min_recall = None, fallback = None):

    # Lấy dữ liệu
    threshold_beta          = config.THRESHOLD_BETA if beta is None else beta
    threshold_min_recall    = config.THRESHOLD_MIN_RECALL if min_recall is None else min_recall
    default_threshold       = config.THRESHOLD if fallback is None else fallback

    # Chuyển đổi dữ liệu
    actual_label        = np.asarray(labels)                                # Nhãn thực tế của một lớp: Giá trị dữ liệu là [0, 1]
    model_prediction    = np.asarray(probabilities, dtype = np.float64)     # Xác suất dự đoán của một lớp

    # Kiểm tra dữ liệu đầu vào
    ## Hai biến phải là vector 1 chiều cùng độ dài [num_samples]
    if actual_label.ndim != 1 or actual_label.shape != model_prediction.shape:
        raise ValueError(
            "[choose_threshold]: Dữ liệu actual_label và model_prediction phải là vector 1 chiều cùng độ dài [num_samples].\n"
            f"Nhận được: actual_label = {actual_label.shape}, model_prediction = {model_prediction.shape}"
        )

    # Lớp chỉ có một loại nhãn (toàn 0 hoặc toàn 1) --> Không chọn được ngưỡng, dùng ngưỡng mặc định
    if np.unique(actual_label).size != 2:
        return float(default_threshold), "insufficient_label_variation"

    # Tính Precision và Recall tại các ngưỡng ứng viên (lấy từ các giá trị xác suất dự đoán, xếp tăng dần)
    ## Phần tử cuối của precision / recall (Precision = 1, Recall = 0) không ứng với ngưỡng nào nên bị bỏ
    precision, recall, candidates = precision_recall_curve(actual_label, model_prediction)
    precision, recall = precision[:-1], recall[:-1]

    # Tính F-beta tại từng ngưỡng (Precision = Recall = 0 --> F-beta = 0)
    beta_squared = float(threshold_beta) ** 2
    f_beta = (1 + beta_squared) * precision * recall / np.maximum(beta_squared * precision + recall, EPSILON)

    # Loại các ngưỡng có Recall thấp hơn Recall tối thiểu
    f_beta[recall < threshold_min_recall] = -1
    valid_index = np.flatnonzero(f_beta >= 0)

    # Không còn ngưỡng nào --> Dùng ngưỡng mặc định
    if len(valid_index) == 0:
        return float(default_threshold), "min_recall_unmet"

    # Chọn ngưỡng có F-beta cao nhất (np.argmax lấy vị trí đầu tiên --> ngưỡng thấp nhất khi bằng điểm)
    best_index = valid_index[np.argmax(f_beta[valid_index])]

    return float(candidates[best_index]), "optimized"

# ============================================================
# Hậu xử lý cho ensemble
# Trả về: (result, report)
def fit_postprocessing(oof_path, output_dir, *, calibration = None, beta = None, min_recall = None, default_threshold = None, patient_col = None, fold_col = "Fold", image_col = None):

    # Lấy dữ liệu
    postprocess_calibration = config.POSTPROCESS_CALIBRATION if calibration is None else calibration
    threshold_beta          = config.THRESHOLD_BETA if beta is None else beta
    threshold_min_recall    = config.THRESHOLD_MIN_RECALL if min_recall is None else min_recall
    default_threshold       = config.THRESHOLD if default_threshold is None else default_threshold

    # Kiểm tra dữ liệu đầu vào
    ## calibration phải là kiểu hiệu chuẩn được hỗ trợ
    if postprocess_calibration not in CALIBRATION_METHODS:
        raise ValueError(
            f"[fit_postprocessing]: calibration phải là một trong {CALIBRATION_METHODS}.\n"
            f"Nhận được: calibration = {postprocess_calibration!r}"
        )

    ## beta phải lớn hơn 0, min_recall phải nằm trong khoảng [0, 1]
    if threshold_beta <= 0 or not 0. <= threshold_min_recall <= 1.:
        raise ValueError(
            "[fit_postprocessing]: Cần beta > 0 và min_recall trong khoảng [0, 1].\n"
            f"Nhận được: beta = {threshold_beta}, min_recall = {threshold_min_recall}"
        )

    # Đọc tệp OOF
    _, classes, actual_label, model_prediction = extract_oof(oof_path, patient_col, fold_col, image_col)

    # Hiệu chuẩn xác suất theo lớp (calibration = "none" --> nhiệt độ T = 1, giữ nguyên xác suất)
    if postprocess_calibration == "temperature":
        temperatures            = fit_temperatures(actual_label, model_prediction)
        calibrated_prediction   = apply_temperature(model_prediction, temperatures)
    else:
        temperatures            = np.ones(len(classes))
        calibrated_prediction   = model_prediction

    # Khởi tạo biến lưu kết quả
    thresholds, selection_status = [], {}

    # Chọn ngưỡng cho từng lớp trên xác suất đã hiệu chuẩn
    for index, name in enumerate(classes):
        threshold, status = choose_threshold(actual_label[:, index], calibrated_prediction[:, index], beta = threshold_beta, min_recall = threshold_min_recall, fallback = default_threshold)
        thresholds.append(threshold)
        selection_status[name] = status

    # Báo cáo trên OOF: Ngưỡng mặc định trên xác suất gốc so với ngưỡng đã chọn trên xác suất đã hiệu chuẩn
    baseline    = class_metrics_dataframe(actual_label, model_prediction, classes, default_threshold)
    tuned       = class_metrics_dataframe(actual_label, calibrated_prediction, classes, thresholds)
    report      = baseline.merge(tuned, on = "class", suffixes = ("_baseline", "_tuned_oof"))
    report["f_beta_tuned_oof"] = [fbeta_score(actual_label[:, index], calibrated_prediction[:, index] >= thresholds[index], beta = threshold_beta, zero_division = 0)
                                  for index in range(len(classes))]
    report["selection_status"] = [selection_status[name] for name in classes]

    # Kết quả hậu xử lý
    result = {
        # Thông tin lớp và hiệu chuẩn
        "schema_version"    : 2,
        "class_order"       : classes,
        "calibration"       : postprocess_calibration,
        "temperatures"      : {name: float(temperature) for name, temperature in zip(classes, temperatures)},
        "thresholds"        : {name: float(threshold) for name, threshold in zip(classes, thresholds)},

        # Cách chọn ngưỡng
        "selection_set"     : "development_oof_only",
        "objective"         : f"F_beta={threshold_beta}",
        "min_recall"        : float(threshold_min_recall),
        "default_threshold" : float(default_threshold),
        "source_file"       : Path(oof_path).name,
        "warning"           : "Điểm F-beta trên OOF là lạc quan. Không bao giờ chọn ngưỡng trên Test.",
    }

    # Lưu kết quả
    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)
    (output_dir / "postprocess.json").write_text(json.dumps(result, indent = 2, ensure_ascii = False) + "\n", encoding = "utf-8")
    report.to_csv(output_dir / "oof_postprocess_report.csv", index = False)

    return result, report

# ============================================================
# Đọc tệp hậu xử lý (postprocess.json)
# Trả về: Nội dung tệp (dict)
def load_postprocess(path, classes):

    # Đọc tệp
    with Path(path).open(encoding = "utf-8") as file:
        data = json.load(file)

    # Kiểm tra dữ liệu đầu vào
    ## Thứ tự lớp trong tệp phải giống đầu ra mô hình
    if list(classes) != data.get("class_order"):
        raise ValueError(
            "[load_postprocess]: Thứ tự lớp trong tệp hậu xử lý phải giống đầu ra mô hình.\n"
            f"Nhận được: Tệp = {data.get('class_order')}, mô hình = {list(classes)}"
        )

    ## Kiểu hiệu chuẩn phải được hỗ trợ
    if data.get("calibration") not in CALIBRATION_METHODS:
        raise ValueError(
            f"[load_postprocess]: Kiểu hiệu chuẩn phải là một trong {CALIBRATION_METHODS}.\n"
            f"Nhận được: calibration = {data.get('calibration')!r}"
        )

    ## Tệp phải có ngưỡng của mọi lớp (và nhiệt độ của mọi lớp khi hiệu chuẩn bằng nhiệt độ)
    required_keys = ("thresholds", "temperatures") if data["calibration"] == "temperature" else ("thresholds",)

    for key in required_keys:
        missing_classes = [name for name in classes if name not in data.get(key, {})]

        if missing_classes:
            raise ValueError(
                f"[load_postprocess]: Tệp hậu xử lý thiếu {key} của một số lớp.\n"
                f"Thiếu: {missing_classes}"
            )

    return data

# ============================================================
# Áp hậu xử lý lên xác suất dự đoán
# Trả về: (xác suất sau hậu xử lý [num_samples, num_classes], ngưỡng [num_classes])
def apply_postprocess(probabilities, data, classes):

    # Chuyển đổi dữ liệu
    model_prediction = np.asarray(probabilities, dtype = np.float64)        # Xác suất dự đoán từ mô hình

    # Kiểm tra dữ liệu đầu vào
    ## Thứ tự lớp phải giống tệp hậu xử lý
    if list(classes) != data["class_order"]:
        raise ValueError(
            "[apply_postprocess]: Thứ tự lớp phải giống tệp hậu xử lý.\n"
            f"Nhận được: classes = {list(classes)}, class_order = {data['class_order']}"
        )

    ## model_prediction phải có 2 chiều [num_samples, num_classes]
    if model_prediction.ndim != 2 or model_prediction.shape[1] != len(classes):
        raise ValueError(
            "[apply_postprocess]: Dữ liệu probabilities phải có 2 chiều [num_samples, num_classes].\n"
            f"Nhận được: probabilities = {model_prediction.shape}, num_classes = {len(classes)}"
        )

    # Hiệu chuẩn xác suất bằng nhiệt độ của từng lớp (calibration = "none" --> giữ nguyên)
    if data["calibration"] == "temperature":
        temperatures            = [data["temperatures"][name] for name in classes]
        calibrated_prediction   = apply_temperature(model_prediction, temperatures)
    else:
        calibrated_prediction   = model_prediction

    # Lấy ngưỡng theo thứ tự lớp
    thresholds = np.asarray([data["thresholds"][name] for name in classes], dtype = np.float64)

    return calibrated_prediction, thresholds

# ============================================================
# Hậu xử lý một Fold
# Trả về: (postprocess, summary)
def fit_fold_thresholds(oof_path, output_dir, *, expected_classes = None, calibration_fraction = None, seed = None, beta = None, min_recall = None, default_threshold = None):

    # Lấy dữ liệu
    threshold_calibration_fraction  = config.THRESHOLD_CALIBRATION_FRACTION if calibration_fraction is None else calibration_fraction
    split_seed                      = config.SEED if seed is None else seed
    threshold_beta                  = config.THRESHOLD_BETA if beta is None else beta
    threshold_min_recall            = config.THRESHOLD_MIN_RECALL if min_recall is None else min_recall
    default_threshold               = config.THRESHOLD if default_threshold is None else default_threshold
    patient_column                  = config.PATIENT_ID_COLUMN

    # Kiểm tra dữ liệu đầu vào
    ## calibration_fraction phải nằm trong khoảng (0, 1)
    if not 0 < threshold_calibration_fraction < 1:
        raise ValueError(
            "[fit_fold_thresholds]: calibration_fraction phải nằm trong khoảng (0, 1).\n"
            f"Nhận được: calibration_fraction = {threshold_calibration_fraction}"
        )

    ## beta phải lớn hơn 0, min_recall phải nằm trong khoảng [0, 1]
    if threshold_beta <= 0 or not 0 <= threshold_min_recall <= 1:
        raise ValueError(
            "[fit_fold_thresholds]: Cần beta > 0 và min_recall trong khoảng [0, 1].\n"
            f"Nhận được: beta = {threshold_beta}, min_recall = {threshold_min_recall}"
        )

    # Đọc tệp OOF của Fold
    frame, classes, actual_label, model_prediction = extract_oof(oof_path)

    # Kiểm tra dữ liệu OOF
    ## Thứ tự lớp phải giống đầu ra mô hình
    if expected_classes is not None and list(classes) != list(expected_classes):
        raise ValueError(
            "[fit_fold_thresholds]: Thứ tự lớp trong tệp OOF phải giống đầu ra mô hình.\n"
            f"Nhận được: Tệp OOF = {list(classes)}, mô hình = {list(expected_classes)}"
        )

    ## Phải có cột mã bệnh nhân để tách không rò rỉ bệnh nhân
    if patient_column not in frame.columns:
        raise ValueError(
            f"[fit_fold_thresholds]: Tệp OOF phải có cột {patient_column} để tách theo bệnh nhân.\n"
            f"Nhận được: Các cột = {list(frame.columns)}"
        )

    ## Cột mã bệnh nhân không được thiếu giá trị
    missing_patient = frame[patient_column].isna().to_numpy()

    if np.any(missing_patient):
        raise ValueError(
            f"[fit_fold_thresholds]: Cột {patient_column} không được thiếu giá trị.\n"
            f"Số dòng bị thiếu: {np.count_nonzero(missing_patient)}.\n"
            f"Lỗi đầu tiên tại dòng {np.flatnonzero(missing_patient)[0]}"
        )

    ## Phải có ít nhất 2 bệnh nhân để tách thành 2 phần
    patient_ids         = frame[patient_column].astype(str).to_numpy()
    number_of_patients  = len(np.unique(patient_ids))

    if number_of_patients < 2:
        raise ValueError(
            "[fit_fold_thresholds]: Cần ít nhất 2 bệnh nhân để tách phần chọn ngưỡng / phần đánh giá.\n"
            f"Nhận được: Số bệnh nhân = {number_of_patients}"
        )

    # Tách theo bệnh nhân: Mỗi bệnh nhân chỉ nằm ở một phần
    splitter = GroupShuffleSplit(n_splits = 1, test_size = 1 - threshold_calibration_fraction, random_state = int(split_seed))
    selection_index, evaluation_index = next(splitter.split(model_prediction, actual_label, groups = patient_ids))

    ## Kiểm tra lại: Không có bệnh nhân nằm ở cả hai phần
    leaked_patients = np.intersect1d(patient_ids[selection_index], patient_ids[evaluation_index])

    if leaked_patients.size > 0:
        raise AssertionError(
            "[fit_fold_thresholds]: Rò rỉ bệnh nhân giữa phần chọn ngưỡng và phần đánh giá.\n"
            f"Số bệnh nhân bị rò rỉ: {leaked_patients.size}.\n"
            f"Bệnh nhân đầu tiên: {leaked_patients[0]}"
        )

    # Khởi tạo biến lưu kết quả
    thresholds, selection_status = [], {}

    # Chọn ngưỡng cho từng lớp trên phần chọn ngưỡng
    for index, name in enumerate(classes):
        threshold, status = choose_threshold(actual_label[selection_index, index], model_prediction[selection_index, index], beta = threshold_beta, min_recall = threshold_min_recall, fallback = default_threshold)
        thresholds.append(threshold)
        selection_status[name] = status

    threshold_array = np.asarray(thresholds, dtype = np.float64)

    # Lấy nhãn và xác suất của phần đánh giá
    evaluation_label        = actual_label[evaluation_index]
    evaluation_prediction   = model_prediction[evaluation_index]

    # Đánh giá trên phần đánh giá: Ngưỡng mặc định so với ngưỡng đã chọn
    baseline        = calculate_metrics(len(classes), evaluation_label, evaluation_prediction, default_threshold)
    tuned           = calculate_metrics(len(classes), evaluation_label, evaluation_prediction, threshold_array)
    baseline_report = class_metrics_dataframe(evaluation_label, evaluation_prediction, classes, default_threshold)
    tuned_report    = class_metrics_dataframe(evaluation_label, evaluation_prediction, classes, threshold_array)
    report          = baseline_report.merge(tuned_report, on = "class", suffixes = ("_default", "_tuned"))
    report["fit_status"] = [selection_status[name] for name in classes]

    # Cảnh báo ghi kèm kết quả
    warning = ("Không dùng phần chọn ngưỡng để báo cáo F1. Fold này cũng đã dùng để chọn checkpoint; "
               "đánh giá khách quan cuối cùng là trên tập Test.")

    # Kết quả hậu xử lý (nội dung postprocess.json)
    postprocess = {
        # Thông tin lớp và hiệu chuẩn
        "class_order"           : classes,
        "calibration"           : "none",
        "temperatures"          : {name: 1.0 for name in classes},
        "thresholds"            : {name: float(threshold) for name, threshold in zip(classes, thresholds)},

        # Cách chọn ngưỡng
        "selection_set"         : "fold_oof_patient_group_calibration_subset",
        "objective"             : f"F_beta={threshold_beta}",
        "min_recall"            : float(threshold_min_recall),
        "default_threshold"     : float(default_threshold),
        "source_file"           : Path(oof_path).name,

        # Cách tách theo bệnh nhân
        "patient_split_seed"    : int(split_seed),
        "calibration_images"    : int(len(selection_index)),
        "evaluation_images"     : int(len(evaluation_index)),
        "warning"               : warning,
    }

    # Tóm tắt kết quả (nội dung fold_threshold_summary.json)
    summary = {
        "fold"                  : int(frame["Fold"].iloc[0]) if "Fold" in frame and frame["Fold"].nunique() == 1 else None,
        "split"                 : "patient-disjoint within fold OOF",
        "calibration_images"    : int(len(selection_index)),
        "evaluation_images"     : int(len(evaluation_index)),
        "baseline"              : {key: value for key, value in baseline.items() if not key.startswith("class_")},
        "optimized"             : {key: value for key, value in tuned.items() if not key.startswith("class_")},
        "warning"               : warning,
    }

    # Lưu kết quả
    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)
    (output_dir / "postprocess.json").write_text(json.dumps(postprocess, ensure_ascii = False, indent = 2) + "\n", encoding = "utf-8")
    (output_dir / "fold_threshold_summary.json").write_text(json.dumps(summary, ensure_ascii = False, indent = 2) + "\n", encoding = "utf-8")
    report.to_csv(output_dir / "fold_threshold_report.csv", index = False)

    # In kết quả
    print("\n[CHỌN NGƯỠNG TỰ ĐỘNG]")
    print(f"Ảnh chọn ngưỡng: {len(selection_index):,} | Ảnh đánh giá: {len(evaluation_index):,}")
    print(f"Macro F1 trên phần đánh giá: {baseline['macro_f1']:.4f} -> {tuned['macro_f1']:.4f}")
    print(f"Đã lưu ngưỡng: {output_dir / 'postprocess.json'}")

    return postprocess, summary
