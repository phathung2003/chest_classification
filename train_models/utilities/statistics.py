# ============================================================
# Thống kê bootstrap theo bệnh nhân (Patient-level Bootstrap)
# Mục tiêu: Ước lượng khoảng tin cậy của AUROC và so sánh ghép cặp hai mô hình trên cùng tập Test.

# Phương pháp: Lấy mẫu lại có hoàn lại theo BỆNH NHÂN (không theo ảnh) vì các ảnh của cùng một bệnh nhân tương quan với nhau.
# - Khoảng tin cậy 95%: Phân vị 2.5% và 97.5% của AUROC trên các mẫu bootstrap.
# - So sánh hai mô hình: Hiệu AUROC (A - B) tính trên cùng các mẫu bootstrap; khoảng tin cậy không chứa 0 --> khác biệt có ý nghĩa.
# ============================================================

import numpy as np
from scipy.stats import rankdata

import config
from utilities.metric import validate_predictions

# ============================================================
# Tính AUROC theo từng lớp
# Trả về: Vector AUROC [num_classes]
def auroc_per_class(targets, probabilities):

    # Chuyển đổi dữ liệu
    actual_label        = np.asarray(targets, dtype = np.float64)            # Nhãn thực tế: Giá trị dữ liệu là [0, 1]
    model_prediction    = np.asarray(probabilities, dtype = np.float64)      # Xác suất dự đoán từ mô hình
    
    # Kiểm tra hai biến phải là hai ma trận [num_samples, num_classes] cùng kích thước
    if actual_label.ndim != 2 or actual_label.shape != model_prediction.shape:
        raise ValueError(
            "[auroc_per_class]: Dữ liệu targets và probabilities phải là hai ma trận [num_samples, num_classes] cùng kích thước.\n"
            f"Nhận được: targets = {actual_label.shape}, probabilities = {model_prediction.shape}"
        )

    # Hạng của xác suất trong từng lớp
    ranks = rankdata(model_prediction, axis = 0)

    # Số mẫu dương / âm của từng lớp
    number_positive = actual_label.sum(axis = 0)
    number_negative = actual_label.shape[0] - number_positive

    # Tính AUROC theo công thức Mann-Whitney
    with np.errstate(divide = "ignore", invalid = "ignore"):
        auroc = ((ranks * actual_label).sum(axis = 0) - number_positive * (number_positive + 1) / 2) / (number_positive * number_negative)

    # Lớp thiếu mẫu dương hoặc mẫu âm --> NaN
    auroc[(number_positive == 0) | (number_negative == 0)] = np.nan

    return auroc

# ============================================================
# Tính trung bình trên các giá trị hữu hạn (macro AUROC bỏ qua lớp NaN)
# Trả về: Giá trị trung bình; không có giá trị hữu hạn --> NaN
def _macro(values):
    values = np.asarray(values, dtype = np.float64)
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else float("nan")

# ============================================================
# Gom chỉ số ảnh theo bệnh nhân
# Trả về: Danh sách mảng chỉ số ảnh, mỗi phần tử là các ảnh của một bệnh nhân
def _patient_index(groups):

    # Chuyển đổi dữ liệu
    patient_ids = np.asarray(groups)

    # Kiểm tra dữ liệu đầu vào
    ## groups phải là mảng 1 chiều, không rỗng
    if patient_ids.ndim != 1 or len(patient_ids) == 0:
        raise ValueError(
            "[_patient_index]: groups (mã bệnh nhân) phải là mảng 1 chiều, không rỗng.\n"
            f"Nhận được: groups = {patient_ids.shape}"
        )

    ## Mã bệnh nhân không được thiếu
    missing_patient = [index for index, patient in enumerate(patient_ids) if patient is None or (isinstance(patient, float) and np.isnan(patient))]

    if missing_patient:
        raise ValueError(
            "[_patient_index]: Mã bệnh nhân không được thiếu.\n"
            f"Số ảnh bị thiếu: {len(missing_patient)}.\n"
            f"Lỗi đầu tiên tại ảnh {missing_patient[0]}"
        )

    # Đánh số bệnh nhân theo mã bệnh nhân (dạng chuỗi)
    _, inverse = np.unique(patient_ids.astype(str), return_inverse = True)

    # Sắp xếp ổn định theo mã bệnh nhân
    order = np.argsort(inverse, kind = "stable")

    # Vị trí đổi bệnh nhân
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1

    # Cắt tại các vị trí đổi bệnh nhân
    return np.split(order, boundaries)

# ============================================================
# Sinh các mẫu bootstrap theo bệnh nhân
# Trả về: Generator, mỗi lần một mảng chỉ số ảnh
def _bootstrap_samples(groups, bootstrap_iteration, seed):

    # Chỉ số ảnh của từng bệnh nhân
    patients = _patient_index(groups)

    # Bộ sinh ngẫu nhiên riêng (không ảnh hưởng bộ sinh ngẫu nhiên toàn cục)
    rng = np.random.default_rng(int(seed))

    # Sinh bootstrap_iteration mẫu
    for _ in range(int(bootstrap_iteration)):

        # Chọn ngẫu nhiên có hoàn lại đủ số bệnh nhân, lấy mọi ảnh của bệnh nhân được chọn
        chosen = rng.integers(0, len(patients), size = len(patients))
        yield np.concatenate([patients[index] for index in chosen])

# ============================================================
# Tính khoảng tin cậy phân vị (percentile)
# Trả về: (cận dưới, cận trên); không có mẫu hữu hạn --> (None, None)
def _interval(samples, alpha):

    # Chỉ giữ các mẫu hữu hạn
    samples = np.asarray(samples, dtype = np.float64)
    samples = samples[np.isfinite(samples)]

    # Không có mẫu hữu hạn --> Không tính được
    if len(samples) == 0:
        return None, None

    # Lấy phân vị hai đầu
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)

# ============================================================
# Tính khoảng tin cậy bootstrap theo bệnh nhân
# Trả về: dict gồm phương pháp, macro_auroc (ước lượng, ci_low, ci_high) và classes (mỗi lớp một dòng)
def bootstrap_auroc_confidence_interval(targets, probabilities, groups, classes, *, bootstrap_iteration = None, seed = None, alpha = None):

    # Lấy dữ liệu
    bootstrap_iteration     = config.BOOTSTRAP_ITERATION if bootstrap_iteration is None else bootstrap_iteration
    bootstrap_seed          = config.BOOTSTRAP_SEED if seed is None else seed
    bootstrap_alpha         = round(1 - config.BOOTSTRAP_CONFIDENCE, 12) if alpha is None else alpha

    # Kiểm tra dữ liệu đầu vào (nhãn và xác suất)
    validate_predictions(targets, probabilities)

    # Chuyển đổi dữ liệu
    actual_label        = np.asarray(targets, dtype = np.float64)            # Nhãn thực tế: Giá trị dữ liệu là [0, 1]
    model_prediction    = np.asarray(probabilities, dtype = np.float64)      # Xác suất dự đoán từ mô hình
    
    ## Kiểm tra mỗi ảnh là một mã bệnh nhân
    if len(groups) != len(actual_label):
        raise ValueError(
            "[bootstrap_auroc_confidence_interval]: Số mã bệnh nhân phải bằng số ảnh.\n"
            f"Nhận được: groups = {len(groups)}, số ảnh = {len(actual_label)}"
        )

    # Tính AUROC trên toàn bộ dữ liệu (ước lượng điểm)
    point = auroc_per_class(actual_label, model_prediction)

    # Khởi tạo biến lưu kết quả
    per_class, macro = [], []

    # Tính AUROC trên từng mẫu bootstrap
    for sample_index in _bootstrap_samples(groups, bootstrap_iteration, bootstrap_seed):
        auroc = auroc_per_class(actual_label[sample_index], model_prediction[sample_index])
        per_class.append(auroc)
        macro.append(_macro(auroc))

    per_class = np.asarray(per_class)

    # Tính khoảng tin cậy của macro AUROC
    macro_low, macro_high = _interval(macro, bootstrap_alpha)

    # Tính khoảng tin cậy của từng lớp
    class_rows = []

    for index, name in enumerate(classes):
        low, high = _interval(per_class[:, index], bootstrap_alpha)
        class_rows.append({"class": name, "auroc": float(point[index]),
                           "ci_low": low, "ci_high": high})

    return {
        "method": "patient-level percentile bootstrap",
        "bootstrap_iteration": int(bootstrap_iteration), "seed": int(bootstrap_seed), "confidence": 1 - bootstrap_alpha,
        "macro_auroc": {"estimate": _macro(point), "ci_low": macro_low, "ci_high": macro_high},
        "classes": class_rows,
    }

# ============================================================
# So sánh ghép cặp hai mô hình trên cùng tập ảnh
# Trả về: dict gồm phương pháp, macro_auroc (a, b, difference, ci_low, ci_high, p_value) và classes (mỗi lớp một dòng)
def paired_bootstrap_auroc(targets, probabilities_a, probabilities_b, groups, classes, *, bootstrap_iteration = None, seed = None, alpha = None):

    # Lấy dữ liệu
    compare_bootstrap_iteration = config.COMPARE_BOOTSTRAP_ITERATION if bootstrap_iteration is None else bootstrap_iteration
    bootstrap_seed              = config.BOOTSTRAP_SEED if seed is None else seed
    bootstrap_alpha             = round(1 - config.BOOTSTRAP_CONFIDENCE, 12) if alpha is None else alpha

    # Kiểm tra dữ liệu đầu vào (nhãn và xác suất của hai mô hình)
    validate_predictions(targets, probabilities_a)
    validate_predictions(targets, probabilities_b)

    # Chuyển đổi dữ liệu
    actual_label    = np.asarray(targets, dtype = np.float64)                # Nhãn thực tế: Giá trị dữ liệu là [0, 1]
    prediction_a    = np.asarray(probabilities_a, dtype = np.float64)        # Xác suất dự đoán của mô hình A
    prediction_b    = np.asarray(probabilities_b, dtype = np.float64)        # Xác suất dự đoán của mô hình B
    
    ## Kiểm tra kích thước của ma trận dự đoán của hai mô hình, nhãn và mã bệnh nhân phải cùng số ảnh và số lớp
    if prediction_a.shape != actual_label.shape or prediction_b.shape != actual_label.shape or len(groups) != len(actual_label):
        raise ValueError(
            "[paired_bootstrap_auroc]: Dự đoán của hai mô hình, nhãn và mã bệnh nhân phải cùng số ảnh và số lớp.\n"
            f"Nhận được: targets = {actual_label.shape}, probabilities_a = {prediction_a.shape}, probabilities_b = {prediction_b.shape}, groups = {len(groups)}"
        )

    # Tính AUROC của từng mô hình trên toàn bộ dữ liệu (ước lượng điểm)
    point_a, point_b = auroc_per_class(actual_label, prediction_a), auroc_per_class(actual_label, prediction_b)

    # Khởi tạo biến lưu kết quả
    difference_class, difference_macro = [], []

    # Tính hiệu AUROC (A - B) trên từng mẫu bootstrap
    for sample_index in _bootstrap_samples(groups, compare_bootstrap_iteration, bootstrap_seed):
        auroc_a = auroc_per_class(actual_label[sample_index], prediction_a[sample_index])
        auroc_b = auroc_per_class(actual_label[sample_index], prediction_b[sample_index])
        difference_class.append(auroc_a - auroc_b)
        difference_macro.append(_macro(auroc_a) - _macro(auroc_b))

    difference_class, difference_macro = np.asarray(difference_class), np.asarray(difference_macro)

    # Hàm tính p-value bootstrap hai phía (Tỉ lệ mẫu nằm phía bên kia của 0, nhân 2, có hiệu chỉnh +1 nên không bao giờ bằng 0)
    def two_sided_p(samples):

        # Chỉ giữ các mẫu hữu hạn
        samples = samples[np.isfinite(samples)]

        # Không có mẫu hữu hạn --> Không tính được
        if len(samples) == 0:
            return None

        # Số mẫu ở phía ít hơn so với 0
        tail = min((samples <= 0).sum(), (samples >= 0).sum())
        return float(min(1.0, 2 * (tail + 1) / (len(samples) + 1)))

    # Tính khoảng tin cậy của hiệu macro AUROC
    macro_low, macro_high = _interval(difference_macro, bootstrap_alpha)

    # Tính khoảng tin cậy và p-value của từng lớp
    rows = []

    for index, name in enumerate(classes):
        low, high = _interval(difference_class[:, index], bootstrap_alpha)
        rows.append({"class": name, "auroc_a": float(point_a[index]), "auroc_b": float(point_b[index]),
                     "difference": float(point_a[index] - point_b[index]),
                     "ci_low": low, "ci_high": high, "p_value": two_sided_p(difference_class[:, index])})

    return {
        "method": "patient-level paired percentile bootstrap",
        "bootstrap_iteration": int(compare_bootstrap_iteration), "seed": int(bootstrap_seed), "confidence": 1 - bootstrap_alpha,
        "macro_auroc": {
            "a": _macro(point_a), "b": _macro(point_b),
            "difference": _macro(point_a) - _macro(point_b),
            "ci_low": macro_low, "ci_high": macro_high,
            "p_value": two_sided_p(difference_macro),
        },
        "classes": rows,
    }
