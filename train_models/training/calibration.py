# ============================================================
# Hiệu chuẩn xác suất (Probability Calibration)
# Mục tiêu: Điều chỉnh xác suất dự đoán của mô hình sao cho mức độ tự tin phù hợp với tỷ lệ dương tính thực tế.

# Ví dụ: Trong một nhóm mẫu có tỷ lệ dương tính thực tế là 70%:
# - Mô hình dự đoán 50%: Thiếu tự tin (Underconfidence).
# - Mô hình dự đoán 90%: Quá tự tin (Overconfidence).
# - Mô hình dự đoán 70%: Hiệu chuẩn tốt (Well-calibrated).

# Phương pháp: Temperature scaling theo từng lớp, xác suất mới = sigmoid(logit(xác suất) / T)
# - T > 1: Giảm độ tự tin (sửa Overconfidence).
# - T < 1: Tăng độ tự tin (sửa Underconfidence).
# - T = 1: Giữ nguyên xác suất.
# ============================================================

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

import config

EPSILON = 1e-6                                                              # Giới hạn xác suất trong [EPSILON, 1 - EPSILON] trước khi lấy logit (tránh logit = ±vô cực)

# ============================================================
# Tìm nhiệt độ tối ưu cho từng lớp: Tối thiểu Binary Cross-Entropy giữa nhãn thực tế và xác suất sau hiệu chuẩn
# Trả về: Vector nhiệt độ [num_classes]
def fit_temperatures(labels, probabilities, min_positive = None, min_negative = None):
    
    # Lấy dữ liệu
    temperature_min_positive     = config.TEMPERATURE_MIN_POSITIVE if min_positive is None else min_positive
    temperature_min_negative     = config.TEMPERATURE_MIN_NEGATIVE if min_negative is None else min_negative

    # Chuyển đổi dữ liệu
    actual_label        = np.asarray(labels)                                # Nhãn thực tế: Giá trị dữ liệu là [0, 1]
    model_prediction    = np.asarray(probabilities, dtype = np.float64)     # Xác suất dự đoán từ mô hình

    # Kiểm tra dữ liệu đầu vào
    ## Hai biến phải có 2 chiều [num_samples, num_classes]
    if actual_label.ndim != 2 or model_prediction.ndim != 2:
        raise ValueError(
            "[fit_temperatures]: Dữ liệu actual_label và model_prediction phải có 2 chiều [num_samples, num_classes].\n"
            f"Nhận được: actual_label = {actual_label.shape}, model_prediction = {model_prediction.shape}"
        )

    ## Hai biến phải có cùng kích thước
    if actual_label.shape != model_prediction.shape:
        raise ValueError(
            "[fit_temperatures]: Kích thước actual_label và model_prediction phải giống nhau.\n"
            f"Nhận được: actual_label = {actual_label.shape}, model_prediction = {model_prediction.shape}"
        )

    ## Dữ liệu không được rỗng
    if actual_label.size == 0:
        raise ValueError(
            "[fit_temperatures]: Dữ liệu đầu vào không được rỗng.\n"
            f"Nhận được: actual_label = {actual_label.shape}"
        )

    ## Hai biến phải chứa dữ liệu số thực hoặc nhãn boolean
    if actual_label.dtype.kind not in "biuf" or model_prediction.dtype.kind not in "biuf":
        raise TypeError(
            "[fit_temperatures]: Dữ liệu actual_label và model_prediction phải có kiểu số thực hoặc số nguyên (actual_label có thể là boolean).\n"
            f"Nhận được: actual_label = {actual_label.dtype}, model_prediction = {model_prediction.dtype}"
        )

    ## actual_label phải chứa nhãn nhị phân (0 hoặc 1)
    invalid_label = ~np.isin(actual_label, [0, 1])

    if np.any(invalid_label):
        row, col = np.argwhere(invalid_label)[0]

        raise ValueError(
            "[fit_temperatures]: Dữ liệu actual_label phải chứa nhãn nhị phân (0 hoặc 1).\n"
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
            "[fit_temperatures]: Dữ liệu model_prediction phải chứa xác suất hữu hạn trong khoảng [0, 1].\n"
            f"Số phần tử không hợp lệ: {np.count_nonzero(invalid_prediction)}.\n"
            f"Lỗi đầu tiên tại [{row}, {col}]: Giá trị = {model_prediction[row, col]!r}"
        )

    # Chuyển xác suất sang logit
    logits = logit(np.clip(model_prediction, EPSILON, 1 - EPSILON))
    fitted = []

    # Kiểm tra số lượng dương tính và âm tính theo lớp
    for index in range(model_prediction.shape[1]):
        
        ## Lớp có quá ít mẫu dương tính hoặc âm tính --> giữ T = 1 (không hiệu chuẩn)
        if actual_label[:, index].sum() < temperature_min_positive or (len(actual_label) - actual_label[:, index].sum()) < temperature_min_negative:
            fitted.append(1.0)
            continue

        # Xây dựng hàm mục tiêu: Binary Cross-Entropy giữa nhãn y và sigmoid(z), với z = logit / T
        def objective(log_temp):
            
            ## Công thức BCE = log(1 + e^z) - y * z, trong đó logaddexp(0, z) = log(1 + e^z) không bị tràn số khi z lớn
            ## Tối ưu theo log(T) thay vì T để T = exp(log(T)) luôn dương
            scaled_logits = logits[:, index] * np.exp(-log_temp)
            return float(np.mean(np.logaddexp(0., scaled_logits) - actual_label[:, index] * scaled_logits))

        # Tối ưu nhiệt độ: Tìm log(T) trong khoảng TEMPERATURE_BOUNDS
        result = minimize_scalar(objective, bounds = (np.log(config.TEMPERATURE_BOUNDS[0]), np.log(config.TEMPERATURE_BOUNDS[1])), method = "bounded")

        # Tìm nghiệm: T = exp(log(T)); tối ưu thất bại --> giữ T = 1
        fitted.append(float(np.exp(result.x)) if result.success else 1.0)

    return np.asarray(fitted, dtype = np.float64)

# ============================================================
# Áp dụng nhiệt độ để điều chỉnh xác suất
# Trả về: Xác suất đã hiệu chuẩn [num_samples, num_classes]
def apply_temperature(probabilities, temperatures):
    
    # Lấy dữ liệu
    probabilities = np.asarray(probabilities, dtype = np.float64)
    temperatures = np.asarray(temperatures, dtype = np.float64)

    # Kiểm tra dữ liệu đầu vào
    ## Kiểm tra kích thước của probabilities
    if probabilities.ndim != 2:
        raise ValueError(
            "[apply_temperature]: Dữ liệu probabilities phải có 2 chiều [num_samples, num_classes].\n"
            f"Nhận được: probabilities = {probabilities.shape}"
        )

    ## Kiểm tra kích thước của temperatures
    if temperatures.shape != (probabilities.shape[1],):
        raise ValueError(
            "[apply_temperature]: Số lượng dữ liệu của temperatures phải bằng số lớp num_classes của probabilities.\n"
            f"Nhận được: temperatures = {temperatures.shape}, probabilities = {probabilities.shape}"
        )

    ## Kiểm tra xác suất hợp lệ
    invalid_probabilities = ~np.isfinite(probabilities) | (probabilities < 0) | (probabilities > 1)

    if np.any(invalid_probabilities):
        row, col = np.argwhere(invalid_probabilities)[0]

        raise ValueError(
            "[apply_temperature]: Dữ liệu probabilities phải chứa xác suất hữu hạn trong khoảng [0, 1].\n"
            f"Số phần tử không hợp lệ: {np.count_nonzero(invalid_probabilities)}.\n"
            f"Lỗi đầu tiên tại [{row}, {col}]: Giá trị = {probabilities[row, col]!r}"
        )

    ## Kiểm tra nhiệt độ hợp lệ (temperatures là vector 1 chiều [num_classes])
    invalid_temperatures = ~np.isfinite(temperatures) | (temperatures <= 0)

    if np.any(invalid_temperatures):
        index = np.flatnonzero(invalid_temperatures)[0]

        raise ValueError(
            "[apply_temperature]: Dữ liệu temperatures phải hữu hạn và lớn hơn 0.\n"
            f"Số phần tử không hợp lệ: {np.count_nonzero(invalid_temperatures)}.\n"
            f"Lỗi đầu tiên tại lớp {index}: Giá trị = {temperatures[index]!r}"
        )

    # Chuyển xác suất sang Logit, sau đó chia cho Temperature
    logits = logit(np.clip(probabilities, EPSILON, 1 - EPSILON))
    scaled_logits = logits / temperatures[None, :]

    # Chuyển Logit đã hiệu chỉnh về xác suất bằng Sigmoid
    return expit(scaled_logits)
