import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

import config

_EPS = 1e-6

# ============================================================
# Temperature scaling
def fit_temperatures(labels, probabilities, min_positive = None, min_negative = None):
    min_positive = config.TEMPERATURE_MIN_POSITIVE if min_positive is None else min_positive
    min_negative = config.TEMPERATURE_MIN_NEGATIVE if min_negative is None else min_negative
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype = np.float64)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("Nhãn và xác suất phải cùng kích thước [N, C]")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Cần nhãn nhị phân và xác suất hợp lệ")

    logits = logit(np.clip(p, _EPS, 1 - _EPS))
    fitted = []
    for k in range(p.shape[1]):
        if y[:, k].sum() < min_positive or (len(y) - y[:, k].sum()) < min_negative:
            fitted.append(1.0)
            continue

        # Tối ưu log(T) để T luôn dương và ổn định số học
        def objective(log_temp):
            z = logits[:, k] * np.exp(-log_temp)
            return float(np.mean(np.logaddexp(0., z) - y[:, k] * z))

        result = minimize_scalar(objective, bounds = (np.log(config.TEMPERATURE_BOUNDS[0]), np.log(config.TEMPERATURE_BOUNDS[1])), method = "bounded")
        fitted.append(float(np.exp(result.x)) if result.success else 1.0)
    return np.asarray(fitted, dtype = np.float64)

# ============================================================
# Áp dụng temperature lên xác suất
def apply_temperatures(probabilities, temperatures):
    p = np.asarray(probabilities, dtype = np.float64)
    temperatures = np.asarray(temperatures, dtype = np.float64)
    if p.ndim != 2 or temperatures.shape != (p.shape[1],) or np.any(temperatures <= 0):
        raise ValueError("Vector temperature không hợp lệ")
    return expit(logit(np.clip(p, _EPS, 1 - _EPS)) / temperatures[None, :])
