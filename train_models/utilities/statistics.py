import numpy as np
from scipy.stats import rankdata

import config

# ============================================================
# AUROC theo từng lớp (vector hóa bằng hạng, xử lý giá trị trùng bằng hạng trung bình)
# Kết quả trùng với sklearn.metrics.roc_auc_score; lớp chỉ có một nhãn trả về NaN.
def auroc_per_class(targets, probabilities):
    y = np.asarray(targets, dtype = np.float64)
    p = np.asarray(probabilities, dtype = np.float64)
    if y.ndim != 2 or y.shape != p.shape:
        raise ValueError(f"Cần hai ma trận [N, C] cùng kích thước, nhận {y.shape} và {p.shape}")
    ranks = rankdata(p, axis = 0)
    n_pos = y.sum(axis = 0)
    n_neg = y.shape[0] - n_pos
    with np.errstate(divide = "ignore", invalid = "ignore"):
        auc = ((ranks * y).sum(axis = 0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    auc[(n_pos == 0) | (n_neg == 0)] = np.nan
    return auc


def _macro(values):
    values = np.asarray(values, dtype = np.float64)
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else float("nan")


# ============================================================
# Chỉ số ảnh của từng bệnh nhân (bootstrap theo bệnh nhân vì một bệnh nhân có nhiều ảnh)
def _patient_index(groups):
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("groups phải là mảng 1 chiều, không rỗng")
    if any(g is None or (isinstance(g, float) and np.isnan(g)) for g in groups):
        raise ValueError("Mã bệnh nhân không được thiếu")
    _, inverse = np.unique(groups.astype(str), return_inverse = True)
    order = np.argsort(inverse, kind = "stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(order, boundaries)


def _bootstrap_samples(groups, n_boot, seed):
    patients = _patient_index(groups)
    rng = np.random.default_rng(int(seed))
    for _ in range(int(n_boot)):
        chosen = rng.integers(0, len(patients), size = len(patients))
        yield np.concatenate([patients[k] for k in chosen])


def _interval(samples, alpha):
    samples = np.asarray(samples, dtype = np.float64)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return None, None
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


# ============================================================
# Khoảng tin cậy bootstrap (percentile) của AUROC từng lớp và macro AUROC
def bootstrap_auroc_ci(targets, probabilities, groups, classes, *, n_boot = None, seed = None, alpha = None):
    n_boot = config.BOOTSTRAP_ITERATIONS if n_boot is None else n_boot
    seed = config.BOOTSTRAP_SEED if seed is None else seed
    alpha = round(1 - config.BOOTSTRAP_CONFIDENCE, 12) if alpha is None else alpha
    y = np.asarray(targets, dtype = np.float64)
    p = np.asarray(probabilities, dtype = np.float64)
    if len(groups) != len(y):
        raise ValueError("Số mã bệnh nhân không khớp số ảnh")
    point = auroc_per_class(y, p)
    per_class, macro = [], []
    for idx in _bootstrap_samples(groups, n_boot, seed):
        auc = auroc_per_class(y[idx], p[idx])
        per_class.append(auc)
        macro.append(_macro(auc))
    per_class = np.asarray(per_class)
    macro_low, macro_high = _interval(macro, alpha)
    class_rows = []
    for i, name in enumerate(classes):
        low, high = _interval(per_class[:, i], alpha)
        class_rows.append({"class": name, "auroc": float(point[i]),
                           "ci_low": low, "ci_high": high})
    return {
        "method": "patient-level percentile bootstrap",
        "n_boot": int(n_boot), "seed": int(seed), "confidence": 1 - alpha,
        "macro_auroc": {"estimate": _macro(point), "ci_low": macro_low, "ci_high": macro_high},
        "classes": class_rows,
    }


# ============================================================
# So sánh ghép cặp hai mô hình trên CÙNG tập ảnh: hiệu AUROC (A - B), khoảng tin cậy và p-value hai phía
def paired_bootstrap_auroc(targets, probabilities_a, probabilities_b, groups, classes, *,
                           n_boot = None, seed = None, alpha = None):
    n_boot = config.COMPARE_BOOTSTRAP_ITERATIONS if n_boot is None else n_boot
    seed = config.BOOTSTRAP_SEED if seed is None else seed
    alpha = round(1 - config.BOOTSTRAP_CONFIDENCE, 12) if alpha is None else alpha
    y = np.asarray(targets, dtype = np.float64)
    pa = np.asarray(probabilities_a, dtype = np.float64)
    pb = np.asarray(probabilities_b, dtype = np.float64)
    if pa.shape != y.shape or pb.shape != y.shape or len(groups) != len(y):
        raise ValueError("Dự đoán của hai mô hình, nhãn và mã bệnh nhân phải cùng số ảnh và số lớp")

    point_a, point_b = auroc_per_class(y, pa), auroc_per_class(y, pb)
    diff_class, diff_macro = [], []
    for idx in _bootstrap_samples(groups, n_boot, seed):
        a, b = auroc_per_class(y[idx], pa[idx]), auroc_per_class(y[idx], pb[idx])
        diff_class.append(a - b)
        diff_macro.append(_macro(a) - _macro(b))
    diff_class, diff_macro = np.asarray(diff_class), np.asarray(diff_macro)

    def two_sided_p(samples):
        samples = samples[np.isfinite(samples)]
        if len(samples) == 0:
            return None
        # p-value bootstrap: tỉ lệ mẫu nằm phía bên kia của 0, nhân 2, có hiệu chỉnh +1
        tail = min((samples <= 0).sum(), (samples >= 0).sum())
        return float(min(1.0, 2 * (tail + 1) / (len(samples) + 1)))

    macro_low, macro_high = _interval(diff_macro, alpha)
    rows = []
    for i, name in enumerate(classes):
        low, high = _interval(diff_class[:, i], alpha)
        rows.append({"class": name, "auroc_a": float(point_a[i]), "auroc_b": float(point_b[i]),
                     "difference": float(point_a[i] - point_b[i]),
                     "ci_low": low, "ci_high": high, "p_value": two_sided_p(diff_class[:, i])})
    return {
        "method": "patient-level paired percentile bootstrap",
        "n_boot": int(n_boot), "seed": int(seed), "confidence": 1 - alpha,
        "macro_auroc": {
            "a": _macro(point_a), "b": _macro(point_b),
            "difference": _macro(point_a) - _macro(point_b),
            "ci_low": macro_low, "ci_high": macro_high,
            "p_value": two_sided_p(diff_macro),
        },
        "classes": rows,
    }
