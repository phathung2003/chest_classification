import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit

import config
from training.calibration import apply_temperatures, fit_temperatures
from utilities.metric import validate_predictions, calculate_metrics, class_metrics_dataframe


# ============================================================
# Đọc tệp dự đoán OOF: lấy danh sách lớp từ cột prob_<lớp>, kiểm tra không trùng ảnh và không rò rỉ bệnh nhân
def extract_oof(oof_path, patient_col = None, fold_col = "Fold", image_col = None):
    patient_col = config.PATIENT_ID_COLUMN if patient_col is None else patient_col
    image_col = config.IMAGE_ID_COLUMN if image_col is None else image_col
    frame = pd.read_csv(oof_path)
    classes = [col[5:] for col in frame if col.startswith("prob_")]
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Thiếu hoặc trùng cột prob_<lớp>")
    if any(c not in frame for c in classes):
        raise ValueError("Thiếu cột nhãn tương ứng với cột prob_<lớp>")
    if image_col in frame and frame[image_col].duplicated().any():
        raise ValueError("Tệp OOF có ảnh trùng lặp")
    if patient_col in frame and fold_col in frame:
        if (frame.groupby(patient_col)[fold_col].nunique() > 1).any():
            raise ValueError("Một bệnh nhân nằm ở nhiều fold")
    targets, scores = validate_predictions(frame[classes].to_numpy(), frame[[f"prob_{c}" for c in classes]].to_numpy())
    return frame, classes, targets, scores


# ============================================================
# Chọn ngưỡng của một lớp: tối đa F-beta trên đường Precision-Recall, có thể ràng buộc recall tối thiểu.
# Nhiều ngưỡng cùng điểm --> chọn ngưỡng thấp nhất (recall cao hơn)
def choose_threshold(y, p, beta = None, min_recall = None, fallback = None):
    beta = config.THRESHOLD_BETA if beta is None else beta
    min_recall = config.THRESHOLD_MIN_RECALL if min_recall is None else min_recall
    fallback = config.THRESHOLD if fallback is None else fallback
    if np.unique(y).size != 2:
        return float(fallback), "insufficient_label_variation"
    precision, recall, candidates = precision_recall_curve(y, p)
    b2 = float(beta) ** 2
    fbeta = (1 + b2) * precision[:-1] * recall[:-1] / np.maximum(b2 * precision[:-1] + recall[:-1], 1e-12)
    fbeta[recall[:-1] < min_recall] = -1
    idx = np.flatnonzero(fbeta >= 0)
    if len(idx) == 0:
        return float(fallback), "min_recall_unmet"
    return float(candidates[int(idx[np.argmax(fbeta[idx])])]), "optimized"


# ============================================================
# Hậu xử lý cho ensemble: calibration (tùy chọn) và ngưỡng từng lớp chọn trên TOÀN BỘ OOF.
# Điểm F-beta trên OOF là lạc quan vì chọn và đo trên cùng dữ liệu; số liệu khách quan là trên Test.
def fit_postprocessing(oof_path, output_dir, *, calibration = None, beta = None, min_recall = None, default_threshold = None, patient_col = None, fold_col = "Fold", image_col = None):
    calibration = config.POSTPROCESS_CALIBRATION if calibration is None else calibration
    beta = config.THRESHOLD_BETA if beta is None else beta
    min_recall = config.THRESHOLD_MIN_RECALL if min_recall is None else min_recall
    default_threshold = config.THRESHOLD if default_threshold is None else default_threshold
    if calibration not in {"none", "temperature"}:
        raise ValueError("calibration phải là none hoặc temperature")
    if beta <= 0 or not 0. <= min_recall <= 1.:
        raise ValueError("Cần beta > 0 và min_recall thuộc [0, 1]")
    _, classes, labels, raw = extract_oof(oof_path, patient_col, fold_col, image_col)

    # Calibration (temperature scaling theo lớp) rồi chọn ngưỡng
    temperatures = fit_temperatures(labels, raw) if calibration == "temperature" else np.ones(len(classes))
    p = apply_temperatures(raw, temperatures) if calibration == "temperature" else raw
    thresholds, status = [], {}
    for i, c in enumerate(classes):
        t, reason = choose_threshold(labels[:, i], p[:, i], beta = beta, min_recall = min_recall, fallback = default_threshold)
        thresholds.append(t)
        status[c] = reason

    # Báo cáo so sánh ngưỡng mặc định và ngưỡng đã chọn trên OOF
    baseline = class_metrics_dataframe(labels, raw, classes, default_threshold)
    tuned = class_metrics_dataframe(labels, p, classes, thresholds)
    report = baseline.merge(tuned, on = "class", suffixes = ("_baseline", "_tuned_oof"))
    report["f_beta_tuned_oof"] = [fbeta_score(labels[:, i], p[:, i] >= thresholds[i], beta = beta, zero_division = 0)
                                  for i in range(len(classes))]
    report["selection_status"] = [status[c] for c in classes]

    result = {
        "schema_version"    : 2,
        "class_order"       : classes,
        "calibration"       : calibration,
        "temperatures"      : {c: float(t) for c, t in zip(classes, temperatures)},
        "thresholds"        : {c: float(t) for c, t in zip(classes, thresholds)},
        "selection_set"     : "development_oof_only",
        "objective"         : f"F_beta={beta}",
        "min_recall"        : float(min_recall),
        "default_threshold" : float(default_threshold),
        "source_file"       : Path(oof_path).name,
        "warning"           : "Điểm F-beta trên OOF là lạc quan. Không bao giờ chọn ngưỡng trên Test.",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)
    (output_dir / "postprocess.json").write_text(json.dumps(result, indent = 2, ensure_ascii = False) + "\n", encoding = "utf-8")
    report.to_csv(output_dir / "oof_postprocess_report.csv", index = False)
    return result, report


# ============================================================
# Đọc tệp hậu xử lý và áp lên xác suất dự đoán
def load_postprocess(path, classes):
    with Path(path).open(encoding = "utf-8") as f:
        data = json.load(f)
    if list(classes) != data["class_order"]:
        raise ValueError("Thứ tự lớp trong tệp hậu xử lý khác đầu ra mô hình")
    if data.get("calibration") not in ("none", "temperature"):
        raise ValueError("Kiểu calibration không được hỗ trợ")
    return data


def apply_postprocess(probabilities, data, classes):
    if list(classes) != data["class_order"]:
        raise ValueError("Thứ tự lớp không khớp")
    raw = np.asarray(probabilities, dtype = np.float64)
    if data["calibration"] == "temperature":
        scores = apply_temperatures(raw, [data["temperatures"][c] for c in classes])
    else:
        scores = raw
    return scores, np.asarray([data["thresholds"][c] for c in classes], dtype = np.float64)


# ============================================================
# Hậu xử lý cho mô hình của một fold: tách OOF của fold theo bệnh nhân thành
# phần chọn ngưỡng (calibration_fraction) và phần đánh giá, để F1 sau chọn ngưỡng không bị lạc quan
def fit_fold_thresholds(oof_path, output_dir, *, expected_classes = None, calibration_fraction = None, seed = None, beta = None, min_recall = None, default_threshold = None):
    calibration_fraction = config.THRESHOLD_CALIBRATION_FRACTION if calibration_fraction is None else calibration_fraction
    seed = config.SEED if seed is None else seed
    beta = config.THRESHOLD_BETA if beta is None else beta
    min_recall = config.THRESHOLD_MIN_RECALL if min_recall is None else min_recall
    default_threshold = config.THRESHOLD if default_threshold is None else default_threshold
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction phải thuộc (0, 1)")
    if beta <= 0 or not 0 <= min_recall <= 1:
        raise ValueError("Cần beta > 0 và min_recall thuộc [0, 1]")

    frame, classes, y, p = extract_oof(oof_path)
    if expected_classes is not None and list(classes) != list(expected_classes):
        raise ValueError("Thứ tự lớp khác đầu ra mô hình")
    if config.PATIENT_ID_COLUMN not in frame or frame[config.PATIENT_ID_COLUMN].isna().any():
        raise ValueError("Cần cột Patient ID đầy đủ để tách không rò rỉ bệnh nhân")
    groups = frame[config.PATIENT_ID_COLUMN].astype(str).to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("Cần ít nhất 2 bệnh nhân để tách phần chọn ngưỡng / phần đánh giá")

    # Tách theo bệnh nhân
    split = GroupShuffleSplit(n_splits = 1, test_size = 1 - calibration_fraction, random_state = int(seed))
    cal_idx, eval_idx = next(split.split(p, y, groups = groups))
    if np.intersect1d(groups[cal_idx], groups[eval_idx]).size:
        raise AssertionError("Rò rỉ bệnh nhân giữa phần chọn ngưỡng và phần đánh giá")

    # Chọn ngưỡng trên phần chọn ngưỡng
    thresholds, reasons = [], {}
    for j, name in enumerate(classes):
        threshold, reason = choose_threshold(y[cal_idx, j], p[cal_idx, j], beta = beta, min_recall = min_recall, fallback = default_threshold)
        thresholds.append(threshold)
        reasons[name] = reason
    threshold_array = np.asarray(thresholds, dtype = np.float64)

    # Đánh giá trên phần còn lại: ngưỡng mặc định so với ngưỡng đã chọn
    baseline = calculate_metrics(len(classes), y[eval_idx], p[eval_idx], default_threshold)
    tuned = calculate_metrics(len(classes), y[eval_idx], p[eval_idx], threshold_array)
    report = class_metrics_dataframe(y[eval_idx], p[eval_idx], classes, default_threshold).merge(class_metrics_dataframe(y[eval_idx], p[eval_idx], classes, threshold_array), on = "class", suffixes = ("_default", "_tuned"))
    report["fit_status"] = [reasons[c] for c in classes]

    warning = ("Không dùng phần chọn ngưỡng để báo cáo F1. Fold này cũng đã dùng để chọn checkpoint; "
               "đánh giá khách quan cuối cùng là trên tập Test.")
    postprocess = {
        "class_order"           : classes,
        "calibration"           : "none",
        "temperatures"          : {c: 1.0 for c in classes},
        "thresholds"            : {c: float(t) for c, t in zip(classes, thresholds)},
        "selection_set"         : "fold_oof_patient_group_calibration_subset",
        "objective"             : f"F_beta={beta}",
        "min_recall"            : float(min_recall),
        "default_threshold"     : float(default_threshold),
        "source_file"           : Path(oof_path).name,
        "patient_split_seed"    : int(seed),
        "calibration_images"    : int(len(cal_idx)),
        "evaluation_images"     : int(len(eval_idx)),
        "warning"               : warning,
    }
    summary = {
        "fold"                  : int(frame["Fold"].iloc[0]) if "Fold" in frame and frame["Fold"].nunique() == 1 else None,
        "split"                 : "patient-disjoint within fold OOF",
        "calibration_images"    : int(len(cal_idx)),
        "evaluation_images"     : int(len(eval_idx)),
        "baseline"              : {k: v for k, v in baseline.items() if not k.startswith("class_")},
        "optimized"             : {k: v for k, v in tuned.items() if not k.startswith("class_")},
        "warning"               : warning,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)
    (output_dir / "postprocess.json").write_text(json.dumps(postprocess, ensure_ascii = False, indent = 2) + "\n", encoding = "utf-8")
    (output_dir / "fold_threshold_summary.json").write_text(json.dumps(summary, ensure_ascii = False, indent = 2) + "\n", encoding = "utf-8")
    report.to_csv(output_dir / "fold_threshold_report.csv", index = False)

    print("\n[CHỌN NGƯỠNG TỰ ĐỘNG]")
    print(f"Ảnh chọn ngưỡng: {len(cal_idx):,} | Ảnh đánh giá: {len(eval_idx):,}")
    print(f"Macro F1 trên phần đánh giá: {baseline['macro_f1']:.4f} -> {tuned['macro_f1']:.4f}")
    print(f"Đã lưu ngưỡng: {output_dir / 'postprocess.json'}")
    return postprocess, summary
