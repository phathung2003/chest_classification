import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_recall_curve
from training.calibration import apply_temperatures, fit_temperatures
from utilities.metric import _validate, class_metrics_dataframe, calculate_metrics
from sklearn.model_selection import GroupShuffleSplit

def extract_oof(oof_path, patient_col="Patient ID", fold_col="Fold", image_col="Image Index"):
    frame = pd.read_csv(oof_path)
    classes = [col[5:] for col in frame if col.startswith("prob_")]
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Missing/duplicate prob_<class> columns")
    if any(c not in frame for c in classes):
        raise ValueError("Missing true labels matching prob_<class> columns")
    if image_col in frame and frame[image_col].duplicated().any():
        raise ValueError("OOF contains duplicate image IDs")
    if patient_col in frame and fold_col in frame:
        if (frame.groupby(patient_col)[fold_col].nunique() > 1).any():
            raise ValueError("Same patient appears in multiple folds")
    targets, scores = _validate(frame[classes].to_numpy(), frame[[f"prob_{c}" for c in classes]].to_numpy())
    return frame, classes, targets, scores

def choose_threshold(y, p, beta=1., min_recall=0., fallback=0.5):
    if np.unique(y).size != 2:
        return float(fallback), "insufficient_label_variation"
    precision, recall, candidates = precision_recall_curve(y, p)
    b2 = float(beta) ** 2
    fbeta = (1 + b2) * precision[:-1] * recall[:-1] / np.maximum(b2 * precision[:-1] + recall[:-1], 1e-12)
    fbeta[recall[:-1] < min_recall] = -1
    idx = np.flatnonzero(fbeta >= 0)
    if len(idx) == 0:
        return float(fallback), "min_recall_unmet"
    # Deterministic tie breaking: choose LOWEST threshold among maximizers -> higher recall
    chosen_idx = int(idx[np.argmax(fbeta[idx])])
    return float(candidates[chosen_idx]), "optimized"

def fit_postprocessing(oof_path, output_dir, *, calibration="none", beta=1., min_recall=0.,
                       default_threshold=0.5, patient_col="Patient ID", fold_col="Fold", image_col="Image Index"):
    if calibration not in {"none", "temperature"}:
        raise ValueError("calibration must be none or temperature")
    if beta <= 0 or not (0. <= min_recall <= 1.):
        raise ValueError("beta > 0, min_recall within [0,1] required")
    frame, classes, labels, raw = extract_oof(oof_path, patient_col, fold_col, image_col)
    temp = fit_temperatures(labels, raw) if calibration == "temperature" else np.ones(len(classes))
    p = apply_temperatures(raw, temp) if calibration == "temperature" else raw
    thresholds, status = [], {}
    for i, c in enumerate(classes):
        t, reason = choose_threshold(labels[:, i], p[:, i], beta=beta, min_recall=min_recall,
                                     fallback=default_threshold)
        thresholds.append(t)
        status[c] = reason
    baseline = class_metrics_dataframe(labels, raw, classes, default_threshold)
    tuned = class_metrics_dataframe(labels, p, classes, thresholds)
    report = baseline.merge(tuned, on="class", suffixes=("_baseline", "_tuned_oof"))
    report["f_beta_tuned_oof"] = [fbeta_score(labels[:, i], p[:, i] >= thresholds[i], beta=beta, zero_division=0)
                                  for i in range(len(classes))]
    report["selection_status"] = [status[c] for c in classes]
    result = {
        "schema_version": 2, "class_order": classes,
        "calibration": calibration, "temperatures": {c: float(t) for c, t in zip(classes, temp)},
        "thresholds": {c: float(t) for c, t in zip(classes, thresholds)},
        "selection_set": "development_oof_only", "objective": f"F_beta={beta}",
        "min_recall": float(min_recall), "default_threshold": float(default_threshold),
        "source_file": Path(oof_path).name,
        "warning": "OOF tuned F-score is optimistic; refit/validate separately for a final retrained model. Do not tune on test.",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "postprocess.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report.to_csv(output_dir / "oof_postprocess_report.csv", index=False)
    return result, report

def load_postprocess(path, classes):
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if list(classes) != data["class_order"]:
        raise ValueError("Postprocess class order differs from model outputs")
    if data.get("calibration") not in ("none", "temperature"):
        raise ValueError("Unsupported calibration")
    return data

def apply_postprocess(probabilities, data, classes):
    if list(classes) != data["class_order"]:
        raise ValueError("Class order mismatch")
    raw = np.asarray(probabilities, dtype=np.float64)
    if data["calibration"] == "temperature":
        temperatures = [data["temperatures"][c] for c in classes]
        scores = apply_temperatures(raw, temperatures)
    else:
        scores = raw
    thresholds = [data["thresholds"][c] for c in classes]
    return scores, np.asarray(thresholds, dtype=np.float64)

def fit_fold_thresholds(oof_path, output_dir, *, expected_classes=None, calibration_fraction=0.75, seed=42, beta=1.0, min_recall=0.0, default_threshold=0.5):
    if not 0 < calibration_fraction < 1:
        raise ValueError('calibration_fraction must be in (0, 1)')
    if beta <= 0 or not 0 <= min_recall <= 1:
        raise ValueError('beta > 0 and min_recall in [0,1] are required')

    frame, classes, y, p = extract_oof(oof_path)
    if expected_classes is not None and list(classes) != list(expected_classes):
        raise ValueError('Class order differs from model output order')
    if 'Patient ID' not in frame:
        raise ValueError('Patient ID column is needed for a leakage-safe split')
    if frame['Patient ID'].isna().any():
        raise ValueError('Missing Patient ID in OOF predictions')
    groups = frame['Patient ID'].astype(str).to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError('At least 2 patients are needed to separate calibration/evaluation')

    split = GroupShuffleSplit(n_splits=1, test_size=1 - calibration_fraction,
                              random_state=int(seed))
    cal_idx, eval_idx = next(split.split(p, y, groups=groups))
    if np.intersect1d(groups[cal_idx], groups[eval_idx]).size:
        raise AssertionError('Patient leakage between threshold calibration and evaluation')

    thresholds, reasons = [], {}
    for j, name in enumerate(classes):
        threshold, reason = choose_threshold(
            y[cal_idx, j], p[cal_idx, j],
            beta=beta, min_recall=min_recall, fallback=default_threshold)
        thresholds.append(threshold)
        reasons[name] = reason
    threshold_array = np.asarray(thresholds, dtype=np.float64)

    baseline = calculate_metrics(len(classes), y[eval_idx], p[eval_idx], default_threshold)
    tuned = calculate_metrics(len(classes), y[eval_idx], p[eval_idx], threshold_array)
    baseline_classes = class_metrics_dataframe(y[eval_idx], p[eval_idx], classes, default_threshold)
    tuned_classes = class_metrics_dataframe(y[eval_idx], p[eval_idx], classes, threshold_array)
    report = baseline_classes.merge(tuned_classes, on='class',
                                    suffixes=('_default', '_tuned'))
    report['fit_status'] = [reasons[c] for c in classes]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    postprocess = {
        'class_order': classes,
        'calibration': 'none',
        'temperatures': {c: 1.0 for c in classes},
        'thresholds': {c: float(t) for c, t in zip(classes, thresholds)},
        'selection_set': 'fold_oof_patient_group_calibration_subset',
        'objective': f'F_beta={beta}',
        'min_recall': float(min_recall),
        'default_threshold': float(default_threshold),
        'source_file': Path(oof_path).name,
        'patient_split_seed': int(seed),
        'calibration_images': int(len(cal_idx)),
        'evaluation_images': int(len(eval_idx)),
        'warning': 'Do not use the threshold-selection data to report unbiased F1. '
                   'This fold was also used for AUROC checkpoint selection; '
                   'the truly independent final evaluation remains the held-out test.',
    }
    summary = {
        'fold': int(frame['Fold'].iloc[0]) if 'Fold' in frame and frame['Fold'].nunique() == 1 else None,
        'split': 'patient-disjoint within fold OOF',
        'calibration_images': int(len(cal_idx)),
        'evaluation_images': int(len(eval_idx)),
        'baseline': {k: v for k, v in baseline.items() if not k.startswith('class_')},
        'optimized': {k: v for k, v in tuned.items() if not k.startswith('class_')},
        'warning': postprocess['warning'],
    }

    (output_dir / 'postprocess.json').write_text(
        json.dumps(postprocess, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'fold_threshold_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report.to_csv(output_dir / 'fold_threshold_report.csv', index=False)
    print('\n[THRESHOLD TỰ ĐỘNG]')
    print(f'Ảnh calibration: {len(cal_idx):,} | ảnh đánh giá: {len(eval_idx):,}')
    print(f'Macro F1 trên phần đánh giá: {baseline["macro_f1"]:.4f} '
          f'-> {tuned["macro_f1"]:.4f}')
    print(f'Đã lưu ngưỡng: {output_dir / "postprocess.json"}')
    return postprocess, summary
