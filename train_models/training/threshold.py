import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_recall_curve
from training.calibration import apply_temperatures, fit_temperatures
from utilities.metric import _validate, class_metrics_dataframe

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
