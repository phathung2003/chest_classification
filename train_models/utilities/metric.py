from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def _validate(targets, probabilities):
    y = np.asarray(targets)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 2 or y.shape != p.shape:
        raise ValueError(f"Expected matching 2D [N,C] matrices, got {y.shape}, {p.shape}")
    if y.shape[0] == 0 or y.shape[1] == 0:
        raise ValueError("Empty classification matrix")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0/1. Map uncertain/missing labels before evaluation.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and within [0,1]")
    return y.astype(np.int8, copy=False), p


def normalize_thresholds(threshold, classes):
    classes = list(classes)
    if isinstance(threshold, Mapping):
        if set(threshold) != set(classes):
            raise ValueError("Threshold dictionary keys do not match classes")
        t = np.array([threshold[c] for c in classes], dtype=np.float64)
    elif np.isscalar(threshold):
        t = np.full(len(classes), float(threshold), dtype=np.float64)
    else:
        t = np.asarray(threshold, dtype=np.float64)
    if t.shape != (len(classes),) or not np.isfinite(t).all() or np.any((t < 0) | (t > 1)):
        raise ValueError(f"Expected {len(classes)} finite thresholds in [0,1]")
    return t


def class_metrics_dataframe(targets, probabilities, classes, threshold=0.5):
    y, p = _validate(targets, probabilities)
    if len(classes) != y.shape[1]:
        raise ValueError("Number of class names does not match prediction columns")
    thresholds = normalize_thresholds(threshold, classes)
    binary = p >= thresholds[None, :]
    rows = []
    for i, name in enumerate(classes):
        yt, scores, pred = y[:, i], p[:, i], binary[:, i]
        evaluable = (np.unique(yt).size == 2)
        rows.append({
            "class": name,
            "positive_support": int(yt.sum()),
            "negative_support": int(len(yt) - yt.sum()),
            "prevalence": float(yt.mean()),
            "threshold": float(thresholds[i]),
            "auroc": float(roc_auc_score(yt, scores)) if evaluable else np.nan,
            "average_precision": float(average_precision_score(yt, scores)) if evaluable else np.nan,
            "precision": float(precision_score(yt, pred, zero_division=0)),
            "recall": float(recall_score(yt, pred, zero_division=0)),
            "f1": float(f1_score(yt, pred, zero_division=0)),
            "tp": int(((yt == 1) & pred).sum()),
            "fp": int(((yt == 0) & pred).sum()),
            "fn": int(((yt == 1) & ~pred).sum()),
            "tn": int(((yt == 0) & ~pred).sum()),
        })
    return pd.DataFrame(rows)


def calculate_metrics(number_of_classes, targets, probabilities, threshold=0.5):
    # Preserve the existing project's API.
    y, p = _validate(targets, probabilities)
    if number_of_classes != y.shape[1]:
        raise ValueError("number_of_classes does not match targets")
    t = normalize_thresholds(threshold, range(y.shape[1])) if not isinstance(threshold, Mapping) else None
    if t is None:
        raise ValueError("Pass a threshold vector here; dictionary requires named classes")
    pred = (p >= t[None, :]).astype(np.uint8)
    aucs, aps = [], []
    for i in range(y.shape[1]):
        valid = np.unique(y[:, i]).size == 2
        aucs.append(float(roc_auc_score(y[:, i], p[:, i])) if valid else float('nan'))
        aps.append(float(average_precision_score(y[:, i], p[:, i])) if valid else float('nan'))
    def mean_finite(values):
        v = np.asarray(values, dtype=np.float64)
        return float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else float('nan')
    return {
        "macro_auroc": mean_finite(aucs),
        "macro_ap": mean_finite(aps),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "class_aurocs": aucs,
        "class_aps": aps,
    }


def with_probabilities(frame, probabilities, classes):
    result = frame.reset_index(drop=True).copy()
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape != (len(result), len(classes)):
        raise ValueError("Predictions do not align with dataframe/class count; disable shuffle on validation/test")
    for i, name in enumerate(classes):
        result[f"prob_{name}"] = p[:, i]
    return result
