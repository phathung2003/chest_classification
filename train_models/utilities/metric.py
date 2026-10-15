import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# ============================================================
# Đánh giá
def calculate_metrics(NUMBER_OF_CLASSES, targets, probabilities, threshold=0.5,):
    predictions = (probabilities >= threshold).astype(np.uint8)
    
    # Per-class AUROC
    class_aurocs = []

    for index in range(NUMBER_OF_CLASSES):
        y_true = targets[:,index]
        y_score = probabilities[:,index]

        if np.unique(y_true).size < 2:
            auc = np.nan

        else:
            auc = roc_auc_score(y_true, y_score)

        class_aurocs.append(auc)

    # Macro AUROC
    macro_auroc = float(np.nanmean(class_aurocs))
    
    # Macro Precision
    macro_precision = float(precision_score(targets, predictions, average="macro", zero_division=0))

    # Macro Recall
    macro_recall = float(recall_score(targets, predictions, average="macro", zero_division=0))

    # Macro F1-Score
    macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))

    # Micro F1-Score
    micro_f1 = float(f1_score(targets, predictions, average="micro", zero_division=0))
    
    # Weight F1-Score
    weighted_f1 = float(f1_score(targets, predictions, average="weighted", zero_division=0))
    
    return {
        "macro_auroc"       : macro_auroc,
        "macro_precision"   : macro_precision,
        "macro_recall"      : macro_recall,
        "macro_f1"          : macro_f1,
        "micro_f1"          : micro_f1,
        "weighted_f1"       : weighted_f1,
        "class_aurocs"      : class_aurocs
    }

# ============================================================
def class_metrics_dataframe(targets, probabilities, classes, threshold=0.5):
    predicted = (probabilities >= threshold).astype(np.uint8)
    rows = []
    for i, name in enumerate(classes):
        y = targets[:, i]
        p = predicted[:, i]
        rows.append({
            "class": name,
            "auroc": roc_auc_score(y, probabilities[:, i]) if np.unique(y).size == 2 else np.nan,
            "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0),
            "positive_support": int(y.sum()),
        })
    return pd.DataFrame(rows)

# ============================================================
def with_probabilities(frame, probabilities, classes):
    result = frame.reset_index(drop=True).copy()
    for i, class_name in enumerate(classes):
        result[f"prob_{class_name}"] = probabilities[:, i]
    return result