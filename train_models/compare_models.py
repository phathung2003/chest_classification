"""So sánh ghép cặp hai mô hình trên CÙNG tập Test bằng bootstrap theo bệnh nhân.

Ví dụ (so sánh hai ensemble 5 fold):
    python compare_models.py ^
        training_results/nih_chest_xray_14/cnn_bimamba_v2_s16_mamba_cross4_skip/run_2/test_ensemble/test_predictions.csv ^
        training_results/nih_chest_xray_14/densenet121_baseline_v1/run_3/test_ensemble/test_predictions.csv ^
        --name-a "CNN BiMamba" --name-b "DenseNet121" --output comparison.json

Kết quả: hiệu macro AUROC (A - B), khoảng tin cậy 95% và p-value hai phía, cho cả từng lớp.
Với 14 lớp, p-value từng lớp nên hiệu chỉnh đa so sánh (cột p_value_holm).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from utilities.statistics import paired_bootstrap_auroc
from utilities.file import save_json


def holm(p_values):
    p = np.asarray([np.nan if v is None else v for v in p_values], dtype = np.float64)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def load_predictions(path):
    frame = pd.read_csv(path)
    classes = [c[5:] for c in frame.columns if c.startswith("prob_")]
    if not classes:
        raise ValueError(f"{path}: không có cột prob_<lớp>")
    return frame, classes


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions_a")
    parser.add_argument("predictions_b")
    parser.add_argument("--name-a", default = "A")
    parser.add_argument("--name-b", default = "B")
    parser.add_argument("--n-boot", type = int, default = int(config.COMPARE_BOOTSTRAP_ITERATIONS))
    parser.add_argument("--seed", type = int, default = int(config.BOOTSTRAP_SEED))
    parser.add_argument("--patient-column", default = config.PATIENT_ID_COLUMN)
    parser.add_argument("--key-column", default = config.IMAGE_ID_COLUMN)
    parser.add_argument("--output", type = Path, default = None)
    args = parser.parse_args()

    a, classes_a = load_predictions(args.predictions_a)
    b, classes_b = load_predictions(args.predictions_b)
    if classes_a != classes_b:
        raise ValueError("Hai tệp dự đoán có danh sách lớp khác nhau")
    classes = classes_a
    for frame, name in ((a, "A"), (b, "B")):
        if frame[args.key_column].duplicated().any():
            raise ValueError(f"Tệp {name} có ảnh trùng lặp")
    if set(a[args.key_column]) != set(b[args.key_column]):
        raise ValueError("Hai tệp dự đoán không cùng tập ảnh test")

    b = b.set_index(args.key_column).loc[a[args.key_column]].reset_index()
    if not (a[classes].to_numpy() == b[classes].to_numpy()).all():
        raise ValueError("Nhãn của hai tệp không khớp nhau")

    result = paired_bootstrap_auroc(a[classes].to_numpy(dtype = np.float64), a[[f"prob_{c}" for c in classes]].to_numpy(dtype = np.float64), b[[f"prob_{c}" for c in classes]].to_numpy(dtype = np.float64), a[args.patient_column].to_numpy(), classes, n_boot = args.n_boot, seed = args.seed)

    adjusted = holm([row["p_value"] for row in result["classes"]])
    for row, value in zip(result["classes"], adjusted):
        row["p_value_holm"] = None if not np.isfinite(value) else float(value)
    result["model_a"] = {"name": args.name_a, "predictions": str(args.predictions_a)}
    result["model_b"] = {"name": args.name_b, "predictions": str(args.predictions_b)}

    macro = result["macro_auroc"]
    print(f"Macro AUROC {args.name_a}: {macro['a']:.4f} | {args.name_b}: {macro['b']:.4f}")
    print(f"Hiệu (A - B): {macro['difference']:+.4f} "
          f"[95% CI {macro['ci_low']:+.4f}, {macro['ci_high']:+.4f}] | p = {macro['p_value']:.4f}")
    print(pd.DataFrame(result["classes"]).round(4).to_string(index = False))
    if args.output is not None:
        save_json(args.output, result)
        print(f"Đã lưu: {args.output}")


if __name__ == "__main__":
    main()
