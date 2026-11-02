"""So sánh ghép cặp hai cấu hình của nghiên cứu CLIS bằng bootstrap theo bệnh nhân (bước 9 của kế hoạch).

Mỗi cấu hình có thể gồm nhiều tệp dự đoán (mỗi seed một tệp, cùng tập ảnh): xác suất được TRUNG BÌNH QUA SEED rồi mới
bootstrap; độ lệch chuẩn giữa các seed (biến thiên do huấn luyện) báo cáo riêng, không gộp với khoảng tin cậy bootstrap
(biến thiên do mẫu). Trước cổng G3 chỉ dùng dự đoán Validation (fold_1/oof_predictions.csv); sau G3 mới dùng Test.

Ví dụ (H1 trên Validation: nhóm tổn thương khu trú L, M-clis so với M-ras, mọi seed đã chạy; mẫu * do script tự mở rộng):
    python analysis/compare_clis.py
        --a "training_results/nih_chest_xray_14/clis_mamba_v1_m_clis_512px/run_*/fold_1/oof_predictions.csv"
        --b "training_results/nih_chest_xray_14/clis_mamba_v1_m_ras_512px/run_*/fold_1/oof_predictions.csv"
        --name-a M-clis --name-b M-ras --output h1_validation.json

Kết quả: hiệu macro AUROC (A - B) trên 14 nhãn, nhóm L, nhóm C và tương tác (ΔL - ΔC), kèm CI 95% percentile và p-value
hai phía / một phía (A > B); từng nhãn kèm p-value Holm; AUROC, AUPRC, độ nhạy tại độ đặc hiệu 90%, ECE, Brier của từng cấu hình.
"""
# Khai báo utilities
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Khai báo thư viện
import argparse
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve

import config
from compare_models import holm, load_predictions
from utilities.file import save_json
from utilities.statistics import _bootstrap_samples, _interval, _macro, auroc_per_class

# Nhóm nhãn của giả thuyết H1 (đăng ký trước): L = tổn thương khu trú / thường một bên, C = lan tỏa / trung tâm.
# Infiltration, Pneumonia, Pleural_Thickening không thuộc nhóm nào (chỉ có trong macro 14 nhãn)
GROUPS = {
    "L": ["Pneumothorax", "Effusion", "Mass", "Nodule", "Consolidation", "Atelectasis"],
    "C": ["Cardiomegaly", "Hernia", "Edema", "Emphysema", "Fibrosis"],
}
PRACTICAL_DELTA = 0.005                                                     # Ngưỡng ý nghĩa thực tiễn: cận dưới CI > 0 và Δ >= 0.005
ECE_BINS = 15


# ============================================================
# Đọc các tệp dự đoán của một cấu hình (mỗi seed một tệp), trả về (khung dữ liệu của tệp đầu, danh sách lớp, xác suất [seed, N, C])
def load_configuration(paths, key_column):
    frames = [load_predictions(path) for path in paths]
    base, classes = frames[0]
    if any(c != classes for _, c in frames):
        raise ValueError("Các tệp của cùng một cấu hình có danh sách lớp khác nhau")
    keys = base[key_column]
    if keys.duplicated().any():
        raise ValueError(f"{paths[0]}: có ảnh trùng lặp")
    probabilities = []
    for path, (frame, _) in zip(paths, frames):
        if set(frame[key_column]) != set(keys):
            raise ValueError(f"{path}: tập ảnh khác tệp đầu tiên của cấu hình")
        frame = frame.set_index(key_column).loc[keys].reset_index()
        if not (frame[classes].to_numpy() == base[classes].to_numpy()).all():
            raise ValueError(f"{path}: nhãn khác tệp đầu tiên")
        probabilities.append(frame[[f"prob_{c}" for c in classes]].to_numpy(dtype = np.float64))
    return base, classes, np.stack(probabilities)


# ============================================================
# Chỉ số từng nhãn của xác suất (trung bình qua seed): AUROC, AUPRC, độ nhạy tại độ đặc hiệu 90%, ECE (15 bin đều), Brier
def label_metrics(targets, probabilities, classes):
    aurocs = auroc_per_class(targets, probabilities)
    rows = []
    for i, name in enumerate(classes):
        y, p = targets[:, i], probabilities[:, i]
        row = {"class": name, "auroc": float(aurocs[i]), "positives": int(y.sum())}
        if 0 < y.sum() < len(y):
            fpr, tpr, _ = roc_curve(y, p, drop_intermediate = False)
            row["auprc"] = float(average_precision_score(y, p))
            row["sensitivity_at_specificity_90"] = float(np.interp(0.10, fpr, tpr))
        bins = np.minimum((p * ECE_BINS).astype(int), ECE_BINS - 1)
        row["ece"] = float(sum(abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean() for b in range(ECE_BINS) if (bins == b).any()))
        row["brier"] = float(np.mean((p - y) ** 2))
        rows.append(row)
    return rows


def macro_of(rows, metric, names = None):
    values = [row.get(metric, np.nan) for row in rows if names is None or row["class"] in names]
    return _macro(values)


# ============================================================
# p-value bootstrap: hai phía (như compare_models.py) và một phía cho giả thuyết A > B, có hiệu chỉnh +1
def p_values(samples):
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return None, None
    two_sided = min(1.0, 2 * (min((samples <= 0).sum(), (samples >= 0).sum()) + 1) / (len(samples) + 1))
    one_sided = ((samples <= 0).sum() + 1) / (len(samples) + 1)
    return float(two_sided), float(one_sided)


# ============================================================
# Bootstrap ghép cặp theo bệnh nhân: cùng mẫu bệnh nhân cho cả hai cấu hình. Nhãn không có ca dương / âm trong một mẫu bootstrap
# (thường là Hernia) bị bỏ khỏi trung bình của mẫu đó; số lần bỏ được đếm và báo cáo
def paired_group_bootstrap(targets, probabilities_a, probabilities_b, patients, classes, *, bootstrap_iteration, seed, alpha):
    index = {name: [classes.index(c) for c in members] for name, members in GROUPS.items()}
    point_a, point_b = auroc_per_class(targets, probabilities_a), auroc_per_class(targets, probabilities_b)

    def summaries(a, b):
        values = {"macro_14": _macro(a) - _macro(b)}
        for name, columns in index.items():
            values[f"group_{name}"] = _macro(a[columns]) - _macro(b[columns])
        values["interaction_L_minus_C"] = values["group_L"] - values["group_C"]
        return values

    point = summaries(point_a, point_b)
    samples = {key: [] for key in point}
    per_class, dropped = [], np.zeros(len(classes), dtype = int)
    for rows in _bootstrap_samples(patients, bootstrap_iteration, seed):
        a, b = auroc_per_class(targets[rows], probabilities_a[rows]), auroc_per_class(targets[rows], probabilities_b[rows])
        dropped += ~np.isfinite(a)
        per_class.append(a - b)
        for key, value in summaries(a, b).items():
            samples[key].append(value)

    result = {}
    for key, values in samples.items():
        values = np.asarray(values)
        low, high = _interval(values, alpha)
        two_sided, one_sided = p_values(values)
        result[key] = {"difference": float(point[key]), "ci_low": low, "ci_high": high, "p_two_sided": two_sided, "p_one_sided_a_greater": one_sided,
                       "practically_significant": bool(low is not None and low > 0 and point[key] >= PRACTICAL_DELTA)}

    per_class = np.asarray(per_class)
    class_rows = []
    for i, name in enumerate(classes):
        low, high = _interval(per_class[:, i], alpha)
        two_sided, one_sided = p_values(per_class[:, i])
        class_rows.append({"class": name, "auroc_a": float(point_a[i]), "auroc_b": float(point_b[i]), "difference": float(point_a[i] - point_b[i]),
                           "ci_low": low, "ci_high": high, "p_two_sided": two_sided, "bootstrap_samples_dropped": int(dropped[i])})
    for row, adjusted in zip(class_rows, holm([row["p_two_sided"] for row in class_rows])):
        row["p_two_sided_holm"] = None if not np.isfinite(adjusted) else float(adjusted)
    return result, class_rows


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", nargs = "+", type = Path, required = True, help = "Tệp dự đoán của cấu hình A (mỗi seed một tệp)")
    parser.add_argument("--b", nargs = "+", type = Path, required = True, help = "Tệp dự đoán của cấu hình B (mỗi seed một tệp)")
    parser.add_argument("--name-a", default = "A")
    parser.add_argument("--name-b", default = "B")
    parser.add_argument("--n-boot", type = int, default = int(config.COMPARE_BOOTSTRAP_ITERATION))
    parser.add_argument("--seed", type = int, default = 0, help = "Seed của bootstrap (đăng ký trước: 0)")
    parser.add_argument("--patient-column", default = config.PATIENT_ID_COLUMN)
    parser.add_argument("--key-column", default = config.IMAGE_ID_COLUMN)
    parser.add_argument("--output", type = Path, default = None)
    args = parser.parse_args()
    alpha = round(1 - config.BOOTSTRAP_CONFIDENCE, 12)
    for side in ("a", "b"):
        paths = [Path(p) for pattern in getattr(args, side) for p in (sorted(glob.glob(str(pattern))) if glob.has_magic(str(pattern)) else [pattern])]
        if not paths:
            parser.error(f"--{side}: không tìm thấy tệp dự đoán nào")
        setattr(args, side, paths)

    if len(args.a) != len(args.b):
        print(f"CẢNH BÁO: số seed khác nhau ({len(args.a)} và {len(args.b)}). Trung bình xác suất nhiều seed là một ensemble, "
              "cấu hình có nhiều seed hơn được lợi; so sánh chính thức phải dùng cùng số seed\n")

    frame_a, classes, seeds_a = load_configuration(args.a, args.key_column)
    frame_b, classes_b, seeds_b = load_configuration(args.b, args.key_column)
    if classes != classes_b or set(frame_a[args.key_column]) != set(frame_b[args.key_column]):
        raise ValueError("Hai cấu hình phải có cùng danh sách lớp và cùng tập ảnh")
    order = frame_b.set_index(args.key_column).index.get_indexer(frame_a[args.key_column])
    seeds_b = seeds_b[:, order]
    if not (frame_b[classes].to_numpy()[order] == frame_a[classes].to_numpy()).all():
        raise ValueError("Nhãn của hai cấu hình không khớp nhau")
    missing = [c for members in GROUPS.values() for c in members if c not in classes]
    if missing:
        raise ValueError(f"Thiếu các lớp của nhóm L / C: {missing}")

    targets = frame_a[classes].to_numpy(dtype = np.float64)
    patients = frame_a[args.patient_column].to_numpy()
    probabilities_a, probabilities_b = seeds_a.mean(axis = 0), seeds_b.mean(axis = 0)
    comparison, class_rows = paired_group_bootstrap(targets, probabilities_a, probabilities_b, patients, classes, bootstrap_iteration = args.n_boot, seed = args.seed, alpha = alpha)

    # Mỗi cấu hình: macro AUROC từng seed (mean ± SD: biến thiên do huấn luyện) và chỉ số của xác suất trung bình qua seed
    configurations = {}
    for name, paths, seeds, probabilities in ((args.name_a, args.a, seeds_a, probabilities_a), (args.name_b, args.b, seeds_b, probabilities_b)):
        per_seed = [_macro(auroc_per_class(targets, s)) for s in seeds]
        rows = label_metrics(targets, probabilities, classes)
        configurations[name] = {
            "predictions": [str(p) for p in paths],
            "macro_auroc_per_seed": per_seed,
            "macro_auroc_seed_mean": float(np.mean(per_seed)),
            "macro_auroc_seed_sd": float(np.std(per_seed, ddof = 1)) if len(per_seed) > 1 else None,
            "seed_averaged": {"macro_auroc": macro_of(rows, "auroc"), "macro_auroc_L": macro_of(rows, "auroc", GROUPS["L"]), "macro_auroc_C": macro_of(rows, "auroc", GROUPS["C"]),
                              "macro_auprc": macro_of(rows, "auprc"), "macro_sensitivity_at_specificity_90": macro_of(rows, "sensitivity_at_specificity_90"),
                              "macro_ece": macro_of(rows, "ece"), "macro_brier": macro_of(rows, "brier")},
            "classes": rows,
        }

    result = {"method": "patient-level paired percentile bootstrap on seed-averaged probabilities", "bootstrap_iteration": args.n_boot, "seed": args.seed,
              "confidence": 1 - alpha, "images": len(targets), "patients": int(pd.Series(patients).nunique()), "groups": GROUPS,
              "practical_delta": PRACTICAL_DELTA, "model_a": args.name_a, "model_b": args.name_b,
              "comparison": comparison, "classes": class_rows, "configurations": configurations}

    for name, info in configurations.items():
        sd = f" ± {info['macro_auroc_seed_sd']:.4f}" if info["macro_auroc_seed_sd"] is not None else ""
        print(f"{name}: macro AUROC theo seed {info['macro_auroc_seed_mean']:.4f}{sd} ({len(info['macro_auroc_per_seed'])} seed) | "
              f"trung bình xác suất: {info['seed_averaged']['macro_auroc']:.4f} (L {info['seed_averaged']['macro_auroc_L']:.4f}, C {info['seed_averaged']['macro_auroc_C']:.4f})")
    print(f"\n{args.name_a} - {args.name_b} | {len(targets):,} ảnh, {result['patients']:,} bệnh nhân, bootstrap {args.n_boot} lần")
    for key, row in comparison.items():
        print(f"  {key:22s} {row['difference']:+.4f} [CI {row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | p hai phía {row['p_two_sided']:.4f}, "
              f"một phía (A > B) {row['p_one_sided_a_greater']:.4f}" + (" | có ý nghĩa thực tiễn" if row["practically_significant"] else ""))
    print(pd.DataFrame(class_rows).round(4).to_string(index = False))
    if args.output is not None:
        save_json(args.output, result)
        print(f"Đã lưu: {args.output}")


if __name__ == "__main__":
    main()
