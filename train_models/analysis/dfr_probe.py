"""Khớp lại đầu Pneumothorax theo nhóm ống dẫn lưu màng phổi trên đặc trưng đóng băng (phương án C, kiểu DFR; Validation, không mở Test).

Vì sao cần: trên Validation fold 1, AUROC Pneumothorax của ảnh có ống dẫn lưu khoảng 0,98, của ảnh không ống khoảng 0,83, ở mọi mô hình
(nhãn ống NEATX, Jiménez-Sánchez et al. 2025). DFR (Kirichenko et al., ICLR 2023) cho thấy khi mô hình dựa vào lối tắt, đặc trưng thật
thường vẫn có trong backbone, chỉ lớp cuối dồn trọng số vào lối tắt. Script giữ nguyên backbone, chỉ khớp lại đầu Pneumothorax.

Đổi so với DFR gốc: ở NIH nhóm PTX+ không ống đã đông hơn nhóm có ống, và nhóm thiểu số đúng nghĩa (PTX- có ống) không có nhãn, nên
không cân bằng số ảnh giữa các nhóm mà cho nhóm có ống trọng số 0.

Cách tính (mỗi seed, đặc trưng 768 chiều đã trích trong analysis/head_refit_probe.py):
  - Tập khớp     : nửa A của Validation (chia đôi bệnh nhân như analysis/pooling_probe.py, seed 0); đặc trưng chuẩn hóa theo nửa A
  - control      : (i) hồi quy logistic L2, dương tính = mọi PTX+ của nửa A (có ống, không ống, chưa nhãn), âm tính = PTX-
  - no_tube      : (ii) như (i) nhưng dương tính = chỉ PTX+ không ống (ảnh có ống và chưa nhãn bỏ khỏi tập khớp)
  - original     : đầu đã huấn luyện của mô hình (cột Pneumothorax)
  - C            : chọn trong lưới bằng kiểm định chéo 5 phần theo bệnh nhân trong nửa A, tiêu chí AUROC PTX+ không ống so với PTX-
                   (cùng tiêu chí cho cả hai đầu)

Đọc kết quả (quy tắc ghi trước khi chạy: prereg/analysis_plan.md mục 7, dòng 2026-11-18 23:40): xác suất nửa B trung bình qua seed;
Δ = AUROC (PTX+ không ống so với PTX-) của (ii) - (i), bootstrap ghép cặp theo bệnh nhân. Đạt khi Δ >= +0,01 và cận dưới CI 95 % > 0.

Ví dụ (Windows hoặc WSL2, chỉ cần numpy, pandas, sklearn):
    python analysis/dfr_probe.py --neatx "E:/New folder/claude_plan/data/neatx/NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv"
Kết quả trong training_results/explore/dfr_neatx/: probabilities.npz (xác suất nửa B), summary.json, summary.txt.
"""
# Khai báo utilities
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Khai báo thư viện
import json
import shutil
import hashlib
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import config
from utilities.statistics import _bootstrap_samples, _interval

FEATURE_DIRECTORY = config.MODELS_OUTPUT_DIRECTORY / "explore" / "head_refit"      # Đặc trưng đã trích (analysis/head_refit_probe.py)
OUTPUT_DIRECTORY = config.MODELS_OUTPUT_DIRECTORY / "explore" / "dfr_neatx"
NEATX_DEFAULT = Path("E:/New folder/claude_plan/data/neatx/NIH-CX14_TubeAnnotations_NonExperts_aggregated.csv")   # Nhãn ống NEATX (ngoài repo, CC BY-NC-SA)
MODEL = "ccmm"                                                              # Chỉ ccmm có đặc trưng đã trích
SEEDS = (42, 43, 44)
TARGET = "Pneumothorax"
SPLIT_SEED = 0                                                              # Seed chia đôi bệnh nhân Validation, giống analysis/pooling_probe.py
C_GRID = (1e-3, 1e-2, 1e-1, 1.0)                                            # Lưới C (nghịch đảo độ mạnh phạt L2) của sklearn
CV_SPLITS = 5                                                               # Số phần kiểm định chéo theo bệnh nhân trong nửa A
MAX_ITERATIONS = 5000                                                       # lbfgs: số vòng lặp tối đa
MIN_GAIN = 0.01                                                             # Quy tắc: Δ AUROC nhóm không ống >= 0,01 và cận dưới CI > 0
HEADS = ("original", "control", "no_tube")                                  # Thứ tự đầu trong kết quả
GROUPS = {"no_tube": 0, "tube": 1}                                          # Giá trị cột Drain của NEATX; ảnh PTX+ không có trong NEATX = -1 (chưa nhãn)


# ============================================================
# Chia đôi bệnh nhân Validation (chép từ analysis/pooling_probe.py để không phải import code mô hình)
# Trả về: Mặt nạ boolean, True = nửa A
def half_a_mask(patients, seed = SPLIT_SEED):
    unique = np.array(sorted(set(patients.tolist())))
    order = np.random.default_rng(int(seed)).permutation(len(unique))
    return np.isin(patients, unique[order[:(len(unique) + 1) // 2]])


# ============================================================
# Mặt nạ ảnh dùng để tính AUROC của một nhóm dương tính: nhóm đó cộng toàn bộ PTX-
# Trả về: dict {tên nhóm: mặt nạ boolean}
def subgroup_masks(label, group):
    negative = label == 0
    return {"no_tube": negative | ((label == 1) & (group == GROUPS["no_tube"])),
            "tube"   : negative | ((label == 1) & (group == GROUPS["tube"])),
            "all"    : np.ones_like(negative)}


# ============================================================
# AUROC của ba nhóm (không ống, có ống, mọi PTX+) so với PTX-
# Trả về: dict {tên nhóm: AUROC}
def subgroup_auroc(label, group, scores):
    return {name: float(roc_auc_score(label[mask], scores[mask])) for name, mask in subgroup_masks(label, group).items()}


# ============================================================
# Mặt nạ tập khớp của một đầu (trong nửa A)
# Trả về: Mặt nạ boolean
def fit_mask(head, label, group):
    if head == "control":
        return np.ones_like(label, dtype = bool)                            # Mọi ảnh: PTX- và mọi PTX+
    return (label == 0) | ((label == 1) & (group == GROUPS["no_tube"]))     # Chỉ PTX- và PTX+ không ống


# ============================================================
# Khớp hồi quy logistic L2 cân bằng lớp
# Trả về: Mô hình sklearn đã khớp
def fit_logistic(features, label, c_value):
    model = LogisticRegression(C = c_value, class_weight = "balanced", solver = "lbfgs", max_iter = MAX_ITERATIONS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", category = UserWarning)              # Không hội tụ thì dừng hẳn, không im lặng
        model.fit(features, label)
    return model


# ============================================================
# Chọn C bằng kiểm định chéo theo bệnh nhân trong nửa A; tiêu chí AUROC PTX+ không ống so với PTX- trên phần giữ lại
# Trả về: (C được chọn, dict {C: AUROC trung bình các phần})
def choose_c(head, features, label, group, patients):
    scores = {}
    for c_value in C_GRID:
        values = []
        for train_rows, held_rows in GroupKFold(n_splits = CV_SPLITS).split(features, label, patients):
            rows = train_rows[fit_mask(head, label[train_rows], group[train_rows])]
            model = fit_logistic(features[rows], label[rows], c_value)
            held = held_rows[subgroup_masks(label[held_rows], group[held_rows])["no_tube"]]
            values.append(roc_auc_score(label[held], model.decision_function(features[held])))
        scores[c_value] = float(np.mean(values))
    best = max(C_GRID, key = lambda c_value: (scores[c_value], -c_value))    # Hòa thì lấy C nhỏ hơn (phạt mạnh hơn)
    return best, scores


# ============================================================
# Khớp hai đầu cho một seed, trả xác suất nửa B của ba đầu
# Trả về: (dict {đầu: xác suất nửa B}, dict chẩn đoán)
def run_seed(path, tube, half_a):
    data = np.load(path, allow_pickle = False)
    classes = [str(name) for name in data["classes"]]
    k = classes.index(TARGET)
    features = data["validation_features"].astype(np.float64)
    label = data["validation_targets"][:, k].astype(int)
    patients = data["validation_patients"]
    group = np.array([tube.get(str(image), -1) for image in data["validation_image_ids"]])
    group = np.where(label == 1, group, -1)                                  # Nhãn ống chỉ có nghĩa với PTX+

    # Đầu gốc: z = W·x + b trên đặc trưng chưa chuẩn hóa
    original = 1 / (1 + np.exp(-(features @ data["weight"][k].astype(np.float64) + float(data["bias"][k]))))

    # Chuẩn hóa theo nửa A
    a, b = half_a, ~half_a
    mean, std = features[a].mean(axis = 0), features[a].std(axis = 0)
    std[std == 0] = 1.0
    standardized = (features - mean) / std

    probabilities, diagnostics, weights = {"original": original[b]}, {}, {}
    for head in ("control", "no_tube"):
        rows = np.flatnonzero(a)
        c_value, cv_scores = choose_c(head, standardized[rows], label[rows], group[rows], patients[rows])
        rows = rows[fit_mask(head, label[rows], group[rows])]
        model = fit_logistic(standardized[rows], label[rows], c_value)
        probabilities[head] = model.predict_proba(standardized[b])[:, 1]
        weights[head] = model.coef_[0]
        diagnostics[head] = {"c": c_value, "cv_no_tube_auroc": {str(key): value for key, value in cv_scores.items()},
                             "fit_images": int(len(rows)), "fit_positives": int(label[rows].sum()), "iterations": int(model.n_iter_[0])}
    diagnostics["cosine_control_no_tube"] = float(weights["control"] @ weights["no_tube"] / (np.linalg.norm(weights["control"]) * np.linalg.norm(weights["no_tube"])))
    return probabilities, diagnostics, {"label": label[b], "group": group[b], "patients": patients[b], "image_ids": data["validation_image_ids"][b]}


# ============================================================
# Bootstrap ghép cặp theo bệnh nhân của hiệu AUROC nhóm không ống (đầu a - đầu b)
# Trả về: dict {difference, ci_low, ci_high}
def paired_bootstrap(label, group, patients, first, second, bootstrap_iteration, seed, alpha):
    def difference(rows):
        mask = subgroup_masks(label[rows], group[rows])["no_tube"]
        target = label[rows][mask]
        if target.min() == target.max():
            return np.nan
        return roc_auc_score(target, first[rows][mask]) - roc_auc_score(target, second[rows][mask])

    point = difference(np.arange(len(label)))
    samples = np.array([difference(rows) for rows in _bootstrap_samples(patients, bootstrap_iteration, seed)])
    low, high = _interval(samples, alpha)
    return {"difference": float(point), "ci_low": low, "ci_high": high, "finite_samples": int(np.isfinite(samples).sum())}


# ============================================================
# Hàm chính
def main():
    parser = argparse.ArgumentParser(description = "Khớp lại đầu Pneumothorax theo nhóm ống dẫn lưu trên đặc trưng đóng băng (Validation, không mở Test)")
    parser.add_argument("--neatx", type = Path, default = NEATX_DEFAULT, help = "File nhãn ống NEATX (aggregated csv)")
    parser.add_argument("--features", type = Path, default = FEATURE_DIRECTORY)
    parser.add_argument("--output", type = Path, default = OUTPUT_DIRECTORY)
    parser.add_argument("--n-boot", type = int, default = int(config.COMPARE_BOOTSTRAP_ITERATION))
    parser.add_argument("--seed", type = int, default = 0, help = "Seed của bootstrap (đăng ký trước: 0)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding = "utf-8")                              # Console Windows cp1252 không in được tiếng Việt
    alpha = 1 - float(config.BOOTSTRAP_CONFIDENCE)

    neatx = pd.read_csv(args.neatx)
    tube = dict(zip(neatx["Image Index"].astype(str), neatx["Drain"].astype(int)))
    args.output.mkdir(parents = True, exist_ok = True)
    shutil.copy2(ROOT_DIR / "prereg" / "analysis_plan.md", args.output / "prereg_snapshot.md")
    neatx_hash = hashlib.sha256(args.neatx.read_bytes()).hexdigest()

    # Từng seed
    per_seed, diagnostics, reference = {}, {}, None
    for seed in SEEDS:
        path = args.features / f"features_{MODEL}_seed{seed}.npz"
        patients = np.load(path, allow_pickle = False)["validation_patients"]
        probabilities, diagnostic, meta = run_seed(path, tube, half_a_mask(patients))
        if reference is None:
            reference = meta
        elif not (reference["image_ids"] == meta["image_ids"]).all():
            raise SystemExit(f"{path}: thứ tự ảnh Validation khác seed đầu tiên.")
        per_seed[seed], diagnostics[seed] = probabilities, diagnostic
        print(f"seed {seed} | C: control {diagnostic['control']['c']:g}, no_tube {diagnostic['no_tube']['c']:g} | cosin {diagnostic['cosine_control_no_tube']:.3f}", flush = True)

    label, group, patients = reference["label"], reference["group"], reference["patients"]
    mean = {head: np.mean([per_seed[seed][head] for seed in SEEDS], axis = 0) for head in HEADS}
    np.savez(args.output / "probabilities.npz", image_ids = reference["image_ids"], label = label, group = group, patients = patients,
             **{f"{head}_mean": mean[head] for head in HEADS}, **{f"{head}_seed{seed}": per_seed[seed][head] for head in HEADS for seed in SEEDS})

    # Chỉ số nửa B
    auroc = {head: subgroup_auroc(label, group, mean[head]) for head in HEADS}
    auroc_seed = {seed: {head: subgroup_auroc(label, group, per_seed[seed][head]) for head in HEADS} for seed in SEEDS}
    primary = paired_bootstrap(label, group, patients, mean["no_tube"], mean["control"], args.n_boot, args.seed, alpha)
    refit = paired_bootstrap(label, group, patients, mean["control"], mean["original"], args.n_boot, args.seed, alpha)
    passed = primary["difference"] >= MIN_GAIN and primary["ci_low"] is not None and primary["ci_low"] > 0
    counts = {"ptx_negative": int((label == 0).sum()), "no_tube": int(((label == 1) & (group == 0)).sum()),
              "tube": int(((label == 1) & (group == 1)).sum()), "unlabelled": int(((label == 1) & (group == -1)).sum()),
              "patients": int(len(set(patients.tolist())))}

    result = {"model": MODEL, "seeds": list(SEEDS), "split_seed": SPLIT_SEED, "c_grid": list(C_GRID), "bootstrap_iteration": args.n_boot,
              "bootstrap_seed": args.seed, "neatx": str(args.neatx), "neatx_sha256": neatx_hash, "half_b_counts": counts,
              "auroc_half_b": auroc, "auroc_half_b_per_seed": {str(seed): value for seed, value in auroc_seed.items()},
              "primary_no_tube_minus_control": primary, "control_minus_original": refit, "rule": {"min_gain": MIN_GAIN}, "passed": bool(passed),
              "diagnostics": {str(seed): value for seed, value in diagnostics.items()}}
    (args.output / "summary.json").write_text(json.dumps(result, indent = 2, ensure_ascii = False), encoding = "utf-8")

    lines = [f"== Khớp lại đầu Pneumothorax theo nhóm ống (NEATX), {MODEL} 3 seed, đặc trưng đóng băng; Validation fold 1 nửa B",
             f"Nửa B: PTX- {counts['ptx_negative']:,} | PTX+ không ống {counts['no_tube']} | có ống {counts['tube']} | chưa nhãn {counts['unlabelled']} | {counts['patients']:,} bệnh nhân",
             "", f"{'đầu':10s} | {'không ống':>9s} | {'có ống':>7s} | {'mọi PTX+':>8s}   (AUROC so với PTX-, xác suất trung bình 3 seed)"]
    lines += [f"{head:10s} | {auroc[head]['no_tube']:9.4f} | {auroc[head]['tube']:7.4f} | {auroc[head]['all']:8.4f}" for head in HEADS]
    lines += ["", f"Δ không ống, (ii) no_tube − (i) control: {primary['difference']:+.4f} [CI {primary['ci_low']:+.4f}, {primary['ci_high']:+.4f}]",
              f"Báo cáo kèm, (i) control − đầu gốc     : {refit['difference']:+.4f} [CI {refit['ci_low']:+.4f}, {refit['ci_high']:+.4f}]",
              "Δ không ống từng seed: " + ", ".join(f"{seed} {auroc_seed[seed]['no_tube']['no_tube'] - auroc_seed[seed]['control']['no_tube']:+.4f}" for seed in SEEDS),
              "C được chọn (control / no_tube): " + ", ".join(f"{seed} {diagnostics[seed]['control']['c']:g} / {diagnostics[seed]['no_tube']['c']:g}" for seed in SEEDS),
              "cosin W(i), W(ii): " + ", ".join(f"{seed} {diagnostics[seed]['cosine_control_no_tube']:.3f}" for seed in SEEDS),
              "", f"QUY TẮC (Δ >= +{MIN_GAIN} và cận dưới CI > 0): " + ("ĐẠT" if passed else "KHÔNG ĐẠT")]
    text = "\n".join(lines)
    (args.output / "summary.txt").write_text(text + "\n", encoding = "utf-8")
    print(text)


if __name__ == "__main__":
    main()
