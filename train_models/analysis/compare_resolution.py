"""Kiểm định H2 của nghiên cứu CLIS (E4): lợi thế của SSM so với GatedConv có tăng theo độ dài chuỗi không.

Δ(R) là hiệu macro AUROC 14 nhãn (SSM - đối chứng) ở độ phân giải R, tính trên xác suất trung bình qua seed; D = Δ(R lớn nhất) - Δ(R nhỏ nhất).
Bootstrap ghép cặp theo bệnh nhân: cùng một mẫu bệnh nhân cho mọi cấu hình ở mọi độ phân giải. Trước cổng G3 chỉ dùng dự đoán Validation; sau G3 mới dùng Test.

Quy tắc đăng ký trước (prereg/analysis_plan.md, mục 2):
- supported: D và Δ(R lớn nhất) đều có cận dưới CI 95% > 0 và giá trị >= ngưỡng thực tiễn, đồng thời |Δ(R nhỏ nhất)| < ngưỡng
- partial  : Hai điều kiện đầu đạt, nhưng Δ(R nhỏ nhất) không xấp xỉ 0
- rejected : Không đạt một trong hai điều kiện đầu

Ví dụ (Validation, seed 42 và 43 ở cả 4 độ phân giải; {size} do script thay, mẫu * do script tự mở rộng):
    python analysis/compare_resolution.py
        --ssm "training_results/nih_chest_xray_14/clismamba_v1_m_ras_{size}px/run_*/fold_1/oof_predictions.csv"
        --control "training_results/nih_chest_xray_14/clismamba_v1_gatedconv_{size}px/run_*/fold_1/oof_predictions.csv"
        --name-ssm M-ras --output h2_validation.json
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

import config
from compare_clis import PRACTICAL_DELTA, load_configuration, p_values
from utilities.file import read_json, save_json
from utilities.statistics import _bootstrap_samples, _interval, _macro, auroc_per_class

DEFAULT_SIZES = (256, 512, 768, 1024)
DEFAULT_SEEDS = (42, 43)                                                    # Hai seed có ở cả 4 độ phân giải (E4): dùng cùng số seed để không cấu hình nào được lợi do ensemble


# ============================================================
# Seed của lượt chạy chứa tệp dự đoán (train_info.json ở thư mục cha gần nhất)
# Trả về: Seed (int)
def seed_of(path):

    # Duyệt các thư mục cha
    for parent in Path(path).parents:
        info_path = parent / "train_info.json"
        if info_path.is_file():
            return int(read_json(info_path)["settings"]["seed"])

    raise FileNotFoundError(f"[seed_of]: Không tìm thấy train_info.json cho {path}")


# ============================================================
# Tệp dự đoán của một cấu hình theo đúng danh sách seed
# Trả về: Danh sách đường dẫn, mỗi seed đúng một tệp, theo thứ tự seeds
def prediction_files(pattern, seeds):

    # Gom các tệp theo seed
    by_seed = {}
    for path in sorted(glob.glob(str(pattern))):
        by_seed.setdefault(seed_of(path), []).append(Path(path))

    # Kiểm tra dữ liệu đầu vào
    ## Mỗi seed phải có đúng một lượt chạy
    wrong = {seed: len(by_seed.get(seed, [])) for seed in seeds if len(by_seed.get(seed, [])) != 1}
    if wrong:
        raise ValueError(
            "[prediction_files]: Mỗi seed phải có đúng một tệp dự đoán.\n"
            f"Nhận được: {pattern}, số tệp theo seed bị thiếu / thừa = {wrong}"
        )

    return [by_seed[seed][0] for seed in seeds]


# ============================================================
# Kết luận H2 theo quy tắc đăng ký trước
# Trả về: "supported" | "partial" | "rejected"
def h2_verdict(trend, longest, shortest, practical_delta):

    # Điều kiện bắt buộc: Δ tăng theo độ dài chuỗi và có lợi thế thật ở chuỗi dài nhất
    rising = trend["ci_low"] is not None and trend["ci_low"] > 0 and trend["difference"] >= practical_delta
    gain = longest["ci_low"] is not None and longest["ci_low"] > 0 and longest["difference"] >= practical_delta
    if not (rising and gain):
        return "rejected"

    # Dự đoán của MambaOut: chuỗi ngắn nhất không có lợi thế đáng kể
    return "supported" if abs(shortest["difference"]) < practical_delta else "partial"


# ============================================================
# Bootstrap ghép cặp theo bệnh nhân qua mọi độ phân giải
# - ssm, control: {độ phân giải: xác suất [N, C]}, cùng thứ tự ảnh với targets
# Trả về: dict gồm delta_<R>, trend (D), verdict
def resolution_bootstrap(targets, ssm, control, patients, sizes, *, bootstrap_iteration, seed, alpha, practical_delta = PRACTICAL_DELTA):
    sizes = sorted(int(size) for size in sizes)

    # Kiểm tra dữ liệu đầu vào
    ## Cần ít nhất hai độ phân giải để nói về xu hướng
    if len(sizes) < 2:
        raise ValueError(f"[resolution_bootstrap]: Cần ít nhất hai độ phân giải. Nhận được: sizes = {sizes}")

    # Δ từng độ phân giải và D trên một tập dòng
    def summaries(rows = None):
        y = targets if rows is None else targets[rows]
        values = {}
        for size in sizes:
            a, b = (ssm[size], control[size]) if rows is None else (ssm[size][rows], control[size][rows])
            values[f"delta_{size}"] = _macro(auroc_per_class(y, a)) - _macro(auroc_per_class(y, b))
        values["trend"] = values[f"delta_{sizes[-1]}"] - values[f"delta_{sizes[0]}"]
        return values

    # Giá trị điểm và các mẫu bootstrap
    point = summaries()
    samples = {key: [] for key in point}
    for rows in _bootstrap_samples(patients, bootstrap_iteration, seed):
        for key, value in summaries(rows).items():
            samples[key].append(value)

    # Khoảng tin cậy và p-value
    result = {}
    for key, values in samples.items():
        values = np.asarray(values)
        low, high = _interval(values, alpha)
        two_sided, one_sided = p_values(values)
        result[key] = {"difference": float(point[key]), "ci_low": low, "ci_high": high, "p_two_sided": two_sided, "p_one_sided_a_greater": one_sided}

    # Kết luận
    result["verdict"] = h2_verdict(result["trend"], result[f"delta_{sizes[-1]}"], result[f"delta_{sizes[0]}"], practical_delta)

    return result


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ssm", required = True, help = "Mẫu đường dẫn tệp dự đoán của cấu hình SSM, chứa {size}")
    parser.add_argument("--control", required = True, help = "Mẫu đường dẫn tệp dự đoán của đối chứng, chứa {size}")
    parser.add_argument("--sizes", nargs = "+", type = int, default = list(DEFAULT_SIZES))
    parser.add_argument("--seeds", nargs = "+", type = int, default = list(DEFAULT_SEEDS))
    parser.add_argument("--name-ssm", default = "SSM")
    parser.add_argument("--name-control", default = "GatedConv")
    parser.add_argument("--n-boot", type = int, default = int(config.COMPARE_BOOTSTRAP_ITERATION))
    parser.add_argument("--seed", type = int, default = 0, help = "Seed của bootstrap (đăng ký trước: 0)")
    parser.add_argument("--practical-delta", type = float, default = PRACTICAL_DELTA, help = "Ngưỡng ε của kế hoạch phân tích")
    parser.add_argument("--patient-column", default = config.PATIENT_ID_COLUMN)
    parser.add_argument("--key-column", default = config.IMAGE_ID_COLUMN)
    parser.add_argument("--output", type = Path, default = None)
    args = parser.parse_args()
    alpha = round(1 - config.BOOTSTRAP_CONFIDENCE, 12)

    # Đọc xác suất trung bình qua seed của từng cấu hình, đưa về cùng thứ tự ảnh của cấu hình đầu tiên
    base, classes, ssm, control, files = None, None, {}, {}, {}
    for size in args.sizes:
        for side, pattern, store in (("ssm", args.ssm, ssm), ("control", args.control, control)):
            paths = prediction_files(pattern.format(size = size), args.seeds)
            files[f"{side}_{size}"] = [str(path) for path in paths]
            frame, frame_classes, seeds = load_configuration(paths, args.key_column)
            if base is None:
                base, classes = frame, frame_classes
            if frame_classes != classes or set(frame[args.key_column]) != set(base[args.key_column]):
                raise ValueError(f"{paths[0]}: danh sách lớp hoặc tập ảnh khác cấu hình đầu tiên")
            order = frame.set_index(args.key_column).index.get_indexer(base[args.key_column])
            if not (frame[classes].to_numpy()[order] == base[classes].to_numpy()).all():
                raise ValueError(f"{paths[0]}: nhãn khác cấu hình đầu tiên")
            store[size] = seeds[:, order].mean(axis = 0)

    # Bootstrap
    targets = base[classes].to_numpy(dtype = np.float64)
    patients = base[args.patient_column].to_numpy()
    comparison = resolution_bootstrap(targets, ssm, control, patients, args.sizes, bootstrap_iteration = args.n_boot, seed = args.seed, alpha = alpha, practical_delta = args.practical_delta)

    # In và lưu kết quả
    print(f"{args.name_ssm} - {args.name_control} | {len(targets):,} ảnh, {pd.Series(patients).nunique():,} bệnh nhân, seed {args.seeds}, bootstrap {args.n_boot} lần")
    for key, row in comparison.items():
        if key != "verdict":
            print(f"  {key:12s} {row['difference']:+.4f} [CI {row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | p hai phía {row['p_two_sided']:.4f}")
    print(f"Kết luận H2 (ngưỡng {args.practical_delta:g}): {comparison['verdict']}")
    if args.output is not None:
        save_json(args.output, {"method": "patient-level paired percentile bootstrap across resolutions on seed-averaged probabilities", "bootstrap_iteration": args.n_boot, "seed": args.seed,
                                "confidence": 1 - alpha, "images": len(targets), "patients": int(pd.Series(patients).nunique()), "seeds": args.seeds, "sizes": args.sizes,
                                "practical_delta": args.practical_delta, "model_ssm": args.name_ssm, "model_control": args.name_control, "comparison": comparison, "files": files})
        print(f"Đã lưu: {args.output}")


if __name__ == "__main__":
    main()
