# ============================================================
# So sánh ghép cặp hai mô hình (Paired Model Comparison)
# Mục tiêu: Kiểm tra mô hình A có thật sự tốt hơn mô hình B trên CÙNG tập Test hay chỉ do ngẫu nhiên, bằng bootstrap ghép cặp theo bệnh nhân; p-value từng lớp được hiệu chỉnh đa so sánh bằng Holm.

# Cách đọc kết quả (ví dụ macro AUROC của A = 0.842, của B = 0.835 --> hiệu 0.007):
# - Khoảng tin cậy 95% của hiệu [0.002, 0.012] không chứa 0 --> A tốt hơn B có ý nghĩa thống kê.
# - Khoảng tin cậy [-0.003, 0.017] chứa 0 --> chưa đủ bằng chứng để kết luận A tốt hơn B.
# ============================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from utilities.statistics import paired_bootstrap_auroc
from utilities.file import save_json

# ============================================================
# Hiệu chỉnh Holm-Bonferroni cho nhiều p-value
# Trả về: Mảng p-value đã hiệu chỉnh, cùng thứ tự đầu vào
def holm(p_values):

    # Chuyển đổi dữ liệu: None --> NaN
    p_value_array = np.asarray([np.nan if value is None else value for value in p_values], dtype = np.float64)

    # Khởi tạo kết quả (mặc định NaN)
    adjusted = np.full_like(p_value_array, np.nan)

    # Thứ tự tăng dần của các p-value hợp lệ
    valid = np.flatnonzero(np.isfinite(p_value_array))
    order = valid[np.argsort(p_value_array[valid])]

    # Nhân hệ số (m - k), cực đại tích lũy để giữ thứ tự, chặn trên ở 1
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_value_array[index]))
        adjusted[index] = running

    return adjusted

# ============================================================
# Đọc tệp dự đoán test_predictions.csv
# Trả về: (frame, classes)
def load_predictions(path):

    # Đọc tệp và lấy danh sách lớp
    frame = pd.read_csv(path)
    classes = [column[5:] for column in frame.columns if column.startswith("prob_")]

    # Kiểm tra dữ liệu đầu vào
    ## Tệp phải có ít nhất một cột prob_<lớp>
    if not classes:
        raise ValueError(
            "[load_predictions]: Tệp dự đoán phải có ít nhất một cột prob_<lớp>.\n"
            f"Nhận được: path = {path}"
        )

    return frame, classes

# ============================================================
# So sánh ghép cặp hai tệp dự đoán
# Trả về: Kết quả paired_bootstrap_auroc, mỗi lớp có thêm p_value_holm
def compare_predictions(predictions_a, predictions_b, *, bootstrap_iteration, seed, patient_column, key_column):

    # Đọc hai tệp dự đoán
    frame_a, classes_a = load_predictions(predictions_a)
    frame_b, classes_b = load_predictions(predictions_b)

    # Kiểm tra dữ liệu đầu vào
    ## Hai tệp phải cùng danh sách lớp
    if classes_a != classes_b:
        raise ValueError(
            "[compare_predictions]: Hai tệp dự đoán phải cùng danh sách lớp.\n"
            f"Nhận được: A = {classes_a}, B = {classes_b}"
        )

    # Danh sách lớp chung
    classes = classes_a

    ## Mỗi tệp không có ảnh trùng lặp
    for frame, name in ((frame_a, "A"), (frame_b, "B")):
        duplicated_images = frame.loc[frame[key_column].duplicated(), key_column]

        if len(duplicated_images) > 0:
            raise ValueError(
                f"[compare_predictions]: Tệp {name} không được có ảnh trùng lặp.\n"
                f"Số ảnh bị trùng: {duplicated_images.nunique()}.\n"
                f"Ảnh bị trùng đầu tiên: {duplicated_images.iloc[0]}"
            )

    ## Hai tệp phải cùng tập ảnh Test
    if set(frame_a[key_column]) != set(frame_b[key_column]):
        raise ValueError(
            "[compare_predictions]: Hai tệp dự đoán phải cùng tập ảnh Test.\n"
            f"Nhận được: Chỉ có ở A = {len(set(frame_a[key_column]) - set(frame_b[key_column]))} ảnh, "
            f"chỉ có ở B = {len(set(frame_b[key_column]) - set(frame_a[key_column]))} ảnh"
        )

    # Sắp xếp tệp B theo thứ tự ảnh của tệp A
    frame_b = frame_b.set_index(key_column).loc[frame_a[key_column]].reset_index()

    # Kiểm tra nhãn
    ## Nhãn của hai tệp phải khớp nhau
    if not (frame_a[classes].to_numpy() == frame_b[classes].to_numpy()).all():
        raise ValueError(
            "[compare_predictions]: Nhãn của hai tệp phải khớp nhau.\n"
            f"Nhận được: Số phần tử khác nhau = {int((frame_a[classes].to_numpy() != frame_b[classes].to_numpy()).sum())}"
        )

    # Bootstrap ghép cặp theo bệnh nhân
    result = paired_bootstrap_auroc(frame_a[classes].to_numpy(dtype = np.float64), frame_a[[f"prob_{name}" for name in classes]].to_numpy(dtype = np.float64), frame_b[[f"prob_{name}" for name in classes]].to_numpy(dtype = np.float64), frame_a[patient_column].to_numpy(), classes, bootstrap_iteration = bootstrap_iteration, seed = seed)

    # Hiệu chỉnh Holm cho p-value từng lớp
    adjusted = holm([row["p_value"] for row in result["classes"]])
    for row, value in zip(result["classes"], adjusted):
        row["p_value_holm"] = None if not np.isfinite(value) else float(value)

    return result

# ============================================================
# Dòng lệnh: So sánh hai tệp dự đoán, in kết quả và lưu JSON (nếu có --output)
def main():

    # Đọc tham số dòng lệnh
    parser = argparse.ArgumentParser(description = "So sánh ghép cặp hai mô hình trên cùng tập Test bằng bootstrap theo bệnh nhân (hiệu macro AUROC, khoảng tin cậy 95%, p-value Holm từng lớp)")
    parser.add_argument("predictions_a")
    parser.add_argument("predictions_b")
    parser.add_argument("--name-a", default = "A")
    parser.add_argument("--name-b", default = "B")
    parser.add_argument("--n-boot", type = int, default = int(config.COMPARE_BOOTSTRAP_ITERATION))
    parser.add_argument("--seed", type = int, default = int(config.BOOTSTRAP_SEED))
    parser.add_argument("--patient-column", default = config.PATIENT_ID_COLUMN)
    parser.add_argument("--key-column", default = config.IMAGE_ID_COLUMN)
    parser.add_argument("--output", type = Path, default = None)
    args = parser.parse_args()

    # So sánh và ghi tên / tệp của hai mô hình vào kết quả (--n-boot: Số lần bootstrap)
    result = compare_predictions(args.predictions_a, args.predictions_b, bootstrap_iteration = args.n_boot, seed = args.seed, patient_column = args.patient_column, key_column = args.key_column)
    result["model_a"] = {"name": args.name_a, "predictions": str(args.predictions_a)}
    result["model_b"] = {"name": args.name_b, "predictions": str(args.predictions_b)}

    # In kết quả: Macro AUROC, hiệu, khoảng tin cậy, p-value và bảng từng lớp
    macro = result["macro_auroc"]
    print(f"Macro AUROC {args.name_a}: {macro['a']:.4f} | {args.name_b}: {macro['b']:.4f}")
    print(f"Hiệu (A - B): {macro['difference']:+.4f} "
          f"[95% CI {macro['ci_low']:+.4f}, {macro['ci_high']:+.4f}] | p = {macro['p_value']:.4f}")
    print(pd.DataFrame(result["classes"]).round(4).to_string(index = False))

    # Lưu JSON
    if args.output is not None:
        save_json(args.output, result)
        print(f"Đã lưu: {args.output}")

if __name__ == "__main__":
    main()
