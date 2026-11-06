# Kiểm thử phần phân tích đăng ký trước của nghiên cứu CLIS: kết luận H2 theo độ phân giải, chọn tệp dự đoán theo seed, khóa Test ở cổng G3,
# không chấm lại Test bằng checkpoint đã đổi, so xác suất khi chấm lại Validation. Dữ liệu tổng hợp, không đọc Test, không cần GPU.
#   python tests/test_clis_analysis.py      (hoặc pytest)

# Khai báo utilities
import sys
import json
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, ROOT_DIR / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Khai báo thư viện
import os
import subprocess
import tempfile
import time
import traceback

import numpy as np
import pandas as pd

from compare_resolution import h2_verdict, prediction_files, resolution_bootstrap, seed_of
from evaluate_locked import already_scored, expand_runs, probability_difference, require_g3, sha256_file


def row(difference, ci_low):
    return {"difference": difference, "ci_low": ci_low, "ci_high": difference + 0.01}


# Quy tắc H2: Bảng chân trị của ba điều kiện
def test_h2_verdict_rules():
    assert h2_verdict(row(0.01, 0.002), row(0.012, 0.004), row(0.002, -0.003), 0.005) == "supported"
    assert h2_verdict(row(0.01, 0.002), row(0.016, 0.004), row(0.006, 0.001), 0.005) == "partial"
    assert h2_verdict(row(0.01, -0.001), row(0.012, 0.004), row(0.002, -0.003), 0.005) == "rejected"           # D có CI chứa 0
    assert h2_verdict(row(0.004, 0.001), row(0.012, 0.004), row(0.002, -0.003), 0.005) == "rejected"            # D dưới ngưỡng thực tiễn
    assert h2_verdict(row(0.01, 0.002), row(0.004, 0.001), row(-0.006, -0.01), 0.005) == "rejected"             # Chuỗi dài nhất không có lợi thế đủ lớn
    assert h2_verdict({"difference": 0.01, "ci_low": None, "ci_high": None}, row(0.012, 0.004), row(0.0, -0.003), 0.005) == "rejected"


# Dữ liệu tổng hợp: điểm = tín hiệu x nhãn + nhiễu, mỗi bệnh nhân một ảnh
def synthetic(signal_by_size, noise = 1.0, images = 4000, classes = 4, seed = 0):
    rng = np.random.default_rng(seed)
    targets = (rng.random((images, classes)) < 0.2).astype(np.float64)
    shared_noise = rng.normal(0, noise, targets.shape)
    return targets, {size: signal * targets + shared_noise for size, signal in signal_by_size.items()}


# H2 tổng hợp: lợi thế tăng theo độ phân giải và bằng 0 ở chuỗi ngắn --> supported
def test_h2_supported_on_synthetic_trend():
    sizes = (256, 1024)
    targets, control = synthetic({256: 1.0, 1024: 1.0}, seed = 1)
    ssm = {256: control[256], 1024: control[1024] + 0.5 * targets}
    result = resolution_bootstrap(targets, ssm, control, np.arange(len(targets)), sizes, bootstrap_iteration = 100, seed = 0, alpha = 0.05)
    assert result["delta_256"]["difference"] == 0
    assert result["delta_1024"]["difference"] > 0.005 and result["trend"]["ci_low"] > 0
    assert result["verdict"] == "supported"


# H2 tổng hợp: hai cấu hình như nhau ở mọi độ phân giải --> rejected
def test_h2_rejected_without_gain():
    sizes = (256, 512, 1024)
    targets, control = synthetic({size: 1.0 for size in sizes}, seed = 2)
    result = resolution_bootstrap(targets, control, control, np.arange(len(targets)), sizes, bootstrap_iteration = 50, seed = 0, alpha = 0.05)
    assert result["trend"]["difference"] == 0
    assert result["verdict"] == "rejected"


# H2 tổng hợp: có lợi thế ở cả chuỗi ngắn và lợi thế tăng theo độ phân giải --> partial
def test_h2_partial_when_short_sequence_also_gains():
    sizes = (256, 1024)
    targets, control = synthetic({256: 1.0, 1024: 1.0}, seed = 3)
    ssm = {256: control[256] + 0.2 * targets, 1024: control[1024] + 0.7 * targets}
    result = resolution_bootstrap(targets, ssm, control, np.arange(len(targets)), sizes, bootstrap_iteration = 100, seed = 0, alpha = 0.05)
    assert abs(result["delta_256"]["difference"]) >= 0.005 and result["trend"]["ci_low"] > 0
    assert result["verdict"] == "partial"


# Cần ít nhất hai độ phân giải
def test_h2_needs_two_sizes():
    targets, control = synthetic({256: 1.0}, seed = 4)
    try:
        resolution_bootstrap(targets, control, control, np.arange(len(targets)), (256,), bootstrap_iteration = 5, seed = 0, alpha = 0.05)
    except ValueError:
        return
    raise AssertionError("Một độ phân giải phải bị từ chối")


# Tạo một lượt chạy giả với seed cho trước
def fake_run(root, name, seed):
    fold_directory = root / name / "fold_1"
    fold_directory.mkdir(parents = True)
    (root / name / "train_info.json").write_text(json.dumps({"settings": {"seed": seed}}), encoding = "utf-8")
    (fold_directory / "oof_predictions.csv").write_text("x\n1\n", encoding = "utf-8")
    return fold_directory / "oof_predictions.csv"


# Chọn đúng seed, đúng thứ tự; bỏ qua seed không dùng; thiếu hoặc thừa --> lỗi
def test_prediction_files_by_seed():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path_42, path_43 = fake_run(root, "run_1", 42), fake_run(root, "run_2", 43)
        fake_run(root, "run_3", 44)
        pattern = str(root / "run_*" / "fold_1" / "oof_predictions.csv")
        assert seed_of(path_43) == 43
        assert prediction_files(pattern, (43, 42)) == [path_43, path_42]
        for seeds in ((42, 45),):
            try:
                prediction_files(pattern, seeds)
            except ValueError:
                continue
            raise AssertionError("Thiếu seed phải báo lỗi")
        fake_run(root, "run_4", 42)
        try:
            prediction_files(pattern, (42, 43))
        except ValueError:
            return
        raise AssertionError("Một seed có hai lượt chạy phải báo lỗi")


# Chạy đầu-cuối bằng dòng lệnh trên tệp dự đoán tổng hợp: đọc đúng seed, ghi kết quả có kết luận
def test_compare_resolution_command_line():
    rng = np.random.default_rng(5)
    labels = (rng.random((3000, 2)) < 0.2).astype(int)
    noise = rng.normal(0, 1, labels.shape)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for model, gain in (("ssm", {256: 0.0, 1024: 0.8}), ("control", {256: 0.0, 1024: 0.0})):
            for size in (256, 1024):
                for run, seed in (("run_1", 42), ("run_2", 43), ("run_3", 44)):
                    fold_directory = root / f"{model}_{size}px" / run / "fold_1"
                    fold_directory.mkdir(parents = True)
                    (root / f"{model}_{size}px" / run / "train_info.json").write_text(json.dumps({"settings": {"seed": seed}}), encoding = "utf-8")
                    frame = pd.DataFrame({"Image Index": [f"{i}.png" for i in range(len(labels))], "Patient ID": np.arange(len(labels)), "A": labels[:, 0], "B": labels[:, 1]})
                    scores = 1 / (1 + np.exp(-((1 + gain[size]) * labels + noise)))
                    frame["prob_A"], frame["prob_B"] = scores[:, 0], scores[:, 1]
                    frame.to_csv(fold_directory / "oof_predictions.csv", index = False)
        output = root / "h2.json"
        pattern = str(root / "{model}_{{size}}px" / "run_*" / "fold_1" / "oof_predictions.csv")
        subprocess.run([sys.executable, str(ROOT_DIR / "analysis" / "compare_resolution.py"), "--ssm", pattern.format(model = "ssm"), "--control", pattern.format(model = "control"),
                        "--sizes", "256", "1024", "--n-boot", "30", "--output", str(output)], check = True, cwd = ROOT_DIR, capture_output = True, env = {**os.environ, "PYTHONIOENCODING": "utf-8"})
        result = json.loads(output.read_text(encoding = "utf-8"))
        assert result["seeds"] == [42, 43] and all(len(paths) == 2 for paths in result["files"].values())
        assert result["comparison"]["delta_256"]["difference"] == 0
        assert result["comparison"]["verdict"] == "supported"


# Test bị khóa: thiếu xác nhận, thiếu tag, hoặc chấm một phần --> dừng
def test_test_set_is_locked():
    for confirmed, tag_commit, limit in ((False, "abc", None), (True, None, None), (True, "abc", 100)):
        try:
            require_g3(confirmed, tag_commit, limit)
        except SystemExit:
            continue
        raise AssertionError(f"Phải dừng: {confirmed}, {tag_commit}, {limit}")
    require_g3(True, "abc")


# Không chấm lại Test bằng checkpoint đã đổi
def test_scoring_once_per_checkpoint():
    assert already_scored(False, None, "hash_a") is False
    assert already_scored(True, {"best_model_sha256": "hash_a"}, "hash_a") is True
    for entry in ({"best_model_sha256": "hash_b"}, None):
        try:
            already_scored(True, entry, "hash_a")
        except SystemExit:
            continue
        raise AssertionError("Checkpoint đã đổi phải dừng")


# So xác suất ghép theo mã ảnh, không phụ thuộc thứ tự dòng
def test_probability_difference_aligns_by_image():
    saved = pd.DataFrame({"Image Index": ["a", "b", "c"], "prob_X": [0.1, 0.2, 0.3], "prob_Y": [0.4, 0.5, 0.6]})
    scored = pd.DataFrame({"Image Index": ["c", "a"], "prob_X": [0.3, 0.1], "prob_Y": [0.6, 0.45]})
    assert abs(probability_difference(saved, scored, ["X", "Y"], "Image Index") - 0.05) < 1e-12
    assert probability_difference(saved, saved, ["X", "Y"], "Image Index") == 0


# Mở rộng mẫu * của thư mục lượt chạy; SHA-256 khớp giá trị đã biết
def test_expand_runs_and_hash():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "run_1").mkdir()
        (root / "run_2").mkdir()
        assert expand_runs([root / "run_*"]) == [root / "run_1", root / "run_2"]
        assert expand_runs([root / "run_9"]) == [root / "run_9"]
        (root / "file").write_bytes(b"abc")
        assert sha256_file(root / "file") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


if __name__ == "__main__":
    failed = []
    for name, test in [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]:
        started = time.perf_counter()
        try:
            test()
            status = "ĐẠT"
        except Exception:
            traceback.print_exc()
            status = "LỖI"
            failed.append(name)
        print(f"[{status}] {name} ({time.perf_counter() - started:.1f}s)", flush = True)
    print(f"\n{len(failed)} lỗi" + (f": {failed}" if failed else ""))
    sys.exit(1 if failed else 0)
