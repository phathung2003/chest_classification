# Kiểm thử phương án P1 (analysis/weight_soup.py) và P2 (models/run_kd.py) của prereg mục 7, dòng 2026-11-20 01:05:
# gộp trọng số đúng công thức, tính lại BatchNorm chỉ bật chế độ train cho BatchNorm, Loss chưng cất đúng nhãn mềm và đúng đường Validation,
# xác suất teacher ghép theo mã ảnh đúng thứ tự train_df. Không cần GPU (WSL2, conda activate mamba-t29 vì cần import mô hình).
#   python tests/test_soup_kd.py      (hoặc pytest)

# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"                         # cuBLAS tất định (DETERMINISTIC_ALGORITHMS)

# Khai báo thư viện
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from analysis.weight_soup import average_state_dicts, recalibrate_batchnorm
from models.run_kd import DistillLoss, distill_build_loss

CLASSES = ["A", "B", "C"]
BCE = {"name": "bce", "pos_weight_alpha": 0.0}


# ============================================================
# Gộp trọng số: số thực lấy trung bình, số nguyên lấy của bản đầu; một bản giữ nguyên; khác tập khóa thì báo lỗi
def test_average_state_dicts():
    first = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor(5)}
    second = {"w": torch.tensor([3.0, 5.0]), "n": torch.tensor(7)}
    soup = average_state_dicts([first, second])
    assert torch.equal(soup["w"], torch.tensor([2.0, 4.0])) and soup["n"].item() == 5 and soup["n"].dtype == first["n"].dtype
    assert all(torch.equal(value, first[key]) for key, value in average_state_dicts([first]).items())
    try:
        average_state_dicts([first, {"w": second["w"]}])
    except ValueError:
        return
    raise AssertionError("Khác tập khóa phải báo lỗi")


# Tính lại BatchNorm: các lớp khác ở chế độ eval khi chạy, thống kê = trung bình tích lũy trên mọi batch, mô hình về eval sau đó
def test_batchnorm_recalibration():
    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.modes = set()
        def forward(self, x):
            self.modes.add(self.training)
            return x

    model = nn.Sequential(Recorder(), nn.BatchNorm1d(2))
    batches = [torch.tensor([[0.0, 2.0], [2.0, 4.0]]), torch.tensor([[4.0, 6.0], [6.0, 8.0]])]
    assert recalibrate_batchnorm(model, [(batch, None) for batch in batches], torch.device("cpu")) == 1
    assert model[0].modes == {False} and not model.training and not model[1].training
    assert torch.allclose(model[1].running_mean, torch.tensor([3.0, 5.0]))
    assert recalibrate_batchnorm(nn.Sequential(nn.Linear(2, 2)), [], torch.device("cpu")) == 0


# Loss chưng cất: không kèm chỉ số = Loss gốc với nhãn cứng; λ = 0 = Loss gốc; λ = 0,5 khớp BCE chép tay với nhãn mềm
def test_distill_loss():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 3, generator = generator)
    targets = (torch.rand(4, 3, generator = generator) > 0.5).float()
    teacher = torch.rand(10, 3, generator = generator)
    indices = torch.tensor([7, 2, 9, 0])
    base = nn.BCEWithLogitsLoss()
    assert torch.allclose(DistillLoss(base, teacher, 0.5)(logits, targets), base(logits, targets))
    assert torch.allclose(DistillLoss(base, teacher, 0.0)(logits, targets, indices), base(logits, targets))
    soft = 0.5 * targets + 0.5 * teacher[indices]
    manual = -(soft * torch.log(torch.sigmoid(logits)) + (1 - soft) * torch.log(1 - torch.sigmoid(logits))).mean()
    assert torch.allclose(DistillLoss(base, teacher, 0.5)(logits, targets, indices), manual, atol = 1e-6)


# Xác suất teacher ghép theo mã ảnh: tệp xáo thứ tự vẫn cho đúng hàng của train_df; thiếu ảnh thì dừng
def test_teacher_alignment():
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({config.IMAGE_ID_COLUMN: [f"img_{i}.png" for i in range(12)], **{name: rng.integers(0, 2, 12) for name in CLASSES}})
    train_df.loc[0, CLASSES], train_df.loc[1, CLASSES] = 1, 0                     # Mỗi lớp có cả mẫu dương lẫn âm (pos_weight của BCE)
    teacher = pd.DataFrame({config.IMAGE_ID_COLUMN: train_df[config.IMAGE_ID_COLUMN], **{f"prob_{name}": rng.random(12) for name in CLASSES}})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "teacher.csv"
        teacher.sample(frac = 1.0, random_state = 1).to_csv(path, index = False)
        criterion, metadata = distill_build_loss(path, 0.5)(BCE, train_df, CLASSES, torch.device("cpu"))
        expected = teacher[[f"prob_{name}" for name in CLASSES]].to_numpy(dtype = np.float32)
        assert np.allclose(criterion.teacher.numpy(), expected) and metadata["distill_lambda"] == 0.5
        assert "teacher" not in criterion.state_dict()                             # Không lưu vào checkpoint
        teacher.iloc[1:].to_csv(path, index = False)
        try:
            distill_build_loss(path, 0.5)(BCE, train_df, CLASSES, torch.device("cpu"))
        except SystemExit:
            return
    raise AssertionError("Thiếu ảnh trong tệp teacher phải dừng")


# ============================================================
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
