# Kiểm thử dữ liệu và giao thức của nghiên cứu CLIS: RLE CheXmask, căn đường giữa, không lật ngang, thư mục ảnh,
# lịch LR warmup + cosine, nhóm tham số không weight decay. Không cần GPU.
#   python tests/test_clis_protocol.py      (hoặc pytest)

# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import copy
import math
import time
import traceback

import numpy as np
import torch
from torchvision import transforms

import config
import models.clis_mamba as clis
import preprocess.align_midline as align
from training.optimizer import build_training_config, make_optimizer, make_scheduler
from utilities.data import image_directory, transform_setting
from utilities.setting import apply_config


# Ghi đè config trong một kiểm thử rồi trả lại giá trị cũ
class override_config:
    def __init__(self, **values):
        self.values, self.saved = values, {}

    def __enter__(self):
        self.saved = {name: copy.deepcopy(getattr(config, name)) for name in self.values}
        apply_config(self.values)

    def __exit__(self, *exc):
        apply_config(self.saved)


# ============================================================
# 1. RLE và căn đường giữa
def test_rle_round_trip():
    rng = np.random.default_rng(0)
    for order in ("C", "F"):
        for one_based in (True, False):
            mask = rng.random((37, 53)) > 0.6
            mask[0, 0], mask[-1, -1] = True, True                           # Run chạm hai đầu mảng
            decoded = align.rle_decode(align.rle_encode(mask, one_based, order), 37, 53, one_based, order)
            assert np.array_equal(decoded, mask), f"RLE order = {order}, one_based = {one_based} không khôi phục được mặt nạ"


def test_rle_known_example():
    # Quy ước mặc định: start tính từ 1, trải phẳng theo hàng. "2 3" trên lưới 2 x 4 = pixel phẳng 1, 2, 3
    expected = np.array([[0, 1, 1, 1], [0, 0, 0, 0]], dtype = bool)
    assert np.array_equal(align.rle_decode("2 3", 2, 4), expected)
    assert np.array_equal(align.rle_decode("1 3", 2, 4, one_based = False), expected)


def test_shift_horizontal_pads_with_zero():
    image = np.arange(1, 13, dtype = np.uint8).reshape(3, 4)
    assert np.array_equal(align.shift_horizontal(image, 1), [[0, 1, 2, 3], [0, 5, 6, 7], [0, 9, 10, 11]])
    assert np.array_equal(align.shift_horizontal(image, -2), [[3, 4, 0, 0], [7, 8, 0, 0], [11, 12, 0, 0]])
    assert np.array_equal(align.shift_horizontal(image, 0), image)
    assert not align.shift_horizontal(image, 9).any()


def test_midline_centres_the_lungs():
    height, width = 100, 1000
    right, left = np.zeros((height, width), bool), np.zeros((height, width), bool)
    right[20:80, 200:350] = True                                            # Phổi phải của bệnh nhân: nửa trái ảnh
    left[20:80, 450:600] = True
    row = {"Dice RCA (Mean)": 0.9, "Height": height, "Width": width, "Left Lung": align.rle_encode(left), "Right Lung": align.rle_encode(right)}
    x_mid, info = align.midline(row, width, 0.7, True, "C")
    assert info["mask_status"] == "ok" and abs(x_mid - 399.5) < 1e-9, f"x_mid = {x_mid}"
    dx = int(round(width / 2 - x_mid))
    shifted = align.shift_horizontal((right | left).astype(np.uint8), dx).astype(bool)
    centre = 0.5 * (np.nonzero(shifted & (np.arange(width) < 500))[1].mean() + np.nonzero(shifted & (np.arange(width) >= 500))[1].mean())
    assert abs(centre - width / 2) <= 1, "Sau khi dịch, đường giữa hai phổi phải ở giữa ảnh"
    assert align.midline({**row, "Dice RCA (Mean)": 0.5}, width, 0.7, True, "C")[1]["mask_status"] == "low_rca"
    assert align.midline(None, width, 0.7, True, "C")[1]["mask_status"] == "missing"


# ============================================================
# 2. Dữ liệu theo giao thức CLIS
def test_clis_transform_has_no_flip():
    with override_config(**config.CLIS_CONFIG):
        train_transform, _ = transform_setting()
    assert not any(isinstance(t, transforms.RandomHorizontalFlip) for t in train_transform.transforms), "Giao thức CLIS không được lật ngang"
    with override_config(FLIP_CHANCE = 0.5):
        train_transform, _ = transform_setting()
    assert isinstance(train_transform.transforms[0], transforms.RandomHorizontalFlip), "Giao thức cũ vẫn phải lật ngang"


def test_image_directory():
    for size, variant, expected in [(224, None, "images"), (448, None, "images_448"), (512, "aligned", "images_512_aligned"), (224, "aligned", "images_224_aligned")]:
        with override_config(IMAGE_SIZE = size, IMAGE_VARIANT = variant):
            assert image_directory(224) == expected, f"{size} / {variant}: {image_directory(224)}"


# ============================================================
# 3. Tối ưu theo giao thức CLIS
def test_warmup_cosine_schedule():
    with override_config(**config.CLIS_CONFIG):
        training_config = build_training_config(config.CLIS_PROTOCOL["optimizer_config"], config.CLIS_PROTOCOL["scheduler_config"], "macro_auroc")
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr = 1.0)
    steps_per_epoch, epochs = 10, training_config["epochs"]
    scheduler = make_scheduler(optimizer, training_config["scheduler"], steps_per_epoch, epochs)
    rates = []
    for _ in range(steps_per_epoch * epochs):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert math.isclose(rates[0], 0.1) and math.isclose(rates[9], 1.0), "Warmup tuyến tính 1 epoch: 0.1, ..., 1.0"
    assert all(a >= b for a, b in zip(rates[9:], rates[10:])), "Sau warmup LR phải giảm dần"
    assert rates[-1] < 1e-2 and rates[-1] > 0, "Cuối lịch LR gần 0"
    assert math.isclose(rates[9 + (len(rates) - 10) // 2], 0.5, abs_tol = 0.02), "Giữa đoạn cosine LR khoảng một nửa"


def test_parameter_groups_without_weight_decay():
    with override_config(**config.CLIS_CONFIG):
        training_config = build_training_config(config.CLIS_PROTOCOL["optimizer_config"], config.CLIS_PROTOCOL["scheduler_config"], "macro_auroc")
    model = clis.CLISMamba(14, mixer = "m_clis", depth = 1, pretrained = False)
    optimizer = make_optimizer(model, torch.nn.BCEWithLogitsLoss(), training_config["optimizer"])
    groups = optimizer.param_groups
    settings = [(g["lr"], g["weight_decay"]) for g in groups]
    expected = [(1e-4, 0.05), (1e-4, 0.0), (5e-4, 0.05), (5e-4, 0.0)]                # (backbone | lớp mới) x (có | không weight decay)
    assert len(settings) == 4 and all(math.isclose(a, c) and math.isclose(b, d) for (a, b), (c, d) in zip(settings, expected)), settings
    by_id = {id(p): name for name, p in model.named_parameters()}
    no_decay = {by_id[id(p)] for g in groups if g["weight_decay"] == 0 for p in g["params"]}
    for name in ("pe.row", "pe.column", "blocks.0.mixer.A_log", "blocks.0.mixer.D", "blocks.0.mixer.dt_bias", "head.bias", "norm.weight"):
        assert name in no_decay, f"{name} không được weight decay"
    for name in ("blocks.0.mixer.x_proj", "blocks.0.mixer.in_proj.weight", "head.weight"):
        assert name not in no_decay, f"{name} phải có weight decay"
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters())), "Mỗi tham số phải thuộc đúng một nhóm"


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
