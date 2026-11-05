# Kiểm thử dữ liệu và giao thức của nghiên cứu CLIS: RLE CheXmask, căn đường giữa, không lật ngang, thư mục ảnh,
# tập con bệnh nhân cho tinh chỉnh, chọn hệ số LR khi loại dần, bảng hệ số LR m* và cách tìm lượt chạy theo m*, tổng kết điểm qua seed,
# lịch LR warmup + cosine, nhóm tham số không weight decay, cấu hình chạy (checkpointing khối, fused AdamW), đọc phiên bản thư viện. Không cần GPU.
#   python tests/test_clis_protocol.py      (hoặc pytest)

# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"                         # cuBLAS tất định (DETERMINISTIC_ALGORITHMS)
os.environ["TRITON_CACHE_AUTOTUNING"] = "1"                                # mamba-ssm >= 2.3 ở chế độ tất định: lưu autotune kernel Triton (CLIS không dùng) thay vì cảnh báo

# Khai báo thư viện
import copy
import json
import math
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

import config
import models.clis_mamba as clis
import preprocess.align_midline as align
from training.optimizer import build_training_config, make_optimizer, make_scheduler
from training.train import subsample_patients
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


def test_subsample_patients():
    # Tập con cho tinh chỉnh siêu tham số: giữ trọn bệnh nhân, đúng tỉ lệ, cố định theo TRAIN_PATIENT_SEED (không theo SEED / thứ tự dòng)
    patient = config.PATIENT_ID_COLUMN
    frame = pd.DataFrame({patient: np.random.default_rng(1).integers(0, 500, 3000), "row": np.arange(3000)})
    assert subsample_patients(frame, 1.0, 2026) is frame, "TRAIN_PATIENT_FRACTION = 1 phải giữ nguyên tập huấn luyện"

    part = subsample_patients(frame, 0.3, 2026)
    kept = set(part[patient])
    assert len(kept) == round(0.3 * frame[patient].nunique()), f"Số bệnh nhân giữ lại: {len(kept)}"
    assert len(part) == int(frame[patient].isin(kept).sum()), "Phải giữ mọi ảnh của bệnh nhân được chọn"
    assert part["row"].is_monotonic_increasing, "Phải giữ thứ tự dòng"

    with override_config(SEED = 7):
        again = subsample_patients(frame.sample(frac = 1, random_state = 3), 0.3, 2026)
    assert set(again[patient]) == kept, "Tập con phải giống nhau với mọi SEED và mọi thứ tự dòng"
    assert set(subsample_patients(frame, 0.3, 2027)[patient]) != kept, "TRAIN_PATIENT_SEED khác phải cho tập con khác"

    for fraction in (0.0, -0.1, 1.5):
        try:
            subsample_patients(frame, fraction, 2026)
        except ValueError:
            continue
        raise AssertionError(f"TRAIN_PATIENT_FRACTION = {fraction} phải báo lỗi")


def test_select_survivors():
    # Loại dần mức LR: giữ các hệ số có điểm tốt nhất, hòa điểm thì giữ thứ tự đã khai báo
    from models.run_clis import select_survivors
    auroc = {5.0: 0.82, 2.0: 0.80, 10.0: 0.82, 20.0: 0.78}
    results = {m: {"best_score": s, "higher_is_better": True} for m, s in auroc.items()}
    assert select_survivors(results, 2) == [5.0, 10.0], select_survivors(results, 2)
    assert select_survivors(results, 1) == [5.0]
    loss = {m: {"best_score": s, "higher_is_better": False} for m, s in {5.0: 0.20, 2.0: 0.18, 10.0: 0.25}.items()}
    assert select_survivors(loss, 2) == [2.0, 5.0], "validation_loss: điểm thấp hơn là tốt hơn"


def test_lr_multiplier_table():
    # Mọi mô hình của hàng đợi có hệ số m* dương; ablation m_clis4 giữ hệ số của m_clis;
    # bảng không vào chữ ký (hệ số của mô hình đang chạy vào chữ ký qua training_config)
    import models.run_clis as run_clis
    from utilities.tracking import config_snapshot
    assert set(config.CLIS_LR_MULTIPLIER) == set(run_clis.NAMES), sorted(set(config.CLIS_LR_MULTIPLIER) ^ set(run_clis.NAMES))
    assert all(math.isfinite(m) and m > 0 for m in config.CLIS_LR_MULTIPLIER.values()), config.CLIS_LR_MULTIPLIER
    assert config.CLIS_LR_MULTIPLIER["m_clis4"] == config.CLIS_LR_MULTIPLIER["m_clis"]
    assert "CLIS_LR_MULTIPLIER" not in config_snapshot()


# Tạo một lượt chạy giả (train_info.json, fold_1, fold_done.json) cho kiểm thử tìm lượt chạy. Trả về thư mục fold_1
def write_run(directory, index, multiplier, learning_rate = 1e-4, seed = 42, done = True):
    fold_directory = directory / f"run_{index}" / "fold_1"
    fold_directory.mkdir(parents = True)
    settings = {"seed": seed, "image_size": 512, "model_kwargs": {"mixer": "m_clis"}, "torch_version": torch.__version__, "config": {},
                "training_config": {"optimizer": {"learning_rate": learning_rate, "new_module_lr_multiplier": multiplier}}}
    (fold_directory.parent / "train_info.json").write_text(json.dumps({"settings": settings}), encoding = "utf-8")
    if done:
        (fold_directory / "fold_done.json").write_text("{}", encoding = "utf-8")
    return fold_directory


def test_runs_found_by_lr_multiplier():
    # Điểm khởi đầu cho độ phân giải cao (find_checkpoint) là lượt 512 px có hệ số m* của biến thể, không phải hệ số mặc định ×5
    # của CLIS_PROTOCOL; tìm lượt chạy theo (seed, hệ số, LR backbone) cho điểm của hàng đợi
    import models.run_clis as run_clis
    from utilities.file import to_snake_case
    multiplier = config.CLIS_LR_MULTIPLIER["m_clis"]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        info_path = root / "dataset_info.json"
        info_path.write_text(json.dumps({"dataset_name": "NIH Test"}), encoding = "utf-8")
        directory = root / to_snake_case("NIH Test") / to_snake_case(clis.model_name("m_clis", 512))
        default = write_run(directory, 1, 5.0)                                     # Hệ số mặc định của CLIS_PROTOCOL
        chosen = write_run(directory, 2, multiplier)                                # m* của m_clis
        lower = write_run(directory, 3, multiplier, learning_rate = 5e-5)           # Cùng m*, LR backbone khác (--lr)
        write_run(directory, 4, multiplier, seed = 43, done = False)                # Seed khác, chưa xong fold
        with override_config(MODELS_OUTPUT_DIRECTORY = root, DATASET_INFO_DIRECTORY = info_path):
            assert run_clis.find_checkpoint("m_clis", 42, 1, 512) == chosen / "best_model.pt"
            assert run_clis.find_fold_directory("m_clis", 42, multiplier, 1, {}, learning_rate = 1e-4) == chosen
            assert run_clis.find_fold_directory("m_clis", 42, multiplier, 1, {}, learning_rate = 5e-5) == lower
            assert run_clis.find_fold_directory("m_clis", 42, 5.0, 1, {}) == default
            assert run_clis.find_fold_directory("m_clis", 44, multiplier, 1, {}) is None
            try:
                run_clis.find_checkpoint("m_clis", 43, 1, 512)
            except SystemExit:
                pass
            else:
                raise AssertionError("Lượt chưa xong fold không được làm điểm khởi đầu")


def test_seed_summary():
    # Tổng kết điểm Validation qua seed: trung bình và độ lệch chuẩn mẫu; một seed --> độ lệch chuẩn 0
    from models.run_clis import seed_summary
    summary = seed_summary({("densenet121", 10.0): {42: 0.82, 43: 0.84, 44: 0.83}, ("convnext_t", 40.0): {42: 0.83}})
    mean, spread = summary[("densenet121", 10.0)]
    assert math.isclose(mean, 0.83) and math.isclose(spread, 0.01), summary
    assert summary[("convnext_t", 40.0)] == (0.83, 0.0), summary


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
# 4. Cấu hình chạy và môi trường
def test_run_child_settings():
    # Biến thể CLIS-Mamba nhận grad_checkpoint theo BLOCK_GRAD_CHECKPOINT (vào chữ ký qua model_kwargs), baseline không nhận;
    # optimizer dùng fused AdamW theo CLIS_CONFIG
    import models.run_clis as run_clis
    saved = {name: copy.deepcopy(getattr(config, name)) for name in set(config.CLIS_CONFIG) | {"SEED", "TRAIN_ALL_FOLD", "SELECTED_FOLD"}}
    captured = []
    original, run_clis.start_train = run_clis.start_train, lambda **kwargs: captured.append(kwargs)
    try:
        run_clis.run_child("m_clis", 42, 1, 2.0, {})
        run_clis.run_child("m_clis", 42, 1, 2.0, {"BLOCK_GRAD_CHECKPOINT": True})
        run_clis.run_child("convnext_t", 42, 1, 20.0, {})
        run_clis.run_child("densenet121", 42, 1, None, {})
        run_clis.run_child("flipdiff", 42, 1, None, {})
    finally:
        run_clis.start_train = original
        apply_config(saved)
    assert captured[0]["model_kwargs"] == {"mixer": "m_clis", "grad_checkpoint": config.CLIS_CONFIG["BLOCK_GRAD_CHECKPOINT"]}, captured[0]["model_kwargs"]
    assert captured[1]["model_kwargs"]["grad_checkpoint"] is True, "--set BLOCK_GRAD_CHECKPOINT=True phải bật lại checkpointing khối"
    assert captured[2]["model_kwargs"] == {}, "Baseline không có checkpointing khối"
    assert captured[0]["optimizer_config"]["new_module_lr_multiplier"] == 2.0
    assert captured[2]["optimizer_config"]["new_module_lr_multiplier"] == 20.0, "--lr-multiplier ghi đè bảng m*"
    for index, name in ((3, "densenet121"), (4, "flipdiff")):
        assert captured[index]["optimizer_config"]["new_module_lr_multiplier"] == config.CLIS_LR_MULTIPLIER[name], f"Không truyền hệ số: {name} phải dùng m* của bảng"
    assert config.CLIS_PROTOCOL["optimizer_config"]["new_module_lr_multiplier"] == 5.0, "run_child không được sửa CLIS_PROTOCOL dùng chung"
    with override_config(**config.CLIS_CONFIG):
        training_config = build_training_config(config.CLIS_PROTOCOL["optimizer_config"], config.CLIS_PROTOCOL["scheduler_config"], "macro_auroc")
    assert training_config["optimizer"]["fused"] is bool(config.CLIS_CONFIG["ADAM_FUSED"]), training_config["optimizer"]


def test_version_parsing():
    # Đọc phiên bản thư viện: bỏ hậu tố build (+cu128) và nhãn (rc, post, dev)
    from utilities.environment import parse_version
    cases = {"2.9.1+cu128": (2, 9, 1), "2.3.2.post1": (2, 3, 2), "1.0.30": (1, 0, 30), "2.10.0rc1": (2, 10, 0), "0.24.1": (0, 24, 1), "dev": ()}
    for text, expected in cases.items():
        assert parse_version(text) == expected, f"{text} --> {parse_version(text)}, cần {expected}"


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
