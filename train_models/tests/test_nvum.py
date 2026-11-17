# Kiểm thử điều chuẩn NVUM (training/loss.py NVUMLoss): tắt thì hàm Loss và chữ ký như cũ, công thức khớp bản chép tay của mã công bố,
# bộ nhớ chỉ cập nhật đúng ảnh của batch, đường Validation là BCE, gradient khớp công thức của bài báo (dạng theo lớp),
# thứ tự chỉ số của EpochSampler khớp thứ tự loader trả về, một epoch huấn luyện cập nhật đúng một lần mỗi ảnh, lưu / nạp trạng thái. Không cần GPU.
#   python tests/test_nvum.py      (hoặc pytest)

# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"                         # cuBLAS tất định (DETERMINISTIC_ALGORITHMS)

# Khai báo thư viện
import copy
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import config
from training.loss import NVUMLoss, build_loss
from training.train import train_one_epoch
from utilities.data import EpochSampler, create_dataloader
from utilities.setting import apply_config
from utilities.tracking import config_snapshot

CLASSES = ["A", "B", "C"]
BCE = {"name": "bce", "pos_weight_alpha": 0.0}


# Ghi đè config trong một kiểm thử rồi trả lại giá trị cũ
class override_config:
    def __init__(self, **values):
        self.values, self.saved = values, {}

    def __enter__(self):
        self.saved = {name: copy.deepcopy(getattr(config, name)) for name in self.values}
        apply_config(self.values)

    def __exit__(self, *exc):
        apply_config(self.saved)


# Bảng nhãn giả: 40 ảnh, 3 lớp, mỗi lớp có cả mẫu dương và mẫu âm
def label_frame(size = 40, seed = 0):
    rng = np.random.default_rng(seed)
    labels = (rng.random((size, len(CLASSES))) < [0.5, 0.2, 0.1]).astype(int)
    labels[0], labels[1] = 1, 0
    return pd.DataFrame(labels, columns = CLASSES)


# Tập dữ liệu giả cùng giao diện với lớp Dataset của pipeline: ảnh là vector 4 chiều mang chỉ số ảnh ở phần tử đầu
class ToyDataset(Dataset):
    def __init__(self, dataframe, classes, transform = None):
        self.labels = torch.tensor(dataframe[list(classes)].to_numpy(), dtype = torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.full((4,), float(index))
        image[1:] = torch.sin(torch.arange(3) + index)
        return image, self.labels[index]


# Mô hình giả: bỏ phần tử mang chỉ số ảnh, tuyến tính trên 3 phần tử còn lại
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, len(CLASSES))

    def forward(self, images):
        return self.linear(images[:, 1:])


# ============================================================
# 1. Tắt NVUM: hàm Loss và ảnh chụp config như cũ
def test_off_keeps_plain_bce_and_signature():
    frame = label_frame()
    assert config.NVUM is None
    criterion, metadata = build_loss(BCE, frame, CLASSES, torch.device("cpu"))
    assert type(criterion) is nn.BCEWithLogitsLoss and "nvum" not in metadata
    assert "NVUM" not in config_snapshot()
    with override_config(NVUM = [1.0, 0.9]):
        assert config_snapshot()["NVUM"] == [1.0, 0.9]
        criterion, metadata = build_loss(BCE, frame, CLASSES, torch.device("cpu"))
        assert isinstance(criterion, NVUMLoss) and criterion.memory.shape == (len(frame), len(CLASSES))
        assert metadata["nvum"]["weight"] == 1.0 and metadata["nvum"]["beta"] == 0.9
        assert np.allclose(metadata["nvum"]["prior"], frame[CLASSES].mean().to_numpy())
        assert not [p for p in criterion.parameters() if p.requires_grad]          # Không thêm tham số học được vào optimizer


# 2. Cấu hình sai bị từ chối
def test_invalid_settings_rejected():
    frame = label_frame()
    for value in ([0.0, 0.9], [1.0, 1.0], [1.0], "x"):
        with override_config(NVUM = value):
            try:
                build_loss(BCE, frame, CLASSES, torch.device("cpu"))
            except ValueError:
                continue
            raise AssertionError(f"NVUM = {value!r} phải bị từ chối")
    with override_config(NVUM = [1.0, 0.9]):
        try:
            build_loss({"name": "asl"}, frame, CLASSES, torch.device("cpu"))
        except ValueError:
            return
        raise AssertionError("NVUM với hàm Loss khác 'bce' phải bị từ chối")


# 3. Công thức: khớp bản chép tay của mã công bố (bộ nhớ cập nhật trước, điều chuẩn theo từng lớp, trung bình theo ảnh x lớp)
def test_formula_matches_reference():
    torch.manual_seed(0)
    frame = label_frame()
    prior = torch.tensor(frame[CLASSES].mean().to_numpy(), dtype = torch.float32)
    with override_config(NVUM = [1.0, 0.9]):
        criterion, _ = build_loss(BCE, frame, CLASSES, torch.device("cpu"))
    indices = torch.tensor([3, 7, 11, 0])
    targets = torch.tensor(frame[CLASSES].to_numpy()[indices], dtype = torch.float32)
    history = torch.zeros(len(frame), len(CLASSES))
    for _ in range(3):                                                      # 3 lần duyệt cùng batch: bộ nhớ tích lũy
        logits = torch.randn(4, len(CLASSES)) * 3
        history[indices] = 0.9 * history[indices] + 0.1 * (logits - (prior + 1e-12).log())
        y_pred = torch.clamp(torch.sigmoid(logits), 1e-4, 1.0 - 1e-4)
        expected = torch.mean(nn.BCEWithLogitsLoss(reduction = "none")(logits, targets) + 1 * (1 - torch.sigmoid(history[indices] * y_pred)).log())
        loss = criterion(logits, targets, indices)
        assert torch.allclose(loss, expected, atol = 1e-6), f"{loss.item()} != {expected.item()}"
    assert torch.allclose(criterion.memory, history, atol = 1e-6)


# 4. Bộ nhớ: chỉ các ảnh của batch được cập nhật; gọi không kèm chỉ số là BCE và không đổi bộ nhớ
def test_memory_rows_and_evaluation_path():
    frame = label_frame()
    with override_config(NVUM = [1.0, 0.9]):
        criterion, _ = build_loss(BCE, frame, CLASSES, torch.device("cpu"))
    logits, targets = torch.randn(5, len(CLASSES)), torch.zeros(5, len(CLASSES))
    plain = F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.equal(criterion(logits, targets), plain) and not criterion.memory.any()
    indices = torch.tensor([4, 9, 2, 30, 17])
    criterion(logits, targets, indices)
    touched = criterion.memory.abs().sum(dim = 1) > 0
    assert sorted(touched.nonzero().flatten().tolist()) == sorted(indices.tolist())
    before = criterion.memory.clone()
    assert torch.equal(criterion(logits, targets), plain) and torch.equal(criterion.memory, before)
    try:
        criterion(logits, targets, indices[:3])
    except ValueError:
        return
    raise AssertionError("Số chỉ số ảnh khác số ảnh của batch phải bị từ chối")


# 5. Gradient của phần điều chuẩn theo logit: g = -sigmoid(t p) p (1 - p) t / (số ảnh x số lớp); không có gradient qua bộ nhớ
def test_gradient_matches_paper():
    torch.manual_seed(1)
    frame = label_frame()
    with override_config(NVUM = [1.0, 0.0]):                                 # beta = 0: bộ nhớ = z - log pi của lần gọi này
        criterion, _ = build_loss(BCE, frame, CLASSES, torch.device("cpu"))
    logits = (torch.randn(6, len(CLASSES)) * 2).requires_grad_()
    targets = torch.zeros(6, len(CLASSES))
    indices = torch.arange(6)
    total = criterion(logits, targets, indices)
    (total - F.binary_cross_entropy_with_logits(logits, targets)).backward()
    memory = criterion.memory[indices]
    p = torch.sigmoid(logits.detach())
    expected = -torch.sigmoid(memory * p) * p * (1 - p) * memory / logits.numel()
    assert torch.allclose(logits.grad, expected, atol = 1e-6)
    assert not criterion.memory.requires_grad


# 6. EpochSampler.indices() là đúng thứ tự loader trả về; __iter__ giữ cách sinh cũ (hoán vị rồi seed, cùng bộ sinh)
def test_sampler_indices_match_loader_order():
    frame = label_frame(size = 37)
    with override_config(NUMBER_OF_WORKERS = 0):
        _, loader = create_dataloader(ToyDataset, CLASSES, torch.device("cpu"), frame, None, True, batch_size = 8, seed = 123)
        for epoch in (1, 2):
            loader.sampler.set_epoch(epoch)
            seen = torch.cat([images[:, 0] for images, _ in loader]).long().tolist()
            assert seen == loader.sampler.indices().tolist(), f"epoch {epoch}"
    sampler = EpochSampler(37, 123)
    sampler.set_epoch(2)
    rng = np.random.default_rng([123, 2])
    order = rng.permutation(37)
    seeds = rng.integers(0, 2 ** 63 - 1, size = 37)
    assert list(sampler) == [(int(seeds[index]), int(index)) for index in order]


# 7. Một epoch qua train_one_epoch: mỗi ảnh được cập nhật đúng một lần với logit của chính nó; lưu / nạp trạng thái giữ bộ nhớ
def test_train_one_epoch_updates_every_sample_once():
    frame = label_frame(size = 37)
    prior = torch.tensor(frame[CLASSES].mean().to_numpy(), dtype = torch.float32)
    training_config = {"grad_accum_steps": 2, "grad_clip_norm": 1.0}
    with override_config(NUMBER_OF_WORKERS = 0, USE_AMP = False, NVUM = [1.0, 0.9]):
        torch.manual_seed(0)
        device = torch.device("cpu")
        model = ToyModel()
        criterion, _ = build_loss(BCE, frame, CLASSES, device)
        _, loader = create_dataloader(ToyDataset, CLASSES, device, frame, None, True, batch_size = 8, seed = 5)
        loader.sampler.set_epoch(1)
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.0)            # LR 0: logit của mỗi ảnh không đổi trong epoch --> kiểm tra được bộ nhớ
        scaler = torch.amp.GradScaler("cuda", enabled = False)
        stats = train_one_epoch(1, device, model, loader, criterion, optimizer, scaler, training_config)
        dataset = ToyDataset(frame, CLASSES)
        with torch.no_grad():
            logits = model(torch.stack([dataset[index][0] for index in range(len(dataset))]))
        assert torch.allclose(criterion.memory, 0.1 * (logits - prior.log()), atol = 1e-6)
        assert np.isfinite(stats["train_loss"])

        # Epoch 2: cập nhật lần hai theo trung bình trượt
        loader.sampler.set_epoch(2)
        train_one_epoch(2, device, model, loader, criterion, optimizer, scaler, training_config)
        assert torch.allclose(criterion.memory, 0.19 * (logits - prior.log()), atol = 1e-6)

        # Lưu / nạp (như criterion_state_dict của last_checkpoint.pt, strict)
        restored, _ = build_loss(BCE, frame, CLASSES, device)
        restored.load_state_dict(copy.deepcopy(criterion.state_dict()), strict = True)
        assert torch.equal(restored.memory, criterion.memory)


# 8. Hàm Loss thường (không NVUM) đi qua train_one_epoch như trước: không cần chỉ số ảnh
def test_plain_loss_path_unchanged():
    frame = label_frame(size = 37)
    training_config = {"grad_accum_steps": 1, "grad_clip_norm": None}
    with override_config(NUMBER_OF_WORKERS = 0, USE_AMP = False):
        torch.manual_seed(0)
        device = torch.device("cpu")
        model = ToyModel()
        criterion, _ = build_loss(BCE, frame, CLASSES, device)
        _, loader = create_dataloader(ToyDataset, CLASSES, device, frame, None, True, batch_size = 8, seed = 5)
        loader.sampler.set_epoch(1)
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.0)
        stats = train_one_epoch(1, device, model, loader, criterion, optimizer, torch.amp.GradScaler("cuda", enabled = False), training_config)
        dataset = ToyDataset(frame, CLASSES)
        with torch.no_grad():
            logits = model(torch.stack([dataset[index][0] for index in range(len(dataset))]))
        per_image = F.binary_cross_entropy_with_logits(logits, dataset.labels, reduction = "none").mean(dim = 1)
        assert abs(stats["train_loss"] - per_image.mean().item()) < 1e-6


# 9. GPU ở chế độ tất định (DETERMINISTIC_ALGORITHMS của pipeline): cập nhật bộ nhớ và backward không bị từ chối; khớp CPU. Không có GPU thì bỏ qua
def test_cuda_deterministic_mode():
    if not torch.cuda.is_available():
        print("  (không có GPU: bỏ qua)")
        return
    frame = label_frame()
    torch.manual_seed(2)
    logits = torch.randn(8, len(CLASSES))
    targets = torch.tensor(frame[CLASSES].to_numpy()[:8], dtype = torch.float32)
    indices = torch.tensor([5, 1, 9, 30, 2, 8, 0, 39])
    results = {}
    torch.use_deterministic_algorithms(True)
    try:
        for device in (torch.device("cpu"), torch.device("cuda")):
            with override_config(NVUM = [1.0, 0.9]):
                criterion, _ = build_loss(BCE, frame, CLASSES, device)
            z = logits.detach().clone().to(device).requires_grad_()
            for _ in range(2):
                loss = criterion(z, targets.to(device), indices.to(device))
            loss.backward()
            results[device.type] = (loss.item(), z.grad.cpu(), criterion.memory.cpu())
    finally:
        torch.use_deterministic_algorithms(False)
    assert abs(results["cpu"][0] - results["cuda"][0]) < 1e-5
    assert torch.allclose(results["cpu"][1], results["cuda"][1], atol = 1e-6) and torch.allclose(results["cpu"][2], results["cuda"][2], atol = 1e-5)


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
