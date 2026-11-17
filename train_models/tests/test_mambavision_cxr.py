# Kiểm thử MambaVision-T CXR (models/mambavision_cxr.py): trùng forward của mã gốc khi cửa sổ phủ cả bản đồ đặc trưng (224 px với cửa sổ tiền huấn luyện,
# và độ phân giải khác), trọng số ImageNet khớp đủ tham số, số tham số, nhóm LR, khởi tạo từ checkpoint, gradient checkpointing, FLOPs phần quét.
# Không cần GPU (selective scan bản tham chiếu của mamba-ssm). WSL2: conda activate mamba-t29 && CUDA_VISIBLE_DEVICES= python tests/test_mambavision_cxr.py
# Có GPU thì chạy thêm test_gpu_*: kernel CUDA so với bản tham chiếu, một bước huấn luyện BF16 ở chế độ tất định (launcher chạy bước này trước khi huấn luyện)

# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TRITON_CACHE_AUTOTUNING"] = "1"

# Khai báo thư viện
import tempfile
import traceback

import torch

import models.clis_mamba as clis
import models.mambavision_cxr as mvx
from training.optimizer import parameter_groups

HEAD_1000 = mvx.FEATURES * 1000 + 1000                                      # Đầu ImageNet của mã gốc (đã bỏ)


def make(seed = 0, **kwargs):
    torch.manual_seed(seed)
    return mvx.MambaVisionCXR(3, pretrained = False, **kwargs)


# Ở 224 px cửa sổ tiền huấn luyện (14, 7) phủ cả bản đồ: features() trùng forward_features của mã gốc
def test_matches_upstream_at_224():
    x = torch.randn(2, 3, 224, 224)
    model = make().eval()
    assert [level.window_size for level in model.backbone.levels] == list(mvx.PRETRAIN_WINDOWS)
    with torch.no_grad():
        assert torch.allclose(model.features(x), model.backbone.forward_features(x), atol = 1e-5)


# Ở độ phân giải khác: trùng mã gốc khi đặt cửa sổ stage 3, 4 bằng cạnh bản đồ (cấu hình 512 px của tác giả là [8, 8, 32, 16])
def test_full_map_window_at_other_sizes():
    for size in (128, 192):
        x = torch.randn(1, 3, size, size)
        model = make(seed = size).eval()
        with torch.no_grad():
            ours = model.features(x)
            model.backbone.levels[2].window_size, model.backbone.levels[3].window_size = size // 16, size // 32
            assert torch.allclose(ours, model.backbone.forward_features(x), atol = 1e-5), size
    assert [size // 16 for size in (512,)] + [size // 32 for size in (512,)] == [32, 16]


# Cấu trúc: stage 1, 2 tích chập; stage 3, 4 nửa đầu Mamba, nửa sau attention. Số tham số: backbone + đầu 1000 lớp = 31,8 triệu như bài báo
def test_structure_and_parameter_count():
    model = make()
    levels = model.backbone.levels
    assert [level.transformer_block for level in levels] == [False, False, True, True]
    kinds = [[type(block.mixer).__name__ for block in level.blocks] for level in levels[2:]]
    assert kinds == [["MambaVisionMixer"] * 4 + ["Attention"] * 4, ["MambaVisionMixer"] * 2 + ["Attention"] * 2]
    backbone = clis.count_parameters(model.backbone)
    assert 31.7e6 < backbone + HEAD_1000 < 31.9e6, backbone + HEAD_1000
    assert clis.count_parameters(model) == backbone + mvx.FEATURES * 3 + 3
    out = model.eval()(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 3) and torch.isfinite(out).all()
    assert mvx.model_name(512) == "MambaVision-T CXR 512px"


# Trọng số ImageNet (nếu tệp đã tải): đúng SHA-256, khớp đủ mọi tham số và buffer của backbone, không còn đầu 1000 lớp
def test_imagenet_weights():
    if not mvx.CHECKPOINT_PATH.is_file():
        print("    (bỏ qua: chưa có tệp trọng số)")
        return
    state = mvx.imagenet_state()
    model = make()
    own = model.backbone.state_dict()
    assert set(state) == set(own) and all(state[name].shape == own[name].shape for name in own)
    loaded = mvx.MambaVisionCXR(3, pretrained = True)
    assert all(torch.equal(loaded.backbone.state_dict()[name], state[name]) for name in state)


# Nhóm LR: backbone ở nhóm backbone, đầu phân loại ở nhóm lớp mới; tham số 1 chiều và A_log / D của SSM không weight decay
def test_parameter_groups():
    model = make()
    groups = parameter_groups(model, torch.nn.BCEWithLogitsLoss(), 1e-4, 40.0, no_weight_decay_1d = True)
    by_id = {id(p): group for group in groups for p in group["params"]}
    assert len(by_id) == len(list(model.parameters()))
    for name, parameter in model.named_parameters():
        group = by_id[id(parameter)]
        assert abs(group["lr"] - (4e-3 if name.startswith("classifier.") else 1e-4)) < 1e-12, name
        exempt = parameter.ndim <= 1 or name.rsplit(".", 1)[-1] in ("A_log", "D")
        assert (group.get("weight_decay") == 0.0) == exempt, name
    exempt_names = model.no_weight_decay()
    assert sum(name.endswith(".A_log") for name in exempt_names) == 6 and sum(name.endswith(".D") for name in exempt_names) == 6


# Khởi tạo từ checkpoint cho cùng đầu ra. Gradient checkpointing không đổi đầu ra và gradient; mọi tham số nhận gradient. FLOPs phần quét
def test_init_from_and_checkpointing():
    x = torch.randn(2, 3, 64, 64)
    source = make(seed = 1).eval()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "best_model.pt"
        torch.save({"model_state_dict": source.state_dict(), "model_kwargs": {}}, path)
        loaded = make(seed = 2, init_from = str(path)).eval()
        with torch.no_grad():
            assert torch.equal(loaded(x), source(x))
    plain, saving = make(seed = 3).train(), make(seed = 3, grad_checkpoint = True).train()
    out_plain, out_saving = plain(x), saving(x)
    assert torch.allclose(out_plain, out_saving, atol = 1e-6)
    out_plain.sum().backward()
    out_saving.sum().backward()
    for (name, a), (_, b) in zip(plain.named_parameters(), saving.named_parameters()):
        assert a.grad is not None and a.grad.abs().sum() > 0, name
        assert torch.allclose(a.grad, b.grad, atol = 1e-5), name
    assert make().uncounted_flops(512, 512) == 9 * 1024 * 160 * 8 * 4 + 9 * 256 * 320 * 8 * 2


# GPU: kernel CUDA của mamba-ssm cho cùng đầu ra với bản tham chiếu dùng trên CPU; bỏ qua khi không có CUDA
def test_gpu_kernel_matches_reference():
    if not torch.cuda.is_available():
        print("    (bỏ qua: không có CUDA)")
        return
    torch.manual_seed(0)
    b, d, n, length = 2, 160, 8, 96
    u, delta, B, C = torch.randn(b, d, length), 0.1 * torch.randn(b, d, length), torch.randn(b, n, length), torch.randn(b, n, length)
    A, D, bias = -torch.rand(d, n) - 0.1, torch.randn(d), 0.1 * torch.randn(d)
    reference = mvx.selective_scan(u, delta, A, B, C, D, delta_bias = bias, delta_softplus = True)
    kernel = mvx.selective_scan(*(t.cuda() for t in (u, delta, A, B, C, D)), delta_bias = bias.cuda(), delta_softplus = True)
    assert torch.allclose(kernel.cpu(), reference, atol = 1e-4, rtol = 1e-4), (kernel.cpu() - reference).abs().max().item()


# GPU: một bước huấn luyện BF16 ở chế độ tất định như pipeline (kernel selective scan, attention, BatchNorm); bỏ qua khi không có CUDA
def test_gpu_training_step():
    if not torch.cuda.is_available():
        print("    (bỏ qua: không có CUDA)")
        return
    torch.use_deterministic_algorithms(True)
    try:
        model = make().cuda().train()
        x = torch.randn(2, 3, 128, 128, device = "cuda")
        with torch.autocast("cuda", dtype = torch.bfloat16):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).float(), torch.ones(2, 3, device = "cuda"))
        loss.backward()
        assert torch.isfinite(loss)
        bad = [name for name, parameter in model.named_parameters() if parameter.grad is None or not torch.isfinite(parameter.grad).all()]
        assert not bad, f"gradient thiếu hoặc không hữu hạn: {bad[:5]}"
    finally:
        torch.use_deterministic_algorithms(False)


if __name__ == "__main__":
    tests = [(name, value) for name, value in list(globals().items()) if name.startswith("test_")]
    failed = []
    for name, test in tests:
        try:
            test()
            print(f"[ĐẠT] {name}", flush = True)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            print(f"[LỖI] {name}", flush = True)
    print(f"\n{len(failed)} lỗi trên {len(tests)} kiểm thử" + (f": {failed}" if failed else ""))
    sys.exit(1 if failed else 0)
