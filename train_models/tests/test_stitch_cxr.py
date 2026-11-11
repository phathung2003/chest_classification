# Kiểm thử ConvNeXt-VMamba CXR (models/stitch_cxr.py) và lớp nối bình phương tối thiểu (analysis/fit_stitch.py): layout một họ trùng baseline,
# đường đi và số tham số của layout lai, khối không dùng bị bỏ, nhóm LR, nạp lớp nối từ tệp (kèm SHA-256), khởi tạo từ checkpoint, gradient checkpointing.
# Không cần GPU (selective scan bản PyTorch của VMamba). WSL2: conda activate mamba-t29 && CUDA_VISIBLE_DEVICES= python tests/test_stitch_cxr.py

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
import models.stitch_cxr as stx
import models.vmamba_cxr as vmx
from analysis.fit_stitch import r_squared, solve_stitch
from models.convnext_tiny import ConvNeXtTinyBaseline
from training.optimizer import parameter_groups

SIZE = 64                                                                   # Ảnh 64 px: lưới stage 1-4 là 16, 8, 4, 2 token mỗi cạnh
vmx.vm.WITH_SELECTIVESCAN_MAMBA = False                                     # Selective scan bản PyTorch của VMamba trên CPU

# Số tham số từng phần (đo trên timm convnext_tiny và VMamba-T [s1l8]): đơn vị stage 1-4 và chuẩn hóa cuối
CNN_UNITS = (4_896 + 237_888, 992_256, 11_112_960, 15_470_592)
MAMBA_UNITS = (43_200 + 201_216, 166_464 + 789_504, 664_704 + 12_509_184, 2_656_512 + 12_447_744)
FINAL_NORM = 1_536


def make(layout = "ccmm", seed = 0, **kwargs):
    torch.manual_seed(seed)
    return stx.StitchCXR(3, layout = layout, pretrained = False, **kwargs)


# Layout "cccc" là ConvNeXt-T baseline: cùng tên tham số (nạp được checkpoint E1) và cùng đầu ra
def test_cccc_is_convnext_baseline():
    x = torch.randn(2, 3, SIZE, SIZE)
    torch.manual_seed(0)
    baseline = ConvNeXtTinyBaseline(3, pretrained = False).eval()
    model = make("cccc").eval()
    assert set(model.state_dict()) == set(baseline.state_dict()), "cccc phải có đúng các tham số của baseline ConvNeXt-T"
    model.load_state_dict(baseline.state_dict(), strict = True)
    with torch.no_grad():
        assert torch.equal(model(x), baseline(x))


# Layout "mmmm" là vm_ras: cùng đầu ra khi chép trọng số backbone và đầu phân loại
def test_mmmm_is_vm_ras():
    x = torch.randn(2, 3, SIZE, SIZE)
    reference = vmx.VMambaCXR(3, mixer = "vm_ras", pretrained = False).eval()
    model = make("mmmm").eval()
    assert len(model.stitches) == 0
    model.mamba.load_state_dict(reference.backbone.state_dict(), strict = True)
    model.classifier.load_state_dict(reference.classifier.state_dict())
    with torch.no_grad():
        assert torch.equal(model(x), reference(x))


# Số tham số của layout lai = các đơn vị stage đang dùng + lớp nối + chuẩn hóa cuối + đầu phân loại (khối không dùng không còn tham số)
def test_parameter_counts():
    for layout in ("cccc", "mmmm", "ccmm", "mmcc", "cmmm", "cccm", "cmcm"):
        units = sum(CNN_UNITS[i] if family == "c" else MAMBA_UNITS[i] for i, family in enumerate(layout))
        stitches = sum(stx.CHANNELS[i] ** 2 + stx.CHANNELS[i] for i in stx.boundaries(layout))
        expected = units + stitches + FINAL_NORM + (stx.FEATURES * 3 + 3)
        assert clis.count_parameters(make(layout)) == expected, f"{layout}: {clis.count_parameters(make(layout))} khác {expected}"
    assert stx.boundaries("ccmm") == [1] and stx.boundaries("cmcm") == [0, 1, 2] and stx.boundaries("mmmm") == []


# Đường đi của "ccmm": ConvNeXt stage 1-2 --> lớp nối --> downsample và khối VSS stage 3-4 của VMamba --> chuẩn hóa + pooling của VMamba --> đầu phân loại
def test_ccmm_path():
    x = torch.randn(2, 3, SIZE, SIZE)
    model = make("ccmm").eval()
    with torch.no_grad():
        y = model.cnn.stages[1](model.cnn.stages[0](model.cnn.stem(x)))
        assert y.shape == (2, 192, SIZE // 8, SIZE // 8)
        assert torch.equal(model.stage_features(x, 1, stitched = False), y)
        y = model.stitches["1"](y)
        assert torch.equal(model.stage_features(x, 1), y)
        y = model.mamba.layers[2].blocks(model.mamba.layers[1].downsample(y))
        y = model.mamba.layers[3].blocks(model.mamba.layers[2].downsample(y))
        assert y.shape == (2, 768, SIZE // 32, SIZE // 32)
        assert torch.equal(model(x), model.classifier(model.mamba.classifier(y)))
    assert isinstance(model.cnn.stages[2], torch.nn.Identity) and isinstance(model.cnn.head, torch.nn.Identity)
    assert isinstance(model.mamba.patch_embed, torch.nn.Identity) and isinstance(model.mamba.layers[1].blocks, torch.nn.Identity)


# Lớp nối khởi tạo đơn vị không đổi đặc trưng; mọi phần trên đường đi nhận gradient
def test_identity_stitch_and_gradients():
    x = torch.randn(2, 3, SIZE, SIZE)
    for layout in ("ccmm", "mmcc"):
        model = make(layout, stitch_init = "identity").train()
        with torch.no_grad():
            assert torch.equal(model.stage_features(x, 1), model.stage_features(x, 1, stitched = False))
        model(x).sum().backward()
        missing = [name for name, parameter in model.named_parameters() if parameter.grad is None or parameter.grad.abs().sum() == 0]
        assert not missing, f"{layout}: tham số không nhận gradient: {missing[:5]}"
    for arguments in ({"layout": "ccm"}, {"layout": "ccxm"}, {"stitch_init": "random"}):
        try:
            make(**arguments)
        except ValueError:
            continue
        raise AssertionError(f"Phải từ chối {arguments}")


# Nhóm LR: hai backbone và lớp nối ở nhóm backbone, chỉ đầu phân loại ở nhóm lớp mới; A_logs / Ds / dt_projs_bias không weight decay
def test_parameter_groups():
    model = make("ccmm")
    groups = parameter_groups(model, torch.nn.BCEWithLogitsLoss(), 1e-4, 40.0, no_weight_decay_1d = True)
    by_id = {id(p): group for group in groups for p in group["params"]}
    assert len(by_id) == len(list(model.parameters()))
    for name, parameter in model.named_parameters():
        group = by_id[id(parameter)]
        assert abs(group["lr"] - (4e-3 if name.startswith("classifier.") else 1e-4)) < 1e-12, name
        exempt = parameter.ndim <= 1 or name.rsplit(".", 1)[-1] in ("A_logs", "Ds", "dt_projs_bias")
        assert (group.get("weight_decay") == 0.0) == exempt, name
    assert any(name.endswith("A_logs") for name in model.no_weight_decay())


# Nghiệm bình phương tối thiểu: khôi phục đúng phép biến đổi tuyến tính (R² gần 1), tốt hơn lớp nối đơn vị
def test_solve_stitch():
    torch.manual_seed(0)
    x = torch.randn(4000, 12)
    true_weight, true_bias = torch.randn(12, 12), torch.randn(12)
    y = x @ true_weight.T + true_bias + 0.01 * torch.randn(4000, 12)
    weight, bias = solve_stitch(x[:3000], y[:3000])
    assert torch.allclose(weight.float(), true_weight, atol = 0.02) and torch.allclose(bias.float(), true_bias, atol = 0.02)
    assert r_squared(x[3000:], y[3000:], weight, bias) > 0.999
    assert r_squared(x[3000:], y[3000:], torch.eye(12, dtype = torch.float64), torch.zeros(12, dtype = torch.float64)) < 0.5


# stitch_init "lstsq": nạp weight / bias từ tệp của đúng ranh giới; sai SHA-256 hoặc thiếu tệp thì báo lỗi
def test_load_stitch_file():
    x = torch.randn(1, 3, SIZE, SIZE)
    original = stx.STITCH_DIRECTORY
    with tempfile.TemporaryDirectory() as directory:
        stx.STITCH_DIRECTORY = Path(directory)
        try:
            model = make("ccmm").eval()
            torch.manual_seed(1)
            weight, bias = torch.randn(192, 192), torch.randn(192)
            path = stx.stitch_path("c", "m", 1)
            torch.save({"weight": weight, "bias": bias, "r2": 0.5}, path)
            stitch = torch.nn.Conv2d(192, 192, 1)
            model.load_stitch(stitch, 1, {path.name: vmx.sha256_file(path)})
            with torch.no_grad():
                features = model.stage_features(x, 1, stitched = False)
                expected = torch.einsum("oc,bchw->bohw", weight, features) + bias.view(1, -1, 1, 1)
                assert torch.allclose(stitch(features), expected, atol = 1e-4)
            for pinned in ({path.name: "0" * 64}, {"vmamba_cxr.py": "0" * 64}):
                try:
                    model.load_stitch(stitch, 1, pinned)
                except RuntimeError:
                    continue
                raise AssertionError("Phải từ chối tệp lớp nối sai hoặc thiếu SHA-256 đã ghim")
            model.load_stitch(stitch, 1, None)                                  # Không ghim: không kiểm tra
            path.unlink()
            try:
                model.load_stitch(stitch, 1, None)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("Phải báo thiếu tệp lớp nối")
        finally:
            stx.STITCH_DIRECTORY = original

    # Mã VMamba CXR đã ghim: đúng SHA-256 thì dựng được, sai thì báo lỗi
    make("ccmm", pinned_sha256 = {stx.VMAMBA_SOURCE.name: vmx.sha256_file(stx.VMAMBA_SOURCE)})
    try:
        make("ccmm", pinned_sha256 = {stx.VMAMBA_SOURCE.name: "0" * 64})
    except RuntimeError:
        return
    raise AssertionError("Phải từ chối models/vmamba_cxr.py khác bản đã ghim")


# Khởi tạo từ checkpoint cùng layout cho cùng đầu ra; khác layout bị từ chối. Gradient checkpointing không đổi đầu ra và gradient
def test_init_from_and_checkpointing():
    x = torch.randn(2, 3, SIZE, SIZE)
    source = make("ccmm", seed = 1).eval()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "best_model.pt"
        torch.save({"model_state_dict": source.state_dict(), "model_kwargs": {"layout": "ccmm"}}, path)
        loaded = make("ccmm", seed = 2, init_from = str(path)).eval()
        with torch.no_grad():
            assert torch.equal(loaded(x), source(x))
        try:
            make("mmcc", init_from = str(path))
        except ValueError:
            pass
        else:
            raise AssertionError("Phải từ chối checkpoint khác layout")
    plain, saving = make("ccmm", seed = 3).train(), make("ccmm", seed = 3, grad_checkpoint = True).train()
    out_plain, out_saving = plain(x), saving(x)
    assert torch.allclose(out_plain, out_saving, atol = 1e-6)
    out_plain.sum().backward()
    out_saving.sum().backward()
    for (name, a), (_, b) in zip(plain.named_parameters(), saving.named_parameters()):
        assert torch.allclose(a.grad, b.grad, atol = 1e-5), name
    assert make("ccmm").uncounted_flops(512, 512) == sum(9 * tokens * 4 * channels * 1 * blocks for tokens, channels, blocks in ((1024, 384, 8), (256, 768, 2)))


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
