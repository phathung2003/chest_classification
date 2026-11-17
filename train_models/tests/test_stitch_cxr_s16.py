# Kiểm thử ConvNeXt-VMamba CXR bước 16 (models/stitch_cxr_s16.py): cùng tham số với ccmm, stage 4 chạy trên lưới bước 16, conv bỏ stride / conv giãn
# trùng phép tính tiền huấn luyện trên các lưới con, đường đi, gradient, ghim mã lớp cha, khởi tạo từ checkpoint ccmm, gradient checkpointing, FLOPs.
# Không cần GPU (selective scan bản PyTorch của VMamba). WSL2: conda activate mamba-t29 && CUDA_VISIBLE_DEVICES= python tests/test_stitch_cxr_s16.py
# Có GPU thì chạy thêm test_gpu_*: một bước huấn luyện BF16 ở chế độ tất định (launcher chạy bước này trước khi huấn luyện)

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
import models.stitch_cxr_s16 as s16
import models.vmamba_cxr as vmx

SIZE = 64                                                                   # Ảnh 64 px: lưới stage 3 là 4 x 4; stage 4 là 4 x 4 ở bước 16 (ccmm: 2 x 2)
vmx.vm.WITH_SELECTIVESCAN_MAMBA = False                                     # Selective scan bản PyTorch của VMamba trên CPU


def make(layout = "ccmm", seed = 0, **kwargs):
    torch.manual_seed(seed)
    return s16.StitchS16CXR(3, layout = layout, pretrained = False, **kwargs)


def make_base(layout = "ccmm", seed = 0, **kwargs):
    torch.manual_seed(seed)
    return stx.StitchCXR(3, layout = layout, pretrained = False, **kwargs)


# Cùng tên, cùng kích thước và cùng số tham số với ccmm: nạp được checkpoint ccmm (strict)
def test_same_parameters_as_ccmm():
    base, model = make_base(seed = 1), make(seed = 2)
    assert {name: value.shape for name, value in model.state_dict().items()} == {name: value.shape for name, value in base.state_dict().items()}
    assert clis.count_parameters(model) == clis.count_parameters(base)
    model.load_state_dict(base.state_dict(), strict = True)
    assert s16.model_name("ccmm", 512) == "ConvNeXt-VMamba CXR ccmm s16 512px"


# Conv giảm mẫu bỏ stride: vị trí chẵn của đầu ra trùng đầu ra stride 2 của ccmm (cùng trọng số)
def test_downsample_keeps_pretrained_samples():
    base, model = make_base(seed = 1).eval(), make(seed = 1).eval()
    model.load_state_dict(base.state_dict(), strict = True)
    y = torch.randn(2, 384, 8, 8)
    with torch.no_grad():
        dense, strided = model.mamba.layers[2].downsample(y), base.mamba.layers[2].downsample(y)
    assert dense.shape == (2, 768, 8, 8) and strided.shape == (2, 768, 4, 4)
    assert torch.allclose(dense[:, :, ::2, ::2], strided, atol = 1e-5)


# Conv depthwise giãn 2: trên mỗi lưới con (chẵn / lẻ) trùng conv gốc trên bản đồ thưa tương ứng
def test_dilated_convolution_on_sublattices():
    base, model = make_base(seed = 1).eval(), make(seed = 1).eval()
    model.load_state_dict(base.state_dict(), strict = True)
    z = torch.randn(2, 768, 8, 8)
    for index in range(len(model.mamba.layers[3].blocks)):
        dilated, original = model.mamba.layers[3].blocks[index].op.conv2d, base.mamba.layers[3].blocks[index].op.conv2d
        assert dilated.dilation == (2, 2) and dilated.padding == (2, 2) and original.dilation == (1, 1)
        with torch.no_grad():
            out = dilated(z)
            assert out.shape == z.shape
            for row in (0, 1):
                for column in (0, 1):
                    assert torch.allclose(out[:, :, row::2, column::2], original(z[:, :, row::2, column::2]), atol = 1e-5), (index, row, column)


# Đường đi: ConvNeXt stage 1-2 --> lớp nối --> stage 3 của VMamba --> downsample stride 1 --> stage 4 trên lưới bước 16 --> chuẩn hóa + pooling --> đầu phân loại
def test_path_and_shapes():
    x = torch.randn(2, 3, SIZE, SIZE)
    model = make().eval()
    with torch.no_grad():
        y = model.stage_features(x, 2)
        assert y.shape == (2, 384, SIZE // 16, SIZE // 16)
        y = model.mamba.layers[3].blocks(model.mamba.layers[2].downsample(y))
        assert y.shape == (2, 768, SIZE // 16, SIZE // 16)
        assert torch.equal(model.stage_features(x, 3), y)
        out = model(x)
        assert torch.equal(out, model.classifier(model.mamba.classifier(y)))
    assert out.shape == (2, 3) and torch.isfinite(out).all()
    assert make_base().eval().stage_features(x, 3).shape == (2, 768, SIZE // 32, SIZE // 32)


# Mọi tham số nhận gradient; layout có stage 4 không thuộc VMamba bị từ chối; mã lớp cha đã ghim: đúng SHA-256 thì dựng được, sai thì báo lỗi
def test_gradients_and_checks():
    x = torch.randn(2, 3, SIZE, SIZE)
    model = make(stitch_init = "identity").train()
    model(x).sum().backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None or parameter.grad.abs().sum() == 0]
    assert not missing, f"tham số không nhận gradient: {missing[:5]}"
    for layout in ("mmcc", "cccc", "ccm"):
        try:
            make(layout)
        except ValueError:
            continue
        raise AssertionError(f"Phải từ chối layout {layout}")
    make(pinned_sha256 = {s16.STITCH_SOURCE.name: vmx.sha256_file(s16.STITCH_SOURCE), stx.VMAMBA_SOURCE.name: vmx.sha256_file(stx.VMAMBA_SOURCE)})
    for pinned in ({s16.STITCH_SOURCE.name: "0" * 64}, {stx.VMAMBA_SOURCE.name: "0" * 64}):
        try:
            make(pinned_sha256 = pinned)
        except RuntimeError:
            continue
        raise AssertionError(f"Phải từ chối mã khác bản đã ghim: {list(pinned)}")


# Khởi tạo từ checkpoint ccmm cho cùng đầu ra với mô hình bước 16 mang cùng trọng số. Gradient checkpointing không đổi đầu ra và gradient. FLOPs phần quét
def test_init_from_and_checkpointing():
    x = torch.randn(2, 3, SIZE, SIZE)
    source = make(seed = 1).eval()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "best_model.pt"
        torch.save({"model_state_dict": source.state_dict(), "model_kwargs": {"layout": "ccmm"}}, path)
        loaded = make(seed = 2, init_from = str(path)).eval()
        with torch.no_grad():
            assert torch.equal(loaded(x), source(x))
    plain, saving = make(seed = 3).train(), make(seed = 3, grad_checkpoint = True).train()
    out_plain, out_saving = plain(x), saving(x)
    assert torch.allclose(out_plain, out_saving, atol = 1e-6)
    out_plain.sum().backward()
    out_saving.sum().backward()
    for (name, a), (_, b) in zip(plain.named_parameters(), saving.named_parameters()):
        assert torch.allclose(a.grad, b.grad, atol = 1e-5), name
    assert make().uncounted_flops(512, 512) == sum(9 * tokens * 4 * channels * 1 * blocks for tokens, channels, blocks in ((1024, 384, 8), (1024, 768, 2)))


# GPU: một bước huấn luyện BF16 ở chế độ tất định (kernel selective scan của mamba-ssm, conv giãn của cuDNN); bỏ qua khi không có CUDA
def test_gpu_training_step():
    if not torch.cuda.is_available():
        print("    (bỏ qua: không có CUDA)")
        return
    vmx.vm.WITH_SELECTIVESCAN_MAMBA = True
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
        vmx.vm.WITH_SELECTIVESCAN_MAMBA = False


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
