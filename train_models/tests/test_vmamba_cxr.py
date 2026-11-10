# Kiểm thử VMamba-T CXR (models/vmamba_cxr.py): bảng tuyến khớp cross scan của VMamba, lõi quét bằng hoán vị gather trùng lõi gốc,
# vm_clis / vm_rand trùng vm_ras ở bước 0, số tham số, nhóm LR, khởi tạo từ checkpoint, gradient checkpointing, trọng số ImageNet.
# Không cần GPU (selective scan bản PyTorch của VMamba). WSL2: conda activate mamba-t29 && CUDA_VISIBLE_DEVICES= python tests/test_vmamba_cxr.py

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
import copy
import tempfile
import traceback

import torch

import models.clis_mamba as clis
import models.vmamba_cxr as vmx
from training.optimizer import parameter_groups

SIZE = 64                                                                   # Ảnh 64 px: lưới stage 3 / 4 là 4 x 4 / 2 x 2 token
vmx.vm.WITH_SELECTIVESCAN_MAMBA = False                                     # Lõi gốc của VMamba cũng dùng selective scan PyTorch trên CPU


def make(mixer = "vm_clis", seed = 0, **kwargs):
    torch.manual_seed(seed)
    return vmx.VMambaCXR(3, mixer = mixer, pretrained = False, **kwargs)


# Lõi gốc của VMamba (cross scan, 4 tuyến) cho các SS2D đã đổi sang hoán vị gather (chỉ dùng được khi chưa thêm tuyến)
def restore_vmamba_core(model):
    for stage in model.route_stages:
        for block in model.backbone.layers[stage].blocks:
            assert block.op.k_group == vmx.BASE_ROUTES
            block.op.forward_core = vmx.vm.partial(block.op.forward_corev2, force_fp32 = False, no_einsum = True)   # Đúng kiểu v05 của VMamba
    return model


# Bảng tuyến: hoán vị gather tái tạo đúng cross scan của VMamba (hàng xuôi, cột xuôi, hàng ngược, cột ngược); tuyến thêm là hoán vị
def test_routes_match_cross_scan():
    for h, w in ((4, 4), (6, 8)):
        x = torch.randn(2, 5, h, w)
        reference = vmx.vm.cross_scan_fn(x, in_channel_first = True, out_channel_first = True, scans = 0, force_torch = True)       # (b, 4, c, l)
        order = torch.stack(vmx.vmamba_routes(h, w))
        ours = clis.permute_routes(x.flatten(2)[:, None].expand(2, 4, 5, h * w), order)
        assert torch.equal(ours, reference), "Thứ tự tuyến phải khớp tham số tiền huấn luyện của VMamba"
        for name in vmx.NAMES:
            for route in vmx.EXTRA_ROUTES[name](h, w):
                assert torch.equal(route.sort().values, torch.arange(h * w))


# Lõi quét bằng hoán vị gather (vm_ras) cho cùng đầu ra với lõi cross scan gốc của VMamba
def test_route_core_matches_vmamba():
    x = torch.randn(2, 3, SIZE, SIZE)
    ours = make("vm_ras").eval()
    reference = restore_vmamba_core(copy.deepcopy(ours)).eval()
    with torch.no_grad():
        assert torch.allclose(ours(x), reference(x), atol = 1e-5), (ours(x) - reference(x)).abs().max()


# vm_clis / vm_rand ở bước 0 trùng vm_ras (tuyến thêm cho đầu ra 0); tuyến thêm vẫn nhận gradient ngay ở hàng C
def test_identity_at_init():
    x = torch.randn(2, 3, SIZE, SIZE)
    base = make("vm_ras").eval()
    for mixer in ("vm_clis", "vm_rand"):
        model = make(mixer).eval()
        assert torch.equal(model.classifier.weight, base.classifier.weight), "Cùng seed phải cho cùng đầu phân loại"
        assert all(c == 0 and d == 0 for _, c, d in model.engagement()) and len(model.engagement()) == 2
        with torch.no_grad():
            assert torch.allclose(model(x), base(x), atol = 1e-5), f"{mixer}: phải trùng vm_ras ở bước 0"
    model = make("vm_clis").train()
    model(x).sum().backward()
    op = model.backbone.layers[2].blocks[0].op
    grad = op.x_proj.weight.grad.view(op.k_group, op.dt_rank + 2 * op.d_state, op.d_inner)
    assert grad[vmx.BASE_ROUTES:, op.dt_rank + op.d_state:].abs().sum() > 0, "Hàng C của tuyến thêm phải có gradient ở bước đầu"
    assert make("vm_ras").engagement() == []


# route_init "copy": tuyến thêm là bản chép đủ của tuyến hàng xuôi / ngược (cả hàng C và D), hoạt động ngay ở bước 0;
# cùng số tham số với "zero"; giá trị khác bị từ chối
def test_copy_route_init():
    x = torch.randn(2, 3, SIZE, SIZE)
    base = make("vm_ras").eval()
    for mixer in ("vm_clis", "vm_rand"):
        model = make(mixer, route_init = "copy").eval()
        op = model.backbone.layers[2].blocks[0].op
        k, r, n, d = op.k_group, op.dt_rank, op.d_state, op.d_inner
        weight, Ds = op.x_proj.weight.view(k, r + 2 * n, d), op.Ds.view(k, d)
        assert torch.equal(weight[4], weight[0]) and torch.equal(weight[5], weight[2]), "Tuyến thêm phải chép đủ x_proj của tuyến hàng"
        assert torch.equal(Ds[4], Ds[0]) and torch.equal(Ds[5], Ds[2]), "Tuyến thêm phải chép D của tuyến hàng"
        assert all(c > 0 and d > 0 for _, c, d in model.engagement())
        # Khởi tạo ngẫu nhiên: D = 1 trội hơn C nhiều, phần D·x của mọi tuyến như nhau và out_norm chuẩn hóa lại, nên đầu ra chỉ đổi rất ít (cỡ 1e-5)
        with torch.no_grad():
            assert not torch.equal(model(x), base(x)), f"{mixer} copy: tuyến thêm phải đổi đầu ra ở bước 0"
        assert clis.count_parameters(model) == clis.count_parameters(make(mixer))
    try:
        make("vm_clis", route_init = "half")
    except ValueError:
        return
    raise AssertionError("Phải từ chối route_init không hợp lệ")


# Phần trộn token (đầu ra - D·x) của tuyến thêm trên đặc trưng `features`, theo dấu tuyến "none" / "signed" của cùng một SS2D
def extra_mixing(op, features, sign):
    op.route_sign, op.route_cache = sign, {}
    with torch.no_grad():
        routes = vmx.route_outputs(op, features)
    b, d, h, w = features.shape
    k = op.k_group
    mixing = routes - op.Ds.view(1, k, d, 1) * features.flatten(2)[:, None]
    return routes, mixing[:, vmx.BASE_ROUTES:]


# route_sign "signed": tuyến gốc không đổi, tuyến thêm quét hiệu hai nửa ảnh. Khởi tạo "zero" vẫn trùng vm_ras ở bước 0.
# Ngữ nghĩa: tỉ lệ ||phần trộn token|| trên đặc trưng đối xứng gương / trên đặc trưng không đối xứng:
#   - không dấu: không triệt tiêu (tỉ lệ > 0,9) ở cả CLIS lẫn cặp ngẫu nhiên (quét tổng)
#   - có dấu: CLIS triệt tiêu mạnh (hai bên của mỗi cặp giống nhau), mạnh hơn rõ so với cặp ngẫu nhiên
#     (cặp ngẫu nhiên cũng giảm một phần: dt khởi tạo nhỏ nên trạng thái gần như cộng dồn cả hàng, mà nửa phải là hoán vị của nửa trái)
#   Dấu đặt sai chỗ (ví dụ đổi dấu cả B, khi đó B·u triệt tiêu dấu) làm mất triệt tiêu ở CLIS
def test_signed_routes():
    x = torch.randn(2, 3, SIZE, SIZE)
    base = make("vm_ras").eval()
    for mixer in ("vm_clis", "vm_rand"):
        with torch.no_grad():
            assert torch.allclose(make(mixer, route_sign = "signed").eval()(x), base(x), atol = 1e-5), f"{mixer} signed + zero: phải trùng vm_ras ở bước 0"

    torch.manual_seed(3)
    half = torch.randn(2, 96 * 4, 4, 4)
    symmetric = torch.cat([half, half.flip(-1)], dim = -1)                    # Lưới 4 x 8 đối xứng gương
    asymmetric = torch.cat([half, torch.randn_like(half)], dim = -1)
    ratio = {}
    for mixer in ("vm_clis", "vm_rand"):
        op = make(mixer, route_init = "copy").eval().backbone.layers[2].blocks[0].op
        for sign in ("none", "signed"):
            routes_sym, mixing_sym = extra_mixing(op, symmetric, sign)
            ratio[mixer, sign] = (mixing_sym.norm() / extra_mixing(op, asymmetric, sign)[1].norm()).item()
            if sign == "none":
                routes_none = routes_sym
        assert torch.equal(routes_none[:, :vmx.BASE_ROUTES], routes_sym[:, :vmx.BASE_ROUTES]), "Tuyến gốc không được đổi theo dấu"
    print("    ||trộn|| đối xứng / không đối xứng: " + ", ".join(f"{mixer} {sign} {value:.3f}" for (mixer, sign), value in ratio.items()))
    assert ratio["vm_clis", "none"] > 0.9 and ratio["vm_rand", "none"] > 0.9, ratio
    assert ratio["vm_clis", "signed"] < 0.35 and ratio["vm_clis", "signed"] < 0.75 * ratio["vm_rand", "signed"], ratio
    try:
        make("vm_clis", route_sign = "minus")
    except ValueError:
        return
    raise AssertionError("Phải từ chối route_sign không hợp lệ")


# Số tham số: vm_clis = vm_rand; chênh với vm_ras đúng bằng tham số của 2 tuyến ở các khối stage 3, 4
def test_parameter_counts():
    sizes = {mixer: clis.count_parameters(vmx.VMambaCXR(14, mixer = mixer, pretrained = False)) for mixer in vmx.NAMES}
    assert sizes["vm_clis"] == sizes["vm_rand"], sizes
    expected = 0
    model = vmx.VMambaCXR(14, mixer = "vm_ras", pretrained = False)
    for stage in vmx.ROUTE_STAGES:
        for block in model.backbone.layers[stage].blocks:
            op = block.op
            expected += 2 * ((op.dt_rank + 2 * op.d_state) * op.d_inner + op.d_inner * op.dt_rank + op.d_inner + op.d_inner * op.d_state + op.d_inner)
    assert sizes["vm_clis"] - sizes["vm_ras"] == expected, (sizes, expected)
    print("    số tham số (triệu): " + ", ".join(f"{mixer} {value / 1e6:.3f}" for mixer, value in sizes.items()))


# Nhóm LR: backbone 1e-4 (gồm tuyến thêm), đầu phân loại x40; A_logs không decay
def test_parameter_groups():
    model = make("vm_clis")
    groups = parameter_groups(model, torch.nn.BCEWithLogitsLoss(), 1e-4, 40.0, no_weight_decay_1d = True)
    rate = {id(p): group["lr"] for group in groups for p in group["params"]}
    decay = {id(p): group.get("weight_decay") for group in groups for p in group["params"]}
    for name, parameter in model.named_parameters():
        assert abs(rate[id(parameter)] - (4e-3 if name.startswith("classifier.") else 1e-4)) < 1e-12, name
        if name.endswith("A_logs"):
            assert decay[id(parameter)] == 0.0, name
    assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))


# Khởi tạo từ checkpoint cùng biến thể (sau khi thêm tuyến); khác biến thể báo lỗi
def test_init_from():
    x = torch.randn(1, 3, SIZE, SIZE)
    source = make("vm_clis", seed = 1)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))                # Tuyến thêm khác 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "best_model.pt"
        torch.save({"model_kwargs": {"mixer": "vm_clis"}, "model_state_dict": source.state_dict()}, path)
        loaded = make("vm_clis", seed = 2, init_from = str(path)).eval()
        with torch.no_grad():
            assert torch.equal(loaded(x), source.eval()(x))
        try:
            make("vm_rand", init_from = str(path))
        except ValueError:
            return
    raise AssertionError("Phải từ chối checkpoint khác biến thể")


# Gradient checkpointing: cùng đầu ra và gradient; đổi độ phân giải; FLOPs phần quét tăng theo số tuyến
def test_grad_checkpoint_and_sizes():
    x = torch.randn(2, 3, SIZE, SIZE)
    plain, checked = make("vm_clis"), make("vm_clis", grad_checkpoint = True)
    plain(x).sum().backward(), checked(x).sum().backward()
    for (name, a), (_, b) in zip(plain.named_parameters(), checked.named_parameters()):
        assert a.grad is None and b.grad is None or torch.allclose(a.grad, b.grad, atol = 1e-6), name
    assert plain(torch.randn(1, 3, 2 * SIZE, 2 * SIZE)).shape == (1, 3)
    assert make("vm_clis").uncounted_flops(512, 512) > make("vm_ras").uncounted_flops(512, 512)


# Trọng số ImageNet (khi đã có tệp): đúng SHA-256, nạp khớp tuyệt đối; vm_clis tiền huấn luyện trùng vm_ras tiền huấn luyện ở bước 0
def test_pretrained_checkpoint():
    if not vmx.CHECKPOINT_PATH.is_file():
        print("    bỏ qua: chưa có", vmx.CHECKPOINT_PATH)
        return
    x = torch.randn(1, 3, SIZE, SIZE)
    torch.manual_seed(0)
    base = vmx.VMambaCXR(3, mixer = "vm_ras").eval()
    torch.manual_seed(0)
    model = vmx.VMambaCXR(3, mixer = "vm_clis").eval()
    with torch.no_grad():
        assert torch.allclose(model(x), base(x), atol = 1e-5)


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
