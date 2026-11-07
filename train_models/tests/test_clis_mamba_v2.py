# Kiểm thử CLIS-Mamba v2 (models/clis_mamba_v2.py): bảng tuyến, convnext_t trùng baseline, khối chèn đồng nhất ở bước 0,
# nhóm LR, số tham số giữa các biến thể, khởi tạo từ checkpoint, đổi độ phân giải. Không cần GPU (selective scan bản tham chiếu).
#   WSL2: conda activate mamba-t29 && CUDA_VISIBLE_DEVICES= python tests/test_clis_mamba_v2.py

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
import models.clis_mamba_v2 as v2
import models.convnext_tiny as convnext
from training.optimizer import parameter_groups

SIZE = 64                                                                   # Ảnh 64 px: lưới stage 3 là 4 x 4 token, đủ cho CPU


def make(mixer = "m_clis", seed = 0, **kwargs):
    torch.manual_seed(seed)
    return v2.CLISMambaV2(3, mixer = mixer, pretrained = False, depth = 2, d_state = 4, grad_checkpoint = False, **kwargs)


def baseline(seed = 0):
    torch.manual_seed(seed)
    return convnext.ConvNeXtTinyBaseline(3, pretrained = False)


# Mọi bảng tuyến là hoán vị; 4 tuyến đầu là quét chéo chung; M-clis / M-rand thêm đúng 2 tuyến ghép cặp
def test_routes():
    for h, w in ((4, 4), (6, 8)):
        tables = {name: v2.ROUTES[name](h, w) for name in v2.ROUTES}
        assert [len(t) for t in tables.values()] == [4, 6, 6]
        for table in tables.values():
            for order in table:
                assert torch.equal(order.sort().values, torch.arange(h * w)), "Mỗi tuyến phải là một hoán vị"
            for route, reference in zip(table[:4], tables["m_ras"]):
                assert torch.equal(route, reference), "4 tuyến đầu phải là quét chéo hàng ±, cột ±"
        assert torch.equal(tables["m_clis"][4], clis.perm_clis(h, w)) and torch.equal(tables["m_clis"][5], clis.perm_clis(h, w).flip(0))
        assert torch.equal(tables["m_rand"][4], clis.perm_random_pair(h, w))
    assert make("m_clis").inserted.blocks[0].mixer.k == 6 and make("m_ras").inserted.blocks[0].mixer.k == 4


# convnext_t: cùng tên tham số, cùng khởi tạo đầu phân loại theo seed, cùng đầu ra với models/convnext_tiny.py
def test_convnext_t_equals_baseline():
    new, old = make("convnext_t").eval(), baseline().eval()
    assert list(new.state_dict()) == list(old.state_dict()), "Tên tham số phải khớp baseline (để nạp checkpoint E1)"
    assert torch.equal(new.classifier.weight, old.classifier.weight), "Cùng seed phải cho cùng đầu phân loại"
    old.load_state_dict(new.state_dict())
    x = torch.randn(2, 3, SIZE, SIZE)
    with torch.no_grad():
        assert torch.equal(new(x), old(x))
    assert new.PRETRAINED_MODULES == ("cnn",) and new.layer_scales() == [] and new.uncounted_flops(512, 512) == 0


# Khối chèn ở bước 0: γ = 0 cho đúng đầu ra baseline; γ = 1e-6 (mặc định) lệch rất nhỏ
def test_identity_at_init():
    x = torch.randn(2, 3, SIZE, SIZE)
    for mixer in ("m_clis", "gatedconv", "attn"):
        old = baseline().eval()
        exact = make(mixer, layer_scale = 0.0).eval()
        assert torch.equal(exact.classifier.weight, old.classifier.weight)
        old.load_state_dict({key: value for key, value in exact.state_dict().items() if not key.startswith("inserted.")})
        default = make(mixer).eval()
        default.load_state_dict(exact.state_dict() | {key: value for key, value in default.state_dict().items() if "gamma" in key})
        with torch.no_grad():
            reference = old(x)
            assert torch.equal(exact(x), reference), f"{mixer}: γ = 0 phải trùng baseline"
            assert (default(x) - reference).abs().max() < 1e-3, f"{mixer}: γ = 1e-6 phải gần baseline"
    assert all(abs(a - 1e-6) < 1e-12 and abs(b - 1e-6) < 1e-12 for a, b in make().layer_scales())


# Nhóm LR: backbone và khối chèn dùng LR backbone; chỉ đầu phân loại nhân hệ số; tham số SSM và γ không decay
def test_parameter_groups():
    model = make("m_clis")
    assert model.PRETRAINED_MODULES == ("cnn", "inserted")
    groups = parameter_groups(model, torch.nn.BCEWithLogitsLoss(), 1e-4, 40.0, no_weight_decay_1d = True)
    rate = {id(p): group["lr"] for group in groups for p in group["params"]}
    decay = {id(p): group.get("weight_decay") for group in groups for p in group["params"]}
    for name, parameter in model.named_parameters():
        expected = 4e-3 if name.startswith("classifier.") else 1e-4
        assert abs(rate[id(parameter)] - expected) < 1e-12, f"{name}: LR {rate[id(parameter)]}, cần {expected}"
        if name.endswith(("A_log", "gamma1", "gamma2")) or name.startswith("inserted.pe."):
            assert decay[id(parameter)] == 0.0, f"{name} không được decay"
    assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))


# Số tham số: M-clis = M-rand tuyệt đối (điều kiện của H1); M-ras ít hơn đúng 2 tuyến mỗi khối; đối chứng không SSM khớp M-ras trong 2%
def test_parameter_counts():
    sizes = {mixer: clis.count_parameters(v2.CLISMambaV2(14, mixer = mixer, pretrained = False)) for mixer in v2.NAMES}
    assert sizes["m_clis"] == sizes["m_rand"], sizes
    two_routes = clis.count_parameters(v2.MultiRouteSS2D(v2.DIM, "m_clis")) - clis.count_parameters(v2.MultiRouteSS2D(v2.DIM, "m_ras"))
    assert sizes["m_clis"] - sizes["m_ras"] == 4 * two_routes, "Chênh lệch phải đúng bằng 2 tuyến x 4 khối"
    for mixer in ("gatedconv", "attn", "flipdiff"):
        inserted = sizes[mixer] - sizes["convnext_t"]
        assert abs(inserted / (sizes["m_ras"] - sizes["convnext_t"]) - 1) < 0.02, f"{mixer}: {inserted} so với {sizes['m_ras'] - sizes['convnext_t']}"
    print("    số tham số (triệu): " + ", ".join(f"{mixer} {value / 1e6:.2f}" for mixer, value in sizes.items()))


# Khởi tạo từ checkpoint: cùng biến thể nạp đúng; khác biến thể báo lỗi; checkpoint baseline E1 (model_kwargs rỗng) nạp vào convnext_t
def test_init_from():
    with tempfile.TemporaryDirectory() as directory:
        source = make("m_clis", seed = 1)
        path = Path(directory) / "best_model.pt"
        torch.save({"model_kwargs": {"mixer": "m_clis"}, "model_state_dict": source.state_dict()}, path)
        loaded = v2.CLISMambaV2(3, mixer = "m_clis", pretrained = False, depth = 2, d_state = 4, grad_checkpoint = False, init_from = str(path))
        assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), source.state_dict().values()))
        try:
            v2.CLISMambaV2(3, mixer = "m_rand", pretrained = False, depth = 2, d_state = 4, init_from = str(path))
        except ValueError:
            pass
        else:
            raise AssertionError("Phải từ chối checkpoint của biến thể khác")
        old = baseline(seed = 2)
        torch.save({"model_kwargs": {}, "model_state_dict": old.state_dict()}, path)
        loaded = v2.CLISMambaV2(3, mixer = "convnext_t", pretrained = False, init_from = str(path))
        assert torch.equal(loaded.classifier.weight, old.classifier.weight)


# Lan truyền xuôi / ngược, đổi độ phân giải (mã hóa vị trí nội suy), FLOPs của phần quét, tham số không decay
def test_forward_backward():
    model = make("m_clis")
    for size in (SIZE, 2 * SIZE):
        logits = model(torch.randn(2, 3, size, size))
        assert logits.shape == (2, 3)
    logits.sum().backward()
    gammas = [p.grad for name, p in model.named_parameters() if "gamma" in name]
    assert gammas and all(g is not None and g.abs().sum() > 0 for g in gammas), "γ phải nhận gradient"
    assert model.uncounted_flops(512, 512) == 2 * 9 * 32 * 32 * 6 * v2.DIM * 4
    assert "inserted.pe.row" in model.no_weight_decay() and "inserted.blocks.0.mixer.A_log" in model.no_weight_decay()


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
