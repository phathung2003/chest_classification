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
    assert new.PRETRAINED_MODULES == ("cnn",) and new.engagement() == [] and new.uncounted_flops(512, 512) == 0


# Khối chèn ở bước 0 (mặc định: khởi tạo 0 phép chiếu ra): đầu ra trùng tuyệt đối baseline với mọi bộ trộn; cách cũ LayerScale chỉ gần trùng
def test_identity_at_init():
    x = torch.randn(2, 3, SIZE, SIZE)
    for mixer in ("m_clis", "m_ras", "gatedconv", "attn", "flipdiff"):
        old = baseline().eval()
        new = make(mixer).eval()
        assert torch.equal(new.classifier.weight, old.classifier.weight)
        assert all(a == 0 and b == 0 for a, b in new.engagement()), f"{mixer}: phép chiếu ra phải bằng 0 lúc khởi tạo"
        old.load_state_dict({key: value for key, value in new.state_dict().items() if not key.startswith("inserted.")})
        with torch.no_grad():
            assert torch.equal(new(x), old(x)), f"{mixer}: phải trùng baseline ở bước 0"
    legacy = make("m_clis", residual_init = "layerscale").eval()
    assert any(name.endswith("gamma1") for name, _ in legacy.named_parameters()) and not any(name.endswith("gamma1") for name, _ in make("m_clis").named_parameters())


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
        if name.endswith("A_log") or name.startswith("inserted.pe."):
            assert decay[id(parameter)] == 0.0, f"{name} không được decay"
    assert decay[id(model.inserted.blocks[0].mixer.out_proj.weight)] != 0.0, "Phép chiếu ra (2 chiều) vẫn có weight decay (mặc định của optimizer)"
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
        torch.save({"model_kwargs": {"mixer": "m_clis", "residual_init": "zero"}, "model_state_dict": source.state_dict()}, path)
        loaded = v2.CLISMambaV2(3, mixer = "m_clis", pretrained = False, depth = 2, d_state = 4, grad_checkpoint = False, init_from = str(path))
        assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), source.state_dict().values()))
        for kwargs in ({"mixer": "m_rand"}, {"mixer": "m_clis", "residual_init": "layerscale"}):
            try:
                v2.CLISMambaV2(3, pretrained = False, depth = 2, d_state = 4, init_from = str(path), **kwargs)
            except ValueError:
                continue
            raise AssertionError(f"Phải từ chối checkpoint khác biến thể hoặc khác cách khởi tạo: {kwargs}")
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
    projections = [layer.weight.grad for block in model.inserted.blocks for layer in (v2.output_projection(block.mixer), block.mlp[2])]
    assert all(g is not None and g.abs().sum() > 0 for g in projections), "Phép chiếu ra (khởi tạo 0) phải nhận gradient ngay ở bước đầu"
    assert model.uncounted_flops(512, 512) == 2 * 9 * 32 * 32 * 6 * v2.DIM * 4
    assert "inserted.pe.row" in model.no_weight_decay() and "inserted.blocks.0.mixer.A_log" in model.no_weight_decay()


# Một bước Adam (không weight decay) trên loss cố định; trả về bản sao giá trị hiệu dụng trước và sau bước
def adam_step(model, x, lr = 1e-3):
    before = {name: value.detach().clone() for name, value in v2.effective_state(model.state_dict(), model.block_lr_scale).items()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr = lr)
    model(x).pow(2).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    after = v2.effective_state(model.state_dict(), model.block_lr_scale)
    return before, after


# block_lr_scale: cùng giá trị hiệu dụng (đầu ra trùng mô hình s = 1), tên tham số thô ".original", tham số SSM vẫn không decay,
# và với Adam bước cập nhật hiệu dụng của khối chèn gấp s lần, của backbone / đầu phân loại không đổi
def test_block_lr_scale():
    x = torch.randn(2, 3, SIZE, SIZE)
    plain, scaled = make("m_clis"), make("m_clis", block_lr_scale = 10.0)
    assert all(name.endswith(".original") for name, _ in scaled.named_parameters() if name.startswith("inserted."))
    assert not any("parametrizations" in name for name, _ in scaled.named_parameters() if not name.startswith("inserted."))
    assert any(name.endswith("A_log.original") for name in scaled.no_weight_decay()) and len(scaled.no_weight_decay()) == len(plain.no_weight_decay())
    groups = parameter_groups(scaled, torch.nn.BCEWithLogitsLoss(), 1e-4, 40.0, no_weight_decay_1d = True)
    assert sum(len(group["params"]) for group in groups) == len(list(scaled.parameters()))
    with torch.no_grad():
        assert torch.allclose(plain.eval()(x), scaled.eval()(x), atol = 1e-6)

    # Bước Adam đầu tiên: |Δ| mỗi phần tử ~ lr, nên tỉ lệ trung bình |Δ hiệu dụng| khối chèn ~ 10, backbone ~ 1
    plain.train(), scaled.train()
    for _ in range(2):                                                       # Bước 1 làm phép chiếu ra (khởi tạo 0) khác 0 để mọi tham số khối chèn có gradient
        (b1, a1), (b2, a2) = adam_step(plain, x), adam_step(scaled, x)
    for key, expected in (("inserted.blocks.0.mlp.0.weight", 10.0), ("inserted.blocks.0.mixer.x_proj", 10.0), ("cnn.stages.3.blocks.0.mlp.fc1.weight", 1.0), ("classifier.weight", 1.0)):
        ratio = (a2[key] - b2[key]).abs().mean() / (a1[key] - b1[key]).abs().mean()
        assert abs(ratio / expected - 1) < 0.15, f"{key}: tỉ lệ bước cập nhật {ratio:.3f}, cần ~{expected}"


# Khởi tạo từ checkpoint có block_lr_scale khác: giá trị hiệu dụng và đầu ra giữ nguyên; convnext_t từ chối block_lr_scale
def test_init_from_across_scales():
    x = torch.randn(2, 3, SIZE, SIZE)
    source = make("m_clis", seed = 3, block_lr_scale = 10.0)
    adam_step(source.train(), x)                                             # Phép chiếu ra khác 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "best_model.pt"
        torch.save({"model_kwargs": {"mixer": "m_clis", "residual_init": "zero", "block_lr_scale": 10.0}, "model_state_dict": source.state_dict()}, path)
        with torch.no_grad():
            reference = source.eval()(x)
            for scale in (1.0, 3.0, 10.0):
                loaded = make("m_clis", seed = 4, block_lr_scale = scale, init_from = str(path)).eval()
                assert torch.allclose(loaded(x), reference, atol = 1e-5), f"Nạp vào block_lr_scale = {scale} phải giữ đầu ra"
                assert any(a > 0 for a, _ in loaded.engagement())
    try:
        make("convnext_t", block_lr_scale = 2.0)
    except ValueError:
        return
    raise AssertionError("convnext_t phải từ chối block_lr_scale")


# freeze_backbone: ConvNeXt-T không nhận gradient; khối chèn và đầu phân loại vẫn học; convnext_t đóng băng chỉ còn đầu phân loại
def test_freeze_backbone():
    x = torch.randn(2, 3, SIZE, SIZE)
    for mixer, trainable in (("m_clis", ("inserted.", "classifier.")), ("convnext_t", ("classifier.",))):
        model = make(mixer, freeze_backbone = True).train()
        model(x).sum().backward()
        for name, parameter in model.named_parameters():
            if name.startswith("cnn."):
                assert not parameter.requires_grad and parameter.grad is None, f"{name} phải đóng băng"
            else:
                assert name.startswith(trainable) and parameter.requires_grad, name
        assert model.classifier.weight.grad is not None
        if mixer == "m_clis":
            assert model.inserted.blocks[0].mlp[2].weight.grad.abs().sum() > 0
        groups = parameter_groups(model, torch.nn.BCEWithLogitsLoss(), 1e-4, 40.0, no_weight_decay_1d = True)
        assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))


# Runner: hậu tố thư mục và khóa model_kwargs của cách huấn luyện khối chèn; lượt thường không thêm khóa (chữ ký giữ dạng cũ)
def test_runner_variants():
    import models.run_clis_v2 as runner
    assert runner.variant_suffix() == "" and runner.training_kwargs() == {}
    assert runner.variant_suffix(10.0, freeze = True) == " lp blr10" and runner.variant_suffix(1.0, lpft = 10.0) == " lpft"
    assert runner.variant_suffix(2.5, lpft = 10.0) == " lpft blr2.5"
    assert runner.training_kwargs(10.0, True) == {"block_lr_scale": 10.0, "freeze_backbone": True}
    assert runner.same_training({}, 1.0, False) and not runner.same_training({"freeze_backbone": True}, 1.0, False)
    assert runner.same_training({"block_lr_scale": 10.0, "freeze_backbone": True}, 10, True)
    assert runner.model_directory("m_clis", 512, 0.3, " lp blr10").name == "clismamba_v2_m_clis_512px_lp_blr10_tune30"


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
