# Kiểm thử cổng G0 của CLIS-Mamba (models/clis_mamba.py): hoán vị, kernel selective scan, độc lập giữa các tuyến,
# khớp số tham số, CAM, nội suy mã hóa vị trí, gradient checkpointing, chế độ tất định.
#   Windows  : python tests/test_clis_mamba.py         (các kiểm thử cần kernel CUDA được bỏ qua)
#   WSL2     : conda activate mamba-t29 && python tests/test_clis_mamba.py      (env mamba-main: torch 2.4 của các lượt tinh chỉnh đầu, cũng phải đạt)
# Chạy được cả bằng pytest. Cổng G0 đạt khi KHÔNG có kiểm thử nào lỗi và không có kiểm thử kernel nào bị bỏ qua (chạy trong WSL2).

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
import tempfile
import time
import traceback
import warnings

import torch
import torch.nn.functional as F

import models.clis_mamba as clis

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
except ImportError:
    selective_scan_ref = None

HAS_CUDA = torch.cuda.is_available()
HAS_KERNEL = clis.selective_scan_fn is not None and HAS_CUDA
GRIDS = [(16, 16), (32, 32), (48, 48), (64, 64), (8, 6)]                   # Lưới 256 / 512 / 768 / 1024 px và một lưới chữ nhật


class Skipped(Exception):
    pass


def skip(reason):
    try:
        import pytest
        pytest.skip(reason)
    except ImportError:
        raise Skipped(reason)


def max_relative_error(actual, expected):
    return ((actual - expected).abs().max() / (expected.abs().max() + 1e-6)).item()


# Đầu vào ngẫu nhiên cho selective scan với k nhóm (tuyến), mỗi nhóm d kênh. A, D, delta_bias luôn FP32 như kernel yêu cầu
def scan_inputs(b = 2, k = 4, d = 32, n = 16, length = 64, device = "cpu", dtype = torch.float32, seed = 0):
    generator = torch.Generator().manual_seed(seed)
    def normal(*shape, scale = 1.0):
        return (torch.randn(*shape, generator = generator) * scale).to(device = device, dtype = dtype)
    u, delta = normal(b, k * d, length), normal(b, k * d, length, scale = 0.5)
    A = (-(torch.rand(k * d, n, generator = generator) + 0.5)).to(device)
    B, C = normal(b, k, n, length), normal(b, k, n, length)
    D, delta_bias = torch.randn(k * d, generator = generator).to(device), (torch.randn(k * d, generator = generator) * 0.5).to(device)
    return [u, delta, A, B, C, D, delta_bias]


def run_scan(backend, u, delta, A, B, C, D, delta_bias):
    return backend(u, delta, A, B, C, D, delta_bias = delta_bias, delta_softplus = True)


# ============================================================
# 1. Hoán vị: song ánh, gather theo hoán vị rồi gather theo hoán vị ngược trả lại đúng thứ tự raster
def test_permutations_are_bijective():
    for h, w in GRIDS:
        for name, routes in clis.ROUTES.items():
            for k, order in enumerate(routes(h, w)):
                assert order.dtype == torch.long and order.shape == (h * w,), f"{name} tuyến {k}: sai kiểu / kích thước"
                assert torch.equal(order.sort().values, torch.arange(h * w)), f"{name} tuyến {k} ({h}x{w}) không phải hoán vị"
                x = torch.randn(h * w)
                assert torch.equal(x[order][torch.argsort(order)], x), f"{name} tuyến {k}: hoán vị ngược sai"


# CLIS: cặp gương (h, w) / (h, W-1-w) luôn cách nhau đúng 1 bước; raster đặt cặp này cách nhau W-1-2w bước
def test_clis_puts_mirror_pairs_adjacent():
    for h, w in GRIDS:
        order = clis.perm_clis(h, w)
        step = torch.empty_like(order)
        step[order] = torch.arange(h * w)
        grid = torch.arange(h * w).view(h, w)
        left, mirror = grid[:, : w // 2], grid[:, w // 2:].flip(1)
        assert (step[left] - step[mirror]).abs().eq(1).all(), f"CLIS {h}x{w}: có cặp gương không liền nhau"
        assert torch.equal((mirror - left)[0], w - 1 - 2 * torch.arange(w // 2)), "Khoảng cách raster của cặp gương phải là W-1-2w"


# Đối chứng m_rand: giữ cấu trúc xen kẽ trái / phải và thứ tự bên trái của CLIS, chỉ phá cặp gương
def test_random_pair_control():
    h, w = 32, 32
    order = clis.perm_random_pair(h, w)
    assert torch.equal(order, clis.perm_random_pair(h, w)), "Hoán vị ngẫu nhiên phải cố định theo seed"
    rows, columns = (order // w).view(h, w), (order % w).view(h, w)
    assert torch.equal(rows, torch.arange(h)[:, None].expand(h, w)), "Phải đi hết từng hàng, từ trên xuống"
    assert (columns[:, 0::2] < w // 2).all() and (columns[:, 1::2] >= w // 2).all(), "Phải xen kẽ trái / phải"
    assert torch.equal(columns[:, 0::2], torch.arange(w // 2).expand(h, -1)), "Bên trái phải giữ thứ tự như CLIS"
    mirror_share = (columns[:, 1::2] == w - 1 - columns[:, 0::2]).float().mean().item()
    assert mirror_share < 0.15, f"Tỉ lệ cặp trùng cặp gương quá cao: {mirror_share:.3f} (kỳ vọng 2 / w = {2 / w:.3f})"


def test_routes_are_distinct():
    for name, routes in clis.ROUTES.items():
        assert torch.unique(torch.stack(routes(32, 32)), dim = 0).size(0) == 4, f"{name}: có hai tuyến trùng nhau"


# RoutePermute (backward = hoán vị ngược, không cộng dồn) cho cùng gradient với autograd của phép lấy chỉ số thông thường
def test_route_permute_gradient():
    torch.manual_seed(0)
    h, w, d = 4, 6, 3
    order = torch.stack(clis.ROUTES["m_clis"](h, w))
    inverse = torch.argsort(order, dim = 1)
    x = torch.randn(2, d, h * w, dtype = torch.float64, requires_grad = True)
    weight = torch.randn(2, 4, d, h * w, dtype = torch.float64)
    spread = clis.RoutePermute.apply(x[:, None].expand(2, 4, d, h * w), order, inverse)
    expected = torch.stack([x[:, :, order[k]] for k in range(4)], dim = 1)
    assert torch.equal(spread, expected), "Sai chuỗi của từng tuyến"
    grad_custom, = torch.autograd.grad((spread * weight).sum(), x)
    grad_reference, = torch.autograd.grad((expected * weight).sum(), x)
    assert torch.allclose(grad_custom, grad_reference), "Gradient của RoutePermute khác gradient của phép lấy chỉ số"
    y = torch.randn(2, 4, d, h * w, dtype = torch.float64, requires_grad = True)
    assert torch.autograd.gradcheck(lambda t: clis.RoutePermute.apply(t, inverse, order), (y,)), "gradcheck RoutePermute thất bại"


# ============================================================
# 2. Selective scan: bản tham chiếu của mô hình khớp selective_scan_ref của mamba-ssm; kernel CUDA khớp bản tham chiếu
def test_reference_matches_mamba_ssm_reference():
    if selective_scan_ref is None:
        skip("chưa cài mamba-ssm")
    inputs = scan_inputs()
    error = (run_scan(clis.selective_scan_reference, *inputs) - run_scan(selective_scan_ref, *inputs)).abs().max().item()
    assert error < 1e-5, f"Sai số {error:.2e}"


def test_kernel_matches_reference_fp32():
    if not HAS_KERNEL:
        skip("không có kernel selective_scan_cuda / CUDA")
    reference = selective_scan_ref or clis.selective_scan_reference
    inputs = [x.requires_grad_(True) for x in scan_inputs(b = 2, k = 4, d = 64, n = 16, length = 256, device = "cuda")]
    y_kernel, y_reference = run_scan(clis.selective_scan_fn, *inputs), run_scan(reference, *inputs)
    error = (y_kernel - y_reference).abs().max().item()
    assert error <= 1e-3, f"Đầu ra lệch {error:.2e} > 1e-3"
    grads_kernel = torch.autograd.grad(y_kernel.square().sum(), inputs)
    grads_reference = torch.autograd.grad(y_reference.square().sum(), inputs)
    for name, a, b in zip(["u", "delta", "A", "B", "C", "D", "delta_bias"], grads_kernel, grads_reference):
        relative = max_relative_error(a, b)
        assert relative <= 1e-2, f"Gradient {name} lệch tương đối {relative:.2e} > 1e-2"
    print(f"    fp32: đầu ra lệch tối đa {error:.2e}")


def test_kernel_matches_reference_bf16():
    if not HAS_KERNEL:
        skip("không có kernel selective_scan_cuda / CUDA")
    reference = selective_scan_ref or clis.selective_scan_reference
    inputs = scan_inputs(b = 2, k = 4, d = 64, n = 16, length = 256, device = "cuda")
    for i in (0, 1, 3, 4):                                                  # u, delta, B, C ở BF16 như khi chạy autocast
        inputs[i] = inputs[i].bfloat16()
    relative = max_relative_error(run_scan(clis.selective_scan_fn, *inputs).float(), run_scan(reference, *inputs).float())
    assert relative <= 5e-2, f"BF16: đầu ra lệch tương đối {relative:.2e} > 5e-2"


# Kernel hiểu B / C 4 chiều (b, nhóm, n, l) là "mỗi tuyến một nhóm": đổi đầu vào của nhóm k chỉ đổi đầu ra của nhóm k.
# Kiểm thử này sai thì mọi so sánh giữa các bảng tuyến đều vô nghĩa
def test_routes_are_independent_groups():
    backends = [("tham chiếu", clis.selective_scan_reference, "cpu")] + ([("kernel", clis.selective_scan_fn, "cuda")] if HAS_KERNEL else [])
    k, d, length = 4, 16, 64
    for label, backend, device in backends:
        inputs = scan_inputs(b = 1, k = k, d = d, n = 8, length = length, device = device)
        base = run_scan(backend, *inputs).view(k, d, length)
        for group in range(k):
            u, delta, A, B, C, D, delta_bias = [x.clone() for x in inputs]
            u[:, group * d:(group + 1) * d] += 1.0
            delta[:, group * d:(group + 1) * d] += 0.3
            B[:, group] += 1.0
            C[:, group] -= 1.0
            changed = (run_scan(backend, u, delta, A, B, C, D, delta_bias).view(k, d, length) - base).abs().amax(dim = (1, 2))
            others = torch.cat([changed[:group], changed[group + 1:]])
            assert changed[group] > 1e-3, f"{label}: đổi nhóm {group} nhưng đầu ra nhóm đó không đổi"
            assert others.max() == 0, f"{label}: đổi nhóm {group} làm đổi nhóm khác (lệch {others.max().item():.2e})"


# Đầu ra tuyến k tại vị trí q chỉ phụ thuộc token đứng trước q trong thứ tự của tuyến k (bắt lỗi nhầm hoán vị / hoán vị ngược)
def test_scan_routes_follow_route_order():
    torch.manual_seed(0)
    h, w, d = 6, 8, 16
    for routes in clis.ROUTES:
        mixer = clis.MultiRouteSS2D(d, routes, d_state = 4).double()
        x = torch.randn(1, d, h * w, dtype = torch.float64)
        _, inverse = mixer.orders(h, w, x.device)
        for position in (0, 13, h * w - 1):
            perturbed = x.clone()
            perturbed[:, :, position] += 1.0
            with torch.no_grad():
                changed = (mixer.scan_routes(perturbed, h, w) - mixer.scan_routes(x, h, w)).abs().amax(dim = 2)[0] > 0   # (k, l)
            for route in range(4):
                expected = inverse[route] >= inverse[route][position]
                assert torch.equal(changed[route], expected), f"{routes} tuyến {route}: phụ thuộc sai thứ tự quét (token {position})"


def test_ss2d_kernel_equals_reference():
    if not HAS_KERNEL:
        skip("không có kernel selective_scan_cuda / CUDA")
    torch.manual_seed(0)
    kernel = clis.selective_scan_fn
    for routes in clis.ROUTES:
        mixer = clis.MultiRouteSS2D(64, routes, d_state = 16).cuda()
        x = torch.randn(2, 16, 16, 64, device = "cuda")
        with torch.no_grad():
            y_kernel = mixer(x)
            clis.selective_scan_fn = None
            try:
                y_reference = mixer(x)
            finally:
                clis.selective_scan_fn = kernel
        relative = max_relative_error(y_kernel, y_reference)
        assert relative <= 1e-3, f"{routes}: SS2D dùng kernel lệch bản tham chiếu {relative:.2e}"


# ============================================================
# 3. Số tham số: 4 bảng tuyến bằng nhau tuyệt đối; gatedconv / attn / flipdiff lệch <= 2% (cả bộ trộn và toàn mô hình)
def test_parameter_matching():
    counts = {}
    for name, kwargs in clis.VARIANTS.items():
        with torch.device("meta"):
            model = clis.CLISMamba(14, pretrained = False, **kwargs)
        counts[name] = (clis.count_parameters(model), clis.count_parameters(model.blocks[0].mixer))
    total, mixer = counts["m_clis"]
    assert mixer == 597_504, f"SS2D (d = 384, n = 16, dt_rank = 24, 4 tuyến) phải có 597.504 tham số, nhận {mixer:,}"
    assert counts["attn"][1] == 591_360, f"Attention phải có 4d^2 + 4d = 591.360 tham số, nhận {counts['attn'][1]:,}"
    for name in ("m_ras", "m_rand", "m_clis4"):
        assert counts[name] == counts["m_clis"], f"{name} phải có đúng số tham số của m_clis"
    for name, (model_total, mixer_total) in counts.items():
        print(f"    {name:10s} toàn mô hình {model_total:>11,} ({(model_total - total) / total:+.3%}) | bộ trộn {mixer_total:>9,} ({(mixer_total - mixer) / mixer:+.3%})")
        assert abs(model_total - total) / total <= 0.02, f"{name}: toàn mô hình lệch quá 2%"
        assert abs(mixer_total - mixer) / mixer <= 0.02, f"{name}: bộ trộn lệch quá 2%"


def test_no_weight_decay_names():
    with torch.device("meta"):
        model = clis.CLISMamba(14, mixer = "m_clis", depth = 2, pretrained = False)
    expected = {"pe.row", "pe.column"} | {f"blocks.{i}.mixer.{name}" for i in range(2) for name in ("A_log", "D", "dt_bias")}
    assert model.no_weight_decay() == expected, f"Sai tập tham số không weight decay: {sorted(model.no_weight_decay() ^ expected)}"


# ============================================================
# 4. Mô hình đầy đủ: CAM chính xác, nội suy mã hóa vị trí, chạy ở mọi độ phân giải
def test_cam_is_exact():
    torch.manual_seed(0)
    model = clis.CLISMamba(14, mixer = "gatedconv", depth = 1, pretrained = False).eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        logits, cam = model(x), model.cam(x)
    assert cam.shape == (2, 14, 8, 8)
    assert torch.allclose(cam.mean(dim = (2, 3)) + model.head.bias, logits, atol = 1e-5), "logits phải bằng trung bình CAM + bias"


def test_init_from_checkpoint():
    torch.manual_seed(0)
    source = clis.CLISMamba(14, mixer = "gatedconv", depth = 1, pretrained = False)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "best_model.pt"
        torch.save({"model_kwargs": {"mixer": "gatedconv"}, "model_state_dict": source.state_dict()}, path)
        model = clis.CLISMamba(14, mixer = "gatedconv", depth = 1, init_from = str(path))
        assert all(torch.equal(a, b) for a, b in zip(source.state_dict().values(), model.state_dict().values())), "Không nạp đúng trọng số"
        try:
            clis.CLISMamba(14, mixer = "attn", depth = 1, init_from = str(path))
        except ValueError:
            pass
        else:
            raise AssertionError("Nạp checkpoint của bộ trộn khác phải báo lỗi")


def test_interpolation_matrix_matches_interpolate():
    for n_out, n_in in [(32, 32), (48, 32), (64, 32), (16, 32), (24, 32), (37, 32)]:
        x = torch.randn(1, 5, n_in, dtype = torch.float64)
        expected = F.interpolate(x, size = n_out, mode = "linear", align_corners = False)
        actual = (clis.interpolation_matrix(n_out, n_in).double() @ x[0].T).T[None]
        assert torch.allclose(actual, expected, atol = 1e-6), f"Nội suy {n_in} --> {n_out} lệch F.interpolate"


def test_forward_all_resolutions():
    torch.manual_seed(0)
    cases = [("gatedconv", "cpu", 256), ("m_clis", "cpu", 128)]                 # m_clis trên CPU: bản tham chiếu
    if HAS_KERNEL:
        cases += [(mixer, "cuda", size) for mixer in ("m_clis", "attn") for size in (512, 768, 1024)]
    for mixer, device, size in cases:
        model = clis.CLISMamba(14, mixer = mixer, depth = 1, pretrained = False).to(device).eval()
        with torch.no_grad(), torch.autocast(device_type = device, dtype = torch.bfloat16, enabled = device == "cuda"):
            logits = model(torch.randn(1, 3, size, size, device = device))
        assert logits.shape == (1, 14) and torch.isfinite(logits).all(), f"{mixer} @ {size} px: đầu ra sai"
        del model


# ============================================================
# 5. Huấn luyện: checkpointing không đổi loss / gradient (BF16), mọi tham số nhận gradient, chạy được ở chế độ tất định
def loss_and_gradients(model, x, y, dtype = torch.bfloat16):
    model.zero_grad(set_to_none = True)
    with torch.autocast(device_type = x.device.type, dtype = dtype, enabled = x.device.type == "cuda"):
        logits = model(x)
    loss = F.binary_cross_entropy_with_logits(logits.float(), y)
    loss.backward()
    return loss.item(), {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}


def test_block_checkpoint_bf16():
    if not HAS_CUDA:
        skip("không có CUDA")
    torch.manual_seed(0)
    model = clis.CLISMamba(14, mixer = "m_clis" if HAS_KERNEL else "gatedconv", depth = 2, pretrained = False).cuda().train()
    x, y = torch.randn(2, 3, 256, 256, device = "cuda"), (torch.rand(2, 14, device = "cuda") > 0.8).float()
    model.grad_checkpoint = False
    loss_plain, grads_plain = loss_and_gradients(model, x, y)
    model.grad_checkpoint = True
    loss_checkpoint, grads_checkpoint = loss_and_gradients(model, x, y)
    missing = [name for name, p in model.named_parameters() if name not in grads_checkpoint]
    assert not missing, f"Tham số không nhận gradient: {missing[:5]}"
    assert abs(loss_plain - loss_checkpoint) / abs(loss_plain) < 1e-2, "Loss đổi khi bật checkpointing"
    worst = max(((grads_checkpoint[n] - g).norm() / (g.norm() + 1e-12)).item() for n, g in grads_plain.items())
    assert worst < 1e-2, f"Gradient đổi khi bật checkpointing (lệch tương đối {worst:.2e})"


def test_backbone_checkpoint_keeps_gradients():
    torch.manual_seed(0)
    model = clis.CLISMamba(14, mixer = "gatedconv", depth = 1, pretrained = False).train()
    x, y = torch.randn(2, 3, 64, 64), (torch.rand(2, 14) > 0.8).float()
    _, plain = loss_and_gradients(model, x, y)
    model.backbone.set_grad_checkpointing(True)
    _, checkpointed = loss_and_gradients(model, x, y)
    assert plain.keys() == checkpointed.keys(), "Checkpointing backbone làm mất gradient của một số tham số"
    worst = max(((checkpointed[n] - g).norm() / (g.norm() + 1e-12)).item() for n, g in plain.items())
    assert worst < 1e-4, f"Gradient backbone đổi khi bật checkpointing (lệch tương đối {worst:.2e})"


def test_deterministic_mode():
    if not HAS_CUDA:
        skip("không có CUDA")
    deterministic, cudnn_deterministic = torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    try:
        for name, kwargs in clis.VARIANTS.items():
            if kwargs["mixer"] in clis.ROUTES and not HAS_KERNEL:
                continue
            model = clis.CLISMamba(14, depth = 1, pretrained = False, **kwargs).cuda().train()
            loss_and_gradients(model, torch.randn(2, 3, 256, 256, device = "cuda"), torch.zeros(2, 14, device = "cuda"))
            del model
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.deterministic = cudnn_deterministic


# ============================================================
if __name__ == "__main__":
    print(f"torch {torch.__version__} | CUDA {HAS_CUDA} | kernel selective_scan_cuda {HAS_KERNEL} | mamba_ssm.selective_scan_ref {selective_scan_ref is not None}")
    failed, skipped = [], []
    for name, test in [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]:
        started = time.perf_counter()
        try:
            test()
            status = "ĐẠT"
        except Skipped as reason:
            status = f"BỎ QUA: {reason}"
            skipped.append(name)
        except Exception:
            traceback.print_exc()
            status = "LỖI"
            failed.append(name)
        print(f"[{status}] {name} ({time.perf_counter() - started:.1f}s)", flush = True)
    print(f"\n{len(failed)} lỗi, {len(skipped)} bỏ qua" + (f": lỗi {failed}" if failed else ""))
    sys.exit(1 if failed else 0)
