# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import argparse
import math
import warnings

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import config
from training.experiment import run_experiment
from utilities.dataset import NIHChestXrayDataset
from utilities.file import load_torch

# Kernel CUDA selective scan của mamba-ssm, chỉ có trên Linux / WSL2 (env conda mamba-main).
# Không có kernel thì dùng selective_scan_reference bên dưới: cùng công thức, PyTorch thuần, bộ nhớ O(B·D·N·L),
# chỉ để kiểm thử trên CPU / Windows, không dùng để huấn luyện.
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

# ============================================================
# CLIS-Mamba: ConvNeXt-T (stage 1-3, stride 16, 384 kênh) + khối MetaFormer với bộ trộn token SS2D 4 tuyến
# --> LayerNorm theo token --> trung bình toàn cục --> Linear.
#
# CLIS (contralateral interleaved scan): trong mỗi hàng, token (h, w) và token gương (h, W-1-w) qua đường giữa đứng
# liền nhau trong chuỗi quét (ảnh đã căn đường giữa cơ thể về giữa ảnh). Quét raster đặt cặp này cách nhau W-1-2w bước.
# CLIS chỉ là một hoán vị thứ tự token: không thêm tham số, không thêm FLOPs.
#
# Các biến thể chỉ khác bảng hoán vị (m_ras, m_clis, m_rand, m_clis4: số tham số bằng nhau tuyệt đối) hoặc bộ trộn
# (gatedconv, attn, flipdiff: khớp số tham số trong 2%). Backbone, mã hóa vị trí, MLP và đầu phân loại giữ nguyên.
# LayerNorm đặt trước pooling nên CAM chính xác: logit = trung bình CAM + bias.
#
# Toàn bộ mô hình nằm trong một tệp để chữ ký thí nghiệm (hash tệp mô hình) phản ánh đúng mọi thay đổi.

VARIANTS = {
    "m_ras"     : {"mixer": "m_ras"},           # Raster hàng + cột, xuôi + ngược (CrossScan kiểu VMamba): đối chứng của H1
    "m_clis"    : {"mixer": "m_clis"},          # Đề xuất: raster xuôi + ngược, CLIS xuôi + ngược
    "m_rand"    : {"mixer": "m_rand"},          # Như m_clis nhưng cột trái j ghép với một cột phải ngẫu nhiên cố định (đối chứng: cặp gương có quan trọng không)
    "m_clis4"   : {"mixer": "m_clis4"},         # Cả 4 tuyến là CLIS: hàng trên xuống, dưới lên, xuôi + ngược
    "gatedconv" : {"mixer": "gatedconv"},       # Gated CNN kiểu MambaOut, không SSM (đối chứng: có cần SSM không)
    "attn"      : {"mixer": "attn"},            # Self-attention toàn cục (SDPA), số tham số tự khớp
    "flipdiff"  : {"mixer": "flipdiff"},        # Gated CNN + đặc trưng |F - lật ngang(F)| (đối chứng: so sánh đối bên tường minh)
}

DIM = 384                                       # Số kênh ConvNeXt-T sau stage 3 (stride 16)


# ============================================================
# Tên mô hình theo biến thể và độ phân giải (mặc định IMAGE_SIZE; dùng làm tên thư mục kết quả)
def model_name(variant, size = None):
    return f"CLIS-Mamba v1 {variant} {size or config.IMAGE_SIZE}px"


# ============================================================
# Hoán vị token cho lưới h x w. Mỗi hàm trả về LongTensor (h * w,): phần tử thứ t là chỉ số raster của token ở bước t
def perm_raster(h, w):
    return torch.arange(h * w)


# Theo cột, từ trên xuống, cột trái sang phải
def perm_column(h, w):
    t = torch.arange(h * w)
    return (t % h) * w + t // h


# CLIS: trong mỗi hàng đi 0, w-1, 1, w-2, ..., w/2-1, w/2 (cặp gương liền nhau, từ ngoài vào trong)
def perm_clis(h, w):
    if w % 2:
        raise ValueError(f"CLIS cần số cột chẵn, nhận w = {w}")
    t = torch.arange(h * w)
    row, j, side = t // w, (t % w) // 2, t % 2
    return row * w + torch.where(side == 0, j, w - 1 - j)


# Đối chứng của CLIS: vẫn xen kẽ trái / phải, nhưng cột trái j ghép với cột phải w/2 + sigma_h(j),
# sigma_h là hoán vị ngẫu nhiên cố định của từng hàng. CLIS là trường hợp sigma(j) = w/2 - 1 - j
def perm_random_pair(h, w, seed = 1234):
    if w % 2:
        raise ValueError(f"Ghép cặp cần số cột chẵn, nhận w = {w}")
    generator = torch.Generator().manual_seed(seed)
    sigma = torch.stack([torch.randperm(w // 2, generator = generator) for _ in range(h)])
    t = torch.arange(h * w)
    row, j, side = t // w, (t % w) // 2, t % 2
    return row * w + torch.where(side == 0, j, w // 2 + sigma[row, j])


# CLIS trong từng hàng nhưng đi các hàng từ dưới lên
def perm_clis_bottom_up(h, w):
    return perm_clis(h, w).view(h, w).flip(0).flatten()


# Bảng 4 tuyến của mỗi biến thể SS2D (thứ tự tuyến cố định: tuyến k luôn dùng tham số x_proj / dt / A / D thứ k)
ROUTES = {
    "m_ras"     : lambda h, w: [perm_raster(h, w), perm_raster(h, w).flip(0), perm_column(h, w), perm_column(h, w).flip(0)],
    "m_clis"    : lambda h, w: [perm_raster(h, w), perm_raster(h, w).flip(0), perm_clis(h, w), perm_clis(h, w).flip(0)],
    "m_rand"    : lambda h, w: [perm_raster(h, w), perm_raster(h, w).flip(0), perm_random_pair(h, w), perm_random_pair(h, w).flip(0)],
    "m_clis4"   : lambda h, w: [perm_clis(h, w), perm_clis(h, w).flip(0), perm_clis_bottom_up(h, w), perm_clis_bottom_up(h, w).flip(0)],
}


# ============================================================
# Selective scan (Mamba-1) bằng PyTorch thuần, cùng chữ ký với kernel mamba-ssm, dùng để kiểm thử:
#   h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t,  y_t = C_t . h_t + D * u_t
# u, delta: (b, d, l) | A: (d, n) | B, C: (b, g, n, l), nhóm g phụ trách d / g kênh liên tiếp | D, delta_bias: (d)
def selective_scan_reference(u, delta, A, B, C, D = None, delta_bias = None, delta_softplus = False):
    dtype = u.dtype
    with torch.autocast(device_type = u.device.type, enabled = False):
        u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
        if delta_bias is not None:
            delta = delta + delta_bias.float()[:, None]
        if delta_softplus:
            delta = F.softplus(delta)
        group_size = u.size(1) // B.size(1)
        B, C = B.repeat_interleave(group_size, dim = 1), C.repeat_interleave(group_size, dim = 1)   # (b, d, n, l)
        decay = torch.exp(delta[:, :, None] * A[None, :, :, None])
        drive = delta[:, :, None] * B * u[:, :, None]
        state = u.new_zeros(u.size(0), u.size(1), A.size(1))
        outputs = []
        for t in range(u.size(2)):
            state = decay[..., t] * state + drive[..., t]
            outputs.append((state * C[..., t]).sum(-1))
        y = torch.stack(outputs, dim = -1)
        if D is not None:
            y = y + u * D.float()[:, None]
    return y.to(dtype)


# Kernel CUDA nếu có và dữ liệu trên GPU, ngược lại bản tham chiếu
def selective_scan(u, delta, A, B, C, D, delta_bias):
    if selective_scan_fn is not None and u.is_cuda:
        return selective_scan_fn(u, delta, A, B, C, D, delta_bias = delta_bias, delta_softplus = True)
    return selective_scan_reference(u, delta, A, B, C, D, delta_bias = delta_bias, delta_softplus = True)


# ============================================================
# Hoán vị theo từng tuyến trên chiều cuối: y (b, k, d, l) --> y[:, j, :, index[j]] cho mỗi tuyến j.
# Hoán vị là song ánh nên gradient của nó cũng chỉ là một phép hoán vị (theo hoán vị ngược), không cần cộng dồn.
# Backward mặc định của gather / index_select là scatter_add / index_add: ở chế độ tất định (config.py) torch dùng thuật toán
# có sắp xếp, làm một bước huấn luyện m_clis chậm khoảng 2 lần (đo: 386 so với 182 ms ở 512 px, batch 8)
def permute_routes(y, index):
    return torch.gather(y, 3, index[None, :, None, :].expand(y.size(0), -1, y.size(2), -1))


class RoutePermute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, index, inverse):
        ctx.save_for_backward(inverse)
        return permute_routes(y, index)

    @staticmethod
    def backward(ctx, grad):
        inverse, = ctx.saved_tensors
        return permute_routes(grad, inverse), None, None


# ============================================================
# Khởi tạo dt như Mamba: dt log-uniform trong [dt_min, dt_max], bias = softplus^-1(dt)
def dt_bias_init(size, dt_min = 0.001, dt_max = 0.1, floor = 1e-4):
    dt = torch.exp(torch.rand(size) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min = floor)
    return dt + torch.log(-torch.expm1(-dt))


# ============================================================
# SS2D đa tuyến (ý tưởng VMamba): chiếu vào --> conv 3x3 theo kênh --> 4 chuỗi theo 4 hoán vị (gather) --> một lần gọi
# selective scan cho cả 4 tuyến (mỗi tuyến một nhóm B / C, tham số dt / A / D riêng) --> hoán vị ngược (gather) --> cộng
# các tuyến --> LayerNorm * SiLU(z) --> chiếu ra. Hoán vị nằm ngoài kernel nên kernel là hộp đen, dùng chung mọi biến thể.
class MultiRouteSS2D(nn.Module):
    def __init__(self, dim, routes = "m_clis", d_state = 16, dt_rank = None):
        super().__init__()
        if routes not in ROUTES:
            raise ValueError(f"Bảng tuyến '{routes}' không hợp lệ: {list(ROUTES)}")
        self.routes, self.k, self.n = routes, 4, d_state
        self.r = dt_rank or math.ceil(dim / 16)

        self.in_proj = nn.Linear(dim, 2 * dim, bias = False)
        self.conv = nn.Conv2d(dim, dim, 3, padding = 1, groups = dim)
        self.x_proj = nn.Parameter(torch.empty(self.k, self.r + 2 * d_state, dim).uniform_(-dim ** -0.5, dim ** -0.5))
        self.dt_proj = nn.Parameter(torch.empty(self.k, dim, self.r).uniform_(-self.r ** -0.5, self.r ** -0.5))
        self.dt_bias = nn.Parameter(dt_bias_init(self.k * dim))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype = torch.float32)).repeat(self.k * dim, 1))   # A = -[1..n] (S4D-real)
        self.D = nn.Parameter(torch.ones(self.k * dim))
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias = False)
        self._orders = {}

    # Bảng hoán vị (k, l) và hoán vị ngược của lưới h x w (tạo một lần cho mỗi kích thước lưới)
    def orders(self, h, w, device):
        key = (h, w, str(device))
        if key not in self._orders:
            order = torch.stack(ROUTES[self.routes](h, w)).to(device)
            self._orders[key] = (order, torch.argsort(order, dim = 1))
        return self._orders[key]

    # Quét 4 tuyến: x (b, d, l) thứ tự raster --> đầu ra từng tuyến (b, k, d, l), đã trả về thứ tự raster
    def scan_routes(self, x, h, w):
        b, d, length = x.shape
        order, inverse = self.orders(h, w, x.device)
        xs = RoutePermute.apply(x[:, None].expand(b, self.k, d, length), order, inverse)                     # (b, k, d, l) theo thứ tự quét
        dt, B, C = torch.split(torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj), [self.r, self.n, self.n], dim = 2)
        dt = torch.einsum("bkrl,kdr->bkdl", dt, self.dt_proj)
        y = selective_scan(xs.view(b, self.k * d, length), dt.reshape(b, self.k * d, length), -torch.exp(self.A_log.float()), B.contiguous(), C.contiguous(), self.D.float(), self.dt_bias.float())
        return RoutePermute.apply(y.view(b, self.k, d, length), inverse, order)

    def forward(self, x):                                                   # x: (b, h, w, d)
        b, h, w, d = x.shape
        x, z = self.in_proj(x).chunk(2, dim = -1)
        x = F.silu(self.conv(x.permute(0, 3, 1, 2))).flatten(2)             # (b, d, l), thứ tự raster
        y = self.scan_routes(x, h, w).sum(1)                                # Cộng 4 tuyến
        y = self.norm(y.transpose(1, 2).reshape(b, h, w, d)) * F.silu(z)
        return self.out_proj(y)


# ============================================================
# Gated CNN kiểu MambaOut (không SSM): chiếu vào thành cổng / giá trị --> conv 7x7 theo kênh trên nhánh giá trị --> SiLU(cổng) * giá trị --> chiếu ra
class GatedConv(nn.Module):
    def __init__(self, dim, hidden, kernel_size = 7):
        super().__init__()
        self.fc1 = nn.Linear(dim, 2 * hidden)
        self.conv = nn.Conv2d(hidden, hidden, kernel_size, padding = kernel_size // 2, groups = hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):                                                   # x: (b, h, w, d)
        gate, value = self.fc1(x).chunk(2, dim = -1)
        value = self.conv(value.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.fc2(F.silu(gate) * value)


# Gated CNN nhận thêm |F - lật ngang(F)| qua một phép chiếu tuyến tính, cộng vào đầu vào.
# Ảnh đã căn đường giữa nên lật ngang = phản chiếu qua đường giữa cơ thể
class FlipDiff(GatedConv):
    def __init__(self, dim, hidden, kernel_size = 7):
        super().__init__(dim, hidden, kernel_size)
        self.diff_proj = nn.Linear(dim, dim)

    def forward(self, x):
        return super().forward(x + self.diff_proj((x - x.flip(2)).abs()))


# Self-attention toàn cục (scaled_dot_product_attention chọn FlashAttention / memory-efficient khi có thể).
# qkv + chiếu ra có bias: 4d^2 + 4d tham số, lệch khoảng 1% so với SS2D (d = 384, n = 16) nên không cần chỉnh độ rộng
class Attention(nn.Module):
    def __init__(self, dim, heads = 6):
        super().__init__()
        if dim % heads:
            raise ValueError("dim phải chia hết cho số head")
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):                                                   # x: (b, h, w, d)
        b, h, w, d = x.shape
        q, k, v = self.qkv(x).view(b, h * w, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(b, h, w, d))


# ============================================================
# Khớp số tham số: đếm trên thiết bị meta (không cấp phát bộ nhớ), tìm nhị phân độ rộng ẩn của Gated CNN / FlipDiff
def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def ss2d_parameters(dim, d_state):
    with torch.device("meta"):
        return count_parameters(MultiRouteSS2D(dim, "m_clis", d_state))


def matched_hidden(mixer_class, dim, target):
    def size(hidden):
        with torch.device("meta"):
            return count_parameters(mixer_class(dim, hidden))
    low, high = 1, 16 * dim                                                 # Số tham số tăng theo độ rộng ẩn
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (middle, high) if size(middle) < target else (low, middle)
    return min((low, high), key = lambda hidden: abs(size(hidden) - target))


def make_mixer(kind, dim, d_state):
    if kind in ROUTES:
        return MultiRouteSS2D(dim, kind, d_state)
    if kind == "attn":
        return Attention(dim)
    if kind in ("gatedconv", "flipdiff"):
        mixer_class = GatedConv if kind == "gatedconv" else FlipDiff
        return mixer_class(dim, matched_hidden(mixer_class, dim, ss2d_parameters(dim, d_state)))
    raise ValueError(f"Bộ trộn '{kind}' không hợp lệ: {list(ROUTES) + ['gatedconv', 'attn', 'flipdiff']}")


# ============================================================
# Ma trận nội suy tuyến tính (n_out, n_in), trùng F.interpolate(mode = "linear", align_corners = False).
# Nội suy bằng phép nhân ma trận nên backward tất định (backward của F.interpolate tuyến tính trên CUDA không tất định)
def interpolation_matrix(n_out, n_in):
    position = ((torch.arange(n_out, dtype = torch.float64) + 0.5) * n_in / n_out - 0.5).clamp(min = 0)
    low = position.floor().long().clamp(max = n_in - 1)
    high = (low + 1).clamp(max = n_in - 1)
    weight = position - low
    matrix = torch.zeros(n_out, n_in, dtype = torch.float64)
    matrix[torch.arange(n_out), low] += 1 - weight
    matrix[torch.arange(n_out), high] += weight
    return matrix.float()


# Mã hóa vị trí hàng + cột học được trên lưới gốc base x base (512 px / 16), nội suy theo kích thước lưới thực tế:
# checkpoint 512 px dùng tiếp được ở 768 / 1024 px
class RowColPE(nn.Module):
    def __init__(self, dim, base = 32):
        super().__init__()
        self.row = nn.Parameter(torch.zeros(base, dim))
        self.column = nn.Parameter(torch.zeros(base, dim))
        nn.init.trunc_normal_(self.row, std = 0.02)
        nn.init.trunc_normal_(self.column, std = 0.02)
        self._matrices = {}

    def matrix(self, n_out, device):
        key = (n_out, str(device))
        if key not in self._matrices:
            self._matrices[key] = interpolation_matrix(n_out, self.row.size(0)).to(device)
        return self._matrices[key]

    def forward(self, x):                                                   # x: (b, h, w, d)
        h, w = x.shape[1:3]
        row = self.matrix(h, x.device) @ self.row
        column = self.matrix(w, x.device) @ self.column
        return x + row[:, None, :] + column[None, :, :]


# ============================================================
# Khối MetaFormer: x + mixer(LN(x)), rồi x + MLP(LN(x))
class Block(nn.Module):
    def __init__(self, dim, mixer, mlp_ratio = 2.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1, self.mixer = nn.LayerNorm(dim), mixer
        self.norm2, self.mlp = nn.LayerNorm(dim), nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# ============================================================
# Mô hình đầy đủ
#   1. ConvNeXt-T (ImageNet-1k, timm) stage 1-3: (b, 384, R / 16, R / 16)
#   2. Mã hóa vị trí hàng / cột, chuyển sang channels-last (b, h, w, 384)
#   3. depth khối MetaFormer cùng một loại bộ trộn (gradient checkpointing từng khối khi huấn luyện)
#   4. LayerNorm theo token --> trung bình toàn cục --> Linear 14 lớp
class CLISMamba(nn.Module):
    PRETRAINED_MODULES = ("backbone",)                                      # Khối dùng trọng số ImageNet (nhóm LR backbone); còn lại là lớp mới

    def __init__(self, num_classes, mixer = "m_clis", depth = 4, d_state = 16, mlp_ratio = 2.0, grad_checkpoint = True, backbone_checkpoint = False, pretrained = True, init_from = None):
        super().__init__()
        self.backbone = timm.create_model("convnext_tiny.fb_in1k", pretrained = pretrained and init_from is None, features_only = True, out_indices = (2,))
        if self.backbone.feature_info.channels() != [DIM] or self.backbone.feature_info.reduction() != [16]:
            raise RuntimeError(f"ConvNeXt-T stage 3 phải có {DIM} kênh, stride 16; nhận {self.backbone.feature_info.channels()}, {self.backbone.feature_info.reduction()}")
        self.backbone.set_grad_checkpointing(backbone_checkpoint)

        self.pe = RowColPE(DIM)
        self.blocks = nn.ModuleList([Block(DIM, make_mixer(mixer, DIM, d_state), mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, num_classes)
        self.grad_checkpoint = grad_checkpoint

        # Khởi tạo từ best_model.pt của cùng biến thể (ví dụ ở 512 px, trước khi tinh chỉnh ở 768 / 1024 px):
        # mã hóa vị trí tự nội suy theo lưới mới
        if init_from is not None:
            saved = load_torch(init_from)
            if saved["model_kwargs"].get("mixer", "m_clis") != mixer:
                raise ValueError(f"{init_from} là checkpoint của bộ trộn {saved['model_kwargs'].get('mixer', 'm_clis')}, không phải {mixer}")
            self.load_state_dict(saved["model_state_dict"])

    # Tham số không áp dụng weight decay (quy ước timm): mã hóa vị trí và tham số SSM (A_log, D, dt_bias) như Mamba / VMamba
    def no_weight_decay(self):
        return {name for name, _ in self.named_parameters() if name.startswith("pe.") or name.rsplit(".", 1)[-1] in ("A_log", "D", "dt_bias")}

    # Token sau các khối và LayerNorm: (b, h, w, d)
    def tokens(self, x):
        x = self.pe(self.backbone(x)[0].permute(0, 2, 3, 1))
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant = False) if self.grad_checkpoint and self.training and torch.is_grad_enabled() else block(x)
        return self.norm(x)

    def forward(self, x):
        return self.head(self.tokens(x).mean(dim = (1, 2)))

    # Bản đồ kích hoạt lớp (b, lớp, h, w): logits = cam.mean((2, 3)) + head.bias, đúng tuyệt đối
    def cam(self, x):
        return torch.einsum("bhwd,cd->bchw", self.tokens(x), self.head.weight)

    # FLOPs của phần quét selective scan cho một ảnh (FlopCounterMode không thấy kernel CUDA tự viết): khoảng 9 FLOP
    # cho mỗi (kênh, trạng thái, bước) theo quy ước của các repo phái sinh VMamba; utilities/tracking.py cộng vào tổng FLOPs
    def uncounted_flops(self, height, width):
        tokens = (height // 16) * (width // 16)
        return sum(9 * tokens * block.mixer.k * DIM * block.mixer.n for block in self.blocks if isinstance(block.mixer, MultiRouteSS2D))


# ============================================================
# Chạy thí nghiệm với giao thức chung của nghiên cứu CLIS (config.CLIS_CONFIG + config.CLIS_PROTOCOL). Ví dụ:
#   python models/clis_mamba.py                     --> biến thể đề xuất m_clis
#   python models/clis_mamba.py m_ras m_rand        --> các biến thể đối chứng
#   python models/clis_mamba.py all                 --> toàn bộ, lần lượt
# Biến thể SS2D cần kernel mamba-ssm: chạy trong WSL2 (conda activate mamba-main)
if __name__ == "__main__":
    from utilities.setting import apply_config

    parser = argparse.ArgumentParser(description = "Huấn luyện CLIS-Mamba và các biến thể đối chứng")
    parser.add_argument("variants", nargs = "*", default = ["m_clis"], help = f"Một hoặc nhiều biến thể trong {list(VARIANTS)}, hoặc all")
    args = parser.parse_args()

    names = list(VARIANTS) if "all" in args.variants else args.variants
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        parser.error(f"Biến thể không hợp lệ: {unknown}. Chọn trong {list(VARIANTS)} hoặc all")
    if selective_scan_fn is None and any(VARIANTS[name]["mixer"] in ROUTES for name in names):
        parser.error("Biến thể SS2D cần kernel selective_scan_cuda của mamba-ssm: chạy trong WSL2 (conda activate mamba-main)")

    apply_config(config.CLIS_CONFIG)
    for name in names:
        run_experiment(model_name = model_name(name), dataset_class = NIHChestXrayDataset, model_factory = CLISMamba, model_kwargs = VARIANTS[name], **config.CLIS_PROTOCOL)
