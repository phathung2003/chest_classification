# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import math

import timm
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.utils.checkpoint import checkpoint

import config
import models.clis_mamba as clis
import models.convnext_tiny as convnext
from utilities.file import load_torch

# ============================================================
# CLIS-Mamba v2: Khối SS2D chèn vào ConvNeXt-T đầy đủ (Inserted SSM blocks)
# Mục tiêu: Mô hình đề xuất = baseline ConvNeXt-T + đúng một thay đổi (các khối trộn token chèn giữa stage 3 và stage 4), nên mọi chênh lệch so với baseline quy được cho các khối chèn
# ** v1 bỏ stage 4 tiền huấn luyện (15,5 triệu tham số ImageNet, 56% backbone) và buộc LR đầu phân loại theo LR khối mới (họ CLIS chọn x2, ConvNeXt-T chọn x40); v2 giữ cả hai như baseline
# ** Không sửa models/clis_mamba.py: tệp đó được hash vào chữ ký các lượt v1. v2 chỉ dùng lại các thành phần (hoán vị, SS2D, bộ trộn đối chứng, mã hóa vị trí)
#
# Kiến trúc:
#   1. ConvNeXt-T (ImageNet-1k, timm): stem --> stage 1-3: (b, 384, R / 16, R / 16)
#   2. depth khối MetaFormer chèn, mỗi khối: x + mixer(LN(x) + PE), rồi x + MLP(LN(x)). Phép chiếu ra cuối của bộ trộn và của MLP khởi tạo bằng 0
#      (như Non-local Networks, ControlNet), PE (hàng + cột) chỉ đi vào nhánh trộn: ở bước 0 mô hình trùng tuyệt đối ConvNeXt-T
#   3. Stage 4 ConvNeXt-T --> đầu timm (pooling trung bình --> LayerNorm) --> Linear 14 lớp: giống hệt models/convnext_tiny.py
#
# ** Không dùng LayerScale γ = 1e-6: dòng đặc trưng sau stage 3 có độ lệch chuẩn khoảng 44 còn nhánh chỉ khoảng 0,07, nên sau 1 epoch (lượt ngày 2026-11-07)
#    γ chỉ đạt 1e-4 và khối chèn đóng góp cỡ 1e-6 so với dòng chính: gradient của γ (vô hướng theo kênh) đổi dấu do nhánh còn ngẫu nhiên, gradient của
#    trọng số bên trong bị γ nhân nhỏ dưới eps của Adam. Khởi tạo 0 phép chiếu ra cho gradient có hướng ngay từ bước đầu (đọc ra tuyến tính từ đặc trưng nhánh).
#    residual_init = "layerscale" giữ lại cách cũ chỉ để tái lập lượt đã bỏ
#
# Bảng tuyến SS2D: Mọi biến thể giữ đủ quét chéo hàng ±, cột ± của VMamba. M-clis / M-rand thêm 2 tuyến ghép cặp (cặp gương CLIS / cặp ngẫu nhiên),
# nên H1 (M-clis so với M-rand) chỉ khác cách ghép cặp và không còn lẫn với việc mất quét cột như v1 (v1: M-clis thay 2 tuyến cột bằng 2 tuyến CLIS)
#
# Nhóm LR: Khối chèn học cùng LR backbone (PRETRAINED_MODULES gồm "inserted"); đầu phân loại là nhóm lớp mới với hệ số m* của ConvNeXt-T (x40, models/run_clis_v2.py).
# Cấu hình tối ưu vì vậy giống hệt baseline, không tinh chỉnh LR mới
#
# Pilot 2026-11-08 (lượt m_clis seed 42 trùng ConvNeXt-T seed 42 từng epoch: khối chèn gần như không được dùng, |W chiếu ra| ~0,004 sau 4 epoch):
#   block_lr_scale = s  : Tham số hóa lại mọi tham số của khối chèn thành s x tham số thô (torch parametrize). Adam bất biến với thang gradient nên bước cập nhật
#                         của tham số hiệu dụng gấp s lần, tức LR khối chèn = s x LR backbone mà không cần nhóm LR thứ ba (training/optimizer.py nằm trong hash pipeline).
#                         Khác LR thật ở weight decay: tham số hiệu dụng vẫn giảm lr x wd mỗi bước (không phải s x lr x wd). s = 1: không tham số hóa, mô hình như cũ
#   freeze_backbone     : Đóng băng ConvNeXt-T (stem, 4 stage); chỉ khối chèn và đầu phân loại học. Giai đoạn 1 của LP-FT (Kumar et al., ICLR 2022):
#                         khối chèn buộc phải mang tín hiệu thích nghi CXR vì backbone không đổi; giai đoạn 2 tinh chỉnh toàn bộ từ checkpoint giai đoạn 1 (init_from)
#   init_from với block_lr_scale khác lượt nguồn: trọng số được quy về giá trị hiệu dụng rồi chia cho s mới, nên mô hình nạp vào tính đúng như mô hình nguồn
#
# Biến thể (tên dùng ở models/run_clis_v2.py):
#   convnext_t                     : Không chèn khối = đúng ConvNeXt-T baseline (dùng để tinh chỉnh baseline 512 px --> 1024 px từ checkpoint E1)
#   m_clis / m_rand / m_ras        : SS2D 6 / 6 / 4 tuyến
#   gatedconv / attn / flipdiff    : Đối chứng không SSM, khớp số tham số với SS2D 4 tuyến như v1 (Gated CNN, self-attention, Gated CNN + |F - lật ngang(F)|)

DIM = clis.DIM                                  # Số kênh ConvNeXt-T sau stage 3 (stride 16): 384
LAYER_SCALE_INIT = 1e-6                         # γ khởi tạo của cách cũ residual_init = "layerscale" (đã bỏ, xem đầu tệp)
RESIDUAL_INITS = ("zero", "layerscale")         # Cách khởi tạo phần dư của khối chèn; mặc định "zero"


# ============================================================
# Quét chéo của VMamba: hàng xuôi, hàng ngược, cột xuôi, cột ngược
def cross_scan(h, w):
    raster, column = clis.perm_raster(h, w), clis.perm_column(h, w)
    return [raster, raster.flip(0), column, column.flip(0)]


# Bảng tuyến của từng biến thể SS2D (tuyến k luôn dùng tham số x_proj / dt / A / D thứ k)
ROUTES = {
    "m_ras"     : lambda h, w: cross_scan(h, w),
    "m_clis"    : lambda h, w: cross_scan(h, w) + [clis.perm_clis(h, w), clis.perm_clis(h, w).flip(0)],
    "m_rand"    : lambda h, w: cross_scan(h, w) + [clis.perm_random_pair(h, w), clis.perm_random_pair(h, w).flip(0)],
}
MIXERS = (*ROUTES, "gatedconv", "attn", "flipdiff")
NAMES = ("convnext_t", *MIXERS)


# ============================================================
# Tên mô hình theo biến thể và độ phân giải (mặc định IMAGE_SIZE; dùng làm tên thư mục kết quả).
# convnext_t dùng đúng tên baseline E1 nên lượt 1024 px nằm cạnh lượt 512 px của baseline
def model_name(name, size = None):
    size = size or config.IMAGE_SIZE
    return f"{convnext.MODEL_NAME} CLIS {size}px" if name == "convnext_t" else f"CLIS-Mamba v2 {name} {size}px"


# ============================================================
# SS2D đa tuyến của v1 với số tuyến bằng số hoán vị trong bảng tuyến v2 (4 hoặc 6). Phép tính (scan_routes, forward) giữ nguyên v1
class MultiRouteSS2D(clis.MultiRouteSS2D):
    def __init__(self, dim, routes = "m_clis", d_state = 16, dt_rank = None):
        nn.Module.__init__(self)

        # Kiểm tra bảng tuyến
        if routes not in ROUTES:
            raise ValueError(f"Bảng tuyến '{routes}' không hợp lệ: {list(ROUTES)}")

        # Số tuyến, số trạng thái, hạng của dt
        self.routes, self.k, self.n = routes, len(ROUTES[routes](2, 2)), d_state
        self.r = dt_rank or math.ceil(dim / 16)

        # Tham số: giống v1, chiều tuyến có self.k phần tử
        self.in_proj = nn.Linear(dim, 2 * dim, bias = False)
        self.conv = nn.Conv2d(dim, dim, 3, padding = 1, groups = dim)
        self.x_proj = nn.Parameter(torch.empty(self.k, self.r + 2 * d_state, dim).uniform_(-dim ** -0.5, dim ** -0.5))
        self.dt_proj = nn.Parameter(torch.empty(self.k, dim, self.r).uniform_(-self.r ** -0.5, self.r ** -0.5))
        self.dt_bias = nn.Parameter(clis.dt_bias_init(self.k * dim))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype = torch.float32)).repeat(self.k * dim, 1))   # A = -[1..n] (S4D-real)
        self.D = nn.Parameter(torch.ones(self.k * dim))
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias = False)
        self._orders = {}

    # Bảng hoán vị (k, l) và hoán vị ngược của lưới h x w theo bảng tuyến v2 (tạo một lần cho mỗi kích thước lưới)
    # Trả về: (order, inverse)
    def orders(self, h, w, device):
        key = (h, w, str(device))
        if key not in self._orders:
            order = torch.stack(ROUTES[self.routes](h, w)).to(device)
            self._orders[key] = (order, torch.argsort(order, dim = 1))
        return self._orders[key]


# Bộ trộn token theo tên: SS2D theo bảng tuyến v2; đối chứng không SSM lấy nguyên của v1 (khớp số tham số với SS2D 4 tuyến)
# Trả về: nn.Module nhận / trả (b, h, w, d)
def make_mixer(kind, dim, d_state):
    if kind in ROUTES:
        return MultiRouteSS2D(dim, kind, d_state)
    if kind in ("gatedconv", "attn", "flipdiff"):
        return clis.make_mixer(kind, dim, d_state)
    raise ValueError(f"Bộ trộn '{kind}' không hợp lệ: {list(MIXERS)}")


# Phép chiếu ra cuối của một bộ trộn (Linear cuối trước khi cộng vào phần dư)
# Trả về: nn.Linear
def output_projection(mixer):
    for name in ("out_proj", "fc2", "proj"):                                        # SS2D | GatedConv, FlipDiff | Attention
        if isinstance(getattr(mixer, name, None), nn.Linear):
            return getattr(mixer, name)
    raise TypeError(f"Không tìm thấy phép chiếu ra của {type(mixer).__name__}")


# ============================================================
# Tham số hóa lại: tham số hiệu dụng = scale x tham số thô (right_inverse: thô = hiệu dụng / scale, nên đăng ký giữ nguyên giá trị hiệu dụng)
class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)

    def forward(self, raw):
        return raw * self.scale

    def right_inverse(self, value):
        return value / self.scale


# Tham số hóa lại mọi tham số của module thành scale x tham số thô (LR hiệu dụng gấp scale lần với Adam / AdamW)
def scale_parameters(module, scale):
    for submodule in list(module.modules()):
        for name in [name for name, _ in submodule.named_parameters(recurse = False)]:
            parametrize.register_parametrization(submodule, name, Scale(scale))


# Tên tham số khi bỏ tham số hóa: "a.parametrizations.w.original" --> "a.w"
def plain_name(name):
    return name.replace("parametrizations.", "").removesuffix(".original")


# state_dict --> giá trị hiệu dụng theo tên thường (tham số thô của khối chèn nhân lại scale của mô hình nguồn)
# Trả về: dict
def effective_state(state, scale):
    return {plain_name(key): (value * scale if key.endswith(".original") else value) for key, value in state.items()}


# ============================================================
# Khối MetaFormer chèn; mã hóa vị trí chỉ cộng vào đầu vào của bộ trộn (không vào dòng chính)
#   residual_init = "zero"       : Phép chiếu ra của bộ trộn và Linear cuối của MLP khởi tạo bằng 0 --> khối là phép đồng nhất ở bước 0
#   residual_init = "layerscale" : Cách cũ (đã bỏ): hai phần dư nhân LayerScale γ khởi tạo LAYER_SCALE_INIT
class Block(nn.Module):
    def __init__(self, dim, mixer, mlp_ratio = 2.0, residual_init = "zero"):
        super().__init__()

        # Kiểm tra cách khởi tạo
        if residual_init not in RESIDUAL_INITS:
            raise ValueError(f"residual_init '{residual_init}' không hợp lệ: {list(RESIDUAL_INITS)}")

        # Nhánh trộn token và nhánh MLP
        hidden = int(dim * mlp_ratio)
        self.norm1, self.mixer = nn.LayerNorm(dim), mixer
        self.norm2, self.mlp = nn.LayerNorm(dim), nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

        # Khởi tạo phần dư
        self.layer_scale = residual_init == "layerscale"
        if self.layer_scale:
            self.gamma1 = nn.Parameter(torch.full((dim,), LAYER_SCALE_INIT))      # LayerScale của nhánh trộn token
            self.gamma2 = nn.Parameter(torch.full((dim,), LAYER_SCALE_INIT))      # LayerScale của nhánh MLP
        else:
            for layer in (output_projection(self.mixer), self.mlp[2]):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x, position):                                                 # x: (b, h, w, d) | position: (1, h, w, d)
        mixed = self.mixer(self.norm1(x) + position)
        x = x + (self.gamma1 * mixed if self.layer_scale else mixed)
        fed = self.mlp(self.norm2(x))
        return x + (self.gamma2 * fed if self.layer_scale else fed)


# Chuỗi khối chèn giữa hai stage ConvNeXt: nhận / trả bản đồ đặc trưng (b, d, h, w) như một stage
class InsertedBlocks(nn.Module):
    def __init__(self, dim, mixer, depth, d_state, mlp_ratio, residual_init, grad_checkpoint):
        super().__init__()
        self.pe = clis.RowColPE(dim)
        self.blocks = nn.ModuleList([Block(dim, make_mixer(mixer, dim, d_state), mlp_ratio, residual_init) for _ in range(depth)])
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x):

        # Chuyển sang channels-last cho bộ trộn; bảng mã hóa vị trí (1, h, w, d) nội suy theo lưới thực tế
        x = x.permute(0, 2, 3, 1)
        position = self.pe(x.new_zeros(1, *x.shape[1:]))

        # Các khối (gradient checkpointing từng khối khi huấn luyện: cùng phép tính, chỉ đổi VRAM / tốc độ)
        for block in self.blocks:
            x = checkpoint(block, x, position, use_reentrant = False) if self.grad_checkpoint and self.training and torch.is_grad_enabled() else block(x, position)
        return x.permute(0, 3, 1, 2)


# ============================================================
# Mô hình đầy đủ (mixer = "convnext_t": không chèn khối, đúng ConvNeXt-T baseline)
class CLISMambaV2(nn.Module):
    def __init__(self, num_classes, mixer = "m_clis", depth = 4, d_state = 16, mlp_ratio = 2.0, residual_init = "zero",
                 grad_checkpoint = True, backbone_checkpoint = False, pretrained = True, init_from = None, block_lr_scale = 1.0, freeze_backbone = False):
        super().__init__()

        # Kiểm tra biến thể và hệ số LR khối chèn
        if mixer not in NAMES:
            raise ValueError(f"Biến thể '{mixer}' không hợp lệ: {list(NAMES)}")
        if not math.isfinite(float(block_lr_scale)) or float(block_lr_scale) <= 0:
            raise ValueError(f"block_lr_scale phải là số dương hữu hạn, nhận được {block_lr_scale}")
        if mixer == "convnext_t" and float(block_lr_scale) != 1.0:
            raise ValueError("convnext_t không có khối chèn nên không dùng block_lr_scale")
        self.mixer = mixer
        self.block_lr_scale = float(block_lr_scale)
        self.freeze_backbone = bool(freeze_backbone)

        # 1. Backbone + đầu phân loại tạo theo đúng thứ tự của baseline (cùng seed --> cùng trọng số khởi tạo đầu phân loại)
        self.cnn = timm.create_model("convnext_tiny.fb_in1k", pretrained = pretrained and init_from is None, num_classes = 0)
        self.classifier = nn.Linear(self.cnn.num_features, num_classes)
        self.cnn.set_grad_checkpointing(backbone_checkpoint)

        # 2. Khối chèn giữa stage 3 và stage 4 (tạo sau đầu phân loại)
        self.inserted = None if mixer == "convnext_t" else InsertedBlocks(DIM, mixer, depth, d_state, mlp_ratio, residual_init, grad_checkpoint)
        if self.inserted is not None and self.block_lr_scale != 1.0:
            scale_parameters(self.inserted, self.block_lr_scale)                    # LR hiệu dụng của khối chèn = block_lr_scale x LR backbone

        # 3. Nhóm LR backbone (training/optimizer.py): backbone ImageNet và khối chèn; còn lại (đầu phân loại) là nhóm lớp mới
        self.PRETRAINED_MODULES = ("cnn",) if self.inserted is None else ("cnn", "inserted")

        # 4. Khởi tạo từ best_model.pt cùng biến thể (ví dụ 512 px, trước khi tinh chỉnh ở 1024 px; hoặc giai đoạn 1 LP-FT);
        #    checkpoint baseline E1 (models/convnext_tiny.py) không có khóa mixer. Trọng số quy về giá trị hiệu dụng rồi chia block_lr_scale của mô hình này
        if init_from is not None:
            saved = load_torch(init_from)
            saved_mixer = saved["model_kwargs"].get("mixer", "convnext_t")
            if saved_mixer != mixer:
                raise ValueError(f"{init_from} là checkpoint của biến thể {saved_mixer}, không phải {mixer}")
            if self.inserted is not None and saved["model_kwargs"].get("residual_init", "layerscale") != residual_init:
                raise ValueError(f"{init_from} dùng cách khởi tạo phần dư khác {residual_init}")
            state = effective_state(saved["model_state_dict"], float(saved["model_kwargs"].get("block_lr_scale", 1.0)))
            expected = {plain_name(key) for key in self.state_dict()}
            if set(state) != expected:
                raise ValueError(f"{init_from}: khóa không khớp mô hình, thiếu {sorted(expected - set(state))[:5]}, thừa {sorted(set(state) - expected)[:5]}")
            self.load_state_dict({key: (state[plain_name(key)] / self.block_lr_scale if key.endswith(".original") else state[plain_name(key)]) for key in self.state_dict()})

        # 5. Giai đoạn 1 LP-FT: backbone ConvNeXt-T không học (không BatchNorm, drop path 0 nên chế độ train không đổi đầu ra của backbone)
        if self.freeze_backbone:
            self.cnn.requires_grad_(False)

    # Tham số không áp dụng weight decay (quy ước timm): mã hóa vị trí và tham số SSM (A_log, D, dt_bias) như Mamba / VMamba.
    # γ LayerScale, bias, norm là tham số 1 chiều: no_weight_decay_1d của CLIS_PROTOCOL đã bỏ decay
    def no_weight_decay(self):
        return {name for name, _ in self.named_parameters() if name.startswith("inserted.pe.") or plain_name(name).rsplit(".", 1)[-1] in ("A_log", "D", "dt_bias")}

    # Bản đồ đặc trưng sau stage 4 (trước pooling)
    # Trả về: (b, 768, R / 32, R / 32)
    def features(self, x):
        x = self.cnn.stem(x)
        x = self.cnn.stages[0](x)
        x = self.cnn.stages[1](x)
        x = self.cnn.stages[2](x)
        if self.inserted is not None:
            x = self.inserted(x)
        x = self.cnn.stages[3](x)
        return self.cnn.norm_pre(x)

    def forward(self, x):
        return self.classifier(self.cnn.forward_head(self.features(x)))

    # Mức tham gia của từng khối chèn: trung bình |trọng số| của phép chiếu ra nhánh trộn và nhánh MLP (bằng 0 ở bước 0 với residual_init "zero")
    # Trả về: Danh sách (nhánh trộn, nhánh MLP) theo khối; rỗng nếu không chèn
    def engagement(self):
        if self.inserted is None:
            return []
        return [(output_projection(block.mixer).weight.detach().abs().mean().item(), block.mlp[2].weight.detach().abs().mean().item()) for block in self.inserted.blocks]

    # FLOPs của phần quét selective scan cho một ảnh, cùng quy ước với v1 (utilities/tracking.py cộng vào tổng FLOPs)
    def uncounted_flops(self, height, width):
        if self.inserted is None:
            return 0
        tokens = (height // 16) * (width // 16)
        return sum(9 * tokens * block.mixer.k * DIM * block.mixer.n for block in self.inserted.blocks if isinstance(block.mixer, clis.MultiRouteSS2D))
