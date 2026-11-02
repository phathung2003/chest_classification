# ============================================================
# Mô hình CNN SymMamba (DenseNet-121 đa tỉ lệ + Mamba-2 quét đối xứng hai bên + Class Query Head)
# Hướng đi: Cho Mamba so sánh trực tiếp hai bên phổi bằng thứ tự quét đối xứng trái / phải.

# Ý tưởng:
# - X-quang ngực gần đối xứng trái / phải, bác sĩ đọc phim bằng cách so sánh hai bên.
# - Mamba xử lý token theo thứ tự quét, nên đặt mỗi token ngay cạnh token đối xứng của nó qua đường giữa giúp trạng thái ẩn so sánh trực tiếp hai bên.
# - Đây chỉ là một phép hoán vị thứ tự token: Không thêm tham số, không thêm FLOPs so với quét chéo bốn hướng của VMamba.

# Thứ tự quét (một hàng của lưới token rộng 6 cột 0..5):
# - Quét thông thường (cross4): 0, 1, 2, 3, 4, 5 --> token 0 và token đối xứng 5 cách nhau 5 bước.
# - Quét đối xứng (sym4): 0, 5, 1, 4, 2, 3 --> mỗi cặp đối xứng (0, 5), (1, 4), (2, 3) đứng liền nhau.
# ============================================================

# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Chế độ tất định của cuBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights

import config
from training.experiment import start_train
from utilities.dataset import NIHChestXrayDataset

# ============================================================
# Cài đặt mô hình
MODEL_NAME      = "CNN SymMamba_v1.1"                  # Tên mô hình


# ============================================================
# Các biến thể của mô hình (Chỉ khác khối trộn token hoặc thứ tự quét, mọi phần khác giữ nguyên)
VARIANTS = {
    "none"          : {"mixer": "none"},                        # Không trộn token (đối chứng)
    "transformer"   : {"mixer": "transformer"},                 # Self-attention, số tham số khớp khối Mamba (đối chứng)
    "bi2"           : {"mixer": "mamba", "scan": "bi2"},        # Hàng xuôi + ngược (kiểu Vision Mamba)
    "cross4"        : {"mixer": "mamba", "scan": "cross4"},     # Hàng + cột, xuôi + ngược (kiểu VMamba)
    "sym4"          : {"mixer": "mamba", "scan": "sym4"},       # Đề xuất: Quét đối xứng hai bên
    "random4"       : {"mixer": "mamba", "scan": "random4"},    # 4 hoán vị ngẫu nhiên cố định (đối chứng: Thứ tự có quan trọng không)
}

# Cùng các biến thể nhưng bỏ đường tắt GAP (<tên>_noskip): Logits chỉ đến từ Class Query Head, khối trộn token gánh toàn bộ quyết định
VARIANTS.update({f"{name}_noskip": {**kwargs, "skip": False} for name, kwargs in list(VARIANTS.items())})

# ============================================================
# Tên mô hình
# Trả về: Tên mô hình
def model_name(variant):
    return f"{MODEL_NAME} {variant}"

# ============================================================
# Tạo thứ tự quét cho lưới h x w
# Trả về: Tensor (số đường quét, h * w), mỗi dòng là một hoán vị chỉ số token
def scan_orders(mode, h, w):

    # Thứ tự theo hàng và theo cột của lưới
    grid = torch.arange(h * w).view(h, w)
    row, column = grid.flatten(), grid.t().flatten()

    # Cột theo cặp đối xứng qua đường giữa, từ ngoài vào trong: 0, w-1, 1, w-2, ... (cột giữa đứng cuối nếu w lẻ)
    pairs = torch.stack([torch.arange(w // 2), torch.arange(w - 1, (w - 1) // 2, -1)], dim = 1).flatten()
    if w % 2:
        pairs = torch.cat([pairs, torch.tensor([w // 2])])

    # Hàng đối xứng: Trong mỗi hàng, token trái và token phải đối xứng đứng liền nhau
    sym_row = grid[:, pairs].flatten()

    # Cột đối xứng: Đi từ trên xuống qua từng cặp cột đối xứng, xen kẽ trái / phải ở mỗi hàng
    sym_column = torch.cat([grid[:, pairs[i:i + 2]].flatten() for i in range(0, len(pairs), 2)])

    # Các đường quét của từng kiểu quét (đường thứ hai thường là đường thứ nhất đảo ngược)
    if mode == "bi2":
        paths = [row, row.flip(0)]                                                  # Hàng xuôi + ngược
    elif mode == "cross4":
        paths = [row, row.flip(0), column, column.flip(0)]                          # Hàng + cột, xuôi + ngược
    elif mode == "sym4":
        paths = [sym_row, sym_row.flip(0), sym_column, sym_column.flip(0)]          # Hàng đối xứng + cột đối xứng, xuôi + ngược
    elif mode == "random4":
        generator = torch.Generator().manual_seed(2024)                             # Seed cố định: Hoán vị giống nhau ở mọi lần chạy
        paths = [torch.randperm(h * w, generator = generator) for _ in range(4)]   # 4 hoán vị ngẫu nhiên
    else:
        raise ValueError(
            "[scan_orders]: Kiểu quét phải là một trong bi2 | cross4 | sym4 | random4.\n"
            f"Nhận được: mode = {mode!r}"
        )

    return torch.stack(paths)

# ============================================================
# SSD của Mamba-2 ở dạng đối ngẫu bậc hai (Quadratic Dual Form)
# - Công thức: y_t = sum_{s <= t} exp(sum_{k = s+1..t} dt_k * A) * (C_t . B_s) * dt_s * x_s
# - Trùng khớp với dạng hồi quy h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t, y_t = C_t . h_t.
# - Với chuỗi ngắn (196 token ở 224 px) dạng này chỉ gồm vài phép nhân ma trận: Nhanh, tất định, không cần CUDA kernel.
# - Kích thước: x: (b, l, h, p) | dt: (b, l, h) | A: (h) | B, C: (b, l, n).
# Trả về: y (b, l, h, p)
def ssd_quadratic(x, dt, A, B, C):

    # Độ dài chuỗi
    length = x.size(1)

    # Tính ở FP32 (tắt autocast) để hàm mũ và tổng tích lũy ổn định số học
    with torch.autocast(device_type = x.device.type, enabled = False):
        x, dt, A, B, C = x.float(), dt.float(), A.float(), B.float(), C.float()

        # Hệ số suy giảm giữa hai bước: exp(tổng dt * A từ s + 1 tới t), bằng 0 khi s > t (nhân quả)
        cumulative = torch.cumsum(dt * A, dim = 1).transpose(1, 2)
        causal = torch.ones(length, length, dtype = torch.bool, device = x.device).tril()
        decay = torch.exp((cumulative[..., :, None] - cumulative[..., None, :]).masked_fill(~causal, float("-inf")))

        # Ma trận trộn (b, h, t, s) = suy giảm * (C_t . B_s), rồi nhân với đầu vào đã rời rạc hóa
        scores = torch.einsum("btn,bsn->bts", C, B)
        return torch.einsum("bhts,bshp->bthp", decay * scores[:, None], x * dt[..., None])

# ============================================================
# Khối Mamba-2 (Dao & Gu, 2024), viết bằng PyTorch thuần theo cấu trúc của mamba_ssm.Mamba2
# - Chiếu vào --> conv1d nhân quả --> SSD --> cổng SiLU + RMSNorm --> chiếu ra.
class Mamba2(nn.Module):

    # Khởi tạo khối Mamba-2
    def __init__(self, d_model, d_state = 64, d_conv = 4, expand = 2, headdim = 64, dt_min = 0.001, dt_max = 0.1):
        super().__init__()
        self.d_inner = expand * d_model                                     # Số kênh bên trong
        self.d_state = d_state                                              # Số chiều trạng thái của SSM
        self.headdim = headdim                                              # Số kênh mỗi head
        self.nheads = self.d_inner // headdim                               # Số head

        # Kiểm tra dữ liệu đầu vào
        ## expand * d_model phải chia hết cho headdim (mỗi head có đúng headdim kênh)
        if self.d_inner % headdim:
            raise ValueError(
                "[Mamba2]: expand * d_model phải chia hết cho headdim.\n"
                f"Nhận được: expand * d_model = {self.d_inner}, headdim = {headdim}"
            )

        # Số kênh đi qua conv1d (x, B, C)
        conv_dim = self.d_inner + 2 * d_state

        # Chiếu vào: z (cổng), x, B, C, dt; conv1d nhân quả theo từng kênh cho x / B / C
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.nheads, bias = False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups = conv_dim, padding = d_conv - 1)

        # dt khởi tạo log-uniform trong [dt_min, dt_max]; dt_bias = softplus^-1(dt)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min = 1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        # A = -exp(A_log) < 0, một giá trị cho mỗi head; D là đường tắt từ x sang y
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.nheads))

        # Trọng số RMSNorm và chiếu ra
        self.norm_weight = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias = False)

    # Lan truyền xuôi (Forward)
    # Trả về: (b, l, d_model)
    def forward(self, u):

        # Kích thước đầu vào (b, l, d_model)
        b, l, _ = u.shape

        # Chiếu vào: Cổng z, nhánh x / B / C (qua conv1d nhân quả) và bước thời gian dt
        z, xBC, dt = torch.split(self.in_proj(u), [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim = -1)
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[..., :l].transpose(1, 2))
        x, B, C = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim = -1)
        x = x.reshape(b, l, self.nheads, self.headdim)

        # SSD và đường tắt D
        dt = F.softplus(dt.float() + self.dt_bias)
        A = -torch.exp(self.A_log.float())
        y = ssd_quadratic(x, dt, A, B, C) + x.float() * self.D[:, None]

        # Cổng SiLU rồi RMSNorm (như RMSNormGated của Mamba-2), chiếu ra
        y = y.reshape(b, l, self.d_inner) * F.silu(z.float())
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim = True) + 1e-5) * self.norm_weight
        return self.out_proj(y)

# ============================================================
# Số tham số của một lớp quét Mamba-2 (dùng để chọn kích thước FFN cho Transformer đối chứng có cùng số tham số)
# Trả về: Số tham số
def mamba_layer_parameters(d_model, d_state, headdim, expand = 2, d_conv = 4):

    # Kích thước bên trong (giống Mamba2)
    d_inner = expand * d_model
    nheads = d_inner // headdim
    conv_dim = d_inner + 2 * d_state

    # in_proj + conv1d (trọng số, bias) + dt_bias / A_log / D + norm_weight + out_proj + LayerNorm (trọng số, bias)
    return d_model * (2 * d_inner + 2 * d_state + nheads) + conv_dim * (d_conv + 1) + 3 * nheads + d_inner + d_inner * d_model + 2 * d_model

# ============================================================
# Một lớp quét: LayerNorm --> hoán vị token theo từng đường quét --> Mamba-2 dùng chung trọng số (kèm embedding đường quét)
# --> trả về thứ tự gốc --> trung bình các đường --> cộng residual
class ScanMambaLayer(nn.Module):

    # Khởi tạo lớp quét
    def __init__(self, d_model, orders, d_state, headdim):
        super().__init__()
        self.register_buffer("orders", orders, persistent = False)                         # Thứ tự quét (số đường, số token)
        self.register_buffer("inverse", torch.argsort(orders, dim = 1), persistent = False)    # Hoán vị ngược: Đưa về thứ tự gốc
        self.path_embed = nn.Parameter(torch.zeros(orders.size(0), 1, 1, d_model))         # Embedding riêng của từng đường quét
        self.norm = nn.LayerNorm(d_model)                                                   # Chuẩn hóa trước khi quét
        self.mamba = Mamba2(d_model, d_state = d_state, headdim = headdim)                  # Mamba-2 dùng chung cho mọi đường quét

    # Lan truyền xuôi (Forward)
    # Trả về: Token sau lớp quét (b, n, d)
    def forward(self, x):

        # Kích thước token và số đường quét
        b, n, d = x.shape
        k = self.orders.size(0)

        # Hoán vị token theo k đường quét: (b, n, d) --> (k, b, n, d), qua Mamba-2 chung (xếp vào chiều batch)
        paths = self.norm(x)[:, self.orders].transpose(0, 1) + self.path_embed
        y = self.mamba(paths.reshape(k * b, n, d)).view(k, b, n, d)

        # Đưa từng đường về thứ tự gốc, trung bình các đường, cộng residual
        y = torch.stack([y[i][:, self.inverse[i]] for i in range(k)]).mean(0)
        return x + y

# ============================================================
# Khối trộn token: Mamba-2 nhiều đường quét | Transformer (cùng số lớp, cùng chiều, cùng số tham số) | không trộn
class TokenMixer(nn.Module):

    # Khởi tạo khối trộn theo kind
    def __init__(self, kind, d_model, n_layers, grid, scan, d_state, headdim):
        super().__init__()

        # Mamba-2 nhiều đường quét
        if kind == "mamba":
            self.layers = nn.Sequential(*[ScanMambaLayer(d_model, scan_orders(scan, grid, grid), d_state, headdim) for _ in range(n_layers)])

        # Transformer (đối chứng): Chọn kích thước FFN để số tham số một lớp Transformer bằng một lớp quét Mamba-2
        elif kind == "transformer":
            attention_parameters = 4 * d_model * d_model + 9 * d_model
            ffn = round((mamba_layer_parameters(d_model, d_state, headdim) - attention_parameters) / (2 * d_model + 1))
            layer = nn.TransformerEncoderLayer(d_model, nhead = 8, dim_feedforward = ffn, dropout = 0.0, activation = "gelu", batch_first = True, norm_first = True)
            self.layers = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor = False)

        # Không trộn token (đối chứng)
        elif kind == "none":
            self.layers = nn.Identity()

        # Khối trộn không được hỗ trợ
        else:
            raise ValueError(
                "[TokenMixer]: Khối trộn phải là một trong mamba | transformer | none.\n"
                f"Nhận được: kind = {kind!r}"
            )

        # Chuẩn hóa đầu ra
        self.norm = nn.LayerNorm(d_model)

    # Lan truyền xuôi (Forward)
    # Trả về: Token đã trộn (b, n, d_model)
    def forward(self, x):
        return self.norm(self.layers(x))

# ============================================================
# Đầu phân loại class-query: Mỗi lớp bệnh có một query, cross-attention lên các token để lấy đặc trưng riêng của lớp
class ClassQueryHead(nn.Module):

    # Khởi tạo đầu phân loại
    def __init__(self, d, num_classes, heads = 4, dropout = 0.1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(num_classes, d) * 0.02)                               # Query của từng lớp [C, D]
        self.norm = nn.LayerNorm(d)                                                             # Chuẩn hóa token trước attention
        self.attn = nn.MultiheadAttention(d, heads, dropout = dropout, batch_first = True)      # Cross-attention: Query của lớp --> token
        self.w = nn.Parameter(torch.randn(num_classes, d) * 0.02)                               # Trọng số phân loại riêng của từng lớp
        self.b = nn.Parameter(torch.zeros(num_classes))                                         # Bias phân loại riêng của từng lớp

    # Lan truyền xuôi (Forward)
    # Trả về: logits [B, C]
    def forward(self, x):
        kv = self.norm(x)
        q = self.q.unsqueeze(0).expand(x.size(0), -1, -1)
        out, _ = self.attn(q, kv, kv)                       # [B, C, D]
        return (out * self.w).sum(-1) + self.b              # [B, C]

# ============================================================
# Phóng to 2 lần kiểu nearest bằng expand + reshape (backward là phép cộng, luôn tất định)
# Trả về: (b, c, 2h, 2w)
def upsample2x(x):
    b, c, h, w = x.shape
    return x[:, :, :, None, :, None].expand(b, c, h, 2, w, 2).reshape(b, c, 2 * h, 2 * w)

# ============================================================
# Mô hình đầy đủ
# 1. DenseNet121 (ImageNet): Đặc trưng stride 16 (sau denseblock3) và stride 32 (sau norm5).
# 2. Token đa tỉ lệ: Chiếu hai mức về d_model, phóng stride 32 lên stride 16, cộng lại --> (IMAGE_SIZE / 16)^2 token.
# 3. Khối trộn token (Mamba-2 nhiều đường quét / Transformer / không trộn) + position embedding.
# 4. Logits = Class Query Head(token) + đường tắt GAP 1024 kênh (y hệt baseline DenseNet121); skip = False bỏ đường tắt.
class CNNSymMamba(nn.Module):

    # Khối dùng trọng số ImageNet (nhóm learning_rate backbone); còn lại là lớp mới
    PRETRAINED_MODULES = ("stage3", "stage4")

    # Khởi tạo mô hình
    def __init__(self, num_classes, mixer = "mamba", scan = "sym4", d_model = 256, n_layers = 2, d_state = 64, headdim = 64, dropout = 0.2, skip = True):
        super().__init__()

        # Kiểm tra dữ liệu đầu vào
        ## IMAGE_SIZE phải chia hết cho 32 (stride của norm5)
        if config.IMAGE_SIZE % 32:
            raise ValueError(
                "[CNNSymMamba]: IMAGE_SIZE phải chia hết cho 32.\n"
                f"Nhận được: IMAGE_SIZE = {config.IMAGE_SIZE}"
            )

        # DenseNet-121: Đến hết denseblock3 (stride 16) | transition3 --> norm5 (stride 32)
        features = densenet121(weights = DenseNet121_Weights.IMAGENET1K_V1).features
        self.stage3, self.stage4 = features[:9], features[9:]

        # Chiếu về d_model kênh: denseblock3 chưa chuẩn hóa --> BN-ReLU-Conv-BN; norm5 đã chuẩn hóa --> ReLU (trong forward) rồi Conv-BN
        self.proj3 = nn.Sequential(nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, d_model, 1, bias = False), nn.BatchNorm2d(d_model))
        self.proj4 = nn.Sequential(nn.Conv2d(1024, d_model, 1, bias = False), nn.BatchNorm2d(d_model))

        # Position embedding của lưới token (IMAGE_SIZE / 16)^2
        self.grid = config.IMAGE_SIZE // 16
        self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, d_model))
        nn.init.trunc_normal_(self.pos, std = 0.02)

        # Khối trộn token, đầu phân loại và đường tắt GAP
        self.mixer = TokenMixer(mixer, d_model, n_layers, self.grid, scan, d_state, headdim)
        self.head = ClassQueryHead(d_model, num_classes)
        self.skip = nn.Linear(1024, num_classes) if skip else None
        self.dropout = nn.Dropout(dropout)

    # Lan truyền xuôi (Forward)
    # Trả về: logits [batch_size, num_classes]
    def forward(self, x):

        # Kiểm tra dữ liệu đầu vào
        ## Ảnh đầu vào phải có dạng [batch_size, 3, H, W]
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError(
                "[CNNSymMamba]: Ảnh đầu vào phải có dạng [batch_size, 3, H, W].\n"
                f"Nhận được: x = {tuple(x.shape)}"
            )

        # Đặc trưng DenseNet-121 ở hai mức
        c3 = self.stage3(x)
        c4 = F.relu(self.stage4(c3))

        # Token đa tỉ lệ: (b, d_model, grid, grid) --> (b, grid * grid, d_model)
        tokens = F.silu(self.proj3(c3) + upsample2x(self.proj4(c4))).flatten(2).transpose(1, 2)

        # Kiểm tra số token
        ## Số token phải khớp position embedding (phụ thuộc IMAGE_SIZE)
        if tokens.size(1) != self.pos.size(1):
            raise ValueError(
                "[CNNSymMamba]: Số token phải khớp position embedding; kiểm tra IMAGE_SIZE.\n"
                f"Nhận được: Số token = {tokens.size(1)}, position embedding = {self.pos.size(1)}"
            )

        # Trộn token --> Class Query Head
        tokens = self.mixer(tokens + self.pos)
        logits = self.head(self.dropout(tokens))

        # Cộng đường tắt GAP (nếu có)
        if self.skip is not None:
            logits = logits + self.skip(self.dropout(c4.mean(dim = (2, 3))))
        return logits

# ============================================================
# Chạy huấn luyện
if __name__ == "__main__":

    # Đọc tham số dòng lệnh
    parser = argparse.ArgumentParser(description = "Huấn luyện CNN SymMamba và các biến thể đối chứng")
    parser.add_argument("variants", nargs = "*", default = ["sym4"], help = f"Một hoặc nhiều biến thể trong {list(VARIANTS)}, hoặc all")
    args = parser.parse_args()

    # Danh sách biến thể cần chạy
    names = list(VARIANTS) if "all" in args.variants else args.variants

    # Kiểm tra biến thể
    ## Biến thể phải hợp lệ
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        parser.error(f"Biến thể không hợp lệ: {unknown}. Chọn trong {list(VARIANTS)} hoặc all")

    # Chạy lần lượt từng biến thể với giao thức chung
    for name in names:
        start_train(model_name = model_name(name), dataset_class = NIHChestXrayDataset, model_factory = CNNSymMamba, model_kwargs = VARIANTS[name], **config.EXPERIMENT_PROTOCOL)
