# ============================================================
# Mô hình CNN BiMamba v2 (DenseNet-121 đa tỉ lệ + quét Mamba 2D + Class Query Head)
# Hướng đi: Trộn token đa tỉ lệ của DenseNet-121 bằng Mamba quét nhiều hướng trên lưới 2D, rồi phân loại bằng Class Query Head (mỗi lớp một query) cộng đường tắt GAP giống baseline.

# Kiến trúc (ảnh 224 x 224, USE_STRIDE16 = True, SCAN_MODE = "cross4"):
# - Token: Trộn stride 16 (denseblock3) và stride 32 (phóng lên) --> lưới 14 x 14 = 196 token [B, 196, 256].
# - Quét 4 hướng (hàng xuôi / ngược, cột xuôi / ngược) dùng chung một Mamba --> gộp 4 hướng --> Class Query Head + đường tắt GAP.
# ============================================================

# Khai báo utilities
import os
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Chế độ tất định của cuBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

from torchvision.models import densenet121, DenseNet121_Weights
from mambapy.mamba import Mamba, MambaConfig

from training.experiment import start_train
from utilities.dataset import NIHChestXrayDataset

# ============================================================
# Điều chỉnh Ablation (Đổi một thứ mỗi lần chạy)
USE_STRIDE16 = True       # True: Trộn token stride 16 (denseblock3) + stride 32 -> 14x14 = 196 token ở 224 | False: Chỉ stride 32 (7 x 7 = 49 token)
SEQ_MODEL    = "mamba"    # "mamba" | "transformer"
SCAN_MODE    = "cross4"   # "bi": Xuôi + ngược theo hàng | "cross4": Hàng + cột, xuôi + ngược
BASELINE_SKIP = True      # True: Đường tắt GAP 1024 kênh + Linear, y hệt baseline DenseNet121

# ============================================================
# Cài đặt mô hình
MODEL_NAME      = "CNN BiMamba_v2"                  # Tên mô hình
MAMBA_DIM       = 256                               # Số chiều của token
MAMBA_LAYERS    = 2                                 # Số lớp Mamba / Transformer
DROPOUT         = 0.2                               # Dropout trước đầu phân loại
dataset_class   = NIHChestXrayDataset               # Lớp Dataset
STRIDE          = 16 if USE_STRIDE16 else 32        # Bước của lưới token so với ảnh
FEAT_HW         = config.IMAGE_SIZE // STRIDE       # Cạnh của lưới token (14 ở 224 px, stride 16)

# ============================================================
# Kiểm tra công tắc ablation
## SEQ_MODEL phải là mamba hoặc transformer
if SEQ_MODEL not in ("mamba", "transformer"):
    raise ValueError(
        "[cnn_bimamba]: SEQ_MODEL phải là mamba hoặc transformer.\n"
        f"Nhận được: SEQ_MODEL = {SEQ_MODEL!r}"
    )

## SCAN_MODE phải là bi hoặc cross4
if SCAN_MODE not in ("bi", "cross4"):
    raise ValueError(
        "[cnn_bimamba]: SCAN_MODE phải là bi hoặc cross4.\n"
        f"Nhận được: SCAN_MODE = {SCAN_MODE!r}"
    )

# Tên mô hình ghép từ điều chỉnh
_tag = (f"s{STRIDE}_{SEQ_MODEL}" + (f"_{SCAN_MODE}" if SEQ_MODEL == "mamba" else "") + ("_skip" if BASELINE_SKIP else ""))
MODEL_NAME = f"{MODEL_NAME}_{_tag}"

# ============================================================
# Quét 2D: Các hướng dùng CHUNG một Mamba (xếp vào chiều batch), có embedding hướng
# - d: Số chiều token, n_layers: Số lớp Mamba, hw: Cạnh của lưới token, mode: "bi" (2 hướng) | "cross4" (4 hướng).
class CrossScanMamba(nn.Module):

    # Khởi tạo khối quét
    def __init__(self, d, n_layers, hw, mode):
        super().__init__()
        self.hw, self.mode = hw, mode                                       # Cạnh của lưới token, kiểu quét
        self.mamba = Mamba(MambaConfig(d_model = d, n_layers = n_layers))   # Mamba dùng chung cho mọi hướng
        n_dir = 4 if mode == "cross4" else 2                                # Số hướng quét
        self.dir_embed = nn.Parameter(torch.zeros(n_dir, 1, 1, d))          # Embedding riêng của từng hướng quét
        self.fuse = nn.Linear(n_dir * d, d)                                 # Gộp đầu ra các hướng
        self.norm = nn.LayerNorm(d)                                         # Chuẩn hóa sau residual

    # Lan truyền xuôi (Forward)
    # Trả về: Token đã trộn [B, N, D]
    def forward(self, x):                                                   # x: [B, N, D], N = H*W

        # Kích thước token và lưới
        B, N, D = x.shape
        H = W = self.hw

        # Các chuỗi quét: Hàng xuôi, hàng ngược (+ cột xuôi, cột ngược)
        sequences = [x, x.flip(1)]
        if self.mode == "cross4":
            column_tokens = x.reshape(B, H, W, D).transpose(1, 2).reshape(B, N, D)   # Theo cột
            sequences += [column_tokens, column_tokens.flip(1)]

        # Mọi hướng qua cùng một Mamba (xếp vào chiều batch): [K, B, N, D] --> [K * B, N, D]
        stacked = torch.stack(sequences, dim = 0) + self.dir_embed            # [K, B, N, D]
        K = stacked.size(0)
        mixed = self.mamba(stacked.reshape(K * B, N, D)).reshape(K, B, N, D)

        # Đưa đầu ra mỗi hướng về thứ tự token gốc (theo hàng)
        outputs = [mixed[0], mixed[1].flip(1)]
        if self.mode == "cross4":
            to_row_order = lambda tokens: tokens.reshape(B, W, H, D).transpose(1, 2).reshape(B, N, D)
            outputs += [to_row_order(mixed[2]), to_row_order(mixed[3].flip(1))]

        # Gộp các hướng, cộng residual, chuẩn hóa
        return self.norm(x + self.fuse(torch.cat(outputs, dim = -1)))

# ============================================================
# Mô hình đối chứng: Transformer (encoder cùng số lớp, cùng chiều)
class TransformerMixer(nn.Module):

    # Khởi tạo Transformer encoder
    def __init__(self, d, n_layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model = d, nhead = 4, dim_feedforward = 2 * d, dropout = 0.1,
            activation = "gelu", batch_first = True, norm_first = True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor = False)
        self.norm = nn.LayerNorm(d)

    # Lan truyền xuôi (Forward)
    # Trả về: Token đã trộn [B, N, D]
    def forward(self, x):
        return self.norm(self.enc(x))

# ============================================================
# Class Query Head
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
    def forward(self, x):                                   # [B, N, D]
        kv = self.norm(x)
        q = self.q.unsqueeze(0).expand(x.size(0), -1, -1)
        out, _ = self.attn(q, kv, kv)                       # [B, C, D]
        return (out * self.w).sum(-1) + self.b              # [B, C]

# ============================================================
# Mô hình: DenseNet121 (đa tỉ lệ) + Mixer chuỗi + Class Query Head + đường tắt baseline
class CNNMamba(nn.Module):

    # Khởi tạo mô hình
    def __init__(self, num_classes):
        super().__init__()

        # DenseNet-121 tiền huấn luyện ImageNet (phần trích đặc trưng)
        f = densenet121(weights = DenseNet121_Weights.IMAGENET1K_V1).features

        # Tách hai mức đặc trưng và chiếu xuống MAMBA_DIM kênh
        if USE_STRIDE16:
            self.stage3, self.stage4 = f[:9], f[9:]         # Đến hết denseblock3 | transition3 -> norm5
            self.proj3 = nn.Sequential(                     # denseblock3 chưa chuẩn hóa -> BN-ReLU-Conv-BN
                nn.BatchNorm2d(1024), nn.ReLU(inplace = False),
                nn.Conv2d(1024, MAMBA_DIM, 1, bias = False), nn.BatchNorm2d(MAMBA_DIM))
        else:
            self.stage3, self.stage4 = nn.Identity(), f
            self.proj3 = None
        self.proj4 = nn.Sequential(
            nn.Conv2d(1024, MAMBA_DIM, 1, bias = False), nn.BatchNorm2d(MAMBA_DIM))

        # Position embedding của lưới token
        self.pos = nn.Parameter(torch.zeros(1, FEAT_HW * FEAT_HW, MAMBA_DIM))
        nn.init.trunc_normal_(self.pos, std = 0.02)

        # Khối trộn token và đầu phân loại
        if SEQ_MODEL == "mamba":
            self.mixer = CrossScanMamba(MAMBA_DIM, MAMBA_LAYERS, FEAT_HW, SCAN_MODE)
        else:
            self.mixer = TransformerMixer(MAMBA_DIM, MAMBA_LAYERS)
        self.head = ClassQueryHead(MAMBA_DIM, num_classes)

        # Đường tắt: GAP 1024 kênh (giống baseline) hoặc GAP của token
        self.skip = nn.Linear(1024 if BASELINE_SKIP else MAMBA_DIM, num_classes)
        self.dropout = nn.Dropout(DROPOUT)

    # Lan truyền xuôi (Forward)
    # Trả về: logits [batch_size, num_classes]
    def forward(self, x):

        # Kiểm tra dữ liệu đầu vào
        ## Ảnh đầu vào phải có dạng [batch_size, 3, H, W]
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError(
                "[CNNMamba]: Ảnh đầu vào phải có dạng [batch_size, 3, H, W].\n"
                f"Nhận được: x = {tuple(x.shape)}"
            )

        # Đặc trưng DenseNet-121 ở hai mức
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        feat4 = F.relu(c4)                                  # [B, 1024, H/32, W/32]
        p4 = self.proj4(feat4)

        # Token đa tỉ lệ: Phóng stride 32 lên stride 16 rồi cộng (hoặc chỉ stride 32)
        if self.proj3 is not None:
            p3 = self.proj3(c3)                             # [B, D, H/16, W/16]
            p4 = F.interpolate(p4, size = p3.shape[-2:], mode = "bilinear", align_corners = False)
            t = F.silu(p3 + p4)
        else:
            t = F.silu(p4)

        # Đầu vào của đường tắt và chuỗi token
        skip_in = feat4.mean(dim = (2, 3)) if BASELINE_SKIP else t.mean(dim = (2, 3))
        tokens = t.flatten(2).transpose(1, 2)               # [B, N, D]

        # Kiểm tra số token
        ## Số token phải khớp position embedding (phụ thuộc IMAGE_SIZE)
        if tokens.size(1) != self.pos.size(1):
            raise ValueError(
                "[CNNMamba]: Số token phải khớp position embedding; kiểm tra IMAGE_SIZE.\n"
                f"Nhận được: Số token = {tokens.size(1)}, position embedding = {self.pos.size(1)}"
            )

        # Trộn token --> Class Query Head + đường tắt
        tokens = self.mixer(tokens + self.pos)

        return self.head(self.dropout(tokens)) + self.skip(self.dropout(skip_in))

# ============================================================
# Chạy huấn luyện
if __name__ == "__main__":
    start_train(
        model_name     = MODEL_NAME,
        dataset_class  = NIHChestXrayDataset,
        model_factory  = CNNMamba,
        model_kwargs   = {},
        **config.EXPERIMENT_PROTOCOL,
    )
