# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

from torchvision.models import densenet121, DenseNet121_Weights
from mambapy.mamba import Mamba, MambaConfig

from training.models import run_experiment
from utilities.dataset import NIHChestXrayDataset

# ============================================================
# CÔNG TẮC ABLATION: đổi MỘT thứ mỗi lần chạy. MODEL_NAME tự đổi theo.
USE_STRIDE16  = True       # True: trộn token stride 16 (denseblock3) + stride 32 -> 14x14=196 token ở 224
                           # False: chỉ stride 32 (7x7=49 token, như v1)
SEQ_MODEL     = "mamba"    # "mamba" | "transformer" (đối chứng: cùng số token, cùng chiều, cùng số lớp)
SCAN_MODE     = "cross4"   # "bi": xuôi + ngược theo hàng (như v1) | "cross4": hàng + cột, xuôi + ngược
BASELINE_SKIP = True       # True: đường tắt GAP 1024 kênh + Linear, y hệt baseline DenseNet121

MAMBA_DIM = 256
MAMBA_LAYERS = 2
DROPOUT = 0.2
dataset_class = NIHChestXrayDataset

STRIDE = 16 if USE_STRIDE16 else 32
FEAT_HW = config.IMAGE_SIZE // STRIDE
assert SEQ_MODEL in ("mamba", "transformer")
assert SCAN_MODE in ("bi", "cross4")

_tag = (f"s{STRIDE}_{SEQ_MODEL}"
        + (f"_{SCAN_MODE}" if SEQ_MODEL == "mamba" else "")
        + ("_skip" if BASELINE_SKIP else ""))
MODEL_NAME = f"CNN BiMamba_v2_{_tag}"

# ============================================================
# Quét 2D: các hướng dùng CHUNG một Mamba (xếp vào chiều batch), có embedding hướng
class CrossScanMamba(nn.Module):
    def __init__(self, d, n_layers, hw, mode):
        super().__init__()
        self.hw, self.mode = hw, mode
        self.mamba = Mamba(MambaConfig(d_model=d, n_layers=n_layers))
        n_dir = 4 if mode == "cross4" else 2
        self.dir_embed = nn.Parameter(torch.zeros(n_dir, 1, 1, d))
        self.fuse = nn.Linear(n_dir * d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):                                   # x: [B, N, D], N = H*W
        B, N, D = x.shape
        H = W = self.hw
        seqs = [x, x.flip(1)]                               # hàng xuôi, hàng ngược
        if self.mode == "cross4":
            xt = x.reshape(B, H, W, D).transpose(1, 2).reshape(B, N, D)   # theo cột
            seqs += [xt, xt.flip(1)]
        s = torch.stack(seqs, dim=0) + self.dir_embed       # [K, B, N, D]
        K = s.size(0)
        y = self.mamba(s.reshape(K * B, N, D)).reshape(K, B, N, D)

        outs = [y[0], y[1].flip(1)]
        if self.mode == "cross4":
            back = lambda t: t.reshape(B, W, H, D).transpose(1, 2).reshape(B, N, D)
            outs += [back(y[2]), back(y[3].flip(1))]
        return self.norm(x + self.fuse(torch.cat(outs, dim=-1)))

# ============================================================
# Đối chứng: Transformer encoder cùng số lớp, cùng chiều
class TransformerMixer(nn.Module):
    def __init__(self, d, n_layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=2 * d, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.enc(x))

# ============================================================
# Class Query Head (giữ nguyên v1)
class ClassQueryHead(nn.Module):
    def __init__(self, d, num_classes, heads=4, dropout=0.1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(num_classes, d) * 0.02)
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.w = nn.Parameter(torch.randn(num_classes, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):                                   # [B, N, D]
        kv = self.norm(x)
        q = self.q.unsqueeze(0).expand(x.size(0), -1, -1)
        out, _ = self.attn(q, kv, kv)                       # [B, C, D]
        return (out * self.w).sum(-1) + self.b              # [B, C]

# ============================================================
# Mô hình: DenseNet121 (đa tỉ lệ) + Mixer chuỗi + Class Query Head + đường tắt baseline
class CNNMamba(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        f = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).features
        if USE_STRIDE16:
            self.stage3, self.stage4 = f[:9], f[9:]         # đến hết denseblock3 | transition3 -> norm5
            self.proj3 = nn.Sequential(                     # denseblock3 chưa chuẩn hóa -> BN-ReLU-Conv-BN
                nn.BatchNorm2d(1024), nn.ReLU(inplace=False),
                nn.Conv2d(1024, MAMBA_DIM, 1, bias=False), nn.BatchNorm2d(MAMBA_DIM))
        else:
            self.stage3, self.stage4 = nn.Identity(), f
            self.proj3 = None
        self.proj4 = nn.Sequential(
            nn.Conv2d(1024, MAMBA_DIM, 1, bias=False), nn.BatchNorm2d(MAMBA_DIM))

        self.pos = nn.Parameter(torch.zeros(1, FEAT_HW * FEAT_HW, MAMBA_DIM))
        nn.init.trunc_normal_(self.pos, std=0.02)

        if SEQ_MODEL == "mamba":
            self.mixer = CrossScanMamba(MAMBA_DIM, MAMBA_LAYERS, FEAT_HW, SCAN_MODE)
        else:
            self.mixer = TransformerMixer(MAMBA_DIM, MAMBA_LAYERS)
        self.head = ClassQueryHead(MAMBA_DIM, num_classes)

        self.skip = nn.Linear(1024 if BASELINE_SKIP else MAMBA_DIM, num_classes)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        feat4 = F.relu(c4)                                  # [B, 1024, H/32, W/32]
        p4 = self.proj4(feat4)
        if self.proj3 is not None:
            p3 = self.proj3(c3)                             # [B, D, H/16, W/16]
            p4 = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
            t = F.silu(p3 + p4)
        else:
            t = F.silu(p4)

        skip_in = feat4.mean(dim=(2, 3)) if BASELINE_SKIP else t.mean(dim=(2, 3))
        tokens = t.flatten(2).transpose(1, 2)               # [B, N, D]
        if tokens.size(1) != self.pos.size(1):
            raise ValueError(f"Số token {tokens.size(1)} khác pos {self.pos.size(1)}; kiểm tra IMAGE_SIZE.")
        tokens = self.mixer(tokens + self.pos)

        return self.head(self.dropout(tokens)) + self.skip(self.dropout(skip_in))

LOSS_CONFIG = {
    "name": "asl",
    "gamma_neg": 4.0,
    "gamma_pos": 0.0,
    "clip": 0.05,
}

if __name__ == "__main__":
    run_experiment(
        model_name      = MODEL_NAME,
        dataset_class   = NIHChestXrayDataset,
        model_factory   = CNNMamba,
        model_kwargs    = {},
        loss_config     = LOSS_CONFIG,
    )