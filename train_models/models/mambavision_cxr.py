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
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint

import config
from models.vmamba_cxr import sha256_file
from utilities.file import load_torch

# ============================================================
# MambaVision-T CXR: đối chứng lai CNN-Mamba tiền huấn luyện trọn khối (prereg/analysis_plan.md mục 7, dòng 2026-11-16 21:05, phần B)
# Mục tiêu: So ccmm (models/stitch_cxr.py, ghép hai backbone tiền huấn luyện riêng bằng lớp nối) với mô hình lai gần nó nhất đã công bố,
# trong cùng giao thức huấn luyện. Đây là đối chứng, không phải mô hình đề xuất
# ** MambaVision-T (Hatamizadeh & Kautz, CVPR 2025): 31,8 triệu tham số, ImageNet-1K 82,3% top-1, mã gốc models/third_party/mambavision (NVIDIA Source Code License-NC)
#
# Kiến trúc:
#   1. Patch embed tích chập (stride 4, 80 kênh) --> stage 1, 2 tích chập (1, 3 khối ConvBlock + BatchNorm; 80, 160 kênh)
#   2. Stage 3, 4 (8, 4 khối; 320, 640 kênh): nửa đầu là khối MambaVisionMixer (selective scan một chiều theo thứ tự raster, d_state 8, nhánh z không quét),
#      nửa sau là khối self-attention
#   3. BatchNorm --> pooling trung bình: (b, 640) --> Linear 640 --> 14 lớp (thay đầu ImageNet 1000 lớp)
#
# Cửa sổ ở stage 3, 4: mã gốc chia bản đồ đặc trưng thành cửa sổ vuông rồi trộn token trong từng cửa sổ. Ở đây cửa sổ phủ CẢ bản đồ đặc trưng
# (32 x 32 và 16 x 16 token ở 512 px), giống lúc tiền huấn luyện ở 224 px (cửa sổ 14, 7 = cả bản đồ) và giống cấu hình 512 px của tác giả
# (window_size = [8, 8, 32, 16]). Khi cửa sổ bằng cả bản đồ, chia cửa sổ chỉ còn là trải bản đồ thành chuỗi raster, nên features() trải trực tiếp
# (kiểm thử: trùng forward của mã gốc)
#
# Selective scan: kernel CUDA của mamba-ssm trên GPU như mã gốc; trên CPU dùng bản tham chiếu PyTorch của mamba-ssm (kiểm thử không cần GPU)
# Nhóm LR: backbone dùng LR backbone; đầu phân loại là nhóm lớp mới (models/run_arch.py: hệ số m* của ConvNeXt-T, như VMamba-T và ccmm)

MODEL_NAME = "MambaVision-T CXR"
VENDORED_PATH = ROOT_DIR / "models" / "third_party" / "mambavision" / "mamba_vision.py"
VENDORED_SHA256 = "11e5a6bf62a6b8dd75e7521fb282f2752a4ce643ec7212f833b09ca643d3910b"          # NOTICE.md: commit 42a4026 + 3 sửa đổi dòng import
CHECKPOINT_PATH = ROOT_DIR / "pretrained" / "mambavision" / "mambavision_tiny_1k.safetensors"
CHECKPOINT_URL = "https://huggingface.co/nvidia/MambaVision-T-1K/resolve/b1de77e17599566d98efb701c0231b1095dc3a67/model.safetensors"
CHECKPOINT_SHA256 = "4f6987be0a2ca2222f386eb750d028a05203b047d3c8dfb664c27e2295d02fc0"
CHECKPOINT_PREFIX = "model."                    # Tệp của Hugging Face bọc mô hình trong thuộc tính `model`
DEPTHS = (1, 3, 8, 4)                           # Cấu hình mamba_vision_T của mã gốc
NUM_HEADS = (2, 4, 8, 16)
DIM = 80
IN_DIM = 32
MLP_RATIO = 4
PRETRAIN_WINDOWS = (8, 8, 14, 7)                # Cửa sổ lúc tiền huấn luyện ở 224 px (stage 1, 2 tích chập không dùng)
FEATURES = DIM * 2 ** (len(DEPTHS) - 1)         # 640 kênh sau stage 4

# Mã MambaVision nhúng phải đúng bản đã ghi (chữ ký thí nghiệm hash tệp mô hình này, nên chữ ký cũng ghim mã MambaVision)
if sha256_file(VENDORED_PATH) != VENDORED_SHA256:
    raise ImportError(f"{VENDORED_PATH} khác bản đã ghi trong NOTICE.md (SHA-256 {VENDORED_SHA256})")
import models.third_party.mambavision.mamba_vision as mv
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as kernel_scan, selective_scan_ref as reference_scan


# ============================================================
# Selective scan của MambaVisionMixer: kernel CUDA của mamba-ssm trên GPU (đúng lời gọi của mã gốc); bản tham chiếu PyTorch ở FP32 trên CPU
# Trả về: (b, d, l) cùng kiểu dữ liệu với u
def selective_scan(u, delta, A, B, C, D = None, z = None, delta_bias = None, delta_softplus = False, return_last_state = None):
    if u.is_cuda:
        return kernel_scan(u, delta, A, B, C, D, z = z, delta_bias = delta_bias, delta_softplus = delta_softplus, return_last_state = return_last_state)
    y = reference_scan(u.float(), delta.float(), A.float(), B.float(), C.float(), None if D is None else D.float(), z = z,
                       delta_bias = None if delta_bias is None else delta_bias.float(), delta_softplus = delta_softplus)
    return y.to(u.dtype)


mv.selective_scan_fn = selective_scan


# ============================================================
# Tên mô hình theo độ phân giải (mặc định IMAGE_SIZE; dùng làm tên thư mục kết quả)
def model_name(size = None):
    return f"{MODEL_NAME} {size or config.IMAGE_SIZE}px"


# MambaVision-T đúng cấu hình mamba_vision_T của mã gốc, không đầu phân loại; drop path tùy chọn (mã gốc tiền huấn luyện với 0,2)
def build_backbone(drop_path_rate):
    return mv.MambaVision(depths = list(DEPTHS), num_heads = list(NUM_HEADS), window_size = list(PRETRAIN_WINDOWS), dim = DIM, in_dim = IN_DIM,
                          mlp_ratio = MLP_RATIO, drop_path_rate = drop_path_rate, num_classes = 0)


# Trọng số ImageNet của MambaVision-T: tải về pretrained/mambavision nếu chưa có, kiểm tra SHA-256, nạp từ safetensors (không dùng pickle), bỏ đầu 1000 lớp
# Trả về: state_dict của backbone
def imagenet_state():
    if not CHECKPOINT_PATH.is_file():
        CHECKPOINT_PATH.parent.mkdir(parents = True, exist_ok = True)
        torch.hub.download_url_to_file(CHECKPOINT_URL, str(CHECKPOINT_PATH))
    if sha256_file(CHECKPOINT_PATH) != CHECKPOINT_SHA256:
        raise RuntimeError(f"{CHECKPOINT_PATH}: SHA-256 khác bản đã ghi {CHECKPOINT_SHA256}")
    state = {name.removeprefix(CHECKPOINT_PREFIX): value for name, value in load_file(str(CHECKPOINT_PATH)).items()}
    return {name: value for name, value in state.items() if not name.startswith("head.")}


# ============================================================
# Mô hình đầy đủ
class MambaVisionCXR(nn.Module):
    def __init__(self, num_classes, drop_path_rate = 0.0, grad_checkpoint = False, pretrained = True, init_from = None):
        super().__init__()
        self.grad_checkpoint = bool(grad_checkpoint)

        # 1. Backbone MambaVision-T, trọng số ImageNet (đã bỏ đầu 1000 lớp)
        self.backbone = build_backbone(drop_path_rate)
        if pretrained and init_from is None:
            self.backbone.load_state_dict(imagenet_state(), strict = True)

        # 2. Đầu phân loại
        self.classifier = nn.Linear(FEATURES, num_classes)

        # 3. Nhóm LR backbone (training/optimizer.py); đầu phân loại là nhóm lớp mới
        self.PRETRAINED_MODULES = ("backbone",)

        # 4. Khởi tạo từ best_model.pt của chính mô hình này (ví dụ 512 px trước khi tinh chỉnh ở 1024 px)
        if init_from is not None:
            self.load_state_dict(load_torch(init_from)["model_state_dict"], strict = True)

    # Tham số không weight decay (ngoài tham số 1 chiều): A_log và D của SSM, theo cờ _no_weight_decay mà mã gốc gắn sẵn
    def no_weight_decay(self):
        return {name for name, parameter in self.named_parameters() if getattr(parameter, "_no_weight_decay", False)}

    # Một stage 3 / 4 với cửa sổ phủ cả bản đồ: (b, c, h, w) --> chuỗi raster (b, h * w, c) --> các khối --> (b, c, h, w) --> downsample (nếu có)
    # Gradient checkpointing từng khối khi huấn luyện (cùng phép tính, giảm VRAM)
    def token_stage(self, level, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        for block in level.blocks:
            tokens = checkpoint(block, tokens, use_reentrant = False) if self.grad_checkpoint and self.training and torch.is_grad_enabled() else block(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x if level.downsample is None else level.downsample(x)

    # Đặc trưng sau pooling (b, 640)
    def features(self, x):
        backbone = self.backbone
        x = backbone.patch_embed(x)
        for level in backbone.levels:
            x = self.token_stage(level, x) if level.transformer_block else level(x)
        return backbone.norm(x).mean((2, 3))

    def forward(self, x):
        return self.classifier(self.features(x))

    # FLOPs của phần quét selective scan cho một ảnh (utilities/tracking.py cộng vào tổng FLOPs), quy ước 9·L·D·N như các mô hình SSM khác của repo:
    # mỗi khối MambaVisionMixer quét một tuyến trên d_inner / 2 kênh với d_state trạng thái
    def uncounted_flops(self, height, width):
        total = 0
        for stage, level in enumerate(self.backbone.levels):
            tokens = (height // (4 * 2 ** stage)) * (width // (4 * 2 ** stage))
            total += sum(9 * tokens * (block.mixer.d_inner // 2) * block.mixer.d_state
                         for block in level.blocks if isinstance(getattr(block, "mixer", None), mv.MambaVisionMixer))
        return total
