# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import timm
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import config
import models.vmamba_cxr as vmx
from utilities.file import load_torch

# ============================================================
# ConvNeXt-VMamba CXR: mô hình lai ghép tầng từ hai backbone tiền huấn luyện ImageNet (khám phá 2026-11-11)
# Mục tiêu: Một backbone lai CNN + Mamba mà MỌI khối đều tiền huấn luyện và nằm trên đường chính (không có nhánh tắt để bỏ qua khối Mamba như v3,
# không có khối trộn token khởi tạo ngẫu nhiên như v1). Mỗi stage chọn từ ConvNeXt-T ("c") hoặc VMamba-T ("m"); hai backbone cùng bề rộng
# 96 / 192 / 384 / 768 và cùng bước 4 / 8 / 16 / 32 nên ghép được ở mọi ranh giới stage.
# ** Căn cứ: CoAtNet (Dai et al., 2021) và MambaVision (Hatamizadeh & Kautz, CVPR 2025) đặt tích chập ở các stage độ phân giải cao, khối trộn token toàn cục
#    ở các stage sau; ghép mô hình tiền huấn luyện bằng lớp nối tuyến tính: Lenc & Vedaldi (2015), Bansal et al. (2021), SN-Net (Pan et al., CVPR 2023)
# ** Phép quét của SSM tuần tự theo số token: stage 1 / 2 của VMamba-T ở 512 px quét 16.384 / 4.096 token, stage 3 / 4 chỉ 1.024 / 256 token
#
# Kiến trúc (layout 4 ký tự, ký tự thứ i là họ của stage i; mặc định "ccmm"):
#   1. Đơn vị stage i của ConvNeXt-T: stem (i = 0) hoặc downsample i, rồi các khối stage i (timm: cnn.stem, cnn.stages[i])
#      Đơn vị stage i của VMamba-T  : patch embed (i = 0) hoặc downsample sau stage i - 1, rồi các khối VSS stage i (layers[i - 1].downsample, layers[i].blocks)
#      Cả hai nhận đặc trưng sau stage i - 1 và trả đặc trưng stage i, nên đổi họ giữa hai stage không cần đổi kích thước
#   2. Lớp nối (stitch) ở mỗi ranh giới đổi họ: conv 1 x 1 cùng số kênh, đưa đặc trưng của họ nguồn về không gian đặc trưng của họ đích.
#      stitch_init "lstsq": nạp nghiệm bình phương tối thiểu (analysis/fit_stitch.py, tệp trong pretrained/stitch) --> đơn vị kế tiếp nhận đầu vào gần với
#      đầu vào lúc tiền huấn luyện; "identity": ma trận đơn vị (đối chứng)
#      pinned_sha256 {tên tệp: SHA-256}: tệp lớp nối và models/vmamba_cxr.py (chữ ký thí nghiệm chỉ hash tệp này, nên các tệp mà phép tính phụ thuộc
#      được ghim qua model_kwargs; models/run_stitch.py tự điền). Tệp khác bản đã ghim thì báo lỗi
#   3. Chuẩn hóa + pooling cuối của họ ở stage 4 (ConvNeXt: GAP --> LayerNorm; VMamba: LayerNorm --> GAP): (b, 768) --> Linear 14 lớp
#   Các khối không dùng của hai backbone được thay bằng Identity (không tính tham số, không vào optimizer)
#
# Nhóm LR: hai backbone và lớp nối dùng LR backbone (lớp nối nằm giữa mạng, LR của đầu phân loại quá lớn với nó); đầu phân loại là nhóm lớp mới
#
# Layout dùng trong nghiên cứu:
#   cccc : ConvNeXt-T (cùng phép tính và cùng tên tham số với models/convnext_tiny.py)
#   mmmm : VMamba-T (cùng phép tính với vm_ras của models/vmamba_cxr.py)
#   ccmm : mô hình lai đề xuất, tích chập ở stage 1-2, SS2D ở stage 3-4
#   mmcc : đối chứng đảo thứ tự

MODEL_NAME = "ConvNeXt-VMamba CXR"
CHANNELS = (96, 192, 384, 768)                  # Số kênh sau stage 1-4 của cả ConvNeXt-T lẫn VMamba-T
FEATURES = CHANNELS[-1]
FAMILIES = ("c", "m")
STITCH_INITS = ("lstsq", "identity")
STITCH_DIRECTORY = ROOT_DIR / "pretrained" / "stitch"
VMAMBA_SOURCE = Path(vmx.__file__).resolve()    # Mã của các khối VMamba mà mô hình lai dùng lại


# ============================================================
# Tên mô hình theo layout và độ phân giải (mặc định IMAGE_SIZE; dùng làm tên thư mục kết quả)
def model_name(layout, size = None):
    return f"{MODEL_NAME} {layout} {size or config.IMAGE_SIZE}px"


# Các ranh giới đổi họ của một layout: chỉ số stage i (0-2) mà stage i và stage i + 1 khác họ
# Trả về: Danh sách chỉ số
def boundaries(layout):
    return [i for i in range(len(layout) - 1) if layout[i] != layout[i + 1]]


# Tệp nghiệm bình phương tối thiểu của lớp nối sau stage i, từ họ `source` sang họ `target` (analysis/fit_stitch.py)
# Trả về: Path
def stitch_path(source, target, stage):
    return STITCH_DIRECTORY / f"stitch_{source}2{target}_stage{stage + 1}.pt"


# ============================================================
# Mô hình đầy đủ
class StitchCXR(nn.Module):
    def __init__(self, num_classes, layout = "ccmm", stitch_init = "lstsq", pinned_sha256 = None, grad_checkpoint = False, pretrained = True, init_from = None, drop_path_rate = 0.0):
        super().__init__()

        # Kiểm tra layout, cách khởi tạo lớp nối và mã VMamba CXR đã ghim
        if len(layout) != len(CHANNELS) or any(family not in FAMILIES for family in layout):
            raise ValueError(f"layout '{layout}' không hợp lệ: cần {len(CHANNELS)} ký tự thuộc {list(FAMILIES)}")
        if stitch_init not in STITCH_INITS:
            raise ValueError(f"stitch_init '{stitch_init}' không hợp lệ: {list(STITCH_INITS)}")
        pinned_sha256 = dict(pinned_sha256 or {})
        if VMAMBA_SOURCE.name in pinned_sha256 and vmx.sha256_file(VMAMBA_SOURCE) != pinned_sha256[VMAMBA_SOURCE.name]:
            raise RuntimeError(f"{VMAMBA_SOURCE}: SHA-256 khác bản đã ghim trong model_kwargs")
        self.layout = layout
        self.grad_checkpoint = bool(grad_checkpoint)
        load_imagenet = bool(pretrained) and init_from is None

        # 1. Hai backbone tiền huấn luyện ImageNet; SS2D ở stage 3, 4 của VMamba quét bằng hoán vị gather như vm_ras
        #    Stochastic depth: mỗi backbone giữ lịch tuyến tính gốc của nó (tính trên đủ 4 stage), khối bị bỏ ở bước 2 không dùng tỉ lệ của mình
        self.cnn = timm.create_model("convnext_tiny.fb_in1k", pretrained = load_imagenet, num_classes = 0, drop_path_rate = drop_path_rate)
        self.mamba = vmx.VMambaCXR(num_classes, mixer = "vm_ras", pretrained = load_imagenet, drop_path_rate = drop_path_rate).backbone

        # 2. Bỏ các khối không nằm trên đường đi của layout
        for stage, family in enumerate(layout):
            if family != "c":
                self.cnn.stages[stage] = nn.Identity()
            if family != "m":
                self.mamba.layers[stage].blocks = nn.Identity()
            if stage > 0 and family != "m":
                self.mamba.layers[stage - 1].downsample = nn.Identity()                 # Downsample dẫn vào stage này thuộc đơn vị stage của VMamba
        if layout[0] != "c":
            self.cnn.stem = nn.Identity()
        if layout[0] != "m":
            self.mamba.patch_embed = nn.Identity()
        if layout[-1] != "c":
            self.cnn.head = nn.Identity()
        if layout[-1] != "m":
            self.mamba.classifier = nn.Identity()

        # 3. Lớp nối ở các ranh giới đổi họ
        self.stitches = nn.ModuleDict()
        for stage in boundaries(layout):
            stitch = nn.Conv2d(CHANNELS[stage], CHANNELS[stage], kernel_size = 1)
            with torch.no_grad():
                stitch.weight.copy_(torch.eye(CHANNELS[stage]).view(CHANNELS[stage], CHANNELS[stage], 1, 1))
                stitch.bias.zero_()
            if stitch_init == "lstsq" and load_imagenet:
                self.load_stitch(stitch, stage, pinned_sha256)
            self.stitches[str(stage)] = stitch

        # 4. Đầu phân loại
        self.classifier = nn.Linear(FEATURES, num_classes)

        # 5. Nhóm LR backbone (training/optimizer.py): hai backbone và lớp nối; đầu phân loại là nhóm lớp mới
        self.PRETRAINED_MODULES = ("cnn", "mamba", "stitches")

        # 6. Khởi tạo từ best_model.pt cùng layout (ví dụ 512 px trước khi tinh chỉnh ở 1024 px)
        if init_from is not None:
            saved = load_torch(init_from)
            if saved["model_kwargs"].get("layout") != layout:
                raise ValueError(f"{init_from} là checkpoint của layout {saved['model_kwargs'].get('layout')}, không phải {layout}")
            self.load_state_dict(saved["model_state_dict"], strict = True)

    # Nạp nghiệm bình phương tối thiểu của lớp nối sau stage `stage` (y = W·x + b theo kênh); pinned {tên tệp: SHA-256} rỗng thì không kiểm tra (dùng tay / kiểm thử)
    def load_stitch(self, stitch, stage, pinned):
        path = stitch_path(self.layout[stage], self.layout[stage + 1], stage)
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu {path}: chạy analysis/fit_stitch.py --layout {self.layout} trước, hoặc dùng stitch_init = 'identity'")
        if pinned and vmx.sha256_file(path) != pinned.get(path.name):
            raise RuntimeError(f"{path}: SHA-256 khác bản đã ghim trong model_kwargs")
        saved = torch.load(path, map_location = "cpu", weights_only = True)
        with torch.no_grad():
            stitch.weight.copy_(saved["weight"].view_as(stitch.weight))
            stitch.bias.copy_(saved["bias"])

    # Tham số không weight decay (ngoài tham số 1 chiều): A_logs, Ds, dt_projs_bias của SSM, như VMamba / Mamba
    def no_weight_decay(self):
        return {name for name, _ in self.named_parameters() if name.rsplit(".", 1)[-1] in ("A_logs", "Ds", "dt_projs_bias")}

    # Đơn vị stage i của họ layout[i]: (b, c[i - 1], h, w) --> (b, c[i], h / 2, w / 2); stage 0 nhận ảnh. Gradient checkpointing từng khối VSS khi huấn luyện
    def stage(self, index, x):
        if self.layout[index] == "c":
            return self.cnn.stages[0](self.cnn.stem(x)) if index == 0 else self.cnn.stages[index](x)
        layers = self.mamba.layers
        x = self.mamba.patch_embed(x) if index == 0 else layers[index - 1].downsample(x)
        for block in layers[index].blocks:
            x = checkpoint(block, x, use_reentrant = False) if self.grad_checkpoint and self.training and torch.is_grad_enabled() else block(x)
        return x

    # Đặc trưng sau stage `last` (0-3), đã qua lớp nối nếu ranh giới sau stage đó đổi họ và stitched = True
    def stage_features(self, x, last, stitched = True):
        for index in range(last + 1):
            x = self.stage(index, x)
            if str(index) in self.stitches and (index < last or stitched):
                x = self.stitches[str(index)](x)
        return x

    # Đặc trưng sau pooling (b, 768)
    def features(self, x):
        x = self.stage_features(x, len(self.layout) - 1)
        return self.cnn.head(self.cnn.norm_pre(x)) if self.layout[-1] == "c" else self.mamba.classifier(x)

    def forward(self, x):
        return self.classifier(self.features(x))

    # FLOPs của phần quét selective scan cho một ảnh ở các stage VMamba đang dùng (utilities/tracking.py cộng vào tổng FLOPs), quy ước 9·L·K·D·N như vm_ras
    def uncounted_flops(self, height, width):
        total = 0
        for stage, family in enumerate(self.layout):
            if family == "m":
                tokens = (height // (4 * 2 ** stage)) * (width // (4 * 2 ** stage))
                total += sum(9 * tokens * block.op.k_group * block.op.d_inner * block.op.d_state for block in self.mamba.layers[stage].blocks)
        return total
