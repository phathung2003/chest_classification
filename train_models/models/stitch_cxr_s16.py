# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import torch.nn as nn

import config
import models.stitch_cxr as stx
from models.vmamba_cxr import sha256_file

# ============================================================
# ConvNeXt-VMamba CXR bước 16 (ccmm_s16): biến thể chuỗi dài của mô hình lai (prereg/analysis_plan.md mục 7, dòng 2026-11-16 21:05, phần A)
# Mục tiêu: Ở 512 px stage 4 của ccmm chỉ quét 16 x 16 = 256 token (bước 32). Giữ bản đồ đặc trưng ở bước 16 để stage 4 quét 32 x 32 = 1.024 token,
# kiểm tra xem chuỗi dài hơn ở khối SS2D có nâng AUROC không (MambaOut, Yu & Wang, CVPR 2025: SSM chỉ có lợi ở chuỗi dài)
#
# Thay đổi duy nhất so với models/stitch_cxr.py (không thêm, không bớt tham số; cùng tên tham số nên nạp được checkpoint ccmm):
#   1. Conv 3 x 3 giảm mẫu giữa stage 3 và stage 4 của VMamba (layers[2].downsample): stride 2 --> 1.
#      Đầu ra ở các vị trí chẵn trùng đúng đầu ra của bản stride 2; các vị trí lẻ là mẫu xen giữa
#   2. Conv depthwise 3 x 3 trong các khối VSS của stage 4: dilation 2, padding 2. Trên lưới dày gấp đôi, hai token cách nhau 2 ô mới cách nhau
#      đúng khoảng cách điểm ảnh lúc tiền huấn luyện, nên trên mỗi lưới con (chẵn / lẻ) conv tính đúng phép tính đã tiền huấn luyện
#      (mạng dư giãn, Yu et al., CVPR 2017; "output stride 16" của DeepLab)
#   Phép quét SS2D không đổi mã: vẫn 4 tuyến hàng / cột, chỉ dài gấp 4 (bước quét giữa hai token liền nhau ngắn bằng nửa lúc tiền huấn luyện;
#   phần này không bù được bằng dilation và để việc tinh chỉnh xử lý)
#
# pinned_sha256 (models/run_arch.py tự điền): ngoài các tệp mà StitchCXR đã ghim (models/vmamba_cxr.py, tệp lớp nối) còn ghim models/stitch_cxr.py,
# vì chữ ký thí nghiệm chỉ hash tệp này trong khi gần hết phép tính nằm ở lớp cha

VARIANT = "s16"
STITCH_SOURCE = Path(stx.__file__).resolve()    # Mã của lớp cha StitchCXR
DOWNSAMPLE_STAGE = 2                            # layers[2].downsample dẫn vào stage 4 (chỉ số stage 0-3)
DILATED_STAGE = 3                               # Stage 4
DILATION = 2


# ============================================================
# Tên mô hình theo layout và độ phân giải (mặc định IMAGE_SIZE; dùng làm tên thư mục kết quả), ví dụ "ConvNeXt-VMamba CXR ccmm s16 512px"
def model_name(layout = "ccmm", size = None):
    return f"{stx.MODEL_NAME} {layout} {VARIANT} {size or config.IMAGE_SIZE}px"


# ============================================================
# Mô hình đầy đủ
class StitchS16CXR(stx.StitchCXR):
    def __init__(self, num_classes, layout = "ccmm", stitch_init = "lstsq", pinned_sha256 = None, grad_checkpoint = False, pretrained = True, init_from = None, drop_path_rate = 0.0):

        # Kiểm tra layout (stage 4 phải là VMamba) và mã lớp cha đã ghim
        if len(layout) != len(stx.CHANNELS) or layout[DILATED_STAGE] != "m":
            raise ValueError(f"layout '{layout}' không hợp lệ cho biến thể bước 16: stage 4 phải thuộc họ 'm'")
        pinned_sha256 = dict(pinned_sha256 or {})
        if STITCH_SOURCE.name in pinned_sha256 and sha256_file(STITCH_SOURCE) != pinned_sha256[STITCH_SOURCE.name]:
            raise RuntimeError(f"{STITCH_SOURCE}: SHA-256 khác bản đã ghim trong model_kwargs")

        # Mô hình ccmm đầy đủ (trọng số ImageNet, lớp nối, hoặc checkpoint init_from)
        super().__init__(num_classes, layout = layout, stitch_init = stitch_init, pinned_sha256 = pinned_sha256, grad_checkpoint = grad_checkpoint,
                         pretrained = pretrained, init_from = init_from, drop_path_rate = drop_path_rate)

        # 1. Bỏ lần giảm mẫu trước stage 4: conv 3 x 3 stride 2 --> stride 1
        reductions = [module for module in self.mamba.layers[DOWNSAMPLE_STAGE].downsample.modules() if isinstance(module, nn.Conv2d)]
        if len(reductions) != 1 or reductions[0].kernel_size != (3, 3) or reductions[0].stride != (2, 2):
            raise RuntimeError("Downsample trước stage 4 không phải một conv 3 x 3 stride 2 như VMamba-T [s1l8]")
        reductions[0].stride = (1, 1)

        # 2. Conv depthwise của các khối VSS stage 4: dilation 2, padding 2 (giữ kích thước bản đồ)
        for block in self.mamba.layers[DILATED_STAGE].blocks:
            depthwise = block.op.conv2d
            if depthwise.kernel_size != (3, 3) or depthwise.groups != depthwise.in_channels or depthwise.padding != (1, 1):
                raise RuntimeError("Conv của khối VSS stage 4 không phải conv depthwise 3 x 3 padding 1 như VMamba-T [s1l8]")
            depthwise.dilation = (DILATION, DILATION)
            depthwise.padding = (DILATION, DILATION)

    # FLOPs của phần quét selective scan: stage 4 quét (h / 16) x (w / 16) token thay vì (h / 32) x (w / 32), gấp 4 lần phần lớp cha đã tính cho stage này
    def uncounted_flops(self, height, width):
        tokens = (height // 32) * (width // 32)
        stage_four = sum(9 * tokens * block.op.k_group * block.op.d_inner * block.op.d_state for block in self.mamba.layers[DILATED_STAGE].blocks)
        return super().uncounted_flops(height, width) + 3 * stage_four
