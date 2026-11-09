# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Khai báo thư viện
import hashlib
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import config
import models.clis_mamba as clis
from utilities.file import load_torch

# ============================================================
# VMamba-T CXR: backbone SSM tiền huấn luyện ImageNet (phương án A, 2026-11-08)
# Mục tiêu: So sánh đường quét trên một backbone mà mọi khối trộn token đều là SS2D đã tiền huấn luyện, thay vì chèn khối SSM ngẫu nhiên vào CNN
# (v2: khối chèn chỉ đổi đặc trưng sau stage 3 khoảng 0,18%, mô hình trùng ConvNeXt-T)
# ** VMamba-T [s1l8] (Liu et al., NeurIPS 2024): 30,2 triệu tham số, ImageNet-1K 82,6% top-1 (ConvNeXt-T 82,1%), mã gốc models/third_party/vmamba (MIT)
# ** Bốn tuyến quét của VMamba (hàng xuôi, cột xuôi, hàng ngược, cột ngược) đúng bằng bảng tuyến M-ras của v2, nên M-ras ở đây là VMamba-T nguyên bản
#
# Kiến trúc:
#   1. VMamba-T: patch embed (stride 4) --> 4 stage (2, 2, 8, 2 khối VSS; 96, 192, 384, 768 kênh) --> LayerNorm --> pooling trung bình: (b, 768)
#   2. Linear 768 --> 14 lớp (thay đầu ImageNet 1000 lớp)
#   3. Ở các stage ROUTE_STAGES (mặc định stage 3, 4: lưới R / 16 và R / 32), SS2D quét bằng hoán vị gather (clis.RoutePermute) thay cho cross scan Triton.
#      vm_clis / vm_rand thêm 2 tuyến ghép cặp (CLIS ± / cặp ngẫu nhiên ±). Tham số của 2 tuyến mới chép từ tuyến hàng xuôi / hàng ngược,
#      riêng hàng C của x_proj và D đặt 0 nên tuyến mới cho đầu ra 0: ở bước 0 vm_clis / vm_rand trùng VMamba-T tiền huấn luyện
#      (cùng ý tưởng khởi tạo 0 phép chiếu ra của v2). C nhận gradient ngay từ bước đầu vì trạng thái ẩn h khác 0
#      route_init = "copy" (2026-11-10): chép đủ cả hàng C và D, tuyến mới hoạt động ngay từ bước 0 (không còn trùng VMamba-T tiền huấn luyện).
#      Lý do: sau huấn luyện full-data với "zero", ||tổng tuyến thêm|| chỉ bằng vài phần trăm ||tổng tuyến gốc|| (analysis/route_engagement.py)
#
# Nhóm LR: backbone (gồm tham số 2 tuyến mới) dùng LR backbone; đầu phân loại là nhóm lớp mới (models/run_vmamba.py: hệ số m* của ConvNeXt-T)
#
# Biến thể (tên dùng ở models/run_vmamba.py):
#   vm_ras  : VMamba-T nguyên bản (4 tuyến mọi stage)
#   vm_clis : 4 tuyến + CLIS ± ở ROUTE_STAGES (6 tuyến)
#   vm_rand : 4 tuyến + ghép cặp ngẫu nhiên ± ở ROUTE_STAGES (6 tuyến); cùng số tham số với vm_clis (đối chứng của H1a)

MODEL_NAME = "VMamba-T CXR"
VENDORED_PATH = ROOT_DIR / "models" / "third_party" / "vmamba" / "vmamba.py"
VENDORED_SHA256 = "10349394653c6e370d58f60919ad981d4e339c010b1684d3bb7280776bcf2671"          # NOTICE.md: commit 2ed52ea + 2 sửa đổi tránh cảnh báo deprecated
CHECKPOINT_PATH = ROOT_DIR / "pretrained" / "vmamba" / "vssm1_tiny_0230s_ckpt_epoch_264.pth"
CHECKPOINT_URL = "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth"
CHECKPOINT_SHA256 = "d855b57224a652bc692804932f88dc3a39b9028e64be99f745a7fd6792ca672a"
FEATURES = 768                                  # Số kênh sau stage 4 của VMamba-T
ROUTE_STAGES = (2, 3)                           # Chỉ số stage (0-3) quét bằng hoán vị gather; vm_clis / vm_rand thêm tuyến ở các stage này
BASE_ROUTES = 4                                 # Số tuyến của VMamba
ROUTE_INITS = ("zero", "copy")                  # Khởi tạo tuyến thêm: "zero" = hàng C và D bằng 0 (trùng VMamba-T ở bước 0); "copy" = chép đủ từ tuyến hàng


# ============================================================
# SHA-256 của một tệp
# Trả về: Chuỗi hex
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Mã VMamba nhúng phải đúng bản đã ghi (hash tệp mô hình này vào chữ ký thí nghiệm, nên chữ ký cũng ghim mã VMamba)
if sha256_file(VENDORED_PATH) != VENDORED_SHA256:
    raise ImportError(f"{VENDORED_PATH} khác bản đã ghi trong NOTICE.md (SHA-256 {VENDORED_SHA256})")
import models.third_party.vmamba.vmamba as vm


# ============================================================
# Bảng tuyến theo đúng thứ tự tham số của VMamba (cross scan, scans = 0): hàng xuôi, cột xuôi, hàng ngược, cột ngược; vm_clis / vm_rand thêm 2 tuyến
EXTRA_ROUTES = {
    "vm_ras"    : lambda h, w: [],
    "vm_clis"   : lambda h, w: [clis.perm_clis(h, w), clis.perm_clis(h, w).flip(0)],
    "vm_rand"   : lambda h, w: [clis.perm_random_pair(h, w), clis.perm_random_pair(h, w).flip(0)],
}
NAMES = tuple(EXTRA_ROUTES)


def vmamba_routes(h, w):
    raster, column = clis.perm_raster(h, w), clis.perm_column(h, w)
    return [raster, column, raster.flip(0), column.flip(0)]


# ============================================================
# Tên mô hình theo biến thể và độ phân giải (mặc định IMAGE_SIZE; dùng làm tên thư mục kết quả)
def model_name(name, size = None):
    return f"{MODEL_NAME} {name} {size or config.IMAGE_SIZE}px"


# ============================================================
# Bảng hoán vị (k, l) và hoán vị ngược của một SS2D trên lưới h x w (tạo một lần cho mỗi kích thước lưới)
# Trả về: (order, inverse)
def route_orders(op, h, w, device):
    key = (h, w, str(device))
    if key not in op.route_cache:
        order = torch.stack(vmamba_routes(h, w) + EXTRA_ROUTES[op.route_variant](h, w)).to(device)
        op.route_cache[key] = (order, torch.argsort(order, dim = 1))
    return op.route_cache[key]


# Phần lõi SS2D của VMamba (forward_corev2, kiểu v05_noz: không ép FP32, đầu ra quét FP32) với cross scan thay bằng hoán vị gather theo bảng tuyến.
# Tham số tuyến k (x_proj nhóm k, dt_projs nhóm k, dt_projs_bias[k], A_logs / Ds khối k) dùng cho hoán vị thứ k như VMamba
# Trả về: (b, d, h, w)
def route_forward_core(op, x):
    b, d, h, w = x.shape
    k, n, r, length = op.k_group, op.d_state, op.dt_rank, h * w
    order, inverse = route_orders(op, h, w, x.device)

    # Chuỗi theo từng tuyến --> chiếu ra dt, B, C theo nhóm tuyến
    xs = clis.RoutePermute.apply(x.flatten(2)[:, None].expand(b, k, d, length), order, inverse)        # (b, k, d, l)
    x_dbl = op.x_proj(xs.reshape(b, -1, length))
    dts, Bs, Cs = torch.split(x_dbl.view(b, k, -1, length), [r, n, n], dim = 2)
    dts = op.dt_projs(dts.contiguous().view(b, -1, length))

    # Selective scan một lần cho mọi tuyến (kernel mamba-ssm trên GPU; bản PyTorch của VMamba trên CPU)
    ys = vm.selective_scan_fn(xs.reshape(b, -1, length), dts.contiguous().view(b, -1, length), -op.A_logs.to(torch.float).exp(),
                              Bs.contiguous(), Cs.contiguous(), op.Ds.to(torch.float), op.dt_projs_bias.view(-1).to(torch.float),
                              True, True, backend = None if x.is_cuda else "torch")

    # Hoán vị ngược về thứ tự raster --> cộng các tuyến --> LayerNorm đầu ra như VMamba
    y = clis.RoutePermute.apply(ys.view(b, k, d, length), inverse, order).sum(1).view(b, d, h, w)
    return op.out_norm(y).to(x.dtype)


# Thêm tuyến cho một SS2D đã nạp trọng số: chép tham số tuyến hàng xuôi (0) / hàng ngược (2) sang tuyến mới
# - init "zero": đặt hàng C của x_proj và D bằng 0 (đầu ra tuyến mới = C·h + D·x = 0 --> SS2D trùng bản 4 tuyến ở bước 0)
# - init "copy": giữ nguyên bản chép (tuyến mới là tuyến hàng đã tiền huấn luyện, chỉ khác thứ tự quét)
def add_routes(op, extra, init = "zero"):
    if extra == 0:
        return
    k, d, n, r = op.k_group, op.d_inner, op.d_state, op.dt_rank
    source = [0, 2] * (extra // 2) + [0] * (extra % 2)

    with torch.no_grad():
        x_proj = op.x_proj.weight.view(k, r + 2 * n, d)
        new_x = x_proj[source].clone()
        if init == "zero":
            new_x[:, r + n:] = 0                                                    # Hàng C của tuyến mới
        dt_projs = op.dt_projs.weight.view(k, d, r)
        A_logs, Ds = op.A_logs.view(k, d, n), op.Ds.view(k, d)
        new_D = torch.zeros_like(Ds[source]) if init == "zero" else Ds[source].clone()

        total = k + extra
        op.x_proj = vm.Linear(d, total * (r + 2 * n), groups = total, bias = False, channel_first = True)
        op.x_proj.weight.copy_(torch.cat([x_proj, new_x]).view(-1, d))
        op.dt_projs = vm.Linear(r, total * d, groups = total, bias = False, channel_first = True)
        op.dt_projs.weight.copy_(torch.cat([dt_projs, dt_projs[source]]).view(-1, r))
        op.dt_projs_bias = nn.Parameter(torch.cat([op.dt_projs_bias.data, op.dt_projs_bias.data[source]]))
        op.A_logs = nn.Parameter(torch.cat([A_logs, A_logs[source]]).view(-1, n))
        op.Ds = nn.Parameter(torch.cat([Ds, new_D]).view(-1))                           # D của tuyến mới
    op.k_group = total


# ============================================================
# VMamba-T [s1l8] đúng cấu hình của vmamba_tiny_s1l8 (models/third_party/vmamba), drop path tùy chọn
def build_vssm(drop_path_rate):
    return vm.VSSM(
        depths = [2, 2, 8, 2], dims = 96, drop_path_rate = drop_path_rate, patch_size = 4, in_chans = 3, num_classes = 1000,
        ssm_d_state = 1, ssm_ratio = 1.0, ssm_dt_rank = "auto", ssm_act_layer = "silu", ssm_conv = 3, ssm_conv_bias = False, ssm_drop_rate = 0.0,
        ssm_init = "v0", forward_type = "v05_noz", mlp_ratio = 4.0, mlp_act_layer = "gelu", mlp_drop_rate = 0.0, gmlp = False,
        patch_norm = True, norm_layer = "ln2d", downsample_version = "v3", patchembed_version = "v2", use_checkpoint = False, posembed = False, imgsize = 224,
    )


# Trọng số ImageNet của VMamba-T: tải về pretrained/vmamba nếu chưa có, kiểm tra SHA-256, nạp không dùng pickle (weights_only)
# Trả về: state_dict
def imagenet_state():
    if not CHECKPOINT_PATH.is_file():
        CHECKPOINT_PATH.parent.mkdir(parents = True, exist_ok = True)
        torch.hub.download_url_to_file(CHECKPOINT_URL, str(CHECKPOINT_PATH))
    if sha256_file(CHECKPOINT_PATH) != CHECKPOINT_SHA256:
        raise RuntimeError(f"{CHECKPOINT_PATH}: SHA-256 khác bản đã ghi {CHECKPOINT_SHA256}")
    return torch.load(CHECKPOINT_PATH, map_location = "cpu", weights_only = True)["model"]


# ============================================================
# Mô hình đầy đủ
class VMambaCXR(nn.Module):
    def __init__(self, num_classes, mixer = "vm_clis", route_stages = ROUTE_STAGES, extra_routes = 2, drop_path_rate = 0.0,
                 grad_checkpoint = False, pretrained = True, init_from = None, route_init = "zero"):
        super().__init__()

        # Kiểm tra biến thể và cách khởi tạo tuyến thêm
        if mixer not in NAMES:
            raise ValueError(f"Biến thể '{mixer}' không hợp lệ: {list(NAMES)}")
        if route_init not in ROUTE_INITS:
            raise ValueError(f"route_init '{route_init}' không hợp lệ: {list(ROUTE_INITS)}")
        self.mixer = mixer
        self.grad_checkpoint = bool(grad_checkpoint)

        # 1. Backbone VMamba-T, trọng số ImageNet (bỏ đầu 1000 lớp); đầu phân loại tạo sau backbone
        self.backbone = build_vssm(drop_path_rate)
        if pretrained and init_from is None:
            self.backbone.load_state_dict(imagenet_state(), strict = True)
        self.backbone.classifier.head = nn.Identity()
        self.classifier = nn.Linear(FEATURES, num_classes)

        # 2. SS2D ở route_stages: quét bằng hoán vị gather; vm_clis / vm_rand thêm tuyến ghép cặp (sau khi nạp trọng số ImageNet)
        self.route_stages = tuple(int(stage) for stage in route_stages)
        extra = 0 if mixer == "vm_ras" else int(extra_routes)
        for stage in self.route_stages:
            for block in self.backbone.layers[stage].blocks:
                op = block.op
                op.route_variant, op.route_cache = mixer, {}
                add_routes(op, extra, route_init)
                op.forward_core = partial(route_forward_core, op)

        # 3. Nhóm LR backbone (training/optimizer.py): backbone ImageNet và tham số tuyến mới; đầu phân loại là nhóm lớp mới
        self.PRETRAINED_MODULES = ("backbone",)

        # 4. Khởi tạo từ best_model.pt cùng biến thể (ví dụ 512 px trước khi tinh chỉnh ở 1024 px)
        if init_from is not None:
            saved = load_torch(init_from)
            if saved["model_kwargs"].get("mixer") != mixer:
                raise ValueError(f"{init_from} là checkpoint của biến thể {saved['model_kwargs'].get('mixer')}, không phải {mixer}")
            self.load_state_dict(saved["model_state_dict"], strict = True)

    # Tham số không weight decay (ngoài tham số 1 chiều): A_logs của SSM, như VMamba / Mamba
    def no_weight_decay(self):
        return {name for name, _ in self.named_parameters() if name.rsplit(".", 1)[-1] in ("A_logs", "Ds", "dt_projs_bias")}

    # Đặc trưng sau pooling (b, 768); gradient checkpointing từng khối VSS khi huấn luyện (cùng phép tính, giảm VRAM)
    def features(self, x):
        backbone = self.backbone
        x = backbone.patch_embed(x)
        for layer in backbone.layers:
            for block in layer.blocks:
                x = checkpoint(block, x, use_reentrant = False) if self.grad_checkpoint and self.training and torch.is_grad_enabled() else block(x)
            x = layer.downsample(x)
        return backbone.classifier(x)

    def forward(self, x):
        return self.classifier(self.features(x))

    # Mức tham gia của tuyến thêm: trung bình |hàng C của x_proj| và |D| của tuyến mới theo stage (0 lúc khởi tạo); rỗng với vm_ras
    # Trả về: Danh sách (stage, |C|, |D|)
    def engagement(self):
        values = []
        for stage in self.route_stages:
            c_rows, d_values = [], []
            for block in self.backbone.layers[stage].blocks:
                op = block.op
                if op.k_group == BASE_ROUTES:
                    continue
                weight = op.x_proj.weight.detach().view(op.k_group, op.dt_rank + 2 * op.d_state, op.d_inner)
                c_rows.append(weight[BASE_ROUTES:, op.dt_rank + op.d_state:].abs().mean().item())
                d_values.append(op.Ds.detach().view(op.k_group, -1)[BASE_ROUTES:].abs().mean().item())
            if c_rows:
                values.append((stage, sum(c_rows) / len(c_rows), sum(d_values) / len(d_values)))
        return values

    # FLOPs của phần quét selective scan cho một ảnh (utilities/tracking.py cộng vào tổng FLOPs), cùng quy ước 9·L·K·D·N với v1 / v2
    def uncounted_flops(self, height, width):
        total = 0
        for stage, layer in enumerate(self.backbone.layers):
            tokens = (height // (4 * 2 ** stage)) * (width // (4 * 2 ** stage))
            total += sum(9 * tokens * block.op.k_group * block.op.d_inner * block.op.d_state for block in layer.blocks)
        return total
