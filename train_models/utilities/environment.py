# ============================================================
# Kiểm tra môi trường chạy (Environment Check)
# Mục tiêu: Phát hiện ngay khi khởi chạy các lỗi do phiên bản thư viện / phần cứng không khớp (thay vì lỗi giữa chừng lúc huấn luyện), báo rõ nguyên nhân và cách sửa.
# ** Không thuộc PIPELINE_SOURCE_FILES (utilities/tracking.py): Sửa tệp này không đổi chữ ký thí nghiệm
# ** Đã kiểm thử: torch 2.4.0 + mamba-ssm 2.2.2 (env mamba-main), torch 2.9.1 + mamba-ssm 2.3.2.post1 (env mamba-t29)
# ** Không chặn cảnh báo: env mamba-t29 không phát FutureWarning nào; cảnh báo autotune Triton của mamba-ssm được xử lý bằng
#    TRITON_CACHE_AUTOTUNING = 1 ở đầu các tệp chạy (models/run_clis.py, analysis/benchmark_clis.py, tests/test_clis_*.py)
# ============================================================

import importlib
import os
import re
import sys

# Phiên bản tối thiểu của từng thư viện
MINIMUM_VERSIONS = {
    "torch"         : (2, 4),                   # torch.amp.GradScaler("cuda"), torch.nn.attention, weights_only của torch.load
    "torchvision"   : (0, 19),
    "timm"          : (0, 9),                   # Tên mô hình có tag (convnext_tiny.fb_in1k), features_only + set_grad_checkpointing
    "numpy"         : (1, 24),
    "pandas"        : (2, 0),
}
TESTED_TORCH = ("2.4", "2.9")                   # Bản torch đã chạy đủ G0, kiểm tra giao thức và e2e

# ============================================================
# Đổi chuỗi phiên bản ("2.9.1+cu128", "1.0.30", "2.3.2.post1") thành tuple số để so sánh
# Trả về: tuple số nguyên (phần không phải số bị bỏ)
def parse_version(text):

    # Lấy phần số đứng trước hậu tố build (+cu128) hoặc nhãn (rc, post, dev): "0rc1" --> 0, dừng ở đoạn có nhãn
    numbers = []
    for part in str(text).split("+")[0].split("."):
        digits = re.match(r"\d+", part)
        if digits is None:
            break
        numbers.append(int(digits.group()))
        if digits.group() != part:
            break

    return tuple(numbers)

# ============================================================
# Phiên bản của một thư viện (None nếu chưa cài)
# Trả về: Chuỗi phiên bản hoặc None
def package_version(name):
    try:
        return getattr(importlib.import_module(name), "__version__", "?")
    except Exception:
        return None

# ============================================================
# Kiểm tra môi trường trước khi huấn luyện / đo
# - require_cuda: Cần GPU CUDA. require_kernel: Cần kernel selective_scan_cuda của mamba-ssm (biến thể SS2D)
# - amp_dtype: "bfloat16" | "float16" | None. deterministic: Chế độ tất định (cần CUBLAS_WORKSPACE_CONFIG trước khi CUDA khởi tạo)
# Trả về: dict phiên bản / thiết bị (báo RuntimeError liệt kê mọi lỗi tìm thấy, không dừng ở lỗi đầu tiên)
def check_environment(require_cuda = True, require_kernel = False, amp_dtype = None, deterministic = None, adam_fused = False, verbose = True):

    problems, notes = [], []
    report = {"python": sys.version.split()[0]}

    # 1. Phiên bản Python và thư viện
    ## Python 3.10 trở lên
    if sys.version_info < (3, 10):
        problems.append(f"Python {report['python']} quá cũ: cần >= 3.10")

    ## Thư viện đủ phiên bản tối thiểu
    for name, minimum in MINIMUM_VERSIONS.items():
        version = package_version(name)
        report[name] = version
        if version is None:
            problems.append(f"Chưa cài {name}: pip install {name}")
        elif parse_version(version)[:len(minimum)] < minimum:
            problems.append(f"{name} {version} quá cũ: cần >= {'.'.join(map(str, minimum))}")

    import torch

    torch_version = parse_version(torch.__version__)
    report.update({"torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None})

    ## torchvision phải đúng cặp với torch (lệch cặp: lỗi "operator torchvision::nms does not exist" khi import)
    vision_version = parse_version(report["torchvision"] or "0")
    if report["torchvision"] and len(torch_version) >= 2 and len(vision_version) >= 2 and vision_version[1] - torch_version[1] != 15:
        problems.append(f"torchvision {report['torchvision']} không khớp torch {torch.__version__}: cài torchvision 0.{torch_version[1] + 15}.x cùng index CUDA với torch")

    ## timm có trọng số convnext_tiny.fb_in1k (backbone của CLIS-Mamba và baseline ConvNeXt-T)
    if report["timm"]:
        import timm
        if "convnext_tiny.fb_in1k" not in timm.list_models("convnext_tiny*", pretrained = True):
            problems.append(f"timm {report['timm']} không có convnext_tiny.fb_in1k: nâng timm (>= 0.9)")

    ## Bản torch chưa kiểm thử: Không chặn, nhắc chạy kiểm thử trước
    if ".".join(map(str, torch_version[:2])) not in TESTED_TORCH:
        notes.append(f"torch {torch.__version__} chưa nằm trong các bản đã kiểm thử {TESTED_TORCH}: chạy tests/test_clis_mamba.py và tests/test_clis_protocol.py trước")

    # 2. GPU CUDA
    if require_cuda:

        ## Phải thấy GPU (WSL2: driver NVIDIA cài phía Windows, không cài driver Linux trong WSL)
        if not torch.cuda.is_available():
            problems.append(f"Không thấy GPU CUDA (torch {torch.__version__}, CUDA {torch.version.cuda}): kiểm tra nvidia-smi và bản torch có CUDA")

        else:
            major, minor = torch.cuda.get_device_capability(0)
            report.update({"gpu": torch.cuda.get_device_name(0), "capability": f"sm_{major}{minor}"})

            ## torch phải có kernel cho kiến trúc GPU (thiếu: lỗi "no kernel image is available")
            try:
                (torch.ones(8, device = "cuda") * 2).sum().item()
            except RuntimeError as error:
                problems.append(f"torch không chạy được trên {report['gpu']} ({report['capability']}): {error}. Cài bản torch có kernel cho {report['capability']}")

            ## BF16 cần GPU Ampere trở lên
            if amp_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                problems.append(f"{report['gpu']} không hỗ trợ bfloat16: đặt AMP_DTYPE = 'float16'")

            ## Chế độ tất định cần CUBLAS_WORKSPACE_CONFIG trước khi CUDA khởi tạo (các tệp chạy đặt ở dòng đầu)
            if deterministic and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
                problems.append("DETERMINISTIC_ALGORITHMS = True nhưng thiếu CUBLAS_WORKSPACE_CONFIG = ':4096:8' (đặt trước khi import torch)")

            ## Fused AdamW lỗi với gradient conv theo kênh dạng channels-last trên torch 2.4
            if adam_fused and torch_version[:2] <= (2, 4):
                problems.append(f"ADAM_FUSED = True lỗi trên torch {torch.__version__} (stride gradient của depthwise conv): đặt ADAM_FUSED = False")

    # 3. Kernel selective scan của mamba-ssm (biến thể SS2D)
    report["mamba_ssm"] = package_version("mamba_ssm")
    if require_kernel:
        report["selective_scan_kernel"] = check_selective_scan(problems)

    # In tóm tắt và báo lỗi
    if verbose:
        kernel = {True: "kernel OK", False: "kernel LỖI", None: "không cần kernel"}[report.get("selective_scan_kernel")]
        print(f"Môi trường: Python {report['python']} | torch {torch.__version__} (CUDA {report['torch_cuda']}, cuDNN {report['cudnn']}) | "
              f"torchvision {report['torchvision']} | timm {report['timm']} | mamba-ssm {report['mamba_ssm']} ({kernel}) | "
              f"GPU {report.get('gpu', '-')} {report.get('capability', '')}", flush = True)
        for note in notes:
            print(f"CẢNH BÁO: {note}", flush = True)

    if problems:
        raise RuntimeError("[check_environment]: Môi trường chưa đúng, sửa các mục sau rồi chạy lại:\n- " + "\n- ".join(problems))

    return report

# ============================================================
# Kernel selective_scan_cuda chạy được và khớp bản tham chiếu PyTorch thuần trên một đầu vào nhỏ
# (phát hiện wheel mamba-ssm lệch ABI / lệch bản torch / thiếu kernel cho kiến trúc GPU)
# Trả về: True nếu đạt (lỗi được thêm vào problems)
def check_selective_scan(problems):

    # Import kernel và hàm quét
    try:
        importlib.import_module("selective_scan_cuda")                  # Bắt lỗi undefined symbol / thiếu tệp .so của kernel
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
    except Exception as error:
        problems.append(f"Không import được kernel selective_scan_cuda của mamba-ssm ({type(error).__name__}: {error}). "
                        "Cài wheel mamba-ssm đúng bản torch / CUDA / cxx11abi (ví dụ mamba_ssm-2.3.2.post1+cu12torch2.9cxx11abiTRUE), "
                        "hoặc MAMBA_KEEP_CUDA_BUILD=TRUE pip install mamba-ssm --no-build-isolation")
        return False

    import torch

    # Chạy kernel và bản tham chiếu trên cùng đầu vào nhỏ
    try:
        generator = torch.Generator(device = "cuda").manual_seed(0)
        def random(*shape):
            return torch.randn(*shape, device = "cuda", generator = generator)
        u, delta, B, C = random(2, 32, 64), random(2, 32, 64), random(2, 1, 16, 64), random(2, 1, 16, 64)
        A, D, delta_bias = -torch.rand(32, 16, device = "cuda", generator = generator) - 0.5, random(32), random(32)
        kernel = selective_scan_fn(u, delta, A, B, C, D, delta_bias = delta_bias, delta_softplus = True)
        reference = selective_scan_ref(u, delta, A, B, C, D, delta_bias = delta_bias, delta_softplus = True)
        error = (kernel - reference).abs().max().item()
    except Exception as error:
        problems.append(f"Kernel selective_scan_cuda lỗi khi chạy ({type(error).__name__}: {error}). Cài lại mamba-ssm đúng bản torch / kiến trúc GPU")
        return False

    # Kiểm tra sai số
    if not error < 1e-3:
        problems.append(f"Kernel selective_scan_cuda lệch bản tham chiếu {error:.2e} (ngưỡng 1e-3): chạy tests/test_clis_mamba.py để xem chi tiết")
        return False

    return True
