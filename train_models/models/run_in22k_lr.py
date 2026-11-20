"""Dò learning rate cho nhánh ImageNet-22k (thăm dò, 19/11): ConvNeXt-T 22k và ccmm 22k với LR backbone khác 1e-4.

Vì sao file riêng: chữ ký thí nghiệm băm tệp mã nguồn của lớp mô hình. Lớp ConvNeXtTinyIn22k / StitchCXRIn22k vẫn nằm trong models/run_in22k.py
(không sửa), nên mô hình giống hệt nhánh 22k gốc; chỉ khác learning_rate (và tùy chọn hệ số LR đầu phân loại) trong training_config.
Thư mục kết quả có hậu tố riêng, ví dụ _in22k_lr5e5 (LR 5e-5, đầu x40) hay _in22k_lr5e5_h10 (đầu x10), không đụng thư mục _in22k.

Lý do: ở LR 1e-4, đầu x40, seed 42, cả hai mô hình 22k đạt đỉnh Validation ở epoch 4-5 rồi giảm (ConvNeXt-T 0,8603 → 0,831 ở epoch 8;
ccmm 0,8596 → 0,837). Dò đối xứng cho cả hai mô hình trên fold 1, seed 42; chọn LR theo macro AUROC Validation fold 1 của từng mô hình.

Ví dụ (WSL2, conda activate mamba-t29):
    python models/run_in22k_lr.py convnext_t --lr 5e-5 --seed 42
    python models/run_in22k_lr.py ccmm --lr 3e-5 --head-multiplier 10 --seed 42
"""
# Khai báo utilities
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Khai báo thư viện
import argparse

import config
import models.run_in22k as base                                                     # Đặt biến môi trường tất định (CUBLAS, Triton) khi import


# ============================================================
# Hậu tố thư mục theo LR: 5e-05 → "lr5e5"; đầu khác x40 thì thêm "_h<hệ số>"
# Trả về: Chuỗi
def run_tag(learning_rate, head_multiplier):
    mantissa, exponent = f"{learning_rate:.0e}".split("e")
    tag = f"{base.RUN_TAG}_lr{mantissa}e{abs(int(exponent))}"
    if head_multiplier != base.HEAD_LR:
        tag += f"_h{head_multiplier:g}"
    return tag


def main():
    parser = argparse.ArgumentParser(description = "Dò LR cho nhánh ImageNet-22k (thăm dò)")
    parser.add_argument("name", choices = ["convnext_t", "ccmm"])
    parser.add_argument("--lr", type = float, required = True, help = "LR backbone (bản gốc 1e-4)")
    parser.add_argument("--head-multiplier", type = float, default = base.HEAD_LR, help = "Hệ số LR đầu phân loại (bản gốc x40)")
    parser.add_argument("--seed", type = int, required = True)
    parser.add_argument("--fold", type = int, default = 1)
    args = parser.parse_args()

    # base.train đọc RUN_TAG, HEAD_LR và config.CLIS_PROTOCOL lúc gọi: đổi trong tiến trình này, không sửa tệp
    base.RUN_TAG = run_tag(args.lr, args.head_multiplier)
    base.HEAD_LR = float(args.head_multiplier)
    config.CLIS_PROTOCOL["optimizer_config"]["learning_rate"] = float(args.lr)
    print(f"Dò LR 22k | {args.name} | LR {args.lr:g} | đầu x{args.head_multiplier:g} | seed {args.seed} | hậu tố {base.RUN_TAG}")
    base.train(args.name, args.seed, args.fold)


if __name__ == "__main__":
    main()
