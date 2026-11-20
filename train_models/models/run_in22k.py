"""Nhánh độ nhạy ImageNet-22k của thiết kế xác nhận v2: ConvNeXt-T và phần CNN của ccmm khởi tạo từ convnext_tiny.fb_in22k_ft_in1k (timm).

Vì sao file riêng: chữ ký thí nghiệm băm mã nguồn file của lớp mô hình (utilities/tracking.py, source_hash) và mọi hằng số config.py.
Sửa models/convnext_tiny.py, models/stitch_cxr.py hay thêm hằng số vào config.py sẽ đổi chữ ký mọi lượt cũ (không chạy tiếp, không tìm lại được).
File này định nghĩa hai lớp con và runner riêng; mã, config và chữ ký của các lượt ImageNet-1k không đổi.

Cách dựng (cùng kiến trúc với bản 1k, chỉ khác trọng số khởi tạo của phần CNN):
  - ConvNeXtTinyIn22k : ConvNeXtTinyBaseline, self.cnn tạo lại từ convnext_tiny.<cnn_weights> (cùng tên tham số với fb_in1k)
  - StitchCXRIn22k    : StitchCXR dựng với lớp nối đơn vị (VMamba-T ImageNet-1k như cũ), rồi nạp trọng số CNN 22k vào các stage còn giữ
                        và lớp nối bình phương tối thiểu khớp riêng cho CNN 22k (tệp stitch_<s>2<t>_stage<n>_<cnn_weights>.pt)
  - Lớp nối 22k       : fit-stitch, cùng cách khớp và cùng tham số với analysis/fit_stitch.py (96 ảnh ngoài fold 1, mỗi bệnh nhân một ảnh, seed 0)

Giao thức huấn luyện: như lượt cũ (CLIS_CONFIG, CLIS_PROTOCOL, LR 1e-4, đầu phân loại x40), RUN_TAG = "in22k" nên thư mục kết quả có hậu tố _in22k.
Quy tắc đọc: prereg/analysis_plan.md mục 8.8 (ghi trước khi chạy).

Ví dụ (WSL2, conda activate mamba-t29):
    python models/run_in22k.py fit-stitch
    python models/run_in22k.py train convnext_t --seed 42
    python models/run_in22k.py train ccmm --seed 42
"""
# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"                                 # cuBLAS tất định (DETERMINISTIC_ALGORITHMS)
os.environ["TRITON_CACHE_AUTOTUNING"] = "1"                                        # mamba-ssm >= 2.3 ở chế độ tất định: lưu autotune kernel Triton thay vì cảnh báo

# Khai báo thư viện
import argparse
import copy

import timm
import torch

import config
import models.convnext_tiny as convnext
import models.stitch_cxr as stx
from models.vmamba_cxr import sha256_file
from training.experiment import start_train
from utilities.dataset import NIHChestXrayDataset
from utilities.file import load_torch
from utilities.setting import apply_config

CNN_WEIGHTS = "fb_in22k_ft_in1k"                                                   # Thẻ trọng số timm của ConvNeXt-T (ImageNet-22k rồi tinh chỉnh ImageNet-1k)
RUN_TAG = "in22k"                                                                  # Hậu tố tên mô hình / thư mục kết quả
LAYOUT = "ccmm"
STITCH_INIT = "lstsq"
HEAD_LR = float(config.CLIS_LR_MULTIPLIER["convnext_t"])                          # Hệ số LR đầu phân loại x40, như lượt 1k của cả hai mô hình


# ============================================================
# Tệp lớp nối của CNN theo thẻ trọng số (bản 1k giữ tên cũ của models/stitch_cxr.py)
# Trả về: Path
def stitch_path(source, target, stage, cnn_weights = CNN_WEIGHTS):
    return stx.STITCH_DIRECTORY / f"stitch_{source}2{target}_stage{stage + 1}_{cnn_weights}.pt"


# Nạp trọng số timm của ConvNeXt-T vào các module CNN còn giữ của mô hình (stage bị thay bằng Identity thì không có khóa)
def load_cnn_weights(cnn, cnn_weights, drop_path_rate):
    source = timm.create_model(f"convnext_tiny.{cnn_weights}", pretrained = True, num_classes = 0, drop_path_rate = drop_path_rate).state_dict()
    target = cnn.state_dict()
    missing = [key for key in target if key not in source]
    if missing:
        raise RuntimeError(f"convnext_tiny.{cnn_weights} thiếu {len(missing)} khóa, ví dụ {missing[:3]}")
    cnn.load_state_dict({key: source[key] for key in target}, strict = True)


# ============================================================
# ConvNeXt-T, phần CNN khởi tạo từ trọng số 22k
class ConvNeXtTinyIn22k(convnext.ConvNeXtTinyBaseline):
    def __init__(self, num_classes, pretrained = True, drop_path_rate = 0.0, init_from = None, cnn_weights = CNN_WEIGHTS):
        super().__init__(num_classes, pretrained = False, drop_path_rate = drop_path_rate)
        if pretrained and init_from is None:
            load_cnn_weights(self.cnn, cnn_weights, drop_path_rate)
        if init_from is not None:
            self.load_state_dict(load_torch(init_from)["model_state_dict"], strict = True)


# ============================================================
# ccmm, phần CNN khởi tạo từ trọng số 22k, lớp nối khớp riêng cho CNN 22k
class StitchCXRIn22k(stx.StitchCXR):
    def __init__(self, num_classes, layout = LAYOUT, stitch_init = STITCH_INIT, pinned_sha256 = None, grad_checkpoint = False, pretrained = True,
                 init_from = None, drop_path_rate = 0.0, cnn_weights = CNN_WEIGHTS):
        super().__init__(num_classes, layout = layout, stitch_init = "identity", pinned_sha256 = pinned_sha256, grad_checkpoint = grad_checkpoint,
                         pretrained = pretrained, init_from = None, drop_path_rate = drop_path_rate)
        self.cnn_weights = cnn_weights
        load_imagenet = bool(pretrained) and init_from is None
        if load_imagenet:
            load_cnn_weights(self.cnn, cnn_weights, drop_path_rate)
            if stitch_init == "lstsq":
                pinned = dict(pinned_sha256 or {})
                for stage in stx.boundaries(layout):
                    path = stitch_path(layout[stage], layout[stage + 1], stage, cnn_weights)
                    if not path.is_file():
                        raise FileNotFoundError(f"Thiếu {path}: chạy python models/run_in22k.py fit-stitch trước")
                    if pinned and sha256_file(path) != pinned.get(path.name):
                        raise RuntimeError(f"{path}: SHA-256 khác bản đã ghim trong model_kwargs")
                    saved = torch.load(path, map_location = "cpu", weights_only = True)
                    with torch.no_grad():
                        self.stitches[str(stage)].weight.copy_(saved["weight"].view_as(self.stitches[str(stage)].weight))
                        self.stitches[str(stage)].bias.copy_(saved["bias"])
        if init_from is not None:
            saved = load_torch(init_from)
            if saved["model_kwargs"].get("layout") != layout:
                raise ValueError(f"{init_from} là checkpoint của layout {saved['model_kwargs'].get('layout')}, không phải {layout}")
            self.load_state_dict(saved["model_state_dict"], strict = True)


# ============================================================
# Khớp lớp nối cho CNN 22k: chạy analysis/fit_stitch.py với lớp mô hình và tên tệp của nhánh 22k (chỉ trong tiến trình này)
def fit_stitch(arguments):
    import analysis.fit_stitch as fit
    stx.StitchCXR = lambda *args, **kwargs: StitchCXRIn22k(*args, **kwargs)        # Layout một họ (cccc / mmmm) không có lớp nối: chỉ khác trọng số CNN
    stx.stitch_path = stitch_path
    sys.argv = ["analysis/fit_stitch.py", "--layout", LAYOUT, *arguments]
    fit.main()


# ============================================================
# Huấn luyện một (mô hình, seed, fold) với giao thức của lượt cũ
def train(name, seed, fold):
    apply_config({**config.CLIS_CONFIG, "RUN_TAG": RUN_TAG, "SEED": int(seed), "TRAIN_ALL_FOLD": False, "SELECTED_FOLD": [int(fold)]})
    protocol = copy.deepcopy(config.CLIS_PROTOCOL)
    protocol["optimizer_config"]["new_module_lr_multiplier"] = HEAD_LR
    if name == "convnext_t":
        factory, kwargs = ConvNeXtTinyIn22k, {"cnn_weights": CNN_WEIGHTS}
        model_name = f"{convnext.MODEL_NAME} CLIS {config.IMAGE_SIZE}px {RUN_TAG}"
    else:
        paths = [stitch_path(LAYOUT[stage], LAYOUT[stage + 1], stage) for stage in stx.boundaries(LAYOUT)]
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            raise SystemExit(f"Thiếu tệp lớp nối {missing}: chạy python models/run_in22k.py fit-stitch")
        factory = StitchCXRIn22k
        kwargs = {"layout": LAYOUT, "stitch_init": STITCH_INIT, "grad_checkpoint": bool(config.BLOCK_GRAD_CHECKPOINT), "cnn_weights": CNN_WEIGHTS,
                  "pinned_sha256": {stx.VMAMBA_SOURCE.name: sha256_file(stx.VMAMBA_SOURCE), **{path.name: sha256_file(path) for path in paths}}}
        model_name = f"{stx.model_name(LAYOUT)} {RUN_TAG}"
    return start_train(model_name = model_name, dataset_class = NIHChestXrayDataset, model_factory = factory, model_kwargs = kwargs, **protocol)


def main():
    parser = argparse.ArgumentParser(description = "Nhánh độ nhạy ImageNet-22k (prereg mục 8.8)")
    commands = parser.add_subparsers(dest = "command", required = True)
    fit = commands.add_parser("fit-stitch", help = "Khớp lớp nối cho CNN 22k (tham số còn lại chuyển cho analysis/fit_stitch.py)")
    fit.add_argument("rest", nargs = argparse.REMAINDER)
    run = commands.add_parser("train", help = "Huấn luyện một mô hình")
    run.add_argument("name", choices = ["convnext_t", "ccmm"])
    run.add_argument("--seed", type = int, required = True)
    run.add_argument("--fold", type = int, default = 1)
    args = parser.parse_args()
    if args.command == "fit-stitch":
        fit_stitch(args.rest)
    else:
        train(args.name, args.seed, args.fold)


if __name__ == "__main__":
    main()
