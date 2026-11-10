"""Đo mức tham gia của các tuyến quét trong VMamba-T CXR trên Validation (không mở Test).

Vì sao cần: vm_clis và vm_rand full-data seed 42 (2026-11-10) chênh nhau 0,0005 macro AUROC, và |C| / |D| của 2 tuyến thêm sau huấn luyện
chỉ khoảng 18% / 0,5% của tuyến gốc. Nếu mô hình gần như không dùng tuyến thêm thì so sánh CLIS với cặp ngẫu nhiên chưa kiểm định được gì.
Script chạy best_model.pt nhiều lần trên cùng ảnh Validation, mỗi lần tắt một nhóm tuyến ở các stage có bảng tuyến (ROUTE_STAGES):
  - on        : mọi tuyến (đúng mô hình đã huấn luyện)
  - extra_off : tắt tuyến thêm (CLIS ± hoặc cặp ngẫu nhiên ±); chỉ vm_clis / vm_rand
  - raster_off: tắt tuyến hàng ± (mốc so sánh: mức giảm khi bỏ 2 tuyến mà mô hình chắc chắn dùng)
  - column_off: tắt tuyến cột ±
Tắt tuyến = nhân đầu ra quét của tuyến đó với 0 trước khi cộng các tuyến (tương đương đặt hàng C của x_proj và D của tuyến bằng 0).
Báo cáo:
  - share_s<k>     : ||tổng tuyến thêm|| / ||tổng tuyến gốc|| ở stage k (chế độ on, trước out_norm), trung bình theo ảnh và khối
  - mix_share_s<k> : như share_s<k> nhưng chỉ phần trộn token (đầu ra tuyến - D·x); D·x là nhánh tắt giống nhau ở mọi thứ tự quét
  - saved_diff của VMamba cỡ 2e-3 (trung bình 1e-4) là bình thường: lõi gốc vmx.route_forward_core cũng lệch đúng mức đó (BF16)
  - auroc_<mode>   : macro AUROC 14 nhãn; L_ / C_<mode>: trung bình AUROC nhóm L / C theo cách chia chính (analysis/compare_clis.py)
  - saved_diff     : độ lệch lớn nhất giữa xác suất "on" và oof_predictions.csv đã lưu (kiểm tra bản sao phần lõi SS2D và việc nạp trọng số)

Ví dụ (WSL2, conda activate mamba-t29):
    python analysis/route_engagement.py --runs "training_results/nih_chest_xray_14/vmambat_cxr_vm_*_512px/run_*"
Kết quả ghi thêm vào training_results/route_engagement.csv (một dòng mỗi lượt chạy, đo lại thì thay dòng cũ).
"""
# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"                           # cuBLAS tất định (DETERMINISTIC_ALGORITHMS)
os.environ["TRITON_CACHE_AUTOTUNING"] = "1"                                  # mamba-ssm >= 2.3 ở chế độ tất định: lưu autotune kernel Triton thay vì cảnh báo

# Khai báo thư viện
import argparse
import glob
from functools import partial

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import config
import models.vmamba_cxr as vmx
from analysis.compare_clis import GROUPS
from utilities.data import create_dataloader, load_dataset, transform_setting
from utilities.dataset import NIHChestXrayDataset
from utilities.file import load_torch, read_json
from utilities.log import timestamp
from utilities.setting import apply_config

OUTPUT_PATH = config.MODELS_OUTPUT_DIRECTORY / "route_engagement.csv"
BASE = vmx.BASE_ROUTES
MODES = {                                                                   # Chế độ --> chỉ số tuyến bị tắt (thứ tự VMamba: hàng, cột, hàng ngược, cột ngược, rồi tuyến thêm)
    "on"        : lambda k: [],
    "extra_off" : lambda k: list(range(BASE, k)),
    "raster_off": lambda k: [0, 2],
    "column_off": lambda k: [1, 3],
}


# ============================================================
# Mở rộng danh sách thư mục lượt chạy (mẫu * do script tự mở rộng)
# Trả về: Danh sách đường dẫn
def expand_runs(patterns):
    return [Path(path).resolve() for pattern in patterns for path in (sorted(glob.glob(str(pattern))) if glob.has_magic(str(pattern)) else [pattern])]


# AUROC từng nhãn (NaN nếu nhãn thiếu mẫu dương hoặc âm)
# Trả về: np.ndarray (số nhãn,)
def auroc_per_label(targets, probabilities):
    return np.array([roc_auc_score(targets[:, k], probabilities[:, k]) if 0 < targets[:, k].sum() < len(targets) else np.nan for k in range(targets.shape[1])])


# ============================================================
# vmx.route_forward_core có thêm mặt nạ tuyến và đo phần đóng góp của tuyến thêm (đầu ra từng tuyến lấy từ vmx.route_outputs, mọi phép tính khác giữ nguyên)
# op.route_mask: (k,) hệ số nhân đầu ra quét của từng tuyến; op.share_log: danh sách ||tuyến thêm|| / ||tuyến gốc|| theo ảnh
# Trả về: (b, d, h, w)
def masked_forward_core(op, x):
    b, d, h, w = x.shape
    k = op.k_group

    # Đầu ra từng tuyến theo thứ tự raster --> đo phần đóng góp --> nhân mặt nạ --> cộng các tuyến
    # Phần trộn token của tuyến k = đầu ra - D_k·x (D·x là nhánh tắt từng token, giống nhau ở mọi thứ tự quét; với tuyến có dấu: s·D·s·x = D·x)
    routes = vmx.route_outputs(op, x)                                                           # (b, k, d, l)
    if op.share_log is not None and k > BASE:
        mixing = routes.float() - op.Ds.float().view(1, k, d, 1) * x.flatten(2).float()[:, None]
        base, extra = routes[:, :BASE].sum(1).float().flatten(1), routes[:, BASE:].sum(1).float().flatten(1)
        base_mix, extra_mix = mixing[:, :BASE].sum(1).flatten(1), mixing[:, BASE:].sum(1).flatten(1)
        op.share_log.append(torch.stack([extra.norm(dim = 1) / base.norm(dim = 1), extra_mix.norm(dim = 1) / base_mix.norm(dim = 1)], dim = 1).cpu())
    y = (routes * op.route_mask.view(1, k, 1, 1).to(routes.dtype)).sum(1).view(b, d, h, w)
    return op.out_norm(y).to(x.dtype)


# Các SS2D ở ROUTE_STAGES của mô hình: [(stage, op)]
def route_ops(model):
    return [(stage, block.op) for stage in model.route_stages for block in model.backbone.layers[stage].blocks]


# Đặt chế độ tắt tuyến cho mọi SS2D (bật / tắt ghi phần đóng góp)
def set_mode(ops, mode, record):
    for _, op in ops:
        mask = torch.ones(op.k_group, device = op.Ds.device)
        mask[MODES[mode](op.k_group)] = 0
        op.route_mask, op.share_log = mask, ([] if record else None)


# ============================================================
# Đo một lượt chạy VMamba-T CXR trên Validation của fold
# Trả về: dict một dòng kết quả
def measure(run_directory, fold, limit, workers, device):
    info = read_json(run_directory / "train_info.json")
    settings = info["settings"]
    kwargs = {key: value for key, value in settings["model_kwargs"].items() if key != "init_from"}
    fold_directory = run_directory / f"fold_{fold}"

    # Cấu hình của lượt chạy: độ phân giải và bộ ảnh
    snapshot = settings.get("config", {})
    apply_config({**config.CLIS_CONFIG, "IMAGE_SIZE": int(settings["image_size"]), "IMAGE_VARIANT": snapshot.get("IMAGE_VARIANT", config.CLIS_CONFIG["IMAGE_VARIANT"]),
                  **({} if workers is None else {"NUMBER_OF_WORKERS": int(workers)})})

    # Ảnh Validation của fold
    train_df, _, _, classes, _ = load_dataset()
    frame = train_df[train_df["Fold"] == int(fold)].reset_index(drop = True)
    frame = frame if limit is None else frame.iloc[:int(limit)].reset_index(drop = True)

    # Nạp trọng số tốt nhất (đúng chữ ký, đúng Fold); thay phần lõi SS2D ở ROUTE_STAGES bằng bản có mặt nạ
    best = load_torch(fold_directory / "best_model.pt", device)
    if best.get("signature") != info["signature"] or int(best.get("fold")) != int(fold):
        raise SystemExit(f"{run_directory}: best_model.pt không khớp chữ ký / Fold của lượt chạy.")
    model = vmx.VMambaCXR(num_classes = len(classes), pretrained = False, **{**kwargs, "grad_checkpoint": False}).to(device).eval()
    model.load_state_dict(best["model_state_dict"], strict = True)
    ops = route_ops(model)
    for _, op in ops:
        op.forward_core = partial(masked_forward_core, op)
    modes = [mode for mode in MODES if mode != "extra_off" or ops[0][1].k_group > BASE]

    # Suy luận mọi chế độ trên cùng lô ảnh
    _, eval_transform = transform_setting()
    _, loader = create_dataloader(NIHChestXrayDataset, classes, device, frame, eval_transform, False, batch_size = int(settings["training_config"]["batch_size"]), seed = int(settings["seed"]) + 900_000)
    probabilities, targets, shares = {mode: [] for mode in modes}, [], {}
    with torch.no_grad(), torch.autocast(device_type = device.type, dtype = torch.bfloat16 if config.AMP_DTYPE == "bfloat16" else torch.float16, enabled = bool(config.USE_AMP and device.type == "cuda")):
        for images, labels in loader:
            images = images.to(device, non_blocking = True)
            for mode in modes:
                set_mode(ops, mode, record = mode == "on")
                probabilities[mode].append(torch.sigmoid(model(images)).float().cpu())
                if mode == "on":
                    for stage, op in ops:
                        shares.setdefault(stage, []).extend(op.share_log)
            targets.append(labels)
    probabilities = {mode: torch.cat(values).numpy() for mode, values in probabilities.items()}
    targets = torch.cat(targets).numpy()

    # Kiểm tra nạp đúng: xác suất "on" so với oof_predictions.csv đã lưu lúc huấn luyện (ghép theo mã ảnh)
    saved_path = fold_directory / "oof_predictions.csv"
    saved_diff = float("nan")
    if saved_path.is_file():
        saved = pd.read_csv(saved_path).set_index("Image Index").loc[frame["Image Index"], [f"prob_{name}" for name in classes]].to_numpy(dtype = np.float64)
        saved_diff = float(np.abs(probabilities["on"] - saved).max())

    # Chỉ số theo chế độ: macro 14 nhãn, nhóm L, nhóm C
    group_index = {name: [classes.index(c) for c in members] for name, members in GROUPS.items()}
    row = {"run": str(run_directory.relative_to(config.MODELS_OUTPUT_DIRECTORY)).replace("\\", "/"), "mixer": kwargs["mixer"], "images": len(frame)}
    for stage, values in sorted(shares.items()):
        if not values:                                                      # vm_ras: không có tuyến thêm
            continue
        values = torch.cat(values).mean(dim = 0)
        row[f"share_s{stage + 1}"], row[f"mix_share_s{stage + 1}"] = float(values[0]), float(values[1])
    for mode in modes:
        per_label = auroc_per_label(targets, probabilities[mode])
        row[f"auroc_{mode}"] = float(np.nanmean(per_label))
        for name, columns in group_index.items():
            row[f"{name}_{mode}"] = float(np.nanmean(per_label[columns]))
        if mode != "on":
            row[f"drop_{mode}"] = row["auroc_on"] - row[f"auroc_{mode}"]
            row[f"prob_change_{mode}"] = float(np.abs(probabilities["on"] - probabilities[mode]).mean())
    row.update({"saved_diff": saved_diff, "measured_at": timestamp()})

    del model, best
    return row


# ============================================================
def main():
    parser = argparse.ArgumentParser(description = "Mức tham gia của các tuyến quét trong VMamba-T CXR (Validation)")
    parser.add_argument("--runs", nargs = "+", required = True, help = "Thư mục lượt chạy (run_N), cho phép mẫu *")
    parser.add_argument("--fold", type = int, default = 1)
    parser.add_argument("--limit", type = int, default = None, help = "Chỉ dùng N ảnh Validation đầu tiên (kiểm tra nhanh)")
    parser.add_argument("--workers", type = int, default = None)
    parser.add_argument("--output", type = Path, default = OUTPUT_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for run_directory in expand_runs(args.runs):
        if not (run_directory / f"fold_{args.fold}" / "fold_done.json").is_file():
            print(f"Bỏ qua {run_directory}: chưa xong fold {args.fold}", flush = True)
            continue
        try:
            row = measure(run_directory, args.fold, args.limit, args.workers, device)
        except (SystemExit, RuntimeError, KeyError, ValueError, OSError) as error:
            print(f"Lỗi khi đo {run_directory}: {error}", flush = True)
            continue
        rows.append(row)
        shares = " ".join(f"{key} {value:.3f}" for key, value in row.items() if key.startswith(("share_", "mix_share_")))
        drops = " | ".join(f"tắt {key[5:]}: giảm {value:+.4f} (L {row['L_on'] - row['L_' + key[5:]]:+.4f}, C {row['C_on'] - row['C_' + key[5:]]:+.4f})"
                           for key, value in row.items() if key.startswith("drop_"))
        print(f"{row['run']} | macro AUROC {row['auroc_on']:.4f} (L {row['L_on']:.4f}, C {row['C_on']:.4f}) | {drops}"
              + (f" | ||tuyến thêm|| / ||tuyến gốc|| {shares}" if shares else "") + f" | lệch so với tệp đã lưu {row['saved_diff']:.1e}", flush = True)

    # Ghi kết quả: thay dòng cũ của cùng lượt chạy, giữ các dòng khác (chỉ khi đo toàn bộ Validation)
    if rows and args.limit is None:
        new = pd.DataFrame(rows)
        if args.output.is_file():
            old = pd.read_csv(args.output)
            new = pd.concat([old[~old["run"].isin(new["run"])], new], ignore_index = True)
        new.to_csv(args.output, index = False)
        print(f"Đã ghi {args.output}", flush = True)


if __name__ == "__main__":
    main()
