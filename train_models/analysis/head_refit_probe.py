"""Khớp lại đầu phân loại trên đặc trưng đóng băng với các hàm Loss cân bằng lớp, cho nhãn hiếm (Validation, không mở Test).

Vì sao cần: các đội đứng đầu CXR-LT 2023 / 2024 đều dùng Loss cân bằng lớp (ASL, focal, CE có trọng số). Với đầu sigmoid riêng từng lớp,
đổi trọng số lớp phần lớn chỉ dịch bias, mà AUROC và AP của một lớp không đổi khi dịch bias. Script giữ nguyên backbone đã huấn luyện,
chỉ khớp lại lớp Linear cuối trên đặc trưng ảnh train (kiểu cRT, Kang et al. 2020) với từng hàm Loss, rồi đo xếp hạng của nhãn hiếm trên
Validation: phần ảnh hưởng còn lại sau khi bỏ tác động dịch bias nằm ở hướng của vector trọng số.

Cách tính (mọi mô hình có lớp cuối model.classifier = Linear):
  - Đặc trưng x  : đầu vào của model.classifier, lấy bằng hook trong đúng lần suy luận của mô hình (biến đổi đánh giá, cùng AMP như lúc
                   huấn luyện), cho ảnh train của Fold (Fold != fold) và ảnh Validation (Fold == fold)
  - original     : đầu đã huấn luyện của mô hình; xác suất trên Validation so với oof_predictions.csv để kiểm tra việc nạp trọng số
  - <loss>_wd<λ> : z = W·x + b, khởi tạo W = 0 và b = logit tỉ lệ dương trong train; L-BFGS toàn lô trên đặc trưng train,
                   Loss của training/loss.py (trung bình trên ảnh × nhãn, như lúc huấn luyện) + (λ/2)·||W||² (không phạt bias).
                   ASL: cùng giá trị Loss với training/loss.py nhưng gradient đi qua cả trọng số (1 - p_t)^gamma (L-BFGS cần gradient đúng)

Đọc kết quả (quy tắc ghi trước khi chạy: prereg/analysis_plan.md mục 7, dòng 2026-11-17 20:20): xác suất trung bình qua seed; bệnh nhân
Validation chia đôi như analysis/pooling_probe.py (seed 0). Trên nửa A chọn (i) đầu BCE tốt nhất theo λ và (ii) đầu cân bằng lớp tốt nhất
(Loss × λ), cả hai theo AP trung bình nhãn hiếm; trên nửa B báo cáo hiệu (ii) - (i) bằng bootstrap ghép cặp theo bệnh nhân.

Ví dụ (WSL2, conda activate mamba-t29):
    python analysis/head_refit_probe.py --runs "training_results/nih_chest_xray_14/convnextvmamba_cxr_ccmm_512px/run_[123]"
    python analysis/head_refit_probe.py --summary-only
Kết quả trong training_results/explore/head_refit/: features_<mô hình>_seed<seed>.npz (đặc trưng và đầu gốc), heads_<mô hình>_seed<seed>.npz
(xác suất Validation của mọi đầu), summary.csv, summary.json, summary.txt.
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
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

import config
import models.vmamba_cxr as vmx
from analysis.evaluate_locked import build_model
from analysis.pooling_probe import SAVED_DIFF_WARNING, expand_runs, half_a_mask, model_tag
from training.loss import build_loss
from utilities.data import create_dataloader, load_dataset, transform_setting
from utilities.dataset import NIHChestXrayDataset
from utilities.file import load_torch, read_json, save_csv, save_json
from utilities.log import timestamp
from utilities.setting import apply_config
from utilities.statistics import _bootstrap_samples, _interval, _macro, auroc_per_class

OUTPUT_DIRECTORY = config.MODELS_OUTPUT_DIRECTORY / "explore" / "head_refit"
LOSSES = {                                                                  # Hàm Loss của đầu khớp lại (training/loss.py); giá trị cố định trong prereg
    "bce"   : {"name": "bce", "pos_weight_alpha": 0.0},                     # BCE không pos_weight: Loss lúc huấn luyện ccmm (đối chứng)
    "wbce"  : {"name": "bce", "pos_weight_alpha": 1.0},                     # BCE cân bằng lớp: pos_weight = số âm / số dương của lớp trong train, không chặn trên
    "focal" : {"name": "focal", "gamma": 2.0},                              # Focal gamma = 2, không trọng số lớp
    "asl"   : {"name": "asl", "gamma_negative": 4.0, "gamma_positive": 0.0, "clip": 0.05},     # ASL (Ridnik et al. 2021), giá trị mặc định của dự án
}
CONTROL = "bce"                                                             # Loss của đầu đối chứng (i); các Loss còn lại là nhóm cân bằng lớp (ii)
WEIGHT_DECAYS = (1e-6, 1e-5, 1e-4, 1e-3)                                   # Lưới λ của phạt L2 (λ/2)·||W||²
ORIGINAL = "original"                                                       # Đầu đã huấn luyện của mô hình
RARE = ["Hernia", "Pneumonia", "Fibrosis", "Edema", "Emphysema", "Cardiomegaly"]       # Nhãn hiếm ghi trước trong prereg (script kiểm tra lại từ dữ liệu)
RARE_PREVALENCE = 0.025                                                     # Định nghĩa nhãn hiếm: tỉ lệ dương trong train của Fold < 2,5 %
NOT_MEASURABLE = "Hernia"                                                   # Báo cáo kèm AP nhãn hiếm không tính nhãn này (29 ca dương ở Validation fold 1)
MAX_ITERATIONS = 2000                                                       # L-BFGS: số vòng lặp tối đa
TOLERANCE_GRAD = 1e-6                                                       # L-BFGS: dừng khi gradient lớn nhất <= mức này
TOLERANCE_CHANGE = 1e-10                                                    # L-BFGS: dừng khi Loss / tham số đổi ít hơn mức này
HISTORY_SIZE = 20                                                           # L-BFGS: số cặp cập nhật giữ lại
MIN_AP_GAIN = 0.005                                                         # Quy tắc: Δ AP trung bình nhãn hiếm trên nửa B >= 0,005 và cận dưới CI > 0
MAX_AUROC_LOSS = 0.001                                                      # Quy tắc: Δ macro AUROC trên nửa B >= -0,001
MEMORY_FRACTION = 0.3                                                       # Phần bộ nhớ GPU tối đa của script (chạy song song với hàng đợi huấn luyện)
PRINT_EVERY = 250                                                           # In tiến độ trích đặc trưng sau mỗi chừng ấy batch
METRICS = ("rare_ap", "rare_ap_measurable", "rare_auroc", "macro_ap", "macro_auroc")     # Chỉ số tóm tắt, theo thứ tự in


# ============================================================
# Tên đầu khớp lại của một Loss và một λ
# Trả về: Chuỗi, ví dụ "asl_wd1e-05"
def head_name(loss, weight_decay):
    return f"{loss}_wd{weight_decay:g}"


# Tên mọi đầu, theo thứ tự lưu trong tệp .npz: đầu gốc rồi Loss × λ
# Trả về: Danh sách tên
def head_names():
    return [ORIGINAL] + [head_name(loss, weight_decay) for loss in LOSSES for weight_decay in WEIGHT_DECAYS]


# Loss của một đầu (None với đầu gốc)
# Trả về: Chuỗi hoặc None
def head_loss(name):
    return None if name == ORIGINAL else name.rsplit("_wd", 1)[0]


# ============================================================
# Trích đặc trưng (đầu vào của lớp Linear cuối) cho ảnh train và Validation của Fold
# Trả về: (đường dẫn tệp .npz, dict thông tin kiểm tra); tệp đã có --> (đường dẫn, None)
def extract(run_directory, fold, limit, workers, batch_size, device, output_directory):
    info = read_json(run_directory / "train_info.json")
    settings = info["settings"]
    fold_directory = run_directory / f"fold_{fold}"
    tag, seed = model_tag(settings), int(settings["seed"])
    path = output_directory / f"features_{tag}_seed{seed}.npz"
    if limit is None and path.is_file():
        return path, None

    # Cấu hình của lượt chạy: độ phân giải và bộ ảnh
    snapshot = settings.get("config", {})
    apply_config({**config.CLIS_CONFIG, "IMAGE_SIZE": int(settings["image_size"]), "IMAGE_VARIANT": snapshot.get("IMAGE_VARIANT", config.CLIS_CONFIG["IMAGE_VARIANT"]),
                  **({} if workers is None else {"NUMBER_OF_WORKERS": int(workers)})})

    # Ảnh train (Fold != fold) và Validation (Fold == fold), cùng cách chia với training/train.py
    train_df, _, _, classes, _ = load_dataset()
    frames = {"train": train_df[train_df["Fold"] != int(fold)].reset_index(drop = True), "validation": train_df[train_df["Fold"] == int(fold)].reset_index(drop = True)}
    if limit is not None:
        frames = {split: frame.iloc[:int(limit)].reset_index(drop = True) for split, frame in frames.items()}

    # Nạp trọng số tốt nhất (đúng chữ ký, đúng Fold); móc vào lớp Linear cuối để lấy đặc trưng của đúng lần suy luận đó
    best = load_torch(fold_directory / "best_model.pt", device)
    if best.get("signature") != info["signature"] or int(best.get("fold")) != int(fold):
        raise SystemExit(f"{run_directory}: best_model.pt không khớp chữ ký / Fold của lượt chạy.")
    model = build_model(settings, len(classes)).to(device).eval()
    model.load_state_dict(best["model_state_dict"], strict = True)
    if not isinstance(getattr(model, "classifier", None), torch.nn.Linear):
        raise SystemExit(f"{type(model).__name__}: lớp cuối model.classifier không phải Linear")
    weight, bias = model.classifier.weight.detach().float(), model.classifier.bias.detach().float()
    captured = {}
    handle = model.classifier.register_forward_hook(lambda module, inputs, output: captured.update(features = inputs[0]))

    # Suy luận: mô hình dưới AMP như lúc huấn luyện; đầu gốc tính lại bằng float32 trên đặc trưng để kiểm tra
    _, eval_transform = transform_setting()
    amp_dtype = torch.bfloat16 if config.AMP_DTYPE == "bfloat16" else torch.float16
    features, targets, speed, model_probabilities, model_diff = {}, {}, {}, [], 0.0
    with torch.no_grad():
        for split, frame in frames.items():
            _, loader = create_dataloader(NIHChestXrayDataset, classes, device, frame, eval_transform, False, batch_size = int(batch_size), seed = seed + 900_000)
            chunks, labels, started, done = [], [], time.time(), 0
            for step, (images, batch_labels) in enumerate(loader, 1):
                images = images.to(device, non_blocking = True)
                with torch.autocast(device_type = device.type, dtype = amp_dtype, enabled = bool(config.USE_AMP and device.type == "cuda")):
                    reference = torch.sigmoid(model(images).float())
                x = captured.pop("features").float()
                model_diff = max(model_diff, float((torch.sigmoid(x @ weight.T + bias) - reference).abs().max()))
                chunks.append(x.cpu())
                labels.append(batch_labels)
                if split == "validation":
                    model_probabilities.append(reference.cpu())
                done += len(images)
                if step % PRINT_EVERY == 0:
                    print(f"   {tag} seed {seed} | {split}: {done:,}/{len(frame):,} ảnh, {done / (time.time() - started):.0f} ảnh/s", flush = True)
            features[split] = torch.cat(chunks).numpy().astype(np.float32)
            targets[split] = torch.cat(labels).numpy()
            speed[split] = len(frame) / (time.time() - started)
            del loader
            if not (targets[split] == frame[classes].to_numpy(dtype = np.float32)).all():
                raise SystemExit(f"{run_directory}: nhãn của loader ({split}) khác khung dữ liệu.")
    handle.remove()

    # Kiểm tra nạp đúng: so với oof_predictions.csv đã lưu lúc huấn luyện (ghép theo mã ảnh)
    ## Đầu ra của mô hình (cùng AMP): kiểm tra trọng số
    ## Đầu gốc float32 trên đặc trưng đã trích: kiểm tra đặc trưng
    original = 1.0 / (1.0 + np.exp(-(features["validation"].astype(np.float64) @ weight.cpu().numpy().T.astype(np.float64) + bias.cpu().numpy().astype(np.float64))))
    saved_path, saved_diff, model_saved_diff = fold_directory / "oof_predictions.csv", float("nan"), float("nan")
    if saved_path.is_file():
        saved = pd.read_csv(saved_path).set_index(config.IMAGE_ID_COLUMN).loc[frames["validation"][config.IMAGE_ID_COLUMN], [f"prob_{name}" for name in classes]].to_numpy(dtype = np.float64)
        saved_diff = float(np.abs(original - saved).max())
        model_saved_diff = float(np.abs(torch.cat(model_probabilities).numpy().astype(np.float64) - saved).max())

    check = {"run": str(run_directory.relative_to(config.MODELS_OUTPUT_DIRECTORY)).replace("\\", "/"), "model": tag, "seed": seed, "fold": int(fold),
             "train_images": len(frames["train"]), "validation_images": len(frames["validation"]), "dimension": int(features["train"].shape[1]),
             "model_diff": model_diff, "saved_diff": saved_diff, "model_saved_diff": model_saved_diff, "batch_size": int(batch_size),
             "train_images_per_second": speed["train"], "validation_images_per_second": speed["validation"], "measured_at": timestamp()}
    del model, best

    # Chỉ ghi tệp khi trích toàn bộ ảnh; ghi tệp tạm rồi đổi tên (tệp dở dang không bị đọc như tệp xong)
    if limit is None:
        output_directory.mkdir(parents = True, exist_ok = True)
        temporary = path.with_name(path.stem + ".partial.npz")
        np.savez(temporary, train_features = features["train"], validation_features = features["validation"],
                 train_targets = targets["train"].astype(np.int8), validation_targets = targets["validation"].astype(np.int8),
                 train_image_ids = frames["train"][config.IMAGE_ID_COLUMN].to_numpy(dtype = str), validation_image_ids = frames["validation"][config.IMAGE_ID_COLUMN].to_numpy(dtype = str),
                 validation_patients = frames["validation"][config.PATIENT_ID_COLUMN].to_numpy(dtype = str), classes = np.array(classes),
                 weight = weight.cpu().numpy(), bias = bias.cpu().numpy(), check = np.array(json.dumps(check)))
        os.replace(temporary, path)
    return path, check


# ============================================================
# Khớp một đầu Linear trên đặc trưng train bằng L-BFGS toàn lô: Loss (trung bình trên ảnh × nhãn) + (λ/2)·||W||²
# Trả về: (W (K, C), b (K,), dict chẩn đoán hội tụ)
def fit_head(features, targets, criterion, weight_decay, initial_bias):
    weight = torch.zeros((targets.shape[1], features.shape[1]), dtype = torch.float32, device = features.device, requires_grad = True)
    bias = initial_bias.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([weight, bias], lr = 1.0, max_iter = MAX_ITERATIONS, max_eval = MAX_ITERATIONS * 5 // 4, tolerance_grad = TOLERANCE_GRAD,
                                  tolerance_change = TOLERANCE_CHANGE, history_size = HISTORY_SIZE, line_search_fn = "strong_wolfe")

    # Hàm mục tiêu (L-BFGS gọi lại nhiều lần trong một bước)
    def objective():
        optimizer.zero_grad()
        loss = criterion(features @ weight.T + bias, targets) + 0.5 * float(weight_decay) * weight.pow(2).sum()
        loss.backward()
        return loss

    # Tối ưu, rồi tính lại Loss và gradient tại điểm cuối
    optimizer.step(objective)
    loss = float(objective())
    gradient = float(torch.cat([weight.grad.flatten(), bias.grad.flatten()]).abs().max())
    state = optimizer.state[weight]
    return weight.detach(), bias.detach(), {"loss": loss, "max_gradient": gradient, "iterations": int(state.get("n_iter", 0)), "evaluations": int(state.get("func_evals", 0))}


# ASL cùng giá trị với training/loss.py, nhưng gradient đi qua cả trọng số (1 - p_t)^gamma. Bản trong training/loss.py tính trọng số trong
# torch.no_grad (như mã của Ridnik et al. khi huấn luyện) nên gradient không phải gradient của giá trị Loss, mà L-BFGS dò bước bằng cả hai.
# Trả về: Hàm (logits, targets) --> Loss
def asl_with_full_gradient(gamma_negative, gamma_positive, clip, epsilon):
    def loss(logits, targets):

        # Xác suất dương; xác suất âm dịch thêm clip (mẫu âm quá dễ có Loss = 0)
        positive_probability = torch.sigmoid(logits)
        negative_probability = (1.0 - positive_probability + clip).clamp(max = 1.0) if clip > 0 else 1.0 - positive_probability

        # Log xác suất của nhãn đúng
        log_negative = torch.log(negative_probability.clamp(min = epsilon)) if clip > 0 else F.logsigmoid(-logits)
        log_probability = targets * F.logsigmoid(logits) + (1.0 - targets) * log_negative

        # Trọng số bất đối xứng, có gradient; gamma = 0 --> trọng số 1 (tránh gradient 0 · x^-1 tại x = 0)
        positive_weight = torch.pow(1.0 - positive_probability, gamma_positive) if gamma_positive > 0 else torch.ones_like(logits)
        negative_weight = torch.pow(1.0 - negative_probability, gamma_negative) if gamma_negative > 0 else torch.ones_like(logits)
        return -(log_probability * (targets * positive_weight + (1.0 - targets) * negative_weight)).mean()
    return loss


# Hàm Loss dùng để khớp: hàm của training/loss.py; riêng ASL thay bằng bản có gradient đầy đủ sau khi kiểm tra cùng giá trị
# Trả về: (hàm Loss, metadata)
def fitting_loss(loss_config, label_frame, classes, device):
    criterion, metadata = build_loss(loss_config, label_frame, classes, device)
    if loss_config["name"] != "asl":
        return criterion, metadata

    # Kiểm tra cùng giá trị Loss trên dữ liệu ngẫu nhiên cố định
    replacement = asl_with_full_gradient(float(metadata["gamma_negative"]), float(metadata["gamma_positive"]), float(metadata["clip"]), float(config.ASYMMETRIC_LOSS_EPSILON))
    generator = torch.Generator().manual_seed(0)
    logits = (3.0 * torch.randn((4096, len(classes)), generator = generator)).to(device)
    targets = (torch.rand((4096, len(classes)), generator = generator) < 0.1).float().to(device)
    expected, actual = float(criterion(logits, targets)), float(replacement(logits, targets))
    if abs(expected - actual) > 1e-6 * max(1.0, abs(expected)):
        raise SystemExit(f"ASL có gradient đầy đủ lệch giá trị so với training/loss.py: {actual} so với {expected}")
    return replacement, {**metadata, "gradient": "full (trọng số có gradient)"}


# Khớp mọi đầu (Loss × λ) của một tệp đặc trưng và tính xác suất Validation; lưu vào heads_<mô hình>_seed<seed>.npz (đã có thì dùng lại)
# Trả về: Đường dẫn tệp
def fit_all(feature_path, device):
    path = feature_path.with_name(feature_path.name.replace("features_", "heads_", 1))
    names = head_names()
    if path.is_file():
        with np.load(path, allow_pickle = False) as saved:
            if [str(name) for name in saved["heads"]] == names:
                return path

    # Kiểm tra: NVUM bật thì build_loss bọc Loss trong NVUM
    if config.NVUM is not None:
        raise SystemExit("config.NVUM khác None: build_loss sẽ bọc Loss trong NVUM; đặt lại NVUM = None")

    # Phép nhân ma trận float32 đầy đủ (không TF32): L-BFGS cần gradient chính xác
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    # Đặc trưng, nhãn train và tỉ lệ dương của từng lớp
    with np.load(feature_path, allow_pickle = False) as saved:
        data = {key: saved[key] for key in saved.files}
    classes = [str(name) for name in data["classes"]]
    train_targets = data["train_targets"].astype(np.float32)
    prevalence = train_targets.mean(axis = 0)
    features = torch.from_numpy(data["train_features"]).to(device)
    targets = torch.from_numpy(train_targets).to(device)
    validation = torch.from_numpy(data["validation_features"]).to(device)
    initial_bias = torch.from_numpy(np.log(prevalence / (1.0 - prevalence)).astype(np.float32)).to(device)
    label_frame = pd.DataFrame(train_targets, columns = classes)                                # build_loss tính pos_weight từ nhãn train

    # Đầu gốc, rồi từng Loss × λ
    with torch.no_grad():
        probabilities = [torch.sigmoid(validation @ torch.from_numpy(data["weight"]).to(device).T + torch.from_numpy(data["bias"]).to(device)).cpu().numpy()]
    diagnostics = [{"head": ORIGINAL}]
    for loss, loss_config in LOSSES.items():
        criterion, metadata = fitting_loss(loss_config, label_frame, classes, device)
        for weight_decay in WEIGHT_DECAYS:
            started = time.time()
            weight, bias, diagnostic = fit_head(features, targets, criterion, weight_decay, initial_bias)
            with torch.no_grad():
                probabilities.append(torch.sigmoid(validation @ weight.T + bias).cpu().numpy())
            diagnostics.append({"head": head_name(loss, weight_decay), **diagnostic, "weight_norm": float(weight.norm()), "seconds": time.time() - started,
                                "loss_metadata": {key: value for key, value in metadata.items() if key != "pos_weight"}})
            print(f"   {feature_path.stem} | {diagnostics[-1]['head']:14s}: {diagnostic['iterations']:4d} vòng lặp, Loss {diagnostic['loss']:.6f}, "
                  f"gradient lớn nhất {diagnostic['max_gradient']:.1e}, ||W|| {diagnostics[-1]['weight_norm']:.2f}", flush = True)

    # Ghi tệp tạm rồi đổi tên
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez(temporary, probabilities = np.stack(probabilities).astype(np.float32), heads = np.array(names), classes = data["classes"],
             targets = data["validation_targets"], image_ids = data["validation_image_ids"], patients = data["validation_patients"],
             train_prevalence = prevalence, diagnostics = np.array(json.dumps(diagnostics)), check = data["check"])
    os.replace(temporary, path)
    return path


# ============================================================
# AP của một lớp, cùng định nghĩa với average_precision_score của sklearn (kể cả cách gộp các điểm bằng nhau), nhanh hơn cho bootstrap
# Trả về: float; lớp không có ca dương hoặc không có ca âm --> NaN
def average_precision(targets, scores):
    order = np.argsort(-scores, kind = "mergesort")
    ordered_scores = scores[order]
    last = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(scores) - 1]                     # Vị trí cuối của mỗi nhóm điểm bằng nhau
    true_positive = np.cumsum(targets[order])[last]
    positive = true_positive[-1]
    if positive == 0 or positive == len(scores):
        return float("nan")
    precision, recall = true_positive / (last + 1), true_positive / positive
    return float(np.sum(np.diff(recall, prepend = 0.0) * precision))


# AP từng lớp
# Trả về: np.ndarray (số lớp,)
def ap_per_class(targets, probabilities):
    return np.array([average_precision(targets[:, k], probabilities[:, k]) for k in range(targets.shape[1])])


# Năm chỉ số tóm tắt: AP và AUROC trung bình nhãn hiếm, AP nhãn hiếm không tính Hernia, macro AP và macro AUROC 14 nhãn
# Trả về: dict
def summary_metrics(targets, probabilities, rare, measurable):
    ap, auroc = ap_per_class(targets, probabilities), auroc_per_class(targets, probabilities)
    return {"rare_ap": _macro(ap[rare]), "rare_ap_measurable": _macro(ap[measurable]), "rare_auroc": _macro(auroc[rare]), "macro_ap": _macro(ap), "macro_auroc": _macro(auroc)}


# Bootstrap ghép cặp theo bệnh nhân của hiệu (đầu a - đầu b) cho các chỉ số tóm tắt
# Trả về: dict {chỉ số: {difference, ci_low, ci_high, p_one_sided_a_greater}}
def paired_bootstrap(targets, a, b, patients, rare, measurable, bootstrap_iteration, seed, alpha):
    def difference(rows):
        first, second = summary_metrics(targets[rows], a[rows], rare, measurable), summary_metrics(targets[rows], b[rows], rare, measurable)
        return {key: first[key] - second[key] for key in first}

    point = difference(slice(None))
    samples = {key: [] for key in point}
    for rows in _bootstrap_samples(patients, bootstrap_iteration, seed):
        for key, value in difference(rows).items():
            samples[key].append(value)
    result = {}
    for key, values in samples.items():
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        low, high = _interval(values, alpha)
        result[key] = {"difference": float(point[key]), "ci_low": low, "ci_high": high, "p_one_sided_a_greater": float(((values <= 0).sum() + 1) / (len(values) + 1))}
    return result


# Ngưỡng tối đa F1 của một lớp (dự đoán dương khi xác suất >= ngưỡng)
# Trả về: float
def best_f1_threshold(targets, scores):
    order = np.argsort(-scores, kind = "mergesort")
    ordered_scores = scores[order]
    last = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(scores) - 1]
    true_positive = np.cumsum(targets[order])[last]
    f1 = 2 * true_positive / (last + 1 + true_positive[-1])                                     # 2TP / (2TP + FP + FN) = 2TP / (số dự đoán dương + số ca dương)
    return float(ordered_scores[last[int(np.argmax(f1))]])


# Precision, recall, F1 của một lớp tại một ngưỡng
# Trả về: dict
def decision_metrics(targets, scores, threshold):
    predicted, positive = scores >= threshold, targets == 1
    true_positive, false_positive, false_negative = int((predicted & positive).sum()), int((predicted & ~positive).sum()), int((~predicted & positive).sum())
    return {"threshold": float(threshold), "predicted_positive": true_positive + false_positive,
            "precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else float("nan"),
            "recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else float("nan"),
            "f1": 2 * true_positive / (2 * true_positive + false_positive + false_negative) if true_positive + false_negative else float("nan")}


# Định dạng một hiệu kèm khoảng tin cậy
# Trả về: Chuỗi
def show(row):
    return f"{row['difference']:+.4f} [{row['ci_low']:+.4f}; {row['ci_high']:+.4f}]"


# ============================================================
# Tổng hợp mọi tệp heads_*.npz: bảng theo đầu, chọn trên nửa A, bootstrap trên nửa B, phân tích ngưỡng, kết luận theo quy tắc
# Trả về: (bảng DataFrame, dict kết quả, danh sách dòng văn bản)
def summarize(output_directory, bootstrap_iteration, seed):
    alpha = round(1 - config.BOOTSTRAP_CONFIDENCE, 12)
    groups = {}
    for path in sorted(output_directory.glob("heads_*_seed*.npz")):
        if not path.name.endswith(".partial.npz"):
            groups.setdefault(path.stem[len("heads_"):].rsplit("_seed", 1)[0], []).append(path)
    if not groups:
        raise SystemExit(f"Không có tệp heads_*.npz nào trong {output_directory}: chạy với --runs trước.")

    table, lines = [], []
    result = {"split_seed": "analysis/pooling_probe.py SPLIT_SEED", "bootstrap_iteration": int(bootstrap_iteration), "bootstrap_seed": int(seed), "confidence": 1 - alpha,
              "rare": RARE, "not_measurable": NOT_MEASURABLE, "losses": LOSSES, "weight_decays": list(WEIGHT_DECAYS),
              "rule": {"min_ap_gain": MIN_AP_GAIN, "max_auroc_loss": MAX_AUROC_LOSS}, "models": {}}
    for tag, paths in groups.items():

        # 1. Đọc các seed, kiểm tra cùng ảnh / nhãn / danh sách đầu, trung bình xác suất qua seed
        files = [np.load(path, allow_pickle = False) for path in paths]
        base = files[0]
        for path, data in zip(paths, files):
            if list(data["heads"]) != list(base["heads"]) or list(data["classes"]) != list(base["classes"]):
                raise ValueError(f"{path}: danh sách đầu / lớp khác tệp đầu tiên")
            if data["image_ids"].shape != base["image_ids"].shape or not (data["image_ids"] == base["image_ids"]).all() or not (data["targets"] == base["targets"]).all():
                raise ValueError(f"{path}: tập ảnh hoặc nhãn khác tệp đầu tiên")
        names, classes = [str(name) for name in base["heads"]], [str(name) for name in base["classes"]]
        targets, patients = base["targets"].astype(np.float64), base["patients"]
        per_seed = np.stack([data["probabilities"].astype(np.float64) for data in files])                # (S, H, N, K)
        probabilities = per_seed.mean(axis = 0)
        checks = [json.loads(str(data["check"])) for data in files]
        diagnostics = [json.loads(str(data["diagnostics"])) for data in files]
        seeds = [check["seed"] for check in checks]
        index = {name: h for h, name in enumerate(names)}

        # 2. Kiểm tra danh sách nhãn hiếm đăng ký trước (tỉ lệ dương trong train < RARE_PREVALENCE) và AP nhanh so với sklearn
        prevalence = base["train_prevalence"]
        found = sorted(name for name, value in zip(classes, prevalence) if value < RARE_PREVALENCE)
        if found != sorted(RARE):
            raise SystemExit(f"Nhãn có tỉ lệ dương < {RARE_PREVALENCE:.1%} trong train là {found}, khác danh sách đăng ký trước {RARE}")
        rare = [classes.index(name) for name in RARE]
        measurable = [classes.index(name) for name in RARE if name != NOT_MEASURABLE]
        reference = np.array([average_precision_score(targets[:, k], probabilities[index[ORIGINAL]][:, k]) for k in range(len(classes))])
        if np.abs(reference - ap_per_class(targets, probabilities[index[ORIGINAL]])).max() > 1e-9:
            raise SystemExit("average_precision lệch average_precision_score của sklearn")

        # 3. Chỉ số của từng đầu trên toàn bộ Validation, nửa A, nửa B
        in_a = half_a_mask(patients)
        halves = {"full": np.ones(len(targets), dtype = bool), "A": in_a, "B": ~in_a}
        metrics = {name: {half: summary_metrics(targets[mask], probabilities[h][mask], rare, measurable) for half, mask in halves.items()} for h, name in enumerate(names)}
        for h, name in enumerate(names):
            per_seed_rare = [summary_metrics(targets, seed_probabilities[h], rare, measurable)["rare_ap"] for seed_probabilities in per_seed]
            table.append({"model": tag, "head": name, "loss": head_loss(name) or ORIGINAL, "seeds": len(paths), "rare_ap_seed_mean": float(np.mean(per_seed_rare)),
                          **{f"{key}_{half}": value for half, values in metrics[name].items() for key, value in values.items()}})

        # 4. Chọn trên nửa A theo AP trung bình nhãn hiếm: (i) đầu BCE, (ii) đầu cân bằng lớp
        control = max((name for name in names if head_loss(name) == CONTROL), key = lambda name: metrics[name]["A"]["rare_ap"])
        balanced = max((name for name in names if head_loss(name) not in (None, CONTROL)), key = lambda name: metrics[name]["A"]["rare_ap"])

        # 5. Nửa B: bootstrap ghép cặp theo bệnh nhân
        b = halves["B"]
        pairs = {"balanced_vs_control": (balanced, control), "control_vs_original": (control, ORIGINAL), "balanced_vs_original": (balanced, ORIGINAL)}
        comparisons = {key: paired_bootstrap(targets[b], probabilities[index[first]][b], probabilities[index[second]][b], patients[b], rare, measurable, bootstrap_iteration, seed, alpha)
                       for key, (first, second) in pairs.items()}
        main_result = comparisons["balanced_vs_control"]
        passed = bool(main_result["rare_ap"]["difference"] >= MIN_AP_GAIN and main_result["rare_ap"]["ci_low"] is not None and main_result["rare_ap"]["ci_low"] > 0
                      and main_result["macro_auroc"]["difference"] >= -MAX_AUROC_LOSS)
        without_hernia = bool(main_result["rare_ap_measurable"]["difference"] >= MIN_AP_GAIN and main_result["rare_ap_measurable"]["ci_low"] is not None
                              and main_result["rare_ap_measurable"]["ci_low"] > 0)
        verdict = "ĐẠT" if passed and without_hernia else ("ĐẠT CHỈ NHỜ HERNIA (không diễn giải là đạt)" if passed else "KHÔNG ĐẠT")

        # 6. Báo cáo kèm: hiệu từng seed (điểm, nửa B), hiệu từng nhãn (nửa B)
        seed_difference = []
        for seed_value, seed_probabilities in zip(seeds, per_seed):
            first = summary_metrics(targets[b], seed_probabilities[index[balanced]][b], rare, measurable)
            second = summary_metrics(targets[b], seed_probabilities[index[control]][b], rare, measurable)
            seed_difference.append({"seed": seed_value, **{key: first[key] - second[key] for key in first}})
        class_difference = {}
        for key, (first, second) in pairs.items():
            ap_first, ap_second = ap_per_class(targets[b], probabilities[index[first]][b]), ap_per_class(targets[b], probabilities[index[second]][b])
            auroc_first, auroc_second = auroc_per_class(targets[b], probabilities[index[first]][b]), auroc_per_class(targets[b], probabilities[index[second]][b])
            class_difference[key] = {name: {"ap": float(ap_first[k] - ap_second[k]), "auroc": float(auroc_first[k] - auroc_second[k])} for k, name in enumerate(classes)}

        # 7. Báo cáo kèm: precision / recall / F1 của nhãn hiếm trên nửa B, ở ngưỡng 0,5 và ở ngưỡng tối đa F1 chọn trên nửa A
        decisions = {}
        for name in (ORIGINAL, control, balanced):
            decisions[name] = {}
            for k in rare:
                scores = probabilities[index[name]][:, k]
                threshold = best_f1_threshold(targets[in_a, k], scores[in_a])
                decisions[name][classes[k]] = {"fixed_0.5": decision_metrics(targets[b, k], scores[b], 0.5), "half_a_f1": decision_metrics(targets[b, k], scores[b], threshold)}

        # 8. Chẩn đoán hội tụ của các lần khớp
        fitted = [row for rows in diagnostics for row in rows if row["head"] != ORIGINAL]
        capped = sum(row["iterations"] >= MAX_ITERATIONS for row in fitted)
        worst_gradient = max(row["max_gradient"] for row in fitted)

        result["models"][tag] = {"seeds": seeds, "checks": checks, "diagnostics": diagnostics, "images": int(len(targets)), "images_half_b": int(b.sum()),
                                 "patients_half_b": int(len(set(patients[b].tolist()))), "rare_positives": {half: {classes[k]: int(targets[mask, k].sum()) for k in rare} for half, mask in halves.items()},
                                 "chosen_on_half_a": {"control": control, "balanced": balanced}, "half_b": comparisons, "passed": passed,
                                 "passed_without_hernia": without_hernia, "verdict": verdict, "seed_difference_half_b": seed_difference,
                                 "class_difference_half_b": class_difference, "decisions_half_b": decisions, "metrics": metrics,
                                 "fits": {"count": len(fitted), "capped": int(capped), "worst_max_gradient": worst_gradient}}

        # 9. Dòng văn bản
        lines.append(f"== {tag} | {len(paths)} seed {seeds} | Validation {len(targets):,} ảnh; nửa B {int(b.sum()):,} ảnh, {len(set(patients[b].tolist())):,} bệnh nhân")
        lines.append(f"   kiểm tra nạp: đầu ra mô hình so với oof đã lưu lệch tối đa {max(check['model_saved_diff'] for check in checks):.1e}; "
                     f"đầu gốc float32 trên đặc trưng so với oof {max(check['saved_diff'] for check in checks):.1e}")
        lines.append(f"   hội tụ: {capped}/{len(fitted)} lần khớp chạm {MAX_ITERATIONS} vòng lặp; gradient lớn nhất ở điểm cuối {worst_gradient:.1e}")
        lines.append("   ca dương nhãn hiếm (nửa A / nửa B): " + ", ".join(f"{classes[k]} {int(targets[in_a, k].sum())}/{int(targets[b, k].sum())}" for k in rare))
        lines.append(f"   {'đầu':16s} | AP hiếm | AP hiếm không Hernia | AUROC hiếm | macro AP | macro AUROC   (toàn bộ Validation, xác suất trung bình seed)")
        for name in names:
            row = metrics[name]["full"]
            lines.append(f"   {name:16s} | {row['rare_ap']:.4f}  | {row['rare_ap_measurable']:.4f}              | {row['rare_auroc']:.4f}     | {row['macro_ap']:.4f}   | {row['macro_auroc']:.4f}")
        lines.append(f"   chọn trên nửa A: (i) {control} (AP hiếm nửa A {metrics[control]['A']['rare_ap']:.4f}), (ii) {balanced} ({metrics[balanced]['A']['rare_ap']:.4f}); "
                     f"đầu gốc {metrics[ORIGINAL]['A']['rare_ap']:.4f}")
        labels = {"balanced_vs_control": "(ii) - (i)      ", "control_vs_original": "(i) - đầu gốc   ", "balanced_vs_original": "(ii) - đầu gốc  "}
        for key, label in labels.items():
            row = comparisons[key]
            lines.append(f"   nửa B {label}: Δ AP hiếm {show(row['rare_ap'])} | không Hernia {show(row['rare_ap_measurable'])} | Δ AUROC hiếm {show(row['rare_auroc'])} | "
                         f"Δ macro AP {show(row['macro_ap'])} | Δ macro AUROC {show(row['macro_auroc'])}")
        lines.append("   từng seed, nửa B (ii) - (i), Δ AP hiếm: " + ", ".join(f"seed {row['seed']} {row['rare_ap']:+.4f}" for row in seed_difference)
                     + " | Δ macro AUROC: " + ", ".join(f"{row['macro_auroc']:+.4f}" for row in seed_difference))
        lines.append("   nửa B (ii) - (i) từng nhãn hiếm, ΔAP / ΔAUROC: " + ", ".join(f"{classes[k]} {class_difference['balanced_vs_control'][classes[k]]['ap']:+.4f} / "
                                                                                     f"{class_difference['balanced_vs_control'][classes[k]]['auroc']:+.4f}" for k in rare))
        lines.append("   ngưỡng, nửa B, precision / recall / F1 (ngưỡng 0,5 | ngưỡng tối đa F1 chọn trên nửa A):")
        for k in rare:
            cells = []
            for name, label in ((ORIGINAL, "gốc"), (control, "(i)"), (balanced, "(ii)")):
                fixed, tuned = decisions[name][classes[k]]["fixed_0.5"], decisions[name][classes[k]]["half_a_f1"]
                cells.append(f"{label} {fixed['precision']:.2f}/{fixed['recall']:.2f}/{fixed['f1']:.2f} | {tuned['precision']:.2f}/{tuned['recall']:.2f}/{tuned['f1']:.2f}")
            lines.append(f"     {classes[k]:13s}: " + "   ".join(cells))
        lines.append(f"   QUY TẮC (Δ AP hiếm >= {MIN_AP_GAIN}, cận dưới CI > 0, Δ macro AUROC >= -{MAX_AUROC_LOSS}; kèm kiểm tra không tính Hernia): {verdict}")
    return pd.DataFrame(table), result, lines


# ============================================================
def main():
    parser = argparse.ArgumentParser(description = "Khớp lại đầu phân loại trên đặc trưng đóng băng với các hàm Loss cân bằng lớp (Validation, không mở Test)")
    parser.add_argument("--runs", nargs = "+", default = [], help = "Thư mục lượt chạy (run_N), cho phép mẫu *")
    parser.add_argument("--fold", type = int, default = 1)
    parser.add_argument("--limit", type = int, default = None, help = "Chỉ dùng N ảnh train và N ảnh Validation đầu tiên (kiểm tra nhanh: không ghi tệp, không khớp đầu)")
    parser.add_argument("--workers", type = int, default = None)
    parser.add_argument("--batch-size", type = int, default = 32, help = "Batch khi trích đặc trưng (chế độ eval: không đổi kết quả ngoài sai số làm tròn)")
    parser.add_argument("--device", default = None, help = "cuda | cpu (mặc định: cuda nếu có)")
    parser.add_argument("--memory-fraction", type = float, default = MEMORY_FRACTION, help = "Phần bộ nhớ GPU tối đa của script")
    parser.add_argument("--output", type = Path, default = OUTPUT_DIRECTORY)
    parser.add_argument("--summary-only", action = "store_true", help = "Không trích đặc trưng; khớp đầu (nếu chưa có) và tổng hợp")
    parser.add_argument("--n-boot", type = int, default = int(config.COMPARE_BOOTSTRAP_ITERATION))
    parser.add_argument("--seed", type = int, default = 0, help = "Seed của bootstrap (đăng ký trước: 0)")
    args = parser.parse_args()
    if not args.runs and not args.summary_only:
        parser.error("cần --runs hoặc --summary-only")

    # Thiết bị; giới hạn bộ nhớ GPU vì chạy song song với hàng đợi huấn luyện
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(args.memory_fraction), torch.cuda.current_device() if device.index is None else device.index)
    else:
        vmx.vm.WITH_SELECTIVESCAN_MAMBA = False                                                 # Selective scan bản PyTorch của VMamba trên CPU

    # 1. Trích đặc trưng từng lượt chạy
    if not args.summary_only:
        written = {}
        for run_directory in expand_runs(args.runs):
            if not (run_directory / f"fold_{args.fold}" / "fold_done.json").is_file():
                print(f"Bỏ qua {run_directory}: chưa xong fold {args.fold}", flush = True)
                continue
            path, check = extract(run_directory, args.fold, args.limit, args.workers, args.batch_size, device, args.output)
            if path in written:
                raise SystemExit(f"{run_directory} và {written[path]} cùng mô hình và seed: tệp {path.name} sẽ bị ghi đè. Chỉ truyền một lượt cho mỗi (mô hình, seed).")
            written[path] = run_directory
            if check is None:
                print(f"Đã có {path.name}: bỏ qua trích đặc trưng", flush = True)
                continue
            print(f"{check['run']} | {check['model']} seed {check['seed']} | train {check['train_images']:,} ảnh ({check['train_images_per_second']:.0f} ảnh/s), "
                  f"Validation {check['validation_images']:,} ảnh | đầu ra mô hình so với oof đã lưu {check['model_saved_diff']:.1e}, "
                  f"đầu gốc float32 so với oof {check['saved_diff']:.1e}, so với đầu ra mô hình {check['model_diff']:.1e}"
                  + ("" if args.limit is None else " | --limit: không ghi tệp"), flush = True)
            if device.type == "cuda" and check["saved_diff"] > SAVED_DIFF_WARNING:
                print(f"CẢNH BÁO: {check['run']} lệch {check['saved_diff']:.1e} so với oof đã lưu (> {SAVED_DIFF_WARNING}): kiểm tra việc nạp trọng số trước khi đọc kết quả", flush = True)
        if args.limit is not None:
            return

    # 2. Khớp đầu cho mọi tệp đặc trưng (đã có thì dùng lại)
    for path in sorted(args.output.glob("features_*_seed*.npz")):
        if not path.name.endswith(".partial.npz"):
            fit_all(path, device)

    # 3. Tổng hợp
    table, result, lines = summarize(args.output, args.n_boot, args.seed)
    save_csv(args.output / "summary.csv", table)
    save_json(args.output / "summary.json", result)
    (args.output / "summary.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")
    print("\n".join(lines))
    print(f"Đã ghi {args.output / 'summary.txt'}, summary.csv, summary.json", flush = True)


if __name__ == "__main__":
    main()
