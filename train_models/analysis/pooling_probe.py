"""Thử pooling theo lớp trên checkpoint đã huấn luyện, KHÔNG huấn luyện lại (Validation, không mở Test).

Vì sao cần: ở 512 px bản đồ đặc trưng cuối có 16 x 16 = 256 token và đầu phân loại lấy trung bình toàn cục (GAP). Tổn thương chiếm 1-4 token
chỉ đóng góp khoảng 1% vào trung bình. Trước khi tốn GPU huấn luyện một đầu pooling theo lớp, script đo xem chỉ đổi cách gộp token
(giữ nguyên mọi trọng số) có làm đổi AP / AUROC trên Validation không. Kết quả âm KHÔNG chứng minh đầu được huấn luyện sẽ vô ích;
kết quả dương cho biết bản đồ token đã mang tín hiệu khu trú mà GAP làm loãng.

Cách tính (cùng một công thức cho mọi họ mô hình): gọi x_t là token trước pooling, h là phần còn lại của đầu phân loại
(VMamba-T, ccmm: LayerNorm đã áp trước pooling nên h = Linear; ConvNeXt-T: h = LayerNorm rồi Linear).
  - Điểm token của lớp k    : s_kt = h(x_t)[k]
  - Logit gộp với trọng số a: z_k(a) = h(sum_t a_kt · x_t)[k], sum_t a_kt = 1. a đều = đúng logit của mô hình đã huấn luyện.
Biến thể:
  - mean          : a đều (đối chứng; so với đầu ra mô hình và oof_predictions.csv để kiểm tra việc nạp trọng số)
  - top<k>        : a đều trên k token có s_kt lớn nhất của lớp k (top1 = max pooling)
  - csra_soft_<w> : z(mean) + w · z(softmax_t s_kt)        (kiểu CSRA, nhiệt độ 1)
  - csra_max_<w>  : z(mean) + w · z(top1)                  (kiểu CSRA, nhiệt độ vô cùng)

Đọc kết quả (quy tắc ghi trước khi chạy: prereg/analysis_plan.md mục 7, dòng 2026-11-14): xác suất trung bình qua seed của từng mô hình;
bệnh nhân Validation chia đôi cố định (seed 0): CHỌN biến thể có macro AP cao nhất trên nửa A, BÁO CÁO hiệu (biến thể - mean) trên nửa B
bằng bootstrap ghép cặp theo bệnh nhân. Chia đôi để mức tăng báo cáo không bị thổi phồng do chọn 1 trong 10 biến thể.

Ví dụ (WSL2, conda activate mamba-t29):
    python analysis/pooling_probe.py --runs "training_results/nih_chest_xray_14/convnextvmamba_cxr_ccmm_512px/run_[123]"
    python analysis/pooling_probe.py --summary-only
Kết quả trong training_results/explore/pooling/: <mô hình>_seed<seed>.npz (xác suất từng biến thể), summary.csv, summary.json, summary.txt.
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

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

import config
import models.convnext_tiny as convnext
import models.stitch_cxr as stx
import models.vmamba_cxr as vmx
from analysis.evaluate_locked import build_model
from utilities.data import create_dataloader, load_dataset, transform_setting
from utilities.dataset import NIHChestXrayDataset
from utilities.file import load_torch, read_json, save_csv, save_json
from utilities.log import timestamp
from utilities.setting import apply_config
from utilities.statistics import _bootstrap_samples, _interval, _macro, auroc_per_class

OUTPUT_DIRECTORY = config.MODELS_OUTPUT_DIRECTORY / "explore" / "pooling"
TOP_K = (1, 4, 16, 64)                                                      # Số token giữ lại của biến thể top<k> (512 px: 256 token)
CSRA_WEIGHTS = (0.2, 0.5, 1.0)                                              # Trọng số w của phần theo lớp trong biến thể kiểu CSRA
BASELINE = "mean"                                                           # Biến thể đối chứng: GAP của mô hình đã huấn luyện
LOCALIZED = ["Nodule", "Mass", "Pneumothorax"]                              # Nhãn tổn thương khu trú báo cáo kèm (ghi trước trong prereg)
SPLIT_SEED = 0                                                              # Seed chia đôi bệnh nhân Validation (nửa A chọn, nửa B báo cáo)
MIN_AP_GAIN = 0.003                                                         # Quy tắc: Δ macro AP trên nửa B >= 0,003 và cận dưới CI > 0
MAX_AUROC_LOSS = 0.001                                                      # Quy tắc: macro AUROC trên nửa B không giảm quá 0,001
SAVED_DIFF_WARNING = 5e-3                                                   # GPU, cùng AMP với lúc huấn luyện: lệch xác suất so với oof_predictions.csv lớn hơn mức này là bất thường (CPU float32 lệch cỡ 1e-2 so với BF16)


# ============================================================
# Tên các biến thể pooling, theo thứ tự lưu trong tệp .npz
# Trả về: Danh sách tên
def variant_names():
    return [BASELINE] + [f"top{k}" for k in TOP_K] + [f"csra_soft_{w}" for w in CSRA_WEIGHTS] + [f"csra_max_{w}" for w in CSRA_WEIGHTS]


# Mở rộng danh sách thư mục lượt chạy (mẫu * do script tự mở rộng)
# Trả về: Danh sách đường dẫn
def expand_runs(patterns):
    return [Path(path).resolve() for pattern in patterns for path in (sorted(glob.glob(str(pattern))) if glob.has_magic(str(pattern)) else [pattern])]


# Tên ngắn của mô hình trong lượt chạy (gộp các seed của cùng một mô hình)
# Trả về: Chuỗi
def model_tag(settings):
    name, kwargs = settings["model_name"], settings["model_kwargs"]
    if name.startswith(stx.MODEL_NAME):
        return kwargs["layout"]
    if name.startswith(vmx.MODEL_NAME):
        return kwargs["mixer"] + ("" if kwargs.get("route_init", "zero") == "zero" else f"_{kwargs['route_init']}") + ("" if kwargs.get("route_sign", "none") == "none" else f"_{kwargs['route_sign']}")
    if name.startswith(convnext.MODEL_NAME):
        return "convnext_t"
    raise ValueError(f"Mô hình '{name}' chưa được hỗ trợ: chỉ ConvNeXt-T baseline, VMamba-T CXR và mô hình ghép tầng")


# ============================================================
# Phần đầu phân loại của ConvNeXt-T sau pooling (timm NormMlpClassifierHead: LayerNorm --> flatten --> Linear 14 lớp của mô hình)
# features: (..., C) --> logit (..., K)
def convnext_logits(model, features):
    head = model.cnn.head
    x = features.reshape(-1, features.shape[-1], 1, 1)
    x = head.fc(head.drop(head.pre_logits(head.flatten(head.norm(x)))))
    return model.classifier(x).reshape(*features.shape[:-1], -1)


# Lớp pooling cuối của mô hình (đầu vào của nó là bản đồ token (b, C, h, w)) và hàm h đưa đặc trưng đã gộp (..., C) về logit (..., K)
# Trả về: (module pooling, hàm h)
def pooling_parts(model):
    if isinstance(model, vmx.VMambaCXR):                                                        # LayerNorm --> GAP --> Linear
        return model.backbone.classifier.avgpool, model.classifier
    if isinstance(model, stx.StitchCXR) and model.layout[-1] == "m":
        return model.mamba.classifier.avgpool, model.classifier
    if isinstance(model, (convnext.ConvNeXtTinyBaseline, stx.StitchCXR)):                       # GAP --> LayerNorm --> Linear
        return model.cnn.head.global_pool, lambda features: convnext_logits(model, features)
    raise ValueError(f"{type(model).__name__}: chưa biết lớp pooling cuối")


# Logit của mọi biến thể pooling từ bản đồ token
# tokens: (b, T, C) float32; to_logits: hàm h
# Trả về: dict {tên biến thể: logit (b, K)}
def pooled_logits(tokens, to_logits):
    count = tokens.shape[1]
    scores = to_logits(tokens).transpose(1, 2)                                                  # s_kt: (b, K, T)

    # z_k(a) = h(sum_t a_kt · x_t)[k]
    def pooled(weights):
        return to_logits(torch.einsum("bkt,btc->bkc", weights, tokens)).diagonal(dim1 = -2, dim2 = -1)

    # a đều trên k token có điểm lớn nhất của từng lớp
    def top(k):
        k = min(int(k), count)
        return torch.zeros_like(scores).scatter_(-1, scores.topk(k, dim = -1).indices, 1.0 / k)

    logits = {BASELINE: pooled(torch.full_like(scores, 1.0 / count))}
    for k in TOP_K:
        logits[f"top{k}"] = pooled(top(k))
    soft = pooled(torch.softmax(scores, dim = -1))
    for weight in CSRA_WEIGHTS:
        logits[f"csra_soft_{weight}"] = logits[BASELINE] + weight * soft
    for weight in CSRA_WEIGHTS:
        logits[f"csra_max_{weight}"] = logits[BASELINE] + weight * logits["top1"]
    return logits


# ============================================================
# Đo một lượt chạy trên Validation của fold: xác suất của mọi biến thể pooling
# Trả về: (đường dẫn tệp .npz, dict thông tin kiểm tra)
def measure(run_directory, fold, limit, workers, device, output_directory):
    info = read_json(run_directory / "train_info.json")
    settings = info["settings"]
    fold_directory = run_directory / f"fold_{fold}"
    tag, seed = model_tag(settings), int(settings["seed"])

    # Cấu hình của lượt chạy: độ phân giải và bộ ảnh
    snapshot = settings.get("config", {})
    apply_config({**config.CLIS_CONFIG, "IMAGE_SIZE": int(settings["image_size"]), "IMAGE_VARIANT": snapshot.get("IMAGE_VARIANT", config.CLIS_CONFIG["IMAGE_VARIANT"]),
                  **({} if workers is None else {"NUMBER_OF_WORKERS": int(workers)})})

    # Ảnh Validation của fold
    train_df, _, _, classes, _ = load_dataset()
    frame = train_df[train_df["Fold"] == int(fold)].reset_index(drop = True)
    frame = frame if limit is None else frame.iloc[:int(limit)].reset_index(drop = True)

    # Nạp trọng số tốt nhất (đúng chữ ký, đúng Fold); móc vào lớp pooling cuối để lấy bản đồ token của đúng lần suy luận đó
    best = load_torch(fold_directory / "best_model.pt", device)
    if best.get("signature") != info["signature"] or int(best.get("fold")) != int(fold):
        raise SystemExit(f"{run_directory}: best_model.pt không khớp chữ ký / Fold của lượt chạy.")
    model = build_model(settings, len(classes)).to(device).eval()
    model.load_state_dict(best["model_state_dict"], strict = True)
    pooling, to_logits = pooling_parts(model)
    captured = {}
    handle = pooling.register_forward_hook(lambda module, inputs, output: captured.update(tokens = inputs[0]))

    # Suy luận: backbone dưới AMP như lúc huấn luyện, phần gộp token tính bằng float32
    _, eval_transform = transform_setting()
    _, loader = create_dataloader(NIHChestXrayDataset, classes, device, frame, eval_transform, False, batch_size = int(settings["training_config"]["batch_size"]), seed = seed + 900_000)
    names = variant_names()
    probabilities, targets, model_diff = {name: [] for name in names}, [], 0.0
    amp_dtype = torch.bfloat16 if config.AMP_DTYPE == "bfloat16" else torch.float16
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking = True)
            with torch.autocast(device_type = device.type, dtype = amp_dtype, enabled = bool(config.USE_AMP and device.type == "cuda")):
                reference = model(images).float()
            tokens = captured.pop("tokens").float().flatten(2).transpose(1, 2)                  # (b, C, h, w) --> (b, T, C)
            logits = pooled_logits(tokens, to_logits)
            model_diff = max(model_diff, float((torch.sigmoid(logits[BASELINE]) - torch.sigmoid(reference)).abs().max()))
            for name in names:
                probabilities[name].append(torch.sigmoid(logits[name]).cpu())
            targets.append(labels)
    handle.remove()
    stacked = np.stack([torch.cat(probabilities[name]).numpy().astype(np.float32) for name in names])               # (V, N, K)
    targets = torch.cat(targets).numpy()
    if not (targets == frame[classes].to_numpy(dtype = np.float32)).all():
        raise SystemExit(f"{run_directory}: nhãn của loader khác khung dữ liệu.")

    # Kiểm tra nạp đúng: xác suất "mean" so với oof_predictions.csv đã lưu lúc huấn luyện (ghép theo mã ảnh)
    saved_path, saved_diff = fold_directory / "oof_predictions.csv", float("nan")
    if saved_path.is_file():
        saved = pd.read_csv(saved_path).set_index(config.IMAGE_ID_COLUMN).loc[frame[config.IMAGE_ID_COLUMN], [f"prob_{name}" for name in classes]].to_numpy(dtype = np.float64)
        saved_diff = float(np.abs(stacked[0] - saved).max())

    check = {"run": str(run_directory.relative_to(config.MODELS_OUTPUT_DIRECTORY)).replace("\\", "/"), "model": tag, "seed": seed, "images": len(frame),
             "tokens": int(tokens.shape[1]), "model_diff": model_diff, "saved_diff": saved_diff, "measured_at": timestamp()}
    del model, best

    # Chỉ ghi tệp khi đo toàn bộ Validation
    path = output_directory / f"{tag}_seed{seed}.npz"
    if limit is None:
        output_directory.mkdir(parents = True, exist_ok = True)
        np.savez_compressed(path, probabilities = stacked, targets = targets.astype(np.int8), variants = np.array(names), classes = np.array(classes),
                            image_ids = frame[config.IMAGE_ID_COLUMN].to_numpy(dtype = str), patients = frame[config.PATIENT_ID_COLUMN].to_numpy(dtype = str),
                            **{key: np.array(value) for key, value in check.items()})
    return path, check


# ============================================================
# AP từng nhãn (NaN nếu nhãn không có ca dương hoặc không có ca âm)
# Trả về: np.ndarray (số nhãn,)
def ap_per_class(targets, probabilities):
    return np.array([average_precision_score(targets[:, k], probabilities[:, k]) if 0 < targets[:, k].sum() < len(targets) else np.nan for k in range(targets.shape[1])])


# Chia đôi bệnh nhân cố định theo seed (không phụ thuộc thứ tự ảnh)
# Trả về: Mặt nạ ảnh thuộc nửa A
def half_a_mask(patients, seed = SPLIT_SEED):
    unique = np.array(sorted(set(patients.tolist())))
    order = np.random.default_rng(int(seed)).permutation(len(unique))
    return np.isin(patients, unique[order[:(len(unique) + 1) // 2]])


# Ba chỉ số tóm tắt của một bộ xác suất: macro AP 14 nhãn, AP trung bình nhóm khu trú, macro AUROC 14 nhãn
# Trả về: dict
def summary_metrics(targets, probabilities, localized):
    ap = ap_per_class(targets, probabilities)
    return {"macro_ap": _macro(ap), "localized_ap": _macro(ap[localized]), "macro_auroc": _macro(auroc_per_class(targets, probabilities))}


# Bootstrap ghép cặp theo bệnh nhân của hiệu (biến thể - đối chứng) cho ba chỉ số tóm tắt
# Trả về: dict {chỉ số: {difference, ci_low, ci_high, p_one_sided_a_greater}}
def paired_bootstrap(targets, variant, baseline, patients, localized, bootstrap_iteration, seed, alpha):
    def difference(rows):
        a, b = summary_metrics(targets[rows], variant[rows], localized), summary_metrics(targets[rows], baseline[rows], localized)
        return {key: a[key] - b[key] for key in a}

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


# Đọc các tệp .npz của cùng một mô hình (mỗi seed một tệp) và trung bình xác suất qua seed
# Trả về: dict {variants, classes, targets, patients, probabilities (V, N, K), per_seed (S, V, N, K), seeds, checks}
def load_model_files(paths):
    files = [np.load(path, allow_pickle = False) for path in paths]
    base = files[0]
    for path, data in zip(paths, files):
        if list(data["variants"]) != list(base["variants"]) or list(data["classes"]) != list(base["classes"]):
            raise ValueError(f"{path}: danh sách biến thể / lớp khác tệp đầu tiên")
        if data["image_ids"].shape != base["image_ids"].shape or not (data["image_ids"] == base["image_ids"]).all() or not (data["targets"] == base["targets"]).all():
            raise ValueError(f"{path}: tập ảnh hoặc nhãn khác tệp đầu tiên")
    per_seed = np.stack([data["probabilities"].astype(np.float64) for data in files])
    return {"variants": [str(v) for v in base["variants"]], "classes": [str(c) for c in base["classes"]], "targets": base["targets"].astype(np.float64),
            "patients": base["patients"], "probabilities": per_seed.mean(axis = 0), "per_seed": per_seed, "seeds": [int(data["seed"]) for data in files],
            "checks": [{key: data[key].item() for key in ("run", "seed", "images", "tokens", "model_diff", "saved_diff")} for data in files]}


# ============================================================
# Tổng hợp mọi tệp .npz trong thư mục kết quả: bảng theo biến thể, chọn trên nửa A, bootstrap trên nửa B, kết luận theo quy tắc
# Trả về: (bảng DataFrame, dict kết quả, danh sách dòng văn bản)
def summarize(output_directory, bootstrap_iteration, seed):
    alpha = round(1 - config.BOOTSTRAP_CONFIDENCE, 12)
    groups = {}
    for path in sorted(output_directory.glob("*_seed*.npz")):
        groups.setdefault(path.stem.rsplit("_seed", 1)[0], []).append(path)
    if not groups:
        raise SystemExit(f"Không có tệp .npz nào trong {output_directory}: chạy với --runs trước.")

    table, result, lines = [], {"split_seed": SPLIT_SEED, "bootstrap_iteration": int(bootstrap_iteration), "bootstrap_seed": int(seed), "confidence": 1 - alpha,
                                "localized": LOCALIZED, "rule": {"min_ap_gain": MIN_AP_GAIN, "max_auroc_loss": MAX_AUROC_LOSS}, "models": {}}, []
    for tag, paths in groups.items():
        data = load_model_files(paths)
        variants, classes, targets, patients = data["variants"], data["classes"], data["targets"], data["patients"]
        localized = [classes.index(name) for name in LOCALIZED]
        in_a = half_a_mask(patients)
        halves = {"full": np.ones(len(targets), dtype = bool), "A": in_a, "B": ~in_a}

        # Bảng theo biến thể: chỉ số của xác suất trung bình qua seed trên toàn bộ Validation, nửa A, nửa B; macro AP từng seed (toàn bộ)
        metrics = {}
        for v, name in enumerate(variants):
            metrics[name] = {half: summary_metrics(targets[mask], data["probabilities"][v][mask], localized) for half, mask in halves.items()}
            per_seed = [_macro(ap_per_class(targets, seed_probabilities[v])) for seed_probabilities in data["per_seed"]]
            table.append({"model": tag, "variant": name, "seeds": len(paths), "macro_ap_seed_mean": float(np.mean(per_seed)),
                          **{f"{key}_{half}": value for half, values in metrics[name].items() for key, value in values.items()}})

        # Chọn trên nửa A (không tính đối chứng), báo cáo trên nửa B
        chosen = max((name for name in variants if name != BASELINE), key = lambda name: metrics[name]["A"]["macro_ap"])
        b = halves["B"]
        v, base = variants.index(chosen), variants.index(BASELINE)
        comparison = paired_bootstrap(targets[b], data["probabilities"][v][b], data["probabilities"][base][b], patients[b], localized, bootstrap_iteration, seed, alpha)
        ap_gain, auroc_change = comparison["macro_ap"], comparison["macro_auroc"]
        passed = bool(ap_gain["difference"] >= MIN_AP_GAIN and ap_gain["ci_low"] is not None and ap_gain["ci_low"] > 0 and auroc_change["difference"] >= -MAX_AUROC_LOSS)
        class_ap = ap_per_class(targets[b], data["probabilities"][v][b]) - ap_per_class(targets[b], data["probabilities"][base][b])
        result["models"][tag] = {"seeds": data["seeds"], "checks": data["checks"], "images": int(len(targets)), "images_half_b": int(b.sum()),
                                 "patients_half_b": int(len(set(patients[b].tolist()))), "chosen_on_half_a": chosen, "half_b": comparison, "passed": passed,
                                 "class_ap_difference_half_b": {name: float(value) for name, value in zip(classes, class_ap)}, "metrics": metrics}

        # Dòng văn bản
        worst = max(data["checks"], key = lambda check: check["saved_diff"])
        lines.append(f"== {tag} | {len(paths)} seed {data['seeds']} | {len(targets):,} ảnh | lệch lớn nhất so với oof đã lưu {worst['saved_diff']:.1e}, so với đầu ra mô hình {max(c['model_diff'] for c in data['checks']):.1e}")
        lines.append(f"   mean (GAP)        : macro AP {metrics[BASELINE]['full']['macro_ap']:.4f} | AP khu trú {metrics[BASELINE]['full']['localized_ap']:.4f} | macro AUROC {metrics[BASELINE]['full']['macro_auroc']:.4f} (toàn bộ Validation)")
        lines.append(f"   chọn trên nửa A   : {chosen} (macro AP nửa A {metrics[chosen]['A']['macro_ap']:.4f} so với mean {metrics[BASELINE]['A']['macro_ap']:.4f})")
        for key, label in (("macro_ap", "Δ macro AP"), ("localized_ap", "Δ AP khu trú"), ("macro_auroc", "Δ macro AUROC")):
            row = comparison[key]
            lines.append(f"   nửa B {label:14s}: {row['difference']:+.4f} [CI {row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | p một phía {row['p_one_sided_a_greater']:.4f}")
        lines.append("   nửa B ΔAP từng nhãn: " + ", ".join(f"{name} {value:+.4f}" for name, value in zip(classes, class_ap)))
        lines.append(f"   QUY TẮC (Δ macro AP >= {MIN_AP_GAIN}, cận dưới CI > 0, Δ macro AUROC >= -{MAX_AUROC_LOSS}): {'ĐẠT' if passed else 'KHÔNG ĐẠT'}")
    return pd.DataFrame(table), result, lines


# ============================================================
def main():
    parser = argparse.ArgumentParser(description = "Thử pooling theo lớp trên checkpoint đã huấn luyện (Validation, không huấn luyện lại)")
    parser.add_argument("--runs", nargs = "+", default = [], help = "Thư mục lượt chạy (run_N), cho phép mẫu *")
    parser.add_argument("--fold", type = int, default = 1)
    parser.add_argument("--limit", type = int, default = None, help = "Chỉ dùng N ảnh Validation đầu tiên (kiểm tra nhanh, không ghi tệp, không tổng hợp)")
    parser.add_argument("--workers", type = int, default = None)
    parser.add_argument("--device", default = None, help = "cuda | cpu (mặc định: cuda nếu có)")
    parser.add_argument("--output", type = Path, default = OUTPUT_DIRECTORY)
    parser.add_argument("--summary-only", action = "store_true", help = "Không suy luận, chỉ tổng hợp lại các tệp .npz đã có")
    parser.add_argument("--n-boot", type = int, default = int(config.COMPARE_BOOTSTRAP_ITERATION))
    parser.add_argument("--seed", type = int, default = 0, help = "Seed của bootstrap (đăng ký trước: 0)")
    args = parser.parse_args()
    if not args.runs and not args.summary_only:
        parser.error("cần --runs hoặc --summary-only")

    # 1. Suy luận từng lượt chạy
    if not args.summary_only:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cpu":
            vmx.vm.WITH_SELECTIVESCAN_MAMBA = False                                             # Selective scan bản PyTorch của VMamba trên CPU
        written = {}
        for run_directory in expand_runs(args.runs):
            if not (run_directory / f"fold_{args.fold}" / "fold_done.json").is_file():
                print(f"Bỏ qua {run_directory}: chưa xong fold {args.fold}", flush = True)
                continue
            try:
                path, check = measure(run_directory, args.fold, args.limit, args.workers, device, args.output)
            except (SystemExit, RuntimeError, KeyError, ValueError, OSError) as error:
                print(f"Lỗi khi đo {run_directory}: {error}", flush = True)
                continue
            if path in written:
                raise SystemExit(f"{run_directory} và {written[path]} cùng mô hình và seed: tệp {path.name} sẽ bị ghi đè. Chỉ truyền một lượt cho mỗi (mô hình, seed).")
            written[path] = run_directory
            print(f"{check['run']} | {check['model']} seed {check['seed']} | {check['images']:,} ảnh, {check['tokens']} token | "
                  f"lệch so với đầu ra mô hình {check['model_diff']:.1e}, so với oof đã lưu {check['saved_diff']:.1e}"
                  + ("" if args.limit is None else " | --limit: không ghi tệp"), flush = True)
            if device.type == "cuda" and check["saved_diff"] > SAVED_DIFF_WARNING:
                print(f"CẢNH BÁO: {check['run']} lệch {check['saved_diff']:.1e} so với oof đã lưu (> {SAVED_DIFF_WARNING}): kiểm tra việc nạp trọng số trước khi đọc kết quả", flush = True)
        if args.limit is not None:
            return

    # 2. Tổng hợp
    table, result, lines = summarize(args.output, args.n_boot, args.seed)
    save_csv(args.output / "summary.csv", table)
    save_json(args.output / "summary.json", result)
    (args.output / "summary.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")
    print("\n".join(lines))
    print(f"Đã ghi {args.output / 'summary.txt'}, summary.csv, summary.json", flush = True)


if __name__ == "__main__":
    main()
