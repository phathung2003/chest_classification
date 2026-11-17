"""Đánh giá ngoài trên Open-i (Indiana University Chest X-ray Collection): nhãn do người gán (mã MeSH <major> trong báo cáo), không dùng để huấn luyện
hay chọn mô hình. Thước đo chính của thí nghiệm phụ NVUM (prereg/analysis_plan.md mục 7, dòng 2026-11-15 23:50).

Bốn bước, mỗi bước một cờ:
1. --labels             Bảng nhãn: ảnh thẳng theo danh sách hướng chụp suy từ t-SNE của torchxrayvision (nlmcxr_tsne_pa.csv.gz, "PA": mỗi báo cáo
                        nhiều nhất một ảnh thẳng; thẻ View Position của DICOM không dùng vì gán nhầm nhiều ảnh nghiêng), 14 nhãn NIH theo bảng ánh xạ của
                        analysis/laterality/laterality_openi.py. Ghi dataset/openi/openi_frontal.csv và openi_classes.json (các lớp có đủ mẫu dương).
                        Chỉ đọc báo cáo và siêu dữ liệu, không cần ảnh, không cần mô hình.
2. --prepare PNG_DIR    Ảnh PNG gốc --> xám, cắt vuông ở giữa, thu về 512 x 512 (LANCZOS) --> dataset/openi/images_512/. Open-i không có mặt nạ phổi
                        CheXmask nên KHÔNG căn đường giữa như bộ ảnh huấn luyện (giới hạn, áp dụng như nhau cho mọi mô hình).
3. --score              Suy luận best_model.pt của các lượt chạy (--runs, --names) --> <output>/<name>.csv (xác suất từng lớp).
4. --compare A B        Macro AUROC trên các lớp đủ mẫu của hai tệp dự đoán (mỗi bên có thể là nhiều tệp: trung bình xác suất) và hiệu A − B,
                        bootstrap ghép cặp theo báo cáo (uid).
   --seed-spread ...    Độ lệch chuẩn macro AUROC giữa các tệp dự đoán (các seed của cùng một mô hình).

Ví dụ:
    python analysis/evaluate_openi.py --labels
    python analysis/evaluate_openi.py --prepare original_dataset/OpenI/NLMCXR_png
    python analysis/evaluate_openi.py --score --runs training_results/nih_chest_xray_14/convnextvmamba_cxr_ccmm_512px/run_1 --names ccmm_b_s42 --output training_results/explore/nvum_pilot/openi
    python analysis/evaluate_openi.py --compare training_results/explore/nvum_pilot/openi/ccmm_nvum_s42.csv training_results/explore/nvum_pilot/openi/ccmm_b_s42.csv
"""
# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "analysis" / "laterality"))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"                           # cuBLAS tất định (DETERMINISTIC_ALGORITHMS)
os.environ["TRITON_CACHE_AUTOTUNING"] = "1"                                  # mamba-ssm >= 2.3 ở chế độ tất định: lưu autotune kernel Triton thay vì cảnh báo

# Khai báo thư viện
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from laterality_openi import codes_xml, matches                              # Bảng ánh xạ nhãn NIH <-- mã MeSH của Open-i (cùng bảng với phân nhóm L/C)

OPENI_SOURCE        = config.ORIGINAL_DATASET_DIRECTORY / "OpenI"
REPORT_DIRECTORY    = OPENI_SOURCE / "reports" / "ecgen-radiology"           # Báo cáo XML của NLM
VIEW_PATH           = OPENI_SOURCE / "nlmcxr_tsne_pa.csv.gz"                 # Hướng chụp suy từ t-SNE (torchxrayvision/data), cột imageid, tsne-view
OPENI_DIRECTORY     = config.DATASET_DIRECTORY / "openi"
LABEL_PATH          = OPENI_DIRECTORY / "openi_frontal.csv"
CLASS_PATH          = OPENI_DIRECTORY / "openi_classes.json"
IMAGE_DIRECTORY     = OPENI_DIRECTORY / "images_512"
FRONTAL_VIEW        = "PA"                                                   # Giá trị tsne-view của ảnh thẳng
MIN_POSITIVE        = 30                                                     # Lớp được tính vào macro AUROC khi có ít nhất chừng này ảnh dương
IMAGE_SIZE          = 512
BOOTSTRAP           = 2000
BOOTSTRAP_SEED      = 0


# ============================================================
# Danh sách 14 lớp NIH theo thứ tự đầu ra của mô hình
def nih_classes():
    return list(json.loads(config.DATASET_INFO_DIRECTORY.read_text(encoding = "utf-8"))["classes"])


# ============================================================
# 1. Bảng nhãn ảnh thẳng
# Trả về: (DataFrame, danh sách lớp đủ mẫu dương)
def build_labels():
    import xml.etree.ElementTree as ET

    classes = nih_classes()

    # Báo cáo: uid, các ảnh của báo cáo, mã MeSH <major>
    rows = {}
    files = sorted(REPORT_DIRECTORY.glob("*.xml"), key = lambda path: int(path.stem))
    for path in files:
        root = ET.parse(path).getroot()
        uid = root.find("uId").attrib["id"]
        codes = codes_xml(path)
        labels = {name: int(any(matches(name, code) for code in codes)) for name in classes}
        for node in root.iter("parentImage"):
            rows[node.attrib["id"]] = {"uid": uid, "mesh_major": ";".join(codes), **labels}

    # Ảnh thẳng theo hướng chụp suy từ t-SNE
    views = pd.read_csv(VIEW_PATH)
    frontal = views.loc[views["tsne-view"] == FRONTAL_VIEW].drop_duplicates("imageid")
    missing = [image for image in frontal["imageid"] if image not in rows]
    frame = pd.DataFrame([{"Image Index": f"{image}.png", **rows[image]} for image in frontal["imageid"] if image in rows])
    frame["image_path"] = "openi/images_512/" + frame["Image Index"]
    frame = frame.sort_values("Image Index").reset_index(drop = True)

    # Lớp đủ mẫu dương (chỉ từ nhãn, trước mọi dự đoán)
    counts = frame[classes].sum()
    kept = [name for name in classes if counts[name] >= MIN_POSITIVE]

    OPENI_DIRECTORY.mkdir(parents = True, exist_ok = True)
    frame.to_csv(LABEL_PATH, index = False)
    CLASS_PATH.write_text(json.dumps({"min_positive": MIN_POSITIVE, "classes": kept, "positives": {name: int(counts[name]) for name in classes},
                                      "images": len(frame), "reports": int(frame["uid"].nunique()), "view_source": VIEW_PATH.name}, ensure_ascii = False, indent = 1), encoding = "utf-8")
    print(f"Báo cáo đọc được: {len(files)} | ảnh thẳng: {len(frame)} của {frame['uid'].nunique()} báo cáo | ảnh thẳng không có báo cáo: {len(missing)}")
    print("Số ảnh dương từng lớp: " + ", ".join(f"{name} {int(counts[name])}" for name in classes))
    print(f"Lớp tính vào macro AUROC (>= {MIN_POSITIVE} ảnh dương): {len(kept)} lớp: {kept}")
    print(f"Đã ghi {LABEL_PATH} và {CLASS_PATH}")
    return frame, kept


# ============================================================
# 2. Chuẩn bị ảnh 512 x 512
def prepare_images(source):
    from PIL import Image

    frame = pd.read_csv(LABEL_PATH)
    available = {path.name: path for path in Path(source).rglob("*.png")}        # Ảnh gốc có thể nằm trong thư mục con sau khi giải nén
    IMAGE_DIRECTORY.mkdir(parents = True, exist_ok = True)
    missing, written = [], 0
    for name in frame["Image Index"]:
        target = IMAGE_DIRECTORY / name
        if target.is_file():
            continue
        path = available.get(name)
        if path is None:
            missing.append(name)
            continue
        with Image.open(path) as image:
            image = image.convert("L")
            side = min(image.size)
            left, top = (image.width - side) // 2, (image.height - side) // 2
            image.crop((left, top, left + side, top + side)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save(target)
        written += 1
    print(f"Đã ghi {written} ảnh vào {IMAGE_DIRECTORY} | đã có sẵn {len(frame) - written - len(missing)} | thiếu ảnh gốc {len(missing)}" + (f", ví dụ {missing[:3]}" if missing else ""))
    if missing:
        raise SystemExit("Thiếu ảnh gốc: kiểm tra thư mục PNG của Open-i (tệp dạng CXR1000_IM-0003-2001.png).")


# ============================================================
# 3. Suy luận một lượt chạy trên Open-i
# Trả về: DataFrame nhãn kèm cột prob_<lớp>
def score_run(run_directory, fold, workers, device):
    from analysis.evaluate_locked import build_model
    from training.evaluate import evaluate
    from utilities.data import create_dataloader, transform_setting
    from utilities.dataset import NIHChestXrayDataset
    from utilities.file import load_torch, read_json
    from utilities.metric import with_probabilities
    from utilities.setting import apply_config

    info = read_json(run_directory / "train_info.json")
    settings = info["settings"]
    classes = list(settings["classes"])
    if int(settings["image_size"]) != IMAGE_SIZE:
        raise SystemExit(f"{run_directory}: lượt chạy ở {settings['image_size']} px, bộ ảnh Open-i đã chuẩn bị ở {IMAGE_SIZE} px.")
    apply_config({**config.CLIS_CONFIG, "IMAGE_SIZE": IMAGE_SIZE, **({} if workers is None else {"NUMBER_OF_WORKERS": int(workers)})})

    frame = pd.read_csv(LABEL_PATH)
    absent = [name for name in frame["Image Index"].head(50) if not (IMAGE_DIRECTORY / name).is_file()]
    if absent:
        raise SystemExit(f"Chưa có ảnh trong {IMAGE_DIRECTORY}: chạy --prepare trước.")

    best = load_torch(run_directory / f"fold_{fold}" / "best_model.pt", device)
    if best.get("signature") != info["signature"] or int(best.get("fold")) != int(fold):
        raise SystemExit(f"{run_directory}: best_model.pt không khớp chữ ký / Fold của lượt chạy.")
    model = build_model(settings, len(classes)).to(device)
    model.load_state_dict(best["model_state_dict"], strict = True)

    _, eval_transform = transform_setting()
    _, loader = create_dataloader(NIHChestXrayDataset, classes, device, frame, eval_transform, False, batch_size = int(settings["training_config"]["batch_size"]), seed = int(settings["seed"]) + 900_000)
    _, _, targets, probabilities = evaluate(device, len(classes), model, loader)
    if not (targets == frame[classes].to_numpy(dtype = np.float32)).all():
        raise SystemExit(f"{run_directory}: nhãn của loader khác bảng nhãn.")
    del model, best
    return with_probabilities(frame, probabilities, classes)


# ============================================================
# 4. Chỉ số và so sánh
# Đọc một bên so sánh (nhiều tệp: trung bình xác suất theo ảnh)
# Trả về: (bảng nhãn, ma trận xác suất [ảnh, lớp đủ mẫu], danh sách lớp)
def read_side(paths, kept):
    frames = [pd.read_csv(path).sort_values("Image Index").reset_index(drop = True) for path in paths]
    for frame in frames[1:]:
        if not frame["Image Index"].equals(frames[0]["Image Index"]):
            raise SystemExit(f"Các tệp dự đoán không cùng danh sách ảnh: {paths}")
    probabilities = np.mean([frame[[f"prob_{name}" for name in kept]].to_numpy(dtype = np.float64) for frame in frames], axis = 0)
    return frames[0], probabilities


# Macro AUROC trên các lớp đủ mẫu
# Trả về: (macro, dict AUROC từng lớp)
def macro_auroc(truth, probabilities, kept):
    scores = {name: float(roc_auc_score(truth[:, index], probabilities[:, index])) for index, name in enumerate(kept)}
    return float(np.mean(list(scores.values()))), scores


# Hiệu macro AUROC A − B, bootstrap ghép cặp theo báo cáo (uid)
# Trả về: dict kết quả
def compare(paths_a, paths_b, name_a = "A", name_b = "B", iterations = BOOTSTRAP, seed = BOOTSTRAP_SEED):
    kept = json.loads(CLASS_PATH.read_text(encoding = "utf-8"))["classes"]
    frame, prob_a = read_side(paths_a, kept)
    frame_b, prob_b = read_side(paths_b, kept)
    if not frame["Image Index"].equals(frame_b["Image Index"]):
        raise SystemExit("Hai bên không cùng danh sách ảnh.")
    truth = frame[kept].to_numpy(dtype = np.int64)
    macro_a, class_a = macro_auroc(truth, prob_a, kept)
    macro_b, class_b = macro_auroc(truth, prob_b, kept)

    # Bootstrap theo báo cáo: lấy lại uid, giữ mọi ảnh của uid; lớp mất hết mẫu dương hoặc âm trong một lần lấy thì lần đó bỏ qua
    groups = frame.groupby("uid").indices
    keys = list(groups)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(int(iterations)):
        rows = np.concatenate([groups[keys[index]] for index in rng.integers(0, len(keys), size = len(keys))])
        sample = truth[rows]
        if ((sample.sum(axis = 0) == 0) | (sample.sum(axis = 0) == len(rows))).any():
            continue
        differences.append(macro_auroc(sample, prob_a[rows], kept)[0] - macro_auroc(sample, prob_b[rows], kept)[0])
    low, high = (float(value) for value in np.percentile(differences, [2.5, 97.5]))
    result = {"name_a": name_a, "name_b": name_b, "files_a": [str(path) for path in paths_a], "files_b": [str(path) for path in paths_b],
              "images": len(frame), "reports": len(keys), "classes": kept, "macro_auroc_a": macro_a, "macro_auroc_b": macro_b,
              "difference": macro_a - macro_b, "ci95": [low, high], "bootstrap_used": len(differences), "bootstrap_seed": int(seed),
              "class_auroc_a": class_a, "class_auroc_b": class_b}
    print(f"Open-i | {len(frame)} ảnh, {len(keys)} báo cáo, {len(kept)} lớp | {name_a} {macro_a:.4f} | {name_b} {macro_b:.4f} | "
          f"Δ = {macro_a - macro_b:+.4f} [{low:+.4f}; {high:+.4f}] (bootstrap {len(differences)}/{int(iterations)})")
    for name in kept:
        print(f"  {name:20s} {class_a[name]:.4f}  {class_b[name]:.4f}  {class_a[name] - class_b[name]:+.4f}")
    return result


# Độ lệch chuẩn macro AUROC giữa các tệp dự đoán (các seed)
# Trả về: (danh sách macro AUROC, SD mẫu)
def seed_spread(paths):
    kept = json.loads(CLASS_PATH.read_text(encoding = "utf-8"))["classes"]
    scores = []
    for path in paths:
        frame, probabilities = read_side([path], kept)
        scores.append(macro_auroc(frame[kept].to_numpy(dtype = np.int64), probabilities, kept)[0])
    spread = float(np.std(scores, ddof = 1)) if len(scores) > 1 else float("nan")
    print("Open-i macro AUROC theo tệp: " + ", ".join(f"{score:.4f}" for score in scores) + f" | SD {spread:.4f}")
    return scores, spread


# ============================================================
def main():
    parser = argparse.ArgumentParser(description = "Đánh giá ngoài trên Open-i")
    parser.add_argument("--labels", action = "store_true", help = "Dựng bảng nhãn ảnh thẳng")
    parser.add_argument("--prepare", default = None, metavar = "PNG_DIR", help = "Thư mục PNG gốc của Open-i")
    parser.add_argument("--score", action = "store_true", help = "Suy luận các lượt chạy trên Open-i")
    parser.add_argument("--runs", nargs = "+", default = [], help = "Thư mục lượt chạy (chứa train_info.json)")
    parser.add_argument("--names", nargs = "+", default = [], help = "Tên tệp dự đoán của từng lượt chạy")
    parser.add_argument("--fold", type = int, default = 1)
    parser.add_argument("--workers", type = int, default = None)
    parser.add_argument("--output", default = str(config.MODELS_OUTPUT_DIRECTORY / "explore" / "nvum_pilot" / "openi"))
    parser.add_argument("--compare", nargs = 2, default = None, metavar = ("A", "B"), help = "Hai bên so sánh; mỗi bên là một tệp hoặc nhiều tệp nối bằng dấu phẩy")
    parser.add_argument("--name-a", default = "A")
    parser.add_argument("--name-b", default = "B")
    parser.add_argument("--json", default = None, help = "Ghi kết quả --compare / --seed-spread ra tệp JSON")
    parser.add_argument("--seed-spread", nargs = "+", default = None, metavar = "CSV", help = "SD macro AUROC giữa các tệp dự đoán")
    args = parser.parse_args()

    if args.labels:
        build_labels()
    if args.prepare:
        prepare_images(args.prepare)
    if args.score:
        from utilities.file import save_csv
        from utilities.setting import device_info
        if len(args.runs) != len(args.names) or not args.runs:
            parser.error("--score cần --runs và --names cùng số phần tử")
        device = device_info()
        output = Path(args.output)
        output.mkdir(parents = True, exist_ok = True)
        for run, name in zip(args.runs, args.names):
            target = output / f"{name}.csv"
            if target.is_file():
                print(f"{name}: đã có {target}, bỏ qua")
                continue
            save_csv(target, score_run(Path(run), args.fold, args.workers, device))
            print(f"{name}: đã ghi {target}")
    if args.compare:
        result = compare(args.compare[0].split(","), args.compare[1].split(","), args.name_a, args.name_b)
        if args.json:
            Path(args.json).write_text(json.dumps(result, ensure_ascii = False, indent = 1), encoding = "utf-8")
    if args.seed_spread:
        scores, spread = seed_spread(args.seed_spread)
        if args.json:
            Path(args.json).write_text(json.dumps({"files": args.seed_spread, "macro_auroc": scores, "sd": spread}, ensure_ascii = False, indent = 1), encoding = "utf-8")


if __name__ == "__main__":
    main()
