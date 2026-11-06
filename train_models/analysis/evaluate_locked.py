"""Chấm các checkpoint đã khóa của nghiên cứu CLIS trên Test, đúng một lần ở cổng G3.

Vì sao cần script riêng: EVALUATE_FOLD_MODELS_ON_TEST nằm trong chữ ký thí nghiệm, nên bật cờ này ở models/run_clis.py sẽ tạo lượt chạy mới
và huấn luyện lại thay vì chấm checkpoint đã khóa. Script này không đổi chữ ký và không ghi đè kết quả huấn luyện.

Quy trình:
1. Kiểm chứng, không mở Test: --split validation chấm lại Validation từ best_model.pt rồi so với oof_predictions.csv đã lưu. Chỉ in độ lệch xác suất, không in chỉ số.
2. Cổng G3: --split test --confirm-g3, cần tag prereg-v1. Ghi SHA-256 của best_model.pt vào training_results/g3_manifest.json trước khi suy luận,
   lưu fold_<k>/test/test_predictions.csv, không in chỉ số. Sau đó phân tích bằng compare_clis.py và compare_resolution.py theo đúng kế hoạch.
   ** Tệp dự đoán đã có và checkpoint không đổi --> bỏ qua. Checkpoint đã đổi --> dừng, không chấm lại.

Ví dụ:
    python analysis/evaluate_locked.py --runs "training_results/nih_chest_xray_14/densenet121_baseline_v1_clis_512px/run_1" --split validation --limit 256 --workers 2
    python analysis/evaluate_locked.py --runs "training_results/nih_chest_xray_14/clismamba_v1_m_clis_1024px/run_*" --split test --confirm-g3
"""
# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"                           # cuBLAS tất định (DETERMINISTIC_ALGORITHMS)
os.environ["TRITON_CACHE_AUTOTUNING"] = "1"                                  # mamba-ssm >= 2.3 ở chế độ tất định: lưu autotune kernel Triton (CLIS không dùng) thay vì cảnh báo

# Khai báo thư viện
import argparse
import glob
import hashlib
import subprocess

import numpy as np
import pandas as pd

import config
import models.clis_mamba as clis
import models.convnext_tiny as convnext
import models.densenet121 as densenet
from training.evaluate import evaluate
from utilities.data import create_dataloader, load_dataset, transform_setting
from utilities.dataset import NIHChestXrayDataset
from utilities.file import load_torch, read_json, save_csv, save_json
from utilities.log import timestamp
from utilities.metric import with_probabilities
from utilities.setting import apply_config, device_info
from utilities.tracking import data_hash

PREREG_TAG = "prereg-v1"
MANIFEST_PATH = config.MODELS_OUTPUT_DIRECTORY / "g3_manifest.json"
MAX_PROBABILITY_DIFFERENCE = 1e-4                                            # Độ lệch tối đa khi chấm lại Validation (cùng trọng số, cùng ảnh; chỉ còn sai số làm tròn)


# ============================================================
# Mở rộng danh sách thư mục lượt chạy (mẫu * do script tự mở rộng)
# Trả về: Danh sách đường dẫn
def expand_runs(patterns):
    return [Path(path) for pattern in patterns for path in (sorted(glob.glob(str(pattern))) if glob.has_magic(str(pattern)) else [pattern])]


# ============================================================
# Chạy lệnh git trong thư mục huấn luyện
# Trả về: Kết quả in ra; None nếu lệnh lỗi
def run_git(*args):
    result = subprocess.run(["git", *args], cwd = ROOT_DIR, capture_output = True, text = True, encoding = "utf-8")
    return result.stdout.strip() if result.returncode == 0 else None


# ============================================================
# Điều kiện mở Test: xác nhận tường minh và đã có tag đăng ký trước
def require_g3(confirmed, tag_commit, limit = None):

    # Kiểm tra điều kiện
    ## Phải xác nhận tường minh rằng đây là lần chạy của cổng G3
    if not confirmed:
        raise SystemExit("Test bị khóa tới cổng G3: cần --confirm-g3. Kiểm chứng đường chạy bằng --split validation.")

    ## Phải có tag đăng ký trước
    if tag_commit is None:
        raise SystemExit(f"Thiếu tag {PREREG_TAG}: không mở Test khi kế hoạch phân tích chưa khóa.")

    ## Không chấm một phần Test
    if limit is not None:
        raise SystemExit("--limit chỉ dùng cho --split validation: Test được chấm đủ một lần.")


# ============================================================
# SHA-256 của một tệp
# Trả về: Chuỗi hex
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ============================================================
# Quyết định với lượt chạy đã có tệp dự đoán Test
# Trả về: True nếu đã chấm với đúng checkpoint này (bỏ qua); False nếu chưa chấm. Checkpoint đã đổi hoặc không có trong manifest --> dừng
def already_scored(predictions_exist, manifest_entry, checkpoint_hash):

    # Chưa có tệp dự đoán
    if not predictions_exist:
        return False

    # Có tệp dự đoán: chỉ hợp lệ nếu đúng checkpoint đã ghi trong manifest
    if manifest_entry is None or manifest_entry.get("best_model_sha256") != checkpoint_hash:
        raise SystemExit("Đã có tệp dự đoán Test nhưng checkpoint khác manifest: không chấm lại Test bằng checkpoint đã đổi.")

    return True


# ============================================================
# Độ lệch xác suất giữa lần chấm lại và tệp đã lưu (ghép theo mã ảnh)
# Trả về: Độ lệch tuyệt đối lớn nhất (float)
def probability_difference(saved, scored, classes, key_column):
    saved = saved.set_index(key_column)
    scored = scored.set_index(key_column)
    columns = [f"prob_{name}" for name in classes]
    return float(np.abs(saved.loc[scored.index, columns].to_numpy(dtype = np.float64) - scored[columns].to_numpy(dtype = np.float64)).max())


# ============================================================
# Dựng mô hình từ thông tin của lượt chạy, không tải trọng số ImageNet khi không cần
# - init_from chỉ dùng lúc khởi tạo huấn luyện: Trọng số cuối nằm trong best_model.pt
# Trả về: Mô hình
def build_model(settings, number_of_classes):
    model_name = settings["model_name"]
    kwargs = {key: value for key, value in settings["model_kwargs"].items() if key != "init_from"}
    if model_name.startswith(densenet.MODEL_NAME):
        return densenet.DenseNetBaseline(num_classes = number_of_classes)
    if model_name.startswith(convnext.MODEL_NAME):
        return convnext.ConvNeXtTinyBaseline(num_classes = number_of_classes, pretrained = False)
    return clis.CLISMamba(num_classes = number_of_classes, pretrained = False, **kwargs)


# ============================================================
# Chấm một lượt chạy trên Validation hoặc Test
# Trả về: (khung dữ liệu đã chấm kèm xác suất, danh sách lớp, thư mục Fold)
def score_run(run_directory, fold, split, limit, workers, device):
    info = read_json(run_directory / "train_info.json")
    settings = info["settings"]
    fold_directory = run_directory / f"fold_{fold}"

    # Cấu hình của lượt chạy: độ phân giải và bộ ảnh
    snapshot = settings.get("config", {})
    apply_config({**config.CLIS_CONFIG, "IMAGE_SIZE": int(settings["image_size"]), "IMAGE_VARIANT": snapshot.get("IMAGE_VARIANT", config.CLIS_CONFIG["IMAGE_VARIANT"]),
                  **({} if workers is None else {"NUMBER_OF_WORKERS": int(workers)})})

    # Đọc dữ liệu và kiểm tra đúng dữ liệu của lượt chạy
    train_df, test_df, _, classes, _ = load_dataset()
    if data_hash(test_df) != settings["test_df_hash"] or data_hash(train_df) != settings["train_df_hash"]:
        raise SystemExit(f"{run_directory}: dữ liệu khác lúc huấn luyện (hash của train_df / test_df không khớp).")
    frame = test_df if split == "test" else train_df[train_df["Fold"] == int(fold)].reset_index(drop = True)
    frame = frame if limit is None else frame.iloc[:int(limit)].reset_index(drop = True)

    # Nạp trọng số tốt nhất (đúng cấu hình, đúng Fold)
    best = load_torch(fold_directory / "best_model.pt", device)
    if best.get("signature") != info["signature"] or int(best.get("fold")) != int(fold):
        raise SystemExit(f"{run_directory}: best_model.pt không khớp chữ ký / Fold của lượt chạy.")
    model = build_model(settings, len(classes)).to(device)
    model.load_state_dict(best["model_state_dict"], strict = True)

    # Suy luận (loader không xáo trộn)
    _, eval_transform = transform_setting()
    _, loader = create_dataloader(NIHChestXrayDataset, classes, device, frame, eval_transform, False, batch_size = int(settings["training_config"]["batch_size"]), seed = int(settings["seed"]) + 900_000)
    _, _, targets, probabilities = evaluate(device, len(classes), model, loader)
    if not (targets == frame[classes].to_numpy(dtype = np.float32)).all():
        raise SystemExit(f"{run_directory}: nhãn của loader khác khung dữ liệu.")

    del model, best
    return with_probabilities(frame, probabilities, classes), classes, fold_directory


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs = "+", required = True, help = "Thư mục lượt chạy (chứa train_info.json); nhận mẫu *")
    parser.add_argument("--fold", type = int, default = 1)
    parser.add_argument("--split", choices = ["validation", "test"], default = "validation")
    parser.add_argument("--limit", type = int, default = None, help = "Validation: chỉ chấm lại N ảnh đầu để kiểm tra nhanh")
    parser.add_argument("--workers", type = int, default = None, help = "Số worker của DataLoader (mặc định theo config)")
    parser.add_argument("--confirm-g3", action = "store_true", help = "Xác nhận đây là lần chấm Test của cổng G3")
    args = parser.parse_args()

    # Kiểm tra trước khi tải mô hình
    runs = expand_runs(args.runs)
    if not runs:
        parser.error("--runs: không tìm thấy lượt chạy nào")
    missing = [str(run) for run in runs if not (run / f"fold_{args.fold}" / "fold_done.json").is_file() or not (run / f"fold_{args.fold}" / "best_model.pt").is_file()]
    if missing:
        raise SystemExit(f"Các lượt chạy chưa xong Fold {args.fold}: {missing}")

    commit = run_git("rev-parse", "HEAD")
    if args.split == "test":
        require_g3(args.confirm_g3, run_git("rev-parse", "-q", "--verify", f"refs/tags/{PREREG_TAG}"), args.limit)
    device = device_info()

    # Kiểm chứng: chấm lại Validation, so với oof_predictions.csv
    if args.split == "validation":
        failed = []
        for run in runs:
            scored, classes, fold_directory = score_run(run, args.fold, "validation", args.limit, args.workers, device)
            difference = probability_difference(pd.read_csv(fold_directory / "oof_predictions.csv"), scored, classes, config.IMAGE_ID_COLUMN)
            passed = difference <= MAX_PROBABILITY_DIFFERENCE
            print(f"{run}: {len(scored):,} ảnh | độ lệch xác suất lớn nhất {difference:.2e} | {'KHỚP' if passed else 'LỆCH'}", flush = True)
            if not passed:
                failed.append(str(run))
        sys.exit(1 if failed else 0)

    # Cổng G3: ghi manifest trước khi suy luận, lưu dự đoán, không in chỉ số
    manifest = read_json(MANIFEST_PATH) if MANIFEST_PATH.is_file() else {"prereg_tag": PREREG_TAG, "runs": {}}
    for run in runs:
        fold_directory = run / f"fold_{args.fold}"
        output_path = fold_directory / "test" / "test_predictions.csv"
        key = run.resolve().relative_to(ROOT_DIR).as_posix()
        checkpoint_hash = sha256_file(fold_directory / "best_model.pt")
        if already_scored(output_path.is_file(), manifest["runs"].get(key), checkpoint_hash):
            print(f"{run}: đã chấm với đúng checkpoint này, bỏ qua", flush = True)
            continue
        manifest["runs"][key] = {"best_model_sha256": checkpoint_hash, "git_commit": commit, "scored_at": timestamp(), "fold": args.fold}
        save_json(MANIFEST_PATH, manifest)
        scored, _, _ = score_run(run, args.fold, "test", None, args.workers, device)
        output_path.parent.mkdir(parents = True, exist_ok = True)
        save_csv(output_path, scored)
        print(f"{run}: đã lưu {output_path} ({len(scored):,} ảnh)", flush = True)


if __name__ == "__main__":
    main()
