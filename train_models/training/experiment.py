import pandas as pd

import config
from training.checkpoint import choose_run
from training.context import ExperimentContext
from training.evaluate import aggregate_fold_tests, cross_validation, evaluate_test
from training.loss import validate_loss_config
from training.optimizer import build_training_config
from training.threshold import fit_fold_thresholds, fit_postprocessing
from training.train import cleanup, train_one_fold
from utilities.data import load_dataset, transform_setting
from utilities.file import read_json, save_json, to_snake_case
from utilities.log import GPUMonitor, capture_run_log, print_model_specification, print_run_summary
from utilities.setting import device_info, keep_system_awake, set_fold, set_seed
from utilities.tracking import launch_record, make_signature, model_specification


# ============================================================
# Chạy K-Fold cho một mô hình:
def run_experiment(model_name, dataset_class, model_factory, model_kwargs = None, *, loss_config, optimizer_config = None, scheduler_config = None, selection_metric = None):
    model_kwargs = dict(model_kwargs or {})
    keep_system_awake()

    # 1. Cấu hình
    loss_config = validate_loss_config(loss_config)
    training_config = build_training_config(optimizer_config, scheduler_config, selection_metric)

    # 2. Dữ liệu
    train_df, test_df, dataset_name, classes, number_of_folds = load_dataset()
    folds = set_fold(number_of_folds)

    # 3. Truy vết
    launch = launch_record()
    if config.REQUIRE_CLEAN_GIT and launch["git_dirty"] is not False:
        raise RuntimeError(f"REQUIRE_CLEAN_GIT=True nhưng còn thay đổi chưa commit: {launch['git_dirty_files'][:10]}")

    # 4. Chữ ký và thư mục lượt chạy
    signature, settings = make_signature(model_name, model_factory, model_kwargs, dataset_name, dataset_class, train_df, test_df, classes, folds, number_of_folds, loss_config, training_config)
    root = config.MODELS_OUTPUT_DIRECTORY / to_snake_case(dataset_name) / to_snake_case(model_name)
    output_directory = choose_run(root, signature, bool(config.RESUME_TRAINING), settings)

    # Lượt chạy cùng cấu hình đã hoàn thành đủ mọi fold và Test --> không huấn luyện lại
    if read_json(output_directory / "train_info.json").get("train_completed", False):
        print(f"Bỏ qua {model_name}: đã hoàn thành tại {output_directory}")
        return pd.read_csv(output_directory / "cross_validation_results.csv")

    with capture_run_log(output_directory, model_name = model_name, dataset_name = dataset_name, train_df = train_df, test_df = test_df, folds = folds, settings = settings, signature = signature, launch = launch):
        update_train_info(output_directory, append_launch = launch)

        # Thiết bị, thông số mô hình, biến đổi ảnh
        device = device_info()
        gpu_monitor = GPUMonitor.get() if device.type == "cuda" else None
        if gpu_monitor is not None:
            gpu_monitor.reset_statistics()
        specification = model_specification(model_factory, classes, device, model_kwargs, training_config["batch_size"])
        print_model_specification(specification)
        update_train_info(output_directory, model_specification = specification)
        train_transform, eval_transform = transform_setting()
        ctx = ExperimentContext(output_directory = output_directory, signature = signature, model_name = model_name, model_factory = model_factory, model_kwargs = model_kwargs, dataset_class = dataset_class, classes = classes, loss_config = loss_config, training_config = training_config, device = device, train_transform = train_transform, eval_transform = eval_transform)
        set_seed()

        # 5. Từng fold
        # Thống kê GPU của lần khởi chạy này được ghi lại sau mỗi fold (lượt chạy bị ngắt vẫn giữ số liệu các fold đã xong)
        results = []
        for fold in folds:
            results.append(run_fold(ctx, fold, train_df, test_df))
            record_gpu_statistics(output_directory, gpu_monitor)

        # 6. Tổng hợp Cross-Validation
        frame, all_done = cross_validation(output_directory, results, number_of_folds, signature)

        # Ngưỡng cho ensemble: chọn trên toàn bộ OOF
        ensemble_postprocess = output_directory / "postprocess.json" if config.ENABLE_POSTPROCESS else None
        if all_done and ensemble_postprocess is not None and not ensemble_postprocess.exists():
            fit_postprocessing(output_directory / "oof_all.csv", output_directory)
            print("Đã lưu postprocess.json chọn trên toàn bộ OOF")

        # Test: mean ± std giữa các fold và ensemble
        test_report = None
        if config.EVALUATE_FOLD_MODELS_ON_TEST:
            test_report = aggregate_fold_tests(output_directory, test_df, classes, number_of_folds, postprocess_path = ensemble_postprocess)

        # Trạng thái lượt chạy và tổng kết
        fold_tests = [f for f in range(1, number_of_folds + 1) if (ctx.fold_directory(f) / "test" / "test_summary.json").is_file()]
        update_train_info(output_directory, folds_completed = frame["fold"].astype(int).tolist(), fold_tests_completed = fold_tests,
            train_completed = bool(all_done and (not config.EVALUATE_FOLD_MODELS_ON_TEST
                                               or len(fold_tests) == number_of_folds)))
        record_gpu_statistics(output_directory, gpu_monitor)
        print_run_summary(frame, test_report)
        return frame


# ============================================================
# Chạy một Fold
def run_fold(ctx, fold, train_df, test_df):
    model, criterion, result = train_one_fold(ctx, fold, train_df)
    fold_directory = ctx.fold_directory(fold)

    # Chọn ngưỡng theo lớp trên OOF của fold, tách theo bệnh nhân thành phần chọn ngưỡng / phần đánh giá
    postprocess_path = None
    if config.ENABLE_POSTPROCESS:
        postprocess_path = fold_directory / "postprocess.json"
        if not postprocess_path.exists():
            fit_fold_thresholds(fold_directory / "oof_predictions.csv", fold_directory, expected_classes = ctx.classes)

    # Đánh giá Test (bỏ qua nếu đã có kết quả từ lần chạy trước)
    test_directory = fold_directory / "test"
    if config.EVALUATE_FOLD_MODELS_ON_TEST and not (test_directory / "test_summary.json").is_file():
        evaluate_test(ctx, test_directory, model, criterion, test_df, title = f"Test Fold {fold}", postprocess_path = postprocess_path)

    del model, criterion
    cleanup()
    return result


# ============================================================
# Cập nhật train_info.json: append_launch thêm một lần khởi chạy, các khóa khác ghi đè
def update_train_info(output_directory, append_launch = None, **values):
    path = output_directory / "train_info.json"
    info = read_json(path)
    if append_launch is not None:
        info.setdefault("launches", []).append(append_launch)
    info.update(values)
    save_json(path, info)


# ============================================================
# Ghi thống kê GPU (xung, nhiệt độ, công suất, thời gian giảm xung) vào lần khởi chạy cuối trong train_info.json
def record_gpu_statistics(output_directory, gpu_monitor):
    if gpu_monitor is None:
        return
    path = output_directory / "train_info.json"
    info = read_json(path)
    info["launches"][-1]["gpu_statistics"] = gpu_monitor.statistics()
    save_json(path, info)
