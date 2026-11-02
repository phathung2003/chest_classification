# ============================================================
# Thực hiện chạy huấn luyện mô hình
# Mục tiêu: Chạy toàn bộ một thí nghiệm cho một mô hình
# Quy trình: Cấu hình --> Dữ liệu --> Truy vết --> Chữ ký --> Lượt chạy --> Chuẩn bị môi trường --> Huấn luyện từng Fold --> Tổng hợp Cross-Validation --> Hậu xử lý --> Đánh giá Test --> Cập nhật trạng thái.
# ============================================================

import pandas as pd

import config
from training.checkpoint import choose_run
from training.context import ExperimentContext
from training.loss import validate_loss_config
from training.optimizer import build_training_config
from training.threshold import fit_fold_thresholds, fit_postprocessing
from training.evaluate import aggregate_fold_tests, cross_validation, evaluate_test

from training.train import cleanup, train_one_fold
from utilities.data import load_dataset, transform_setting
from utilities.file import read_json, save_json, to_snake_case
from utilities.setting import device_info, keep_system_awake, set_fold, set_seed
from utilities.tracking import launch_record, make_signature, model_specification
from utilities.log import GPUMonitor, capture_run_log, print_model_specification, print_run_summary

# ============================================================
# Chạy K-Fold cho một mô hình
# Trả về: DataFrame kết quả Cross-Validation (mỗi dòng một Fold)
def start_train(model_name, dataset_class, model_factory, model_kwargs = None, *, loss_config, optimizer_config = None, scheduler_config = None, selection_metric = None):
    
    # Chuẩn hóa tham số
    model_kwargs = dict(model_kwargs or {})
    
    # Cấu hình giữ máy chạy
    keep_system_awake()

    # 1. Kiểm tra cấu hình huấn luyện
    ## Kiểm tra cấu hình hàm Loss
    loss_config = validate_loss_config(loss_config)
    
    ## Tổng hợp tham số cấu hình huấn luyện (optimizer, scheduler, metric)
    training_config = build_training_config(optimizer_config, scheduler_config, selection_metric)

    # 2. Xử lý dữ liệu
    ## Đọc thông tin tập dữ liệu
    train_df, test_df, dataset_name, classes, number_of_folds = load_dataset()
    
    ## Đọc Fold cần thực hiện
    folds = set_fold(number_of_folds)

    # 3. Truy vết thí nghiệm (Bằng Git)
    
    ## Ghi nhận thông tin môi trường hiện tại
    launch = launch_record()

    ## Ngăn chạy thí nghiệm nếu mã nguồn chỉnh sửa chưa commit (Nếu REQUIRE_CLEAN_GIT = True trong config)
    if config.REQUIRE_CLEAN_GIT and launch["git_dirty"] is not False:
        raise RuntimeError(
            "[start_train]: REQUIRE_CLEAN_GIT = True nhưng còn thay đổi chưa commit.\n"
            f"Tệp chưa commit: {launch['git_dirty_files'][:10]}"
        )

    # 4. Tạo chữ ký
    signature, settings = make_signature(model_name, model_factory, model_kwargs, dataset_name, dataset_class, train_df, test_df, classes, folds, number_of_folds, loss_config, training_config)
    
    # 5. Xác định thư mục lượt chạy
    root = config.MODELS_OUTPUT_DIRECTORY / to_snake_case(dataset_name) / to_snake_case(model_name)
    output_directory = choose_run(root, signature, bool(config.RESUME_TRAINING), settings)

    ## Kiểm tra thí nghiệm đã hoàn thành (Hoàn thành thì không huấn luyện lại)
    if read_json(output_directory / "train_info.json").get("train_completed", False):
        print(f"Bỏ qua {model_name}: đã hoàn thành tại {output_directory}")
        return pd.read_csv(output_directory / "cross_validation_results.csv")

    # 6. Chuẩn bị môi trường huấn luyện
    ## Quản lý ghi log
    with capture_run_log(output_directory, model_name = model_name, dataset_name = dataset_name, train_df = train_df, test_df = test_df, folds = folds, settings = settings, signature = signature, launch = launch):
        
        ## Cập nhật thông tin lần chạy
        update_train_info(output_directory, append_launch = launch)

        ## Thông tin thiết bị
        device = device_info()
        
        ## Chuẩn bị GPU Monitor
        gpu_monitor = GPUMonitor.get() if device.type == "cuda" else None
        if gpu_monitor is not None:
            gpu_monitor.reset_statistics()
        
        ## Thông tin về thông số mô hình    
        specification = model_specification(model_factory, classes, device, model_kwargs, training_config["batch_size"])
        
        ## In thông tin mô hình
        print_model_specification(specification)
        
        ## Lưu thông tin
        update_train_info(output_directory, model_specification = specification)
        
        ## Tạo hàm làm giàu ảnh (Transform)
        train_transform, eval_transform = transform_setting()
        
        ## Tập hợp thông tin huấn luyện
        experiment_context = ExperimentContext(output_directory = output_directory, signature = signature, model_name = model_name, model_factory = model_factory, model_kwargs = model_kwargs, dataset_class = dataset_class, classes = classes, loss_config = loss_config, training_config = training_config, device = device, train_transform = train_transform, eval_transform = eval_transform)
        
        ## Thiết lập seed
        set_seed()

        # 7. Huấn luyện từng Fold
        ## Khởi tạo biến lưu kết quả
        results = []
        
        ## Quy trình huấn luyện từng Fold
        for fold in folds:
            
            ### Chạy huấn luyện
            results.append(run_fold(experiment_context, fold, train_df, test_df))
            
            ### Thông số về GPU
            record_gpu_statistics(output_directory, gpu_monitor)

        # 8. Tổng hợp K-Fold cross validation
        ## Tổng hợp kết quả các Fold (đọc fold_done.json, gộp OOF khi đủ Fold)
        frame, all_done = cross_validation(output_directory, results, number_of_folds, signature)

        # 9. Hậu xử lý
        ## Đường dẫn lưu kết quả
        ensemble_postprocess = output_directory / "postprocess.json" if config.ENABLE_POSTPROCESS else None
        
        ## Kiểm tra xem quá trình K-Fold hoàn tất trên toàn bộ Fold chưa
        if all_done and ensemble_postprocess is not None and not ensemble_postprocess.exists():
            
            ### Chọn ngưỡng (và calibration) cho ensemble trên toàn bộ OOF
            fit_postprocessing(output_directory / "oof_all.csv", output_directory)
            print("Đã lưu postprocess.json chọn trên toàn bộ OOF")

        # 10. Đánh giá trên tập kiểm tra
        ## Khởi tạo biến lưu kết quả
        test_report = None
        
        ## Thực hiện đánh giá trên tập kiểm tra (Nếu EVALUATE_FOLD_MODELS_ON_TEST = True trong config)
        if config.EVALUATE_FOLD_MODELS_ON_TEST:
            test_report = aggregate_fold_tests(output_directory, test_df, classes, number_of_folds, postprocess_path = ensemble_postprocess)

        # 11. Cập nhật trạng thái lượt chạy hoàn thành
        ## Lấy danh sách những Fold có file test_summary.json
        fold_tests = [fold for fold in range(1, number_of_folds + 1) if (experiment_context.fold_directory(fold) / "test" / "test_summary.json").is_file()]
        
        ## Cập nhật trạng thái huấn luyện
        update_train_info(output_directory, folds_completed = frame["fold"].astype(int).tolist(), fold_tests_completed = fold_tests,
            train_completed = bool(all_done and (not config.EVALUATE_FOLD_MODELS_ON_TEST or len(fold_tests) == number_of_folds)))
        
        ## Thống kê GPU
        record_gpu_statistics(output_directory, gpu_monitor)
        
        ## In kết quả huấn luyện
        print_run_summary(frame, test_report)

        return frame

# ============================================================
# Chạy một Fold
# Trả về: Kết quả của Fold (dict)
def run_fold(experiment_context, fold, train_df, test_df):
    
    # Huấn luyện mô hình
    model, criterion, result = train_one_fold(experiment_context, fold, train_df)
    
    # Xác định thư mục Fold
    fold_directory = experiment_context.fold_directory(fold)

    # Tối ưu Threshold (Nếu ENABLE_POSTPROCESS = True trong config)
    postprocess_path = None
    if config.ENABLE_POSTPROCESS:
        
        # Đường dẫn lưu file JSON hậu xử lý
        postprocess_path = fold_directory / "postprocess.json"
        
        # Nếu chưa có --> Chạy hậu xử lý điều chỉnh Threshold
        if not postprocess_path.exists():
            fit_fold_thresholds(fold_directory / "oof_predictions.csv", fold_directory, expected_classes = experiment_context.classes)

    # Đánh giá mô hình trên tập kiểm tra (bỏ qua nếu đã có kết quả từ lần chạy trước)
    test_directory = fold_directory / "test"
    if config.EVALUATE_FOLD_MODELS_ON_TEST and not (test_directory / "test_summary.json").is_file():
        evaluate_test(experiment_context, test_directory, model, criterion, test_df, title = f"Test Fold {fold}", postprocess_path = postprocess_path)

    # Giải phóng bộ nhớ
    del model, criterion
    cleanup()

    return result

# ============================================================
# Cập nhật thông tin huấn luyện
def update_train_info(output_directory, append_launch = None, **values):
    
    # Đường dẫn file JSON lưu thông tin huấn luyện
    train_info_path = output_directory / "train_info.json"
    
    # Đọc thông tin
    info = read_json(train_info_path)

    # Bổ sung lịch sử chạy
    if append_launch is not None:
        info.setdefault("launches", []).append(append_launch)

    # Cập nhật các trường dữ liệu
    info.update(values)
    
    # Lưu bản cập nhật
    save_json(train_info_path, info)

# ============================================================
# Thống kê thông số GPU
def record_gpu_statistics(output_directory, gpu_monitor):
    
    # Kiểm tra GPU Monitor (Không có thì bỏ qua)
    if gpu_monitor is None:
        return

    # Đường dẫn file JSON lưu thông tin huấn luyện
    path = output_directory / "train_info.json"
    
    # Đọc thông tin
    info = read_json(path)
    
    # Ghi thông tin thống kê GPU
    info["launches"][-1]["gpu_statistics"] = gpu_monitor.statistics()
    
    # Lưu bản cập nhật
    save_json(path, info)
