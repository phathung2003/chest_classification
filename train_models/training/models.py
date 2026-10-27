import json
from copy import deepcopy
import numpy as np
import config
from utilities.runtime import runtime
from utilities.data import dataset_info, transform_setting
from utilities.setting import device_info, set_seed, set_fold
from utilities.check import check_file_directory, validate_data
from utilities.other import to_snake_case
from utilities.log import capture_run_log, model_specification
from training.checkpoint import save_json, read_json, choose_run
from training.train import train_one_fold, train_final_model, cleanup
from training.evaluate import evaluate_test
from utilities.hash import make_signature
from training.evaluate import cross_validation
from training.threshold import fit_postprocessing, fit_fold_thresholds
from training.loss import check_loss
from training.optimizer import normalize_training_config

def run_experiment(model_name, dataset_class, model_factory, model_kwargs, loss_factory=None, *,
                   loss_config=None, training_config=None, optimizer_factory=None,
                   optimizer_config=None, scheduler_config=None, selection_metric=None):
    
    # Kiểm tra hàm Loss
    check_loss(loss_config, loss_factory)
    
    # Accept flat per-model settings while preserving legacy training_config.
    # LR, weight decay, batch size and patience remain global in config.py.
    flat_options = (optimizer_config, scheduler_config, selection_metric)
    if training_config is not None and any(value is not None for value in flat_options):
        raise ValueError(
            "Do not combine training_config with optimizer_config, "
            "scheduler_config or selection_metric."
        )

    if training_config is None:
        training_config = {
            "optimizer": optimizer_config,
            "scheduler": scheduler_config,
        }
        if selection_metric is not None:
            training_config["selection_metric"] = selection_metric

    training_config = normalize_training_config(training_config)
    if optimizer_factory is not None and training_config["optimizer"]:
        raise ValueError("Select optimizer_factory OR optimizer_config, not both")
    runtime.BATCH_SIZE = training_config["batch_size"]

    # Kiểm tra đường dẫn
    required = [config.TRAIN_FOLDS_DIRECTORY, config.TEST_DIRECTORY, config.DATASET_INFO_DIRECTORY]
    check_file_directory(required)
    
    # Đọc thông tin tập dữ liệu
    with open(config.DATASET_INFO_DIRECTORY, 'r', encoding='utf-8') as f:
        dataset_config = json.load(f)
    
    # Lấy tập dữ liệu
    train_df, test_df, dataset_name, classes, number_of_folds = dataset_info(dataset_config)
    validate_data(train_df, test_df, classes, number_of_folds)
    
    # Cấu hình Fold
    folds = set_fold(number_of_folds)
    complete_selection = sorted(folds) == list(range(1, number_of_folds + 1))
    include_final = bool(getattr(config, 'TRAIN_FINAL_MODEL', True))
    single_fold_test = bool(getattr(config, 'EVALUATE_SINGLE_FOLD_TEST', False))
    if single_fold_test and (len(folds) != 1 or include_final):
        raise ValueError(
            'EVALUATE_SINGLE_FOLD_TEST=True requires exactly one selected fold '
            'and TRAIN_FINAL_MODEL=False.'
        )
    
    # Tạo chữ ký
    signature, settings = make_signature(model_name, dataset_name, train_df, test_df, classes, folds, number_of_folds, model_factory, dataset_class, model_kwargs, loss_factory, include_final, loss_config=loss_config, training_config=training_config, optimizer_factory=optimizer_factory)
    
    # Đường dẫn lưu lần chạy
    root = (config.MODELS_OUTPUT_DIRECTORY / to_snake_case(dataset_name) / to_snake_case(model_name))
    output_directory = choose_run(root, signature, bool(getattr(config, 'RESUME_TRAINING', True)), settings)
    
    # Ghi log
    with capture_run_log(output_directory, model_name = model_name, dataset_name = dataset_name, train_df = train_df, test_df = test_df, folds = folds, include_final = include_final, settings = settings):
        runtime.MODEL_NAME = model_name
        runtime.DATASET_NAME = dataset_name
        runtime.RUN_NO = int(output_directory.name[4:])
        runtime.REMAINING_FOLD = list(folds)
        device = device_info()
        model_specification(model_factory, classes, device, model_kwargs)
        train_transform, eval_transform = transform_setting()
        results = []
        set_seed(int(runtime.SEED))
        
        # Bắt đầu huấn luyện từng Fold
        for fold in folds:
            model, criterion, result = train_one_fold(output_directory, fold, train_transform, eval_transform, dataset_class, train_df, classes, model_factory, model_kwargs, device, signature, model_name, loss_factory, loss_config=loss_config, training_config=training_config, optimizer_factory=optimizer_factory)
            
            if not single_fold_test:
                del model, criterion
                cleanup()
                
            results.append(result)
            if bool(getattr(config, 'ENABLE_POSTPROCESS', True)):
                fold_directory = output_directory / f'fold_{fold}'
                oof_path = fold_directory / 'oof_predictions.csv'
                postprocess_path = fold_directory / 'postprocess.json'
                if not oof_path.exists():
                    raise FileNotFoundError(f'Thiếu dự đoán OOF cho fold {fold}: {oof_path}')
                if not postprocess_path.exists():
                    fit_fold_thresholds(
                        oof_path, fold_directory,
                        expected_classes=classes,
                        calibration_fraction=0.75,
                        seed=int(runtime.SEED),
                        beta=float(getattr(config, 'THRESHOLD_BETA', 1.0)),
                        min_recall=float(getattr(config, 'THRESHOLD_MIN_RECALL', 0.0)),
                        default_threshold=float(runtime.THRESHOLD),
                    )
        frame, all_done = cross_validation(output_directory, results, folds, number_of_folds, signature)
        
        # Chạy test fold đó
        if single_fold_test:
            single_fold_directory = output_directory / f'fold_{folds[0]}' / 'single_fold_test'
            single_fold_directory.mkdir(parents=True, exist_ok=True)
            evaluate_test(single_fold_directory, dataset_class, model, criterion, test_df, classes, eval_transform, device)
            del model, criterion
            cleanup()

        # Fit A on the COMPLETE development OOF only; never on held-out test.
        if all_done and bool(getattr(config, 'ENABLE_POSTPROCESS', True)):
            pp_path = output_directory / 'postprocess.json'
            if not pp_path.exists():
                fit_postprocessing(
                    output_directory / 'oof_all.csv', output_directory,
                    calibration=str(config.POSTPROCESS_CALIBRATION),
                    beta=float(config.THRESHOLD_BETA),
                    min_recall=float(config.THRESHOLD_MIN_RECALL),
                    default_threshold=float(runtime.THRESHOLD))
                print('Đã lưu postprocess.json dựa trên OOF')
        
        # Huấn luyện lần cuối
        if include_final and all_done:
            if (output_directory / 'test_summary.json').exists():
                print('Đã hoàn tất đánh giá trên tập Test độc lập. Bỏ qua đánh giá lại.')
            else:
                median_epoch = max(1, int(np.median(frame['best_epoch'].to_numpy(dtype=int))))
                model, criterion = train_final_model(output_directory, dataset_class, train_df, classes, train_transform, model_factory, model_kwargs, device, median_epoch, signature, model_name, loss_factory, loss_config=loss_config, training_config=training_config, optimizer_factory=optimizer_factory)
                evaluate_test(output_directory, dataset_class, model, criterion,
                              test_df, classes, eval_transform, device)
                del model, criterion
                cleanup()
                
        info = read_json(output_directory / 'train_info.json')
        info['train_completed'] = bool(all_done and (not include_final or (output_directory / 'test_summary.json').exists()))
        info['folds_completed'] = frame['fold'].astype(int).tolist()
        info['final_test_completed'] = (output_directory / 'test_summary.json').exists()
        if single_fold_test:
            info['single_fold_test_completed'] = (
                output_directory / f'fold_{folds[0]}' / 'single_fold_test' / 'test_summary.json'
            ).is_file()
            info['single_fold_test_fold'] = int(folds[0])
        save_json(output_directory / 'train_info.json', info)
        return frame
