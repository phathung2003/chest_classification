# Khai báo utilities
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from utilities.training import train_one_fold, train_final_model, evaluate_test, cleanup
import json


# ============================================================
# Chạy mô hình
def run_experiment(DEVICE, CLASSES, dataset_class, FOLD_TRAIN, train_transform, test_transform, OUTPUT_DIRECTORY, train_folds_df, test_df, model_factory, model_kwargs):
    results = []
    oof_frames = []
    
    # Huấn luyện mô hình
    for fold in [FOLD_TRAIN]:
        results.append(train_one_fold(OUTPUT_DIRECTORY, fold, len([FOLD_TRAIN]), train_transform, test_transform, dataset_class, train_folds_df, CLASSES, model_factory, model_kwargs, DEVICE))
        
        if len([FOLD_TRAIN]) > 1:
            oof_frames.append(pd.read_csv(OUTPUT_DIRECTORY / f"fold_{fold}" / "oof_predictions.csv"))
        else:
            pd.DataFrame(results).to_csv(OUTPUT_DIRECTORY / "cross_validation_results.csv", index=False)

    # Kết quả
    result_df = pd.DataFrame(results)
    
    print("\n=== CROSS VALIDATION RESULTS ===")
    print(result_df.to_string(index=False))
    
    if len([FOLD_TRAIN]) > 1:
        pd.concat(oof_frames, ignore_index=True).to_csv(OUTPUT_DIRECTORY / "oof_all.csv", index=False)
        print(f"Mean AUROC: {result_df['best_macro_auroc'].mean():.6f} ± {result_df['best_macro_auroc'].std(ddof=1):.6f}")

    # Chạy huấn luyện lượt cuối (Nếu chạy trên 1 Fold)

    # Chỉ dùng best epoch từ cross-validation ĐẦY ĐỦ vừa chạy.
    final_epochs = max(1, int(np.median(result_df["best_epoch"].to_numpy())))
    final_model, final_criterion = train_final_model(OUTPUT_DIRECTORY, dataset_class, train_folds_df, CLASSES, train_transform, model_factory, model_kwargs, DEVICE, final_epochs)

    # Đánh giá mô hình    
    evaluate_test(OUTPUT_DIRECTORY, dataset_class, final_model, final_criterion, test_df, CLASSES, test_transform, DEVICE)
    del final_model, final_criterion
    cleanup()
    
    return result_df
