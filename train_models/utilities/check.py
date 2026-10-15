# Khai báo utilities
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import NUMBER_OF_FOLDS

def check_file_directory(file_directory):
    for file_path in file_directory:
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy:\n {file_path}")

def check_dataset(CLASSES, train_folds_df, test_df, NUMBER_OF_FOLDS = NUMBER_OF_FOLDS):
    required_train_columns = (["image_path", "Fold"] + CLASSES)
    required_test_columns = (["image_path"] + CLASSES)
    
    missing_train_columns = [
        column
        for column in required_train_columns
        if column not in train_folds_df.columns
    ]


    missing_test_columns = [
        column
        for column in required_test_columns
        if column not in test_df.columns
    ]
    
    if missing_train_columns:
        raise ValueError(f"train_folds.csv thiếu cột: {missing_train_columns}")

    if missing_test_columns:
        raise ValueError(f"test.csv thiếu cột: {missing_test_columns}")


    actual_folds = set(train_folds_df["Fold"].astype(int).unique())
    expected_folds = set(range(1, NUMBER_OF_FOLDS + 1))

    if actual_folds != expected_folds:

        raise ValueError(
            f"Số fold thực tế: {actual_folds}\n"
            f"Số fold mong đợi: {expected_folds}"
        )
