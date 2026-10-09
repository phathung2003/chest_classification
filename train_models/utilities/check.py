# Khai báo utilities
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utilities.config import NUMBER_OF_FOLDS, TRAIN_ALL_FOLD, SELECTED_FOLD

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

def folds_setting(number_of_folds):
    # Kiểm tra tổng số lượng Folds
    if not isinstance(number_of_folds, int) or number_of_folds < 2:
        raise ValueError("NUMBER_OF_FOLDS phải >= 2")
    
    # Kiểm tra xem chọn huấn luyện toàn bộ Fold hay 1 Fold duy nhất
    if isinstance(TRAIN_ALL_FOLD, bool):
        raise ValueError("TRAIN_ALL_FOLD phải là biến bool")
    
    # Huấn luyện toàn bộ Fold
    if TRAIN_ALL_FOLD == True:
        return list(range(1, number_of_folds + 1))
    
    # Huấn luyện 1 Fold
    if type(SELECTED_FOLD) is not int or not 1 <= SELECTED_FOLD <= number_of_folds:
        raise ValueError(f"SELECTED_FOLD phải là số nguyên từ 1 đến {number_of_folds}")
    
    return [SELECTED_FOLD]
