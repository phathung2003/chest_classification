# Khai báo utilities
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import NUMBER_OF_FOLDS
import numpy as np
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


def validate_data(train_df, test_df, classes, nfolds):
    check_dataset(classes, train_df, test_df, NUMBER_OF_FOLDS=nfolds)
    if train_df.empty or test_df.empty:
        raise ValueError('Train and test dataframes must be nonempty.')
    if set(train_df['image_path'].astype(str)) & set(test_df['image_path'].astype(str)):
        raise ValueError('Train/test image path leakage detected.')
    for name, frame in [('train', train_df), ('test', test_df)]:
        values = frame[classes].to_numpy()
        if not bool(np.isin(values, [0, 1]).all()):
            raise ValueError(f'{name} contains nonbinary or missing label values.')
    # If patient identity is available, enforce patient-disjoint folds/test.
    candidates = ('Patient ID', 'PatientID', 'patient_id', 'patientId', 'patientid')
    patient_col = next((c for c in candidates if c in train_df and c in test_df), None)
    if patient_col is not None:
        if train_df[patient_col].isna().any() or test_df[patient_col].isna().any():
            raise ValueError('Patient ID must not contain missing values.')
        a = set(train_df[patient_col].astype(str))
        b = set(test_df[patient_col].astype(str))
        if a & b:
            raise ValueError(f'Patient leakage across train/test detected in {patient_col}.')
        group_count = train_df.groupby(patient_col)['Fold'].nunique()
        if (group_count > 1).any():
            raise ValueError(f'Patient leakage across folds detected in {patient_col}.')
    else:
        print('WARNING: No common patient-id column found; patient-level split cannot be verified.')