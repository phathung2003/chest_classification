from pathlib import Path

# Đường dẫn
ROOT_DIRECTORY              = (Path.cwd().parent / "chest_classification" / "train_models").resolve()
ORIGINAL_DATASET_DIRECTORY  = (ROOT_DIRECTORY / "original_dataset").resolve()
DATASET_DIRECTORY           = (ROOT_DIRECTORY / "dataset").resolve()

# Cấu hình huấn luyện
IMAGE_SIZE                  = 224
NUMBER_OF_FOLDS             = 5
RANDOM_SEED                 = 42

# Khác
IMAGE_EXTENSION             = {".png", ".jpg", ".jpeg"}

## Cấu trúc text: [RESULT]: TITLE: MESSAGE
MIN_RESULT_WIDTH            = 10
MIN_TITLE_WIDTH             = 30
MIN_MESSAGE_WIDTH           = 20
MIN_NUMBER_WIDTH            = 8
    
    