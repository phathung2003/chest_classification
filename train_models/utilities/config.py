from pathlib import Path

# Đường dẫn
ROOT_DIRECTORY              = Path(__file__).resolve().parents[1]                                       # Đường dẫn thư mục huấn luyện mô hình

ORIGINAL_DATASET_DIRECTORY  = (ROOT_DIRECTORY / "original_dataset").resolve()                           # Tập dữ liệu ban đầu

DATASET_DIRECTORY           = (ROOT_DIRECTORY / "dataset").resolve()                                    # Tập dữ liệu sau khi xử lý
FOLDS_DIRECTORY             = (DATASET_DIRECTORY / "folds").resolve()

TRAIN_FOLDS_DIRECTORY       = (DATASET_DIRECTORY / "train_folds.csv").resolve()                         # Dữ liệu huấn luyện (Đã chia Fold)                                   
TEST_DIRECTORY              = (DATASET_DIRECTORY / "test.csv").resolve()                                # Dữ liệu kiểm tra
DATASET_INFO_DIRECTORY      = (DATASET_DIRECTORY / "dataset_info.json").resolve()                       # Thông tin chính trong tập dữ liệu

MODELS_OUTPUT_DIRECTORY     = (ROOT_DIRECTORY / "training_results").resolve()                           # Đường dẫn đầu ra mô hình

# Cấu hình huấn luyện
NUMBER_OF_FOLDS             = 5
TRAIN_ALL_FOLD              = False
SELECTED_FOLD               = 1

IMAGE_SIZE                  = 224

SEED                        = 42
BATCH_SIZE                  = 16
EPOCHS                      = 30
LEARNING_RATE               = 1e-4
WEIGHT_DECAY                = 1e-4
PATIENCE                    = 5
THRESHOLD                   = 0.5
NUMBBER_OF_WORKERS          = 8
USE_AMP                     = True

# Cấu hình tái lập
CUDNN_DETERMINISTIC         = True                                                                      # cuDNN tái lập chính xác
CUDNN_BENCHMARK             = False                                                                     # Cho phép thử nhiều thuật toán (False cho dễ tái lập)
DETERMINISTIC_ALGORITHMS    = True                                                                      # Cho phép PyTorch operation nondeterministic

# Cấu hình làm giàu ảnh
DERGEES                     = 5
TRANSLATE                   = (0.02, 0.02)
SCALE                       = (0.95, 1.05)

TRANSFORM_NORMALIZE_MEAN    = [0.485, 0.456, 0.406]
TRANSFORM_NORMALIZE_STD     = [0.229, 0.224, 0.225]

## Cấu trúc text: [RESULT]: TITLE: MESSAGE
MIN_RESULT_WIDTH            = 10
MIN_TITLE_WIDTH             = 30
MIN_MESSAGE_WIDTH           = 20
MIN_NUMBER_WIDTH            = 8
    
# Khác
IMAGE_EXTENSION             = {".png", ".jpg", ".jpeg"}