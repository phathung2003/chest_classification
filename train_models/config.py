from pathlib import Path

# Đường dẫn
ROOT_DIRECTORY              = Path(__file__).resolve().parents[0]                                       # Đường dẫn thư mục huấn luyện mô hình

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
SELECTED_FOLD               = [1]

# Train 1 Fold
TRAIN_FINAL_MODEL           = False
EVALUATE_SINGLE_FOLD_TEST   = True
ENABLE_POSTPROCESS          = False

PERSISTENT_WORKERS          = True                                                                      # Giữ worker chạy giữa các epoch 
PREFETCH_FACTOR             = 8                                                                         # Hệ số tải trước
NUMBBER_OF_WORKERS          = 8                                                                         # Số luồng tải dữ liệu song song

IMAGE_SIZE                  = 224


BATCH_SIZE                  = 8
GRAD_ACCUM_STEPS            = 2                                                                        # Hệ số batch size / epoch

SEED                        = 42

EPOCHS                      = 30
LEARNING_RATE               = 0.0001
WEIGHT_DECAY                = 0.0005
PATIENCE                    = 5
THRESHOLD                   = 0.5

USE_AMP                     = True
SCHEDULER_PATIENCE          = 3
SCHEDULER_FACTOR            = 0.5
MIN_DELTA                   = 0

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

BRIGHTNESS                  = 0.10
CONTRAST                    = 0.10

FLIP_CHANCE                 = 0.5
# Huấn luyện lại
RESUME_TRAINING             = True

## Cấu trúc text: [RESULT]: TITLE: MESSAGE
MIN_RESULT_WIDTH            = 10
MIN_TITLE_WIDTH             = 30
MIN_MESSAGE_WIDTH           = 20
MIN_NUMBER_WIDTH            = 8
    
# Khác
IMAGE_EXTENSION             = {".png", ".jpg", ".jpeg"}

POSTPROCESS_CALIBRATION     = "none"       # none | temperature
THRESHOLD_BETA              = 1.0          # F1; co the chon F2 = 2.0
THRESHOLD_MIN_RECALL        = 0.0          # Hoac rang buoc recall min tren OOF
