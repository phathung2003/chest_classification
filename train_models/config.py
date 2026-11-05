# ============================================================
# Cấu hình huấn luyện (Configuration)
# ============================================================

from pathlib import Path

# ============================================================
# Đường dẫn
ROOT_DIRECTORY              = Path(__file__).resolve().parents[0]                                       # Thư mục huấn luyện mô hình
ORIGINAL_DATASET_DIRECTORY  = (ROOT_DIRECTORY / "original_dataset").resolve()                           # Tập dữ liệu gốc
DATASET_DIRECTORY           = (ROOT_DIRECTORY / "dataset").resolve()                                    # Tập dữ liệu sau tiền xử lý
FOLDS_DIRECTORY             = (DATASET_DIRECTORY / "folds").resolve()                                   # Thống kê từng Fold (do notebook tiền xử lý tạo)
TRAIN_FOLDS_DIRECTORY       = (DATASET_DIRECTORY / "train_folds.csv").resolve()                         # Tập huấn luyện đã gán Fold
TEST_DIRECTORY              = (DATASET_DIRECTORY / "test.csv").resolve()                                # Tập Test chính thức (chỉ dùng để báo cáo)
DATASET_INFO_DIRECTORY      = (DATASET_DIRECTORY / "dataset_info.json").resolve()                       # Tên tập dữ liệu, danh sách lớp, số Fold
MODELS_OUTPUT_DIRECTORY     = (ROOT_DIRECTORY / "training_results").resolve()                           # Kết quả huấn luyện

# ============================================================
# Dữ liệu và Fold
NUMBER_OF_FOLDS             = 5                                                                         # Số Fold khi chia dữ liệu (notebook tiền xử lý)
TRAIN_ALL_FOLD              = False                                                                     # True: chạy đủ mọi Fold (bắt buộc cho kết quả chính thức)
SELECTED_FOLD               = [1]                                                                       # Fold chạy khi TRAIN_ALL_FOLD = False
TRAIN_PATIENT_FRACTION      = 1.0                                                                       # Tỉ lệ bệnh nhân của tập huấn luyện được giữ (Validation giữ nguyên); < 1: tinh chỉnh siêu tham số với ngân sách nhỏ
TRAIN_PATIENT_SEED          = 2026                                                                      # Seed chọn tập con bệnh nhân (cố định, không theo SEED: mọi mô hình, seed và learning_rate dùng chung một tập con)
IMAGE_SIZE                  = 224                                                                       # Kích thước ảnh đầu vào (ảnh đã resize sẵn)
IMAGE_VARIANT               = None                                                                      # Bộ ảnh: None = images_<IMAGE_SIZE>/ (images/ ở kích thước gốc); "aligned" = images_<IMAGE_SIZE>_aligned/ đã căn đường giữa (preprocess/align_midline.py)
IMAGE_EXTENSION             = {".png", ".jpg", ".jpeg"}                                                 # Định dạng ảnh hợp lệ

# ============================================================
# DataLoader
BATCH_SIZE                  = 32                                                                        # Số ảnh mỗi batch
GRAD_ACCUM_STEPS            = 1                                                                         # Số batch cộng dồn gradient trước mỗi lần cập nhật
NUMBER_OF_WORKERS           = 4                                                                         # Số tiến trình tải dữ liệu song song
PREFETCH_FACTOR             = 8                                                                         # Số batch mỗi worker tải trước
PERSISTENT_WORKERS          = True                                                                      # Giữ worker suốt Fold (Train và Validation), không khởi động lại mỗi epoch

# ============================================================
# Huấn luyện
EPOCHS                      = 100                                                                       # Số epoch tối đa
LEARNING_RATE               = 0.0001                                                                    # Learning rate ban đầu
WEIGHT_DECAY                = 0.0005                                                                    # Weight decay (mặc định cho AdamW)
NEW_MODULE_LR_MULTIPLIER    = 1.0                                                                       # LR lớp mới (neck, khối trộn, đầu phân loại) = LEARNING_RATE x hệ số; backbone tiền huấn luyện (PRETRAINED_MODULES của mô hình) giữ LEARNING_RATE; 1.0 = một LR chung

GRAD_CLIP_NORM              = None                                                                      # Ngưỡng clip gradient (ví dụ 1.0); None = không clip
USE_AMP                     = True                                                                      # Huấn luyện mixed precision trên GPU (kiểu AMP_DTYPE)
AMP_DTYPE                   = "float16"                                                                 # Kiểu AMP: float16 (kèm GradScaler) | bfloat16 (cùng dải số với FP32, không cần GradScaler)
EMA_DECAY                   = 0.9995                                                                    # EMA trọng số: hệ số tối đa (khởi động (1 + n) / (10 + n)); Validation, chọn và lưu mô hình bằng trọng số EMA; None = tắt
PATIENCE                    = 5                                                                         # Early stopping: số epoch không cải thiện tối đa
MIN_DELTA                   = 0                                                                         # Early stopping: mức cải thiện tối thiểu
SCHEDULER_PATIENCE          = 2                                                                         # ReduceLROnPlateau: số epoch chờ trước khi giảm learning_rate
SCHEDULER_FACTOR            = 0.5                                                                       # ReduceLROnPlateau: hệ số giảm learning_rate
OPTIMIZER_NAME              = "adamw"                                                                   # Optimizer mặc định: adam | adamw | sgd
ADAM_BETAS                  = (0.9, 0.999)                                                              # Adam / AdamW: hệ số trung bình động của gradient và bình phương gradient
ADAM_EPSILON                = 1e-8                                                                      # Adam / AdamW: hằng số ổn định số học
ADAM_FUSED                  = True                                                                      # Adam / AdamW: cập nhật mọi tham số trong một kernel (cùng thuật toán, nhanh hơn, không đồng bộ CPU / GPU khi dùng AMP)
SGD_MOMENTUM                = 0.0                                                                       # SGD: momentum
SGD_NESTEROV                = False                                                                     # SGD: dùng Nesterov momentum
SCHEDULER_NAME              = "reduce_on_plateau"                                                       # Scheduler mặc định: reduce_on_plateau | warmup_cosine | none
SCHEDULER_THRESHOLD         = 1e-4                                                                      # ReduceLROnPlateau: mức thay đổi tối thiểu được tính là cải thiện
SCHEDULER_WARMUP_EPOCHS     = 1                                                                         # warmup_cosine: LR tăng tuyến tính trong số epoch này rồi giảm theo cosine về 0 ở epoch EPOCHS (cập nhật mỗi bước tối ưu)
THRESHOLD                   = 0.5                                                                       # Ngưỡng mặc định để tính Precision/Recall/F1
RESUME_TRAINING             = True                                                                      # Chạy tiếp lượt chạy dở nếu chữ ký cấu hình khớp
PAUSE_AFTER_EPOCH           = None                                                                      # Tạm dừng Fold sau epoch này (checkpoint đã lưu, chạy lại để tiếp tục; loại dần mức learning_rate khi tinh chỉnh); None = không tạm dừng; không vào chữ ký

# ============================================================
# Tái lập kết quả
SEED                        = 42                                                                        # Seed gốc (Fold và epoch được cộng thêm vào seed này)
CUDNN_DETERMINISTIC         = True                                                                      # cuDNN chọn thuật toán tất định
CUDNN_BENCHMARK             = False                                                                     # Tắt tự dò thuật toán (dò sẽ làm mất tính tái lập)
DETERMINISTIC_ALGORITHMS    = True                                                                      # Báo lỗi nếu dùng phép toán không tất định

# ============================================================
# Làm giàu ảnh
FLIP_CHANCE                 = 0.5                                                                       # Xác suất lật ngang
DEGREES                     = 5                                                                         # Góc xoay tối đa (độ)
TRANSLATE                   = (0.02, 0.02)                                                              # Tịnh tiến tối đa theo tỉ lệ kích thước ảnh
SCALE                       = (0.95, 1.05)                                                              # Khoảng phóng to / thu nhỏ
BRIGHTNESS                  = 0.10                                                                      # Biên độ thay đổi độ sáng
CONTRAST                    = 0.10                                                                      # Biên độ thay đổi độ tương phản
TRANSFORM_NORMALIZE_MEAN    = [0.485, 0.456, 0.406]                                                     # Chuẩn hóa theo ImageNet
TRANSFORM_NORMALIZE_STD     = [0.229, 0.224, 0.225]                                                     # Độ lệch chuẩn theo ImageNet (cùng thứ tự kênh R, G, B với TRANSFORM_NORMALIZE_MEAN)

# ============================================================
# Đánh giá trên tập kiểm tra
EVALUATE_FOLD_MODELS_ON_TEST = True                                                                     # Đánh giá mô hình tốt nhất của mỗi Fold, tổng hợp mean ± std và ensemble
BOOTSTRAP_ITERATION         = 1000                                                                      # Số lần bootstrap theo bệnh nhân (khoảng tin cậy 95%); 0 = bỏ qua
BOOTSTRAP_SEED              = 2024                                                                      # Seed của bootstrap
PATIENT_ID_COLUMN           = "Patient ID"                                                              # Cột mã bệnh nhân dùng để bootstrap
IMAGE_ID_COLUMN             = "Image Index"                                                             # Cột mã ảnh (khóa ghép dự đoán giữa các mô hình)
BOOTSTRAP_CONFIDENCE        = 0.95                                                                      # Mức tin cậy của khoảng bootstrap
COMPARE_BOOTSTRAP_ITERATION = 2000                                                                      # Số lần bootstrap khi so sánh ghép cặp hai mô hình (compare_models.py)
INFERENCE_WARMUP_STEPS      = 10                                                                        # Đo tốc độ suy luận: số lần chạy làm nóng (không tính giờ)
INFERENCE_TIMED_STEPS       = 30                                                                        # Đo tốc độ suy luận: số lần chạy tính giờ (lấy trung vị)
SELECTION_METRIC            = "macro_auroc"                                                             # Tiêu chí chọn mô hình tốt nhất: macro_auroc | validation_loss

# ============================================================
# Hậu xử lý: chọn ngưỡng theo lớp trên dự đoán OOF
ENABLE_POSTPROCESS          = True                                                                      # Bật chọn ngưỡng theo lớp
POSTPROCESS_CALIBRATION     = "none"                                                                    # none | temperature
THRESHOLD_BETA              = 1.0                                                                       # Tối ưu F-beta (1.0 = F1, 2.0 = F2)
THRESHOLD_MIN_RECALL        = 0.0                                                                       # Ràng buộc recall tối thiểu khi chọn ngưỡng
THRESHOLD_CALIBRATION_FRACTION = 0.75                                                                   # Tỉ lệ bệnh nhân trong OOF của Fold dùng để chọn ngưỡng (phần còn lại để đánh giá)
TEMPERATURE_MIN_POSITIVE    = 20                                                                        # Temperature scaling: số mẫu dương tối thiểu để hiệu chỉnh một lớp
TEMPERATURE_MIN_NEGATIVE    = 20                                                                        # Temperature scaling: số mẫu âm tối thiểu để hiệu chỉnh một lớp
TEMPERATURE_BOUNDS          = (0.3, 10.0)                                                               # Temperature scaling: khoảng tìm nhiệt độ T

# ============================================================
# Truy vết thí nghiệm
REQUIRE_CLEAN_GIT           = False                                                                     # True cho lần chạy chính thức: từ chối chạy khi còn thay đổi chưa commit

# ============================================================
# Giá trị mặc định của các hàm Loss
BCE_WITH_LOGITS_ALPHA       = 1.0                                                                       # BCE (pos_weight_alpha): số mũ của pos_weight = (âm/dương) ^ alpha
BCE_WITH_LOGITS_CLIP        = None                                                                      # BCE (max_pos_weight): giới hạn trên của pos_weight; None = không giới hạn
SIGMOID_FOCAL_LOSS_GAMMA    = 2.0                                                                       # Focal (gamma): mức tập trung vào mẫu khó
SIGMOID_FOCAL_LOSS_ALPHA    = None                                                                      # Focal (pos_weight_alpha): trọng số lớp dương; None = không cân bằng
ASYMMETRIC_LOSS_GAMMA_NEGATIVE = 4.0                                                                    # ASL (gamma_negative): gamma cho mẫu âm
ASYMMETRIC_LOSS_GAMMA_POSITIVE = 0.0                                                                    # ASL (gamma_positive): gamma cho mẫu dương
ASYMMETRIC_LOSS_CLIP        = 0.05                                                                      # ASL (clip): dịch xác suất mẫu âm (probability margin)
ASYMMETRIC_LOSS_EPSILON     = 1e-8                                                                      # ASL: chặn dưới trước khi lấy log

# ============================================================
# Cấu hình huấn luyện chung cho các mô hình
EXPERIMENT_PROTOCOL = {
    "loss_config": {
        "name"              : "asl",
        "gamma_negative"    : ASYMMETRIC_LOSS_GAMMA_NEGATIVE,
        "gamma_positive"    : ASYMMETRIC_LOSS_GAMMA_POSITIVE,
        "clip"              : ASYMMETRIC_LOSS_CLIP,
    },
    "optimizer_config": {
        "name"              : OPTIMIZER_NAME,
        "learning_rate"     : LEARNING_RATE,
        "weight_decay"      : WEIGHT_DECAY,
        "new_module_lr_multiplier": NEW_MODULE_LR_MULTIPLIER,
    },
    "scheduler_config": {
        "name"              : SCHEDULER_NAME,
        "mode"              : "min" if SELECTION_METRIC == "validation_loss" else "max",   # Theo dõi cùng chỉ số với SELECTION_METRIC
        "factor"            : SCHEDULER_FACTOR,
        "patience"          : SCHEDULER_PATIENCE,
    },
    "selection_metric"      : SELECTION_METRIC,
}

# ============================================================
# Giao thức CHUNG của nghiên cứu CLIS-Mamba (models/clis_mamba.py, models/run_clis.py) cho mọi mô hình được so sánh trong
# nghiên cứu này: DenseNet121, ConvNeXt-T và các biến thể CLIS-Mamba. CLIS_CONFIG ghi đè các hằng số ở trên trước khi chạy
# (utilities.setting.apply_config); CLIS_PROTOCOL thay cho EXPERIMENT_PROTOCOL. Giá trị thực tế vào chữ ký qua config / training_config.
CLIS_CONFIG = {
    "IMAGE_SIZE"            : 512,
    "IMAGE_VARIANT"         : "aligned",            # Ảnh đã căn đường giữa cơ thể về giữa ảnh (mặt nạ phổi CheXmask)
    "FLIP_CHANCE"           : 0.0,                  # Không lật ngang: giữ bất đối xứng giải phẫu trái / phải
    "DEGREES"               : 7,
    "TRANSLATE"             : (0.04, 0.04),
    "SCALE"                 : (0.9, 1.1),
    "EPOCHS"                : 20,
    "PATIENCE"              : 4,
    "BATCH_SIZE"            : 8,                    # Micro-batch
    "GRAD_ACCUM_STEPS"      : 4,                    # Batch hiệu dụng 32
    "GRAD_CLIP_NORM"        : 1.0,
    "AMP_DTYPE"             : "bfloat16",
    "ADAM_FUSED"            : True,                 # Fused AdamW: cùng thuật toán với foreach (lệch ở mức làm tròn float32), cần torch >= 2.5 (env mamba-t29); torch 2.4 (env mamba-main) lỗi với gradient conv theo kênh dạng channels-last: --set ADAM_FUSED=False
    "EVALUATE_FOLD_MODELS_ON_TEST": False,          # Khóa tập Test tới cổng G3 (đăng ký trước kế hoạch phân tích)
}
CLIS_PROTOCOL = {
    "loss_config"           : {"name": "bce", "pos_weight_alpha": 0.0},                                         # BCE không pos_weight
    "optimizer_config"      : {"name": "adamw", "learning_rate": 1e-4, "weight_decay": 0.05, "new_module_lr_multiplier": 5.0, "no_weight_decay_1d": True},   # Backbone 1e-4, lớp mới 5e-4
    "scheduler_config"      : {"name": "warmup_cosine", "warmup_epochs": 1},
    "selection_metric"      : "macro_auroc",
}

# ============================================================
# Hiển thị console. Cấu trúc dòng: [RESULT] TITLE: MESSAGE
MIN_RESULT_WIDTH            = 10                                                                        # Độ rộng tối thiểu của cột RESULT
MIN_TITLE_WIDTH             = 30                                                                        # Độ rộng tối thiểu của cột TITLE
MIN_MESSAGE_WIDTH           = 20                                                                        # Độ rộng tối thiểu của cột MESSAGE
MIN_NUMBER_WIDTH            = 8                                                                         # Độ rộng tối thiểu khi in số
CONSOLE_WIDTH               = 100                                                                       # Độ dài dòng "=" / "-" của tiêu đề
GPU_MONITOR_INTERVAL        = 1.0                                                                       # Chu kỳ đọc GPU usage / VRAM / điện năng / nhiệt độ cho thanh tiến trình (giây)
PROGRESS_SYNC_STEPS         = 20                                                                        # Số batch giữa hai lần đọc số liệu từ GPU (thanh tiến trình, kiểm tra Loss); mỗi lần đọc buộc CPU chờ GPU
