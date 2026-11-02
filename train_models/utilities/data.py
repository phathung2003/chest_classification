# ============================================================
# Dữ liệu huấn luyện (Data Pipeline)
# Mục tiêu: Đọc và kiểm tra tập dữ liệu đã tiền xử lý, tạo biến đổi ảnh và DataLoader có seed để thứ tự ảnh và làm giàu ảnh tái lập được giữa các lần chạy.
# ============================================================

import json
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import config

# ============================================================
# Đọc tập dữ liệu
# Trả về: (train_df, test_df, tên tập dữ liệu, danh sách lớp, số Fold)
def load_dataset():

    # Kiểm tra dữ liệu đầu vào
    ## Các tệp của tập dữ liệu phải tồn tại
    missing_files = [str(path) for path in (config.DATASET_INFO_DIRECTORY, config.TRAIN_FOLDS_DIRECTORY, config.TEST_DIRECTORY) if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "[load_dataset]: Không tìm thấy tệp của tập dữ liệu (chạy notebook tiền xử lý trước).\n"
            f"Thiếu: {missing_files}"
        )

    # Đọc thông tin tập dữ liệu
    with open(config.DATASET_INFO_DIRECTORY, "r", encoding = "utf-8") as file:
        dataset_config = json.load(file)

    # Lấy tên tập dữ liệu, danh sách lớp và số Fold
    name = dataset_config["dataset_name"]
    classes = list(dataset_config["classes"])
    number_of_folds = int(dataset_config["number_of_folds"])

    # Đọc tập huấn luyện (đã gán Fold) và tập Test
    train_df = pd.read_csv(config.TRAIN_FOLDS_DIRECTORY)
    test_df = pd.read_csv(config.TEST_DIRECTORY)

    # Kiểm tra dữ liệu
    validate_data(train_df, test_df, classes, number_of_folds)

    # Kiểm tra ảnh trên đĩa
    check_image_files(train_df, test_df)

    return train_df, test_df, name, classes, number_of_folds

# ============================================================
# Kiểm tra ảnh trong tập dữ liệu
def check_image_files(*frames):

    # Khởi tạo biến lưu kết quả
    listed_directories, missing_images = {}, []                                 # Danh sách tệp của từng thư mục, ảnh bị thiếu

    # Duyệt từng ảnh của mọi bảng dữ liệu
    for frame in frames:
        for relative_path in frame["image_path"].astype(str):
            path = config.DATASET_DIRECTORY / relative_path

            # Đọc danh sách tệp của mỗi thư mục một lần
            if path.parent not in listed_directories:
                listed_directories[path.parent] = set(os.listdir(path.parent)) if path.parent.is_dir() else set()

            # Ảnh không có trong thư mục --> Ghi nhận ảnh bị thiếu
            if path.name not in listed_directories[path.parent]:
                missing_images.append(relative_path)

    # Kiểm tra dữ liệu đầu vào
    ## Không được thiếu ảnh
    if missing_images:
        raise FileNotFoundError(
            f"[check_image_files]: Thiếu ảnh trên đĩa. image_path phải là đường dẫn tương đối theo {config.DATASET_DIRECTORY} (dạng images/x.png); dữ liệu có đường dẫn tuyệt đối của máy khác thì chạy lại notebook tiền xử lý.\n"
            f"Số ảnh bị thiếu: {len(missing_images):,}."
        )

    ## Kích thước ảnh phải bằng IMAGE_SIZE
    for frame in frames:
        if frame.empty:
            continue

        first_image = str(frame["image_path"].iloc[0])

        with Image.open(config.DATASET_DIRECTORY / first_image) as image:
            image_size = image.size

        if image_size != (config.IMAGE_SIZE, config.IMAGE_SIZE):
            raise ValueError(
                "[check_image_files]: Kích thước ảnh phải bằng IMAGE_SIZE trong config.py (ảnh được đọc nguyên kích thước, không resize khi huấn luyện).\n"
                f"Nhận được: Ảnh {first_image} có kích thước {image_size}, Mong muốn: IMAGE_SIZE = {config.IMAGE_SIZE}"
            )

# ============================================================
# Kiểm tra tập dữ liệu
def validate_data(train_df, test_df, classes, number_of_folds):

    # Kiểm tra dữ liệu đầu vào
    ## Đủ cột bắt buộc
    missing_train = [column for column in ["image_path", "Fold", *classes] if column not in train_df.columns]
    missing_test = [column for column in ["image_path", *classes] if column not in test_df.columns]

    if missing_train or missing_test:
        raise ValueError(
            "[validate_data]: Thiếu cột bắt buộc.\n"
            f"Thiếu cột: train_folds.csv = {missing_train}, test.csv = {missing_test}"
        )

    ## Hai tập không được rỗng
    if train_df.empty or test_df.empty:
        raise ValueError(
            "[validate_data]: Tập huấn luyện và tập Test không được rỗng.\n"
            f"Nhận được: train_folds.csv = {len(train_df)} dòng, test.csv = {len(test_df)} dòng"
        )

    ## Cột Fold phải có đủ các Fold từ 1 tới number_of_folds
    actual_folds = set(train_df["Fold"].astype(int).unique())
    expected_folds = set(range(1, int(number_of_folds) + 1))

    if actual_folds != expected_folds:
        raise ValueError(
            "[validate_data]: Cột Fold phải có đủ các Fold từ 1 tới number_of_folds.\n"
            f"Nhận được: Fold thực tế = {[int(fold) for fold in sorted(actual_folds)]}, Fold mong đợi = {sorted(expected_folds)}"
        )

    ## Nhãn phải là nhị phân (0 hoặc 1) và không bị thiếu
    for split_name, frame in (("train", train_df), ("test", test_df)):
        label_matrix = frame[classes].to_numpy()
        invalid_label = ~np.isin(label_matrix, [0, 1])

        if np.any(invalid_label):
            row, col = np.argwhere(invalid_label)[0]

            raise ValueError(
                f"[validate_data]: Tập {split_name} phải có nhãn nhị phân (0 hoặc 1) và không bị thiếu.\n"
                f"Số phần tử không hợp lệ: {np.count_nonzero(invalid_label)}.\n"
                f"Lỗi đầu tiên tại dòng {row}, lớp {classes[col]}: Giá trị = {label_matrix[row, col]!r}"
            )

    ## Không có ảnh nằm ở cả tập huấn luyện và tập Test
    shared_images = set(train_df["image_path"].astype(str)) & set(test_df["image_path"].astype(str))

    if shared_images:
        raise ValueError(
            "[validate_data]: Không được có ảnh nằm ở cả tập huấn luyện và tập Test.\n"
            f"Số ảnh bị trùng: {len(shared_images)}.\n"
            f"Ví dụ: {sorted(shared_images)[0]}"
        )

    # Kiểm tra chia theo bệnh nhân (cần cột mã bệnh nhân ở cả hai tập)
    patient_column = config.PATIENT_ID_COLUMN

    ## Không có cột mã bệnh nhân --> Cảnh báo và bỏ qua
    if patient_column not in train_df or patient_column not in test_df:
        print(f"CẢNH BÁO: không có cột {patient_column}, không kiểm tra được việc chia theo bệnh nhân.")
        return

    ## Mã bệnh nhân không được thiếu
    if train_df[patient_column].isna().any() or test_df[patient_column].isna().any():
        raise ValueError(
            f"[validate_data]: Cột {patient_column} không được thiếu giá trị.\n"
            f"Số dòng bị thiếu: train_folds.csv = {int(train_df[patient_column].isna().sum())}, test.csv = {int(test_df[patient_column].isna().sum())}"
        )

    ## Không có bệnh nhân nằm ở cả tập huấn luyện và tập Test
    shared_patients = set(train_df[patient_column].astype(str)) & set(test_df[patient_column].astype(str))

    if shared_patients:
        raise ValueError(
            "[validate_data]: Không được có bệnh nhân nằm ở cả tập huấn luyện và tập Test.\n"
            f"Số bệnh nhân bị trùng: {len(shared_patients)}.\n"
            f"Ví dụ: {sorted(shared_patients)[0]}"
        )

    ## Mỗi bệnh nhân chỉ nằm ở một Fold
    folds_per_patient = train_df.groupby(patient_column)["Fold"].nunique()
    leaked_patients = folds_per_patient.index[folds_per_patient > 1]

    if len(leaked_patients) > 0:
        raise ValueError(
            "[validate_data]: Một bệnh nhân không được nằm ở nhiều Fold (rò rỉ bệnh nhân).\n"
            f"Số bệnh nhân bị rò rỉ: {len(leaked_patients)}.\n"
            f"Bệnh nhân đầu tiên: {leaked_patients[0]}"
        )

# ============================================================
# Biến đổi ảnh
# Trả về: (train_transform, eval_transform)
def transform_setting():

    # Chuẩn hóa theo mean / std của ImageNet
    normalize = transforms.Normalize(mean = config.TRANSFORM_NORMALIZE_MEAN, std = config.TRANSFORM_NORMALIZE_STD)

    # Làm giàu ảnh (Tập huấn luyện)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p = config.FLIP_CHANCE),
        transforms.RandomAffine(degrees = config.DEGREES, translate = config.TRANSLATE, scale = config.SCALE, fill = 0, interpolation = InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness = config.BRIGHTNESS, contrast = config.CONTRAST),
        normalize,
    ])

    # Biến đổi ảnh cho tập kiểm thử / kiểm tra (không làm giàu ảnh, chỉ chuẩn hóa)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform

# ============================================================
# Đặt seed cho từng worker
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ============================================================
# Thứ tự ảnh của tập huấn luyện theo epoch
class EpochSampler(Sampler):

    # Khởi tạo Sampler
    def __init__(self, size, seed):

        # Chuyển đổi dữ liệu
        self.size = int(size)                                                   # Số ảnh
        self.seed = int(seed)                                                   # Seed của Fold
        self.epoch = None                                                       # Epoch hiện tại (đặt bằng set_epoch)

        # Kiểm tra dữ liệu đầu vào
        ## Tập dữ liệu phải có ít nhất một ảnh, seed không âm (yêu cầu của np.random.default_rng)
        if self.size < 1 or self.seed < 0:
            raise ValueError(
                "[EpochSampler]: Cần size >= 1 và seed >= 0.\n"
                f"Nhận được: size = {self.size}, seed = {self.seed}"
            )

    # Đặt epoch (Gọi trước khi duyệt loader của mỗi epoch)
    def set_epoch(self, epoch):

        # Chuyển đổi dữ liệu
        epoch = int(epoch)

        # Kiểm tra dữ liệu đầu vào
        ## epoch không âm (yêu cầu của np.random.default_rng)
        if epoch < 0:
            raise ValueError(
                "[EpochSampler.set_epoch]: epoch phải lớn hơn hoặc bằng 0.\n"
                f"Nhận được: epoch = {epoch}"
            )

        self.epoch = epoch

    # Số ảnh mỗi epoch
    def __len__(self):
        return self.size

    # Thứ tự ảnh của epoch hiện tại
    # Trả về: Iterator các cặp (seed làm giàu ảnh, chỉ số ảnh)
    def __iter__(self):

        # Kiểm tra dữ liệu đầu vào
        ## Phải gọi set_epoch() trước khi duyệt
        if self.epoch is None:
            raise RuntimeError(
                "[EpochSampler]: Gọi set_epoch() trước khi duyệt train loader.\n"
                "Nhận được: epoch = None"
            )

        # Bộ sinh ngẫu nhiên theo (seed, epoch)
        rng = np.random.default_rng([self.seed, self.epoch])

        # Xáo trộn thứ tự ảnh
        order = rng.permutation(self.size)

        # Sinh seed làm giàu ảnh của từng ảnh (cùng bộ sinh)
        sample_seeds = rng.integers(0, 2 ** 63 - 1, size = self.size)

        return iter([(int(sample_seeds[index]), int(index)) for index in order])

# ============================================================
# Đặt seed cho bộ sinh ngẫu nhiên CPU của torch
class SeededSamples(Dataset):

    # Khởi tạo (bọc tập dữ liệu gốc)
    def __init__(self, dataset):
        self.dataset = dataset

    # Số ảnh của tập dữ liệu gốc
    def __len__(self):
        return len(self.dataset)

    # Đọc một ảnh với seed làm giàu ảnh riêng (key = (seed làm giàu ảnh, chỉ số ảnh) do EpochSampler sinh ra)
    # Trả về: (ảnh đã biến đổi, nhãn) của tập dữ liệu gốc
    def __getitem__(self, key):

        # Tách seed làm giàu ảnh và chỉ số ảnh
        sample_seed, index = key

        # fork_rng: Đặt seed tạm thời, trạng thái ngẫu nhiên của worker được khôi phục sau khi đọc ảnh
        with torch.random.fork_rng(devices = []):
            torch.default_generator.manual_seed(sample_seed)
            return self.dataset[index]

# ============================================================
# Tạo DataLoader
# Trả về: (dataset, loader)
def create_dataloader(dataset_class, classes, device, dataframe, transform, shuffle, *, batch_size, seed):

    # Kiểm tra dữ liệu đầu vào
    ## batch_size phải là số nguyên dương
    if int(batch_size) < 1:
        raise ValueError(
            "[create_dataloader]: batch_size phải là số nguyên dương.\n"
            f"Nhận được: batch_size = {batch_size}"
        )

    # Tạo tập dữ liệu
    dataset = dataset_class(dataframe = dataframe, classes = classes, transform = transform)
    sampler = None

    # Tập huấn luyện: Thêm EpochSampler và seed làm giàu ảnh của từng ảnh
    if shuffle:
        sampler = EpochSampler(len(dataset), seed)
        dataset = SeededSamples(dataset)

    # Bộ sinh ngẫu nhiên của DataLoader (seed của worker lấy từ đây)
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    # Số worker đọc dữ liệu
    workers = int(config.NUMBER_OF_WORKERS)

    # Tạo DataLoader
    ## Worker khởi tạo bằng spawn trên mọi hệ điều hành: Windows chỉ có spawn; fork (mặc định của Linux) sao chép tiến trình đang có luồng chạy (theo dõi GPU, tqdm) cùng các khóa của chúng, có thể làm worker bị treo
    loader = DataLoader(dataset, batch_size = int(batch_size), sampler = sampler, num_workers = workers, prefetch_factor = int(config.PREFETCH_FACTOR) if workers > 0 else None, pin_memory = device.type == "cuda", drop_last = False, worker_init_fn = _seed_worker, generator = generator, persistent_workers = bool(config.PERSISTENT_WORKERS and workers > 0), multiprocessing_context = "spawn" if workers > 0 else None)

    return dataset, loader
