# ============================================================
# Tập dữ liệu NIH ChestX-ray14 (Dataset)
# Mục tiêu: Đọc từng ảnh X-quang và vector nhãn nhị phân của ảnh đó cho DataLoader của PyTorch.
# ============================================================

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import config

# ============================================================
# Tập dữ liệu NIH ChestX-ray14
class NIHChestXrayDataset(Dataset):

    # Khởi tạo tập dữ liệu
    def __init__(self, dataframe, classes, transform = None):

        # Kiểm tra dữ liệu đầu vào
        ## dataframe phải có cột image_path và cột nhãn của mọi lớp
        missing_columns = [column for column in ["image_path", *classes] if column not in dataframe.columns]

        if missing_columns:
            raise ValueError(
                "[NIHChestXrayDataset]: dataframe phải có cột image_path và cột nhãn của mọi lớp.\n"
                f"Thiếu cột: {missing_columns}"
            )

        # Lấy dữ liệu
        self.df = dataframe.reset_index(drop = True).copy()                     # Bảng dữ liệu (đánh lại chỉ số từ 0)
        self.classes = classes                                                  # Danh sách lớp theo thứ tự đầu ra của mô hình
        self.transform = transform                                              # Biến đổi ảnh (làm giàu ảnh / chuẩn hóa)

        # Chuẩn bị đường dẫn ảnh và nhãn của mọi dòng
        ## image_path tương đối theo DATASET_DIRECTORY (dạng images/x.png)
        self.image_paths = [str(config.DATASET_DIRECTORY / path) for path in self.df["image_path"].astype(str)]
        self.labels = self.df[classes].to_numpy(dtype = np.float32)

    # Số ảnh của tập dữ liệu
    def __len__(self):
        return len(self.df)

    # Đọc một ảnh
    # Trả về: (ảnh đã biến đổi, nhãn [num_classes])
    def __getitem__(self, index):

        # Lấy đường dẫn ảnh
        image_path = Path(self.image_paths[index])
        
        ## Kiểm tra tệp ảnh có trong máy
        if not image_path.exists():
            raise FileNotFoundError(
                "[NIHChestXrayDataset]: Không tìm thấy ảnh.\n"
                f"Nhận được: image_path = {image_path}"
            )

        # Đọc ảnh xám --> Chuyển sang 3 kênh RGB --> Biến đổi ảnh
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)

        return image, torch.from_numpy(self.labels[index])
