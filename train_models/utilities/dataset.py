from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# Tập dữ liệu NIH ChestX-ray14: mỗi dòng của dataframe là một ảnh, nhãn là vector nhị phân 14 lớp
class NIHChestXrayDataset(Dataset):

    def __init__(self, dataframe, classes, transform = None):
        self.df = dataframe.reset_index(drop = True).copy()
        self.classes = classes
        self.transform = transform
        self.image_paths = self.df["image_path"].astype(str).tolist()
        self.labels = self.df[classes].to_numpy(dtype = np.float32)

    def __len__(self):
        return len(self.df)

    # Đọc ảnh xám, chuyển sang 3 kênh (mô hình tiền huấn luyện trên ImageNet) rồi biến đổi
    def __getitem__(self, index):
        image_path = Path(self.image_paths[index])
        if not image_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh:\n {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)

        return image, torch.from_numpy(self.labels[index])
