
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

# ============================================================
# Tập dữ liệu: NIH Chest X-ray 14
class NIHChestXrayDataset(Dataset):

    def __init__(self, dataframe, classes, transform=None,):

        self.df = (dataframe.reset_index(drop=True).copy())
        self.classes = classes
        self.transform = transform
        self.image_paths = (self.df["image_path"].astype(str).tolist())
        self.labels = (self.df[classes].to_numpy(dtype=np.float32))

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image_path = Path(self.image_paths[index])
        
        if not image_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh:\n {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)


        labels = torch.from_numpy(self.labels[index])
        
        return (image, labels)