utilities/
│
├── check.py         # Kiểm tra dữ liệu, đường dẫn và tính hợp lệ của Fold
├── data.py          # Xử lý dữ liệu (Đọc metadata, thiết lập DataLoader, hàm làm giàu ảnh `Trans và pos_weight)
├── dataset.py       # Định nghĩa NIHChestXrayDataset
├── transform.py     # Augmentation và chuẩn hóa ảnh
├── metric.py        # Tính AUROC, Precision, Recall, F1 và kết quả dự đoán
├── training.py      # Huấn luyện epoch, Fold và final model
├── evaluation.py    # Đánh giá Validation và Test
├── experiment.py    # Điều phối thí nghiệm và K-Fold Cross-Validation
├── setting.py       # Thiết lập thiết bị, seed và lựa chọn Fold
├── notice.py        # Định dạng thông báo console
├── progress.py      # Tùy chỉnh thanh tiến trình
└── text.py          # Các hàm xử lý chuỗi