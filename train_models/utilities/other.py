import re
import json
from pathlib import Path

# ============================================================
# Đổi kiểu chữ rắn
def to_snake_case(text):
    text = text.replace("-", "").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")

    return text

# ============================================================
# Cập nhật file JSON
def update_json(file_path: str, data: dict):
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as file:
        current_data = json.load(file)

    current_data.update(data)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(current_data, file, indent=4, ensure_ascii=False)

# ============================================================
# Tạo file json  
def create_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)