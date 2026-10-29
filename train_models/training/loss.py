import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

# ============================================================
# Các khóa hợp lệ của từng hàm Loss (khóa sai tên sẽ báo lỗi thay vì bị bỏ qua)
ALLOWED_LOSS_KEYS = {
    "bce"           : {"name", "alpha", "clip"},
    "chexnet_bce"   : {"name"},
    "focal"         : {"name", "gamma", "alpha"},
    "asl"           : {"name", "gamma_negative", "gamma_positive", "clip"},
}


# ============================================================
# Kiểm tra cấu hình Loss, trả về bản sao đã chuẩn hóa tên
def validate_loss_config(loss_config):
    if not isinstance(loss_config, dict) or not isinstance(loss_config.get("name"), str):
        raise TypeError("loss_config phải là dict có khóa 'name'")

    loss_config = deepcopy(loss_config)
    loss_config["name"] = loss_config["name"].lower()
    name = loss_config["name"]

    # Hàm Loss có được hỗ trợ không
    if name not in ALLOWED_LOSS_KEYS:
        raise ValueError(f"Hàm Loss '{name}' không được hỗ trợ. Các hàm Loss hợp lệ: {sorted(ALLOWED_LOSS_KEYS)}")

    # Khóa lạ --> báo lỗi
    unknown = set(loss_config) - ALLOWED_LOSS_KEYS[name]
    if unknown:
        raise ValueError(f"Hàm Loss '{name}' không nhận các khóa {sorted(unknown)}. "
                         f"Các khóa hợp lệ: {sorted(ALLOWED_LOSS_KEYS[name])}")
    return loss_config


# ============================================================
# Xây dựng hàm Loss. train_df là phần huấn luyện của fold (pos_weight tính trên đó, không rò rỉ)
def build_loss(loss_config, train_df, classes, device):
    loss_config = validate_loss_config(loss_config)
    name = loss_config["name"]

    # BCEWithLogits theo bài báo CheXNet: tổng theo lớp, trung bình theo batch
    if name == "chexnet_bce":
        criterion = CheXNetBCELoss()
        metadata = {"name": "BCEWithLogitsLoss (CheXNet)", "reduction": "sum_classes_mean_batch"}

    # BCEWithLogits có pos_weight tính từ tập huấn luyện
    elif name == "bce":
        alpha = float(loss_config.get("alpha", config.BCE_WITH_LOGITS_ALPHA))
        clip = loss_config.get("clip", config.BCE_WITH_LOGITS_CLIP)
        weights = positive_weights(train_df, classes, alpha, clip).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight = weights)
        metadata = {"name": "BCEWithLogitsLoss", "alpha": alpha, "clip": clip,
                    "pos_weight": weights.detach().cpu().tolist()}

    # Sigmoid Focal Loss
    elif name == "focal":
        criterion = SigmoidFocalLoss(gamma = loss_config.get("gamma", config.SIGMOID_FOCAL_LOSS_GAMMA), alpha = loss_config.get("alpha", config.SIGMOID_FOCAL_LOSS_ALPHA))
        metadata = {"name": "Sigmoid Focal Loss", "gamma": criterion.gamma, "alpha": criterion.alpha}

    # Asymmetric Loss
    else:
        criterion = AsymmetricLoss(gamma_negative = loss_config.get("gamma_negative", config.ASYMMETRIC_LOSS_GAMMA_NEGATIVE), gamma_positive = loss_config.get("gamma_positive", config.ASYMMETRIC_LOSS_GAMMA_POSITIVE), clip = loss_config.get("clip", config.ASYMMETRIC_LOSS_CLIP), epsilon = config.ASYMMETRIC_LOSS_EPSILON)
        metadata = {"name": "Asymmetric Loss", "gamma_negative": criterion.gamma_negative,
                    "gamma_positive": criterion.gamma_positive, "clip": criterion.clip}

    return criterion.to(device), metadata


# ============================================================
# Trọng số lớp dương cho BCE: pos_weight = (số âm / số dương) ^ alpha, có thể giới hạn bởi clip
def positive_weights(train_df, classes, alpha, clip):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError(f"BCEWithLogitsLoss: alpha phải thuộc [0, 1], nhận {alpha}")

    # Nhãn nhị phân của tập huấn luyện
    target = train_df[list(classes)].to_numpy(dtype = np.float64)
    if target.ndim != 2 or len(target) == 0 or not np.isin(target, [0, 1]).all():
        raise ValueError("BCEWithLogitsLoss: dữ liệu huấn luyện phải có nhãn nhị phân 0 hoặc 1")

    # Đếm mẫu dương / âm từng lớp
    positive = target.sum(axis = 0)
    negative = len(target) - positive
    invalid = (positive == 0) | (negative == 0)
    if invalid.any():
        rare = [c for i, c in enumerate(classes) if invalid[i]]
        raise ValueError(f"BCEWithLogitsLoss: không tính được pos_weight cho {rare} vì thiếu mẫu dương hoặc âm")

    # Tỉ lệ mất cân bằng, điều chỉnh theo alpha, giới hạn theo clip
    weights = np.power(negative / positive, alpha)
    if clip is not None:
        clip = float(clip)
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("BCEWithLogitsLoss: clip phải là số hữu hạn và lớn hơn 0")
        weights = np.minimum(weights, clip)

    return torch.tensor(weights, dtype = torch.float32)


# ============================================================
# BCEWithLogits theo CheXNet
class CheXNetBCELoss(nn.Module):
    def forward(self, logits, targets):
        loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction = "none")
        return loss.sum(dim = 1).mean()


# ============================================================
# Sigmoid Focal Loss: BCE nhân (1 - p_t) ^ gamma, có thể cân bằng lớp bằng alpha
class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        if self.gamma < 0:
            raise ValueError(f"Sigmoid Focal Loss: gamma phải >= 0, nhận {self.gamma}")
        if self.alpha is not None and not 0 <= self.alpha <= 1:
            raise ValueError(f"Sigmoid Focal Loss: alpha phải thuộc [0, 1] hoặc None, nhận {self.alpha}")

    def forward(self, logits, targets):
        logits, targets = logits.float(), targets.float()

        # BCE từng phần tử
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction = "none")

        # Xác suất của lớp đúng p_t và trọng số tập trung (1 - p_t) ^ gamma
        probability = torch.sigmoid(logits)
        target_probability = probability * targets + (1 - probability) * (1 - targets)
        loss = bce * (1.0 - target_probability) ** self.gamma

        # Cân bằng lớp dương / âm
        if self.alpha is not None:
            loss = loss * (self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets))

        return loss.mean()


# ============================================================
# Asymmetric Loss (Ridnik et al., 2021): gamma riêng cho mẫu dương / âm và dịch xác suất mẫu âm
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_negative, gamma_positive, clip, epsilon):
        super().__init__()
        self.gamma_negative = float(gamma_negative)
        self.gamma_positive = float(gamma_positive)
        self.clip = float(clip)
        self.eps = float(epsilon)
        if self.gamma_negative < 0 or self.gamma_positive < 0:
            raise ValueError("Asymmetric Loss: gamma_negative và gamma_positive phải >= 0")
        if not 0 <= self.clip <= 1:
            raise ValueError(f"Asymmetric Loss: clip phải thuộc [0, 1], nhận {self.clip}")

    def forward(self, logits, targets):
        logits, targets = logits.float(), targets.float()

        # Xác suất dương / âm; dịch xác suất âm thêm clip (bỏ qua mẫu âm quá dễ)
        positive_probability = torch.sigmoid(logits)
        negative_probability = 1.0 - positive_probability
        if self.clip > 0:
            negative_probability = (negative_probability + self.clip).clamp(max = 1.0)

        # Log xác suất của nhãn đúng
        log_positive = F.logsigmoid(logits)
        if self.clip > 0:
            log_negative = torch.log(negative_probability.clamp(min = self.eps))
        else:
            log_negative = F.logsigmoid(-logits)
        log_probability = targets * log_positive + (1.0 - targets) * log_negative

        # Trọng số tập trung bất đối xứng (không lan truyền gradient qua trọng số)
        with torch.no_grad():
            target_probability = positive_probability * targets + negative_probability * (1.0 - targets)
            gamma = self.gamma_positive * targets + self.gamma_negative * (1.0 - targets)
            weight = torch.pow(1.0 - target_probability, gamma)

        return -(log_probability * weight).mean()
