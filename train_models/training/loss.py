# ============================================================
# Hàm Loss cho phân loại đa nhãn (Multi-label Loss)
# Mục tiêu: Kiểm tra cấu hình và xây dựng hàm Loss: BCE, BCE theo CheXNet, Focal, Asymmetric (ASL).
# ============================================================

import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import config

# ============================================================
# Các khóa hợp lệ của từng hàm Loss (khóa sai tên sẽ báo lỗi thay vì bị bỏ qua)
ALLOWED_LOSS_KEYS = {
    "bce"           : {"name", "pos_weight_alpha", "max_pos_weight"},
    "chexnet_bce"   : {"name"},
    "focal"         : {"name", "gamma", "pos_weight_alpha"},
    "asl"           : {"name", "gamma_negative", "gamma_positive", "clip"},
}


# ============================================================
# Kiểm tra cấu hình Loss
# Trả về: dict(loss_config)
def validate_loss_config(loss_config):

    ## Kiểm tra loss_config phải là dict có khóa 'name' kiểu chuỗi
    if not isinstance(loss_config, dict) or not isinstance(loss_config.get("name"), str):
        raise TypeError(
            "[validate_loss_config]: loss_config phải là dict có khóa 'name' kiểu chuỗi.\n"
            f"Nhận được: loss_config = {loss_config!r}"
        )

    # Chuẩn hóa tên hàm Loss
    loss_config = deepcopy(loss_config)
    loss_config["name"] = loss_config["name"].lower()
    name = loss_config["name"]

    # Kiểm tra hàm Loss có được hỗ trợ hay không
    if name not in ALLOWED_LOSS_KEYS:
        raise ValueError(
            f"[validate_loss_config]: Hàm Loss phải là một trong {sorted(ALLOWED_LOSS_KEYS)}.\n"
            f"Nhận được: name = {name!r}"
        )

    ## Kiểm tra không có khóa lạ
    unknown_keys = set(loss_config) - ALLOWED_LOSS_KEYS[name]

    if unknown_keys:
        raise ValueError(
            f"[validate_loss_config]: Hàm Loss '{name}' chỉ nhận các khóa {sorted(ALLOWED_LOSS_KEYS[name])}.\n"
            f"Khóa lạ: {sorted(unknown_keys)}"
        )

    return dict(loss_config)


# ============================================================
# Kiểm tra logits và targets của một batch trước khi tính Loss
# - Hai tensor phải cùng kích thước [batch_size, num_classes]: Tránh broadcast âm thầm làm Loss sai.
def _check_loss_inputs(name, logits, targets):

    ## logits và targets cùng kích thước
    if logits.shape != targets.shape:
        raise ValueError(
            f"[{name}]: logits và targets phải cùng kích thước [batch_size, num_classes].\n"
            f"Nhận được: logits = {tuple(logits.shape)}, targets = {tuple(targets.shape)}"
        )

# ============================================================
# Xây dựng hàm Loss
# Trả về: (criterion, metadata)
def build_loss(loss_config, train_df, classes, device):
    
    # Kiểm tra dữ liệu đầu vào
    loss_config = validate_loss_config(loss_config)
    
    # Lấy tên hàm Loss
    name = loss_config["name"]
    
    
    # Khai báo hàm loss (Theo name)
    ## Hàm Loss: BCEWithLogits (Theo bài báo CheXNet)
    if name == "chexnet_bce":
        criterion, metadata = build_BCEWithLogits_CheXNet()
    
    ## Hàm Loss: BCEWithLogits
    elif name == "bce":
        criterion, metadata = build_BCEWithLogits(loss_config, train_df, classes, device)

    ## Hàm Loss: Sigmoid Focal Loss
    elif name == "focal":
        criterion, metadata = build_sigmoid_focal_Loss(loss_config)

    ## Hàm Loss: Asymmetric Loss
    elif name == "asl":
        criterion, metadata = build_asymmetric_loss(loss_config)
    
    return criterion.to(device), metadata

# ============================================================
# Hàm Loss: BCEWithLogits
# Trả về: (criterion, metadata)
def build_BCEWithLogits(loss_config, train_df, classes, device):

    # Đọc giá trị (thiếu khóa --> lấy mặc định trong config.py)
    pos_weight_alpha    = float(loss_config.get("pos_weight_alpha", config.BCE_WITH_LOGITS_ALPHA))
    max_pos_weight      = loss_config.get("max_pos_weight", config.BCE_WITH_LOGITS_CLIP)
    
    # Kiểm tra giá trị
    ## pos_weight_alpha phải nằm trong khoảng [0, 1]
    if not math.isfinite(pos_weight_alpha) or not 0 <= pos_weight_alpha <= 1:
        raise ValueError(
            "[build_BCEWithLogits]: pos_weight_alpha phải nằm trong khoảng [0, 1].\n"
            f"Nhận được: pos_weight_alpha = {pos_weight_alpha}"
        )

    # Hàm tính trọng số
    def pos_weight_from_train(train_df, classes, pos_weight_alpha = 1.0, max_pos_weight = None):
        
        # Lấy nhãn của các lớp từ tập huấn luyện
        target = train_df[list(classes)].to_numpy(dtype = np.float64)
        
        # Kiểm tra tập huấn luyện phải có dữ liệu và nhãn nhị phân (0 hoặc 1)

        if target.ndim != 2 or len(target) == 0 or not np.isin(target, [0, 1]).all():
            raise ValueError(
                "[build_BCEWithLogits]: Dữ liệu huấn luyện phải có ít nhất một dòng và nhãn nhị phân (0 hoặc 1).\n"
                f"Nhận được: Kích thước nhãn = {target.shape}"
            )

        # Đếm số mẫu dương tính và âm tính của từng lớp
        positive = target.sum(axis = 0)
        negative = len(target) - positive
        
        # Kiểm tra các lớp không có mẫu dương tính hoặc âm tính
        invalid = (positive == 0) | (negative == 0)
        
        if invalid.any():
            rare_classes = [c for i,c in enumerate(classes) if positive[i] == 0 or negative[i] == 0]
            raise ValueError(
                "[build_BCEWithLogits]: Không tính được pos_weight vì lớp thiếu mẫu dương hoặc mẫu âm.\n"
                f"Lớp không hợp lệ: {rare_classes}"
            )
        
        # Tính tỷ lệ mất cân bằng giữa mẫu âm tính và dương tính
        imbalance_ratio = negative / positive
        
        # Điều chỉnh trọng số theo hệ số pos_weight_alpha
        weights = np.power(imbalance_ratio, pos_weight_alpha)
        
        # Giới hạn trọng số tối đa nếu có cấu hình max_pos_weight
        if max_pos_weight is not None:
            max_pos_weight = float(max_pos_weight)

            if not math.isfinite(max_pos_weight) or max_pos_weight <= 0:
                raise ValueError(
                    "[build_BCEWithLogits]: max_pos_weight phải là số dương hữu hạn hoặc None.\n"
                    f"Nhận được: max_pos_weight = {max_pos_weight}"
                )

            # Chặn trên trọng số bằng max_pos_weight
            weights = np.minimum(weights, max_pos_weight)
        
        return torch.tensor(weights, dtype = torch.float32)
    
    # Tính trọng số Pos Weight từ tập huấn luyện
    weights     = pos_weight_from_train(train_df, classes, pos_weight_alpha, max_pos_weight)
    
    # Chuyển trọng số sang thiết bị huấn luyện
    weights     = weights.to(device)
    
    criterion   = nn.BCEWithLogitsLoss(pos_weight = weights)
    metadata    = {
        "name"              : "BCEWithLogitsLoss", 
        "pos_weight_alpha"  : pos_weight_alpha,
        "max_pos_weight"    : max_pos_weight, 
        "pos_weight"        : weights.detach().cpu().tolist()
    }
    
    return criterion, metadata

# ============================================================
# Hàm Loss: BCEWithLogits (Theo bài báo CheXNet)
# Trả về: (criterion, metadata)
def build_BCEWithLogits_CheXNet():
    
    # Dựng hàm BCEWithLogits (Theo bài báo CheXNet)
    class BCEWithLogits_CheXNet(nn.Module):
        def forward(self, logits, targets):

            # Kiểm tra dữ liệu đầu vào
            _check_loss_inputs("BCEWithLogits_CheXNet", logits, targets)

            # BCE từng phần tử (FP32) --> Tổng theo lớp --> Trung bình theo batch
            x = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
            
            return x.sum(dim = 1).mean()
    
    criterion   = BCEWithLogits_CheXNet()
    metadata    = {
        "name": "BCEWithLogitsLoss (Theo CheXNet)", 
        "reduction": "sum_classes_mean_batch"
    }
    
    return criterion, metadata

# ============================================================
# Hàm Loss: Sigmoid Focal Loss
# Trả về: (criterion, metadata)
def build_sigmoid_focal_Loss(loss_config):

    # Đọc giá trị (thiếu khóa --> lấy mặc định trong config.py)
    gamma                   = float(loss_config.get("gamma", config.SIGMOID_FOCAL_LOSS_GAMMA))
    pos_weight_alpha        = loss_config.get("pos_weight_alpha", config.SIGMOID_FOCAL_LOSS_ALPHA)
    
    # Kiểm tra giá trị
    ## Giá trị gamma không âm
    if gamma < 0:
        raise ValueError(
            "[build_sigmoid_focal_Loss]: gamma phải lớn hơn hoặc bằng 0.\n"
            f"Nhận được: gamma = {gamma}"
        )
        
    ## Giá trị pos_weight_alpha nằm trong khoảng [0, 1] hoặc None
    if pos_weight_alpha is not None and not (0 <= pos_weight_alpha <= 1):
        raise ValueError(
            "[build_sigmoid_focal_Loss]: pos_weight_alpha phải nằm trong khoảng [0, 1] hoặc None.\n"
            f"Nhận được: pos_weight_alpha = {pos_weight_alpha}"
        )
    
    # Dựng hàm Sigmoid Focal Loss
    class SigmoidFocalLoss(nn.Module):
        def __init__(self, gamma = 2.0, pos_weight_alpha = None):
            super().__init__()
            self.gamma  = float(gamma)
            self.pos_weight_alpha = pos_weight_alpha

        def forward(self, logits, targets):

            # Kiểm tra dữ liệu đầu vào
            _check_loss_inputs("SigmoidFocalLoss", logits, targets)

            # Tính ở FP32
            logits      = logits.float()
            targets     = targets.float()
            
            # Binary Cross Entropy
            binary_cross_entropy    = F.binary_cross_entropy_with_logits(logits, targets, reduction = "none")
            
            # Xác suất dương tính
            positive_probability    = torch.sigmoid(logits)

            # Xác suất của nhãn đúng
            target_probability      = positive_probability * targets + (1 - positive_probability) * (1 - targets)
            
            # Focusing Weight
            focusing_weight         = (1.0 - target_probability) ** self.gamma
            
            # Focal Loss
            loss                    = binary_cross_entropy * focusing_weight

            # Cân bằng pos_weight_alpha
            if self.pos_weight_alpha is not None:
                alpha_weight        = self.pos_weight_alpha * targets + (1.0 - self.pos_weight_alpha) * (1.0 - targets)
                loss                = loss * alpha_weight
            
            return loss.mean()

    criterion   = SigmoidFocalLoss(gamma = gamma, pos_weight_alpha = pos_weight_alpha)
    metadata    = {
        "name": "Sigmoid Focal Loss", 
        "gamma": gamma, 
        "pos_weight_alpha": pos_weight_alpha
    }
    
    return criterion, metadata
    
# ============================================================
# Hàm Loss: Asymmetric Loss
# Trả về: (criterion, metadata)
def build_asymmetric_loss(loss_config):

    # Đọc giá trị (thiếu khóa --> lấy mặc định trong config.py)
    gamma_negative  = float(loss_config.get("gamma_negative", config.ASYMMETRIC_LOSS_GAMMA_NEGATIVE))
    gamma_positive  = float(loss_config.get("gamma_positive", config.ASYMMETRIC_LOSS_GAMMA_POSITIVE))
    clip            = float(loss_config.get("clip", config.ASYMMETRIC_LOSS_CLIP))

    # Kiểm tra giá trị
    ## Giá trị gamma_negative không âm
    if gamma_negative < 0:
        raise ValueError(
            "[build_asymmetric_loss]: gamma_negative phải lớn hơn hoặc bằng 0.\n"
            f"Nhận được: gamma_negative = {gamma_negative}"
        )
    
    ## Giá trị gamma_positive không âm
    if gamma_positive < 0:
        raise ValueError(
            "[build_asymmetric_loss]: gamma_positive phải lớn hơn hoặc bằng 0.\n"
            f"Nhận được: gamma_positive = {gamma_positive}"
        )
    
    ## Giá trị clip (dịch xác suất mẫu âm tính) nằm trong khoảng [0, 1]
    if not (0 <= clip <= 1):
        raise ValueError(
            "[build_asymmetric_loss]: clip phải nằm trong khoảng [0, 1].\n"
            f"Nhận được: clip = {clip}"
        )

    # Dựng hàm Asymmetric Loss
    class AsymmetricLoss(nn.Module):
        def __init__(self, gamma_negative = 4., gamma_positive = 0., clip = 0.05, epsilon = 1e-8):
            super().__init__()
            self.gamma_negative = float(gamma_negative)
            self.gamma_positive = float(gamma_positive)
            self.clip           = float(clip)
            self.eps            = float(epsilon)

        def forward(self, logits, targets):

            # Kiểm tra dữ liệu đầu vào
            _check_loss_inputs("AsymmetricLoss", logits, targets)

            # Tính ở FP32
            logits                          = logits.float()
            targets                         = targets.float()

            # Xác suất dương tính và âm tính
            positive_probability            = torch.sigmoid(logits)
            negative_probability            = 1.0 - positive_probability

            # Dịch xác suất âm tính thêm clip (Asymmetric Clipping): Mẫu âm tính quá dễ có loss = 0
            if self.clip > 0:
                negative_probability        = (negative_probability + self.clip).clamp(max = 1.0)

            # Log xác suất dương tính
            log_positive_probability        = F.logsigmoid(logits)

            # Log xác suất âm tính
            if self.clip > 0:
                log_negative_probability    = torch.log(negative_probability.clamp(min = self.eps))
            else:
                log_negative_probability    = F.logsigmoid(-logits)

            # Log xác suất của nhãn đúng
            log_probability                = targets * log_positive_probability + (1.0 - targets) * log_negative_probability
            
            # Asymmetric Focusing
            with torch.no_grad():
                probability_target  = positive_probability * targets + negative_probability * (1.0 - targets)
                gamma               = self.gamma_positive * targets + self.gamma_negative * (1.0 - targets)
                weight              = torch.pow(1.0 - probability_target, gamma)
            
            return -(log_probability * weight).mean()
        
    criterion   = AsymmetricLoss(gamma_negative, gamma_positive, clip, config.ASYMMETRIC_LOSS_EPSILON)
    metadata    = {
        "name"              : "Asymmetric Loss",
        "gamma_negative"    : gamma_negative,
        "gamma_positive"    : gamma_positive,
        "clip"              : clip
    }

    return criterion, metadata

