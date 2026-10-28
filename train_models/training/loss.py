import math
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Xây dựng hàm loss
def build_loss(loss_config, train_df, classes, device):
    config = dict(loss_config)
    name = config.get("name", "bce").lower()
    
    # Hàm Loss: BCEWithLogits (Theo bài báo CheXNet)
    if name == "chexnet_bce":
        criterion, metadata = build_BCEWithLogits_CheXNet()
    
    # Hàm Loss: BCEWithLogits
    elif name == "bce":
        criterion, metadata = build_BCEWithLogits(config, train_df, classes, device)
    
    # Hàm Loss: Sigmoid Focal Loss
    elif name == "focal":
        criterion, metadata = build_sigmoid_focal_Loss(config)
    
    # Hàm Loss: Asymmetric Loss
    elif name == "asl":
       criterion, metadata = build_asymmetric_loss(config)
       
    # Hàm Loss khác --> Thông báo lỗi
    else:
        raise ValueError(f"Hàm Loss {name} hiện tại không hỗ trợ.")
    
    return criterion.to(device), metadata

# ============================================================
# Kiểm tra hàm Loss
def check_loss(loss_config, loss_factory):
    
    # Kiểm tra chỉ được truyền một trong hai phương thức khởi tạo Loss
    if (loss_config is None) == (loss_factory is None):
        raise ValueError('Phải truyền đúng một trong loss_config hoặc loss_factory.')
    
    # Kiểm tra cấu hình Loss nếu sử dụng loss_config    
    if loss_config is not None:
        if not isinstance(loss_config, dict) or not isinstance(loss_config.get('name'), str):
            raise TypeError('loss_config phải là dict chứa tên hàm Loss hợp lệ.')
        
        # Kiểm tra hàm Loss có được hỗ trợ hay không
        supported_losses = {"bce", "chexnet_bce", "focal", "asl"}
        if loss_config["name"] not in supported_losses:
            raise ValueError(f"Hàm Loss '{loss_config['name']}' không được hỗ trợ. Các hàm Loss hợp lệ gồm: {', '.join(sorted(supported_losses))}.")
        
        # Sao chép cấu hình để tránh thay đổi dữ liệu gốc
        loss_config = deepcopy(loss_config)
    
    # Kiểm tra phương thức khởi tạo Loss nếu sử dụng loss_factory
    elif not callable(loss_factory):
        raise TypeError('loss_factory phải là một hàm hoặc đối tượng có thể gọi.')
    
# ============================================================
# Hàm Loss: BCEWithLogits
def build_BCEWithLogits(config, train_df, classes, device):
    
    # Đọc giá trị
    alpha       = float(config.get("pos_weight_alpha", 1.0))
    clip        = config.get("max_pos_weight")
    
    # Kiểm tra giá trị
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError(f"BCEWithLogitsLoss: Giá trị alpha phải nằm trong khoảng [0, 1], nhưng giá trị nhận được là {alpha}.")


    # Hàm tính trọng số
    def pos_weight_from_train(train_df, classes, alpha = 1.0, clip = None):
        
        # Lấy nhãn của các lớp từ tập huấn luyện
        target = train_df[list(classes)].to_numpy(dtype = np.float64)
        
        # Kiểm tra dữ liệu nhãn nhị phân
        if target.ndim != 2 or len(target) == 0 or not np.isin(target, [0, 1]).all():
            raise ValueError("BCEWithLogitsLoss: Dữ liệu huấn luyện phải có nhãn nhị phân 0 hoặc 1.")
        
        # Đếm số mẫu dương tính và âm tính của từng lớp
        positive = target.sum(axis = 0)
        negative = len(target) - positive
        
        # Kiểm tra các lớp không có mẫu dương tính hoặc âm tính
        invalid = (positive == 0) | (negative == 0)
        
        if invalid.any():
            rare = [c for i,c in enumerate(classes) if positive[i] == 0 or negative[i] == 0]
            raise ValueError(f"BCEWithLogitsLoss: không thể tính Pos Weight cho các lớp {rare} vì không có mẫu dương tính hoặc âm tính.")
        
        # Tính tỷ lệ mất cân bằng giữa mẫu âm tính và dương tính
        imbalance_ratio = negative / positive
        
        # Điều chỉnh trọng số theo hệ số alpha
        weights = np.power(imbalance_ratio, alpha)
        
        # Giới hạn trọng số tối đa nếu có cấu hình clip
        if clip is not None:
            clip = float(clip)

            if not math.isfinite(clip) or clip <= 0:
                raise ValueError("BCEWithLogitsLoss: Giá trị max_pos_weight phải là số hữu hạn và lớn hơn 0.")

            weights = np.minimum(weights, clip)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    # Tính trọng số Pos Weight từ tập huấn luyện
    weights     = pos_weight_from_train(train_df, classes, alpha, clip)
    
    # Chuyển trọng số sang thiết bị huấn luyện
    weights     = weights.to(device)
    
    criterion   = nn.BCEWithLogitsLoss(pos_weight = weights)
    metadata    = {
        "name"              : "BCEWithLogitsLoss", 
        "pos_weight_alpha"  : alpha,
        "max_pos_weight"    : clip, 
        "pos_weight"        : weights.detach().cpu().tolist()}
    
    return criterion, metadata

# ============================================================
# Hàm Loss: BCEWithLogits (Theo bài báo CheXNet)
def build_BCEWithLogits_CheXNet():
    
    # Dựng hàm BCEWithLogits (Theo bài báo CheXNet)
    class BCEWithLogits_CheXNet(nn.Module):
        def forward(self, logits, targets):
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
def build_sigmoid_focal_Loss(config):
    # Đọc giá trị
    gamma       = float(config.get("gamma", 2.0))
    alpha       = config.get("alpha")
    
    # Kiểm tra giá trị
    if gamma < 0:
        raise ValueError(f"Sigmoid Focal Loss: gamma phải >= 0, nhưng giá trị nhận được là {gamma}.")

    if alpha is not None and not (0 <= alpha <= 1):
        raise ValueError(f"Sigmoid Focal Loss: phải nằm trong khoảng [0, 1] hoặc None, nhưng giá trị nhận được là {alpha}.")
    
    # Dựng hàm Sigmoid Focal Loss
    class SigmoidFocalLoss(nn.Module):
        def __init__(self, gamma = 2.0, alpha = None):
            super().__init__()
            self.gamma  = float(gamma)
            self.alpha = alpha

        def forward(self, logits, targets):
            logits      = logits.float()
            targets     = targets.float()
            
            # Binary Cross Entropy
            binary_cross_entropy    = F.binary_cross_entropy_with_logits(logits, targets, reduction = "none")
            
            # Xác xuất đúng
            positive_probability    = torch.sigmoid(logits)
            
            # Xác xuất lớp đúng
            target_probability      = positive_probability * targets + (1 - positive_probability) * (1 - targets)
            
            # Focusing Weight
            focusing_weight         = (1.0 - target_probability) ** self.gamma
            
            # Focal Loss
            loss                    = binary_cross_entropy * focusing_weight

            # Cân bằng Alpha
            if self.alpha is not None:
                alpha_weight        = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
                loss                = loss * alpha_weight
            
            return loss.mean()

    criterion   = SigmoidFocalLoss(gamma = gamma, alpha = alpha)
    metadata    = {
        "name": "Sigmoid Focal Loss", 
        "gamma": gamma, 
        "alpha": alpha
        }
    
    return criterion, metadata
    
# ============================================================
# Hàm Loss: Asymmetric Loss
def build_asymmetric_loss(config):
    # Đọc giá trị
    gamma_negative  = float(config.get("gamma_negative", 4.0))
    gamma_positive  = float(config.get("gamma_positive", 0.0))
    clip            = float(config.get("clip", 0.05))
    
    # Kiểm tra giá trị
    if gamma_negative < 0:
        raise ValueError(f"Asymmetric Loss: gamma_negative phải >= 0, nhưng giá trị nhận được là {gamma_negative}.")

    if gamma_positive < 0:
        raise ValueError(f"Asymmetric Loss: gamma_positive phải >= 0, nhưng giá trị nhận được là {gamma_positive}.")

    if not (0 <= clip <= 1):
        raise ValueError(f"Asymmetric Loss: clip phải thuộc [0, 1], nhưng giá trị nhận được là {clip}.")
    
    # Dựng hàm Asymmetric Loss
    class AsymmetricLoss(nn.Module):
        def __init__(self, gamma_negative = 4., gamma_positive = 0., clip = 0.05, epsilon = 1e-8):
            super().__init__()
            self.gamma_negative = float(gamma_negative)
            self.gamma_positive = float(gamma_positive)
            self.clip           = float(clip)
            self.eps            = float(epsilon)

        def forward(self, logits, targets):
            logits                          = logits.float()
            targets                         = targets.float()
            
            # Xác suất đúng và sai
            positive_probability            = torch.sigmoid(logits)
            negative_probability            = 1.0 - positive_probability
            
            # Xác suất Asymmetric Clipping
            if self.clip > 0:
                negative_probability        = (negative_probability + self.clip).clamp(max = 1.0)
                
            # Xác suất Log đúng
            log_positive_probability        = F.logsigmoid(logits)
            
            # Xác suất Log sai
            if self.clip > 0:
                log_negative_probability    = torch.log(negative_probability.clamp(min = self.eps))
            else:
                log_negative_probability    = F.logsigmoid(-logits)

            # Xác suất Log
            log_probability                 = targets * log_positive_probability + (1.0 - targets) * log_negative_probability
            
            # Asymmetric Focusing
            with torch.no_grad():
                probability_target  = positive_probability * targets + negative_probability * (1.0 - targets)
                gamma               = self.gamma_positive * targets + self.gamma_negative * (1.0 - targets)
                weight              = torch.pow(1.0 - probability_target, gamma)
            
            return -(log_probability * weight).mean()
        
    criterion   = AsymmetricLoss(gamma_negative, gamma_positive, clip)
    metadata    = {
        "name"      : "Asymmetric Loss", 
        "gamma_negative" : gamma_negative, 
        "gamma_positive" : gamma_positive, 
        "clip"      : clip
        }
    
    return criterion, metadata

