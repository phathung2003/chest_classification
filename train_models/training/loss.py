import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha

    def forward(self, logits, targets):
        logits, targets = logits.float(), targets.float()
        raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)
        loss = raw * ((1 - pt) ** self.gamma)
        if self.alpha is not None:
            loss = loss * (self.alpha * targets + (1 - self.alpha) * (1 - targets))
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """Ridnik et al. ICCV 2021 ASL with detached asymmetrical focusing weights."""
    def __init__(self, gamma_neg=4., gamma_pos=0., clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits, targets):
        x, y = logits.float(), targets.float()
        p_pos = torch.sigmoid(x)
        p_neg = 1. - p_pos
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.)
        log_prob = y * torch.log(p_pos.clamp(min=self.eps)) + (1.-y)*torch.log(p_neg.clamp(min=self.eps))
        with torch.no_grad():
            pt = p_pos * y + p_neg * (1. - y)
            gamma = self.gamma_pos * y + self.gamma_neg * (1. - y)
            weight = torch.pow(1. - pt, gamma)
        return -(log_prob * weight).mean()


def pos_weight_from_train(train_df, classes, alpha=1., clip=None):
    """Counts are computed ONLY within the train partition for each fold.
    0=none, 0.5=sqrt, 1=neg/pos. `clip` caps upper bound, not a lower bound.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("pos_weight exponent alpha must be [0,1]")
    target = train_df[list(classes)].to_numpy(dtype=np.float64)
    if target.ndim != 2 or len(target) == 0 or not np.isin(target, [0, 1]).all():
        raise ValueError("Training labels must be binary 0/1; map missing/uncertain first")
    positives = target.sum(axis=0)
    negatives = len(target) - positives
    if (positives == 0).any() or (negatives == 0).any():
        rare = [c for i,c in enumerate(classes) if positives[i] == 0 or negatives[i] == 0]
        raise ValueError(f"Cannot derive pos_weight for absent/constant labels: {rare}")
    weights = np.power(negatives / positives, alpha)
    if clip is not None:
        if not math.isfinite(float(clip)) or clip <= 0:
            raise ValueError("clip must be finite > 0")
        weights = np.minimum(weights, float(clip))
    return torch.tensor(weights, dtype=torch.float32)

# ============================================================
# Xây dựng hàm loss
def build_loss(loss_config, train_df, classes, device):
    """Return (criterion, serializable metadata) for a NEW training run."""
    cfg = dict(loss_config)
    name = cfg.get("name", "bce").lower()
    if name == "chexnet_bce":
        # Unweighted multi-label BCE summed over classes, averaged over samples.
        class CheXNetBCE(nn.Module):
            def forward(self, logits, targets):
                x = F.binary_cross_entropy_with_logits(
                    logits.float(), targets.float(), reduction="none"
                )
                return x.sum(dim=1).mean()
        criterion = CheXNetBCE()
        metadata = {"name": "chexnet_bce", "reduction": "sum_classes_mean_batch"}
    elif name == "bce":
        alpha = float(cfg.get("pos_weight_alpha", 1.0))
        clip = cfg.get("max_pos_weight")
        weights = pos_weight_from_train(train_df, classes, alpha, clip).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
        metadata = {"name": "bce", "pos_weight_alpha": alpha,
                    "max_pos_weight": clip, "pos_weight": weights.detach().cpu().tolist()}
    elif name == "focal":
        gamma = float(cfg.get("gamma", 2.0))
        pos_alpha = cfg.get("alpha")
        if gamma < 0 or (pos_alpha is not None and not (0 <= pos_alpha <= 1)):
            raise ValueError("Invalid focal loss gamma/alpha")
        criterion = SigmoidFocalLoss(gamma=gamma, alpha=pos_alpha)
        metadata = {"name": "focal", "gamma": gamma, "alpha": pos_alpha}
    elif name == "asl":
        gamma_neg = float(cfg.get("gamma_neg", 4.0))
        gamma_pos = float(cfg.get("gamma_pos", 0.0))
        clip = float(cfg.get("clip", 0.05))
        if min(gamma_neg, gamma_pos, clip) < 0 or clip > 1:
            raise ValueError("ASL parameters must be nonnegative; clip <= 1")
        criterion = AsymmetricLoss(gamma_neg, gamma_pos, clip)
        metadata = {"name": "asl", "gamma_neg": gamma_neg, "gamma_pos": gamma_pos, "clip": clip}
    else:
        raise ValueError(f"Unknown loss name: {name}")
    return criterion.to(device), metadata

# ============================================================
# Kiểm tra hàm Loss
def check_loss(loss_config, loss_factory):
    if (loss_config is None) == (loss_factory is None):
        raise ValueError('Phải truyền đúng một trong loss_config hoặc loss_factory.')
    
    if loss_config is not None:
        if not isinstance(loss_config, dict) or not isinstance(loss_config.get('name'), str):
            raise TypeError('loss_config phải là dict chứa name: bce, chexnet_bce, focal hoặc asl.')
        
        loss_config = deepcopy(loss_config)
        
    elif not callable(loss_factory):
        raise TypeError('loss_factory phải là callable(weights) -> nn.Module.')