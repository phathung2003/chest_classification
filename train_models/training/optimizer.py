import math
from copy import deepcopy

import torch

import config

SELECTION_METRICS = ("macro_auroc", "validation_loss")
STEP_SCHEDULERS = ("warmup_cosine",)                                       # Scheduler cập nhật sau mỗi bước tối ưu (các scheduler khác: mỗi epoch)


# ============================================================
# Cấu hình huấn luyện đầy đủ: điền mọi giá trị mặc định từ config.py vào MỘT dict duy nhất.
# Dict này vừa dùng để huấn luyện, vừa được in ra log và đưa vào chữ ký, nên ba nơi luôn khớp nhau.
def build_training_config(optimizer_config = None, scheduler_config = None, selection_metric = None):
    selection_metric = config.SELECTION_METRIC if selection_metric is None else selection_metric
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric phải thuộc {SELECTION_METRICS}")

    training_config = {
        "batch_size"            : int(config.BATCH_SIZE),
        "epochs"                : int(config.EPOCHS),
        "early_stop_patience"   : int(config.PATIENCE),
        "min_delta"             : float(config.MIN_DELTA),
        "grad_accum_steps"      : int(config.GRAD_ACCUM_STEPS),
        "grad_clip_norm"        : None if config.GRAD_CLIP_NORM is None else float(config.GRAD_CLIP_NORM),
        "ema_decay"             : None if config.EMA_DECAY is None else float(config.EMA_DECAY),
        "selection_metric"      : selection_metric,
        "optimizer"             : _complete_optimizer_config(optimizer_config),
        "scheduler"             : _complete_scheduler_config(scheduler_config, selection_metric),
    }

    # Kiểm tra giá trị
    if training_config["batch_size"] < 1 or training_config["epochs"] < 1:
        raise ValueError("BATCH_SIZE và EPOCHS phải là số nguyên dương")
    if training_config["early_stop_patience"] < 1 or training_config["grad_accum_steps"] < 1:
        raise ValueError("PATIENCE và GRAD_ACCUM_STEPS phải là số nguyên dương")
    if training_config["min_delta"] < 0:
        raise ValueError("MIN_DELTA phải >= 0")
    clip = training_config["grad_clip_norm"]
    if clip is not None and (not math.isfinite(clip) or clip <= 0):
        raise ValueError("GRAD_CLIP_NORM phải là số dương hoặc None")
    ema_decay = training_config["ema_decay"]
    if ema_decay is not None and not 0 < ema_decay < 1:
        raise ValueError("EMA_DECAY phải nằm trong khoảng (0, 1) hoặc None")

    return training_config


# ============================================================
# Optimizer: điền mặc định và kiểm tra khóa
def _complete_optimizer_config(optimizer_config):
    cfg = deepcopy(dict(optimizer_config or {}))
    name = str(cfg.pop("name", config.OPTIMIZER_NAME)).lower()

    # Weight decay mặc định chỉ áp dụng cho AdamW (Adam/SGD mặc định không decay)
    result = {
        "name"          : name,
        "lr"            : float(cfg.pop("lr", config.LEARNING_RATE)),
        "weight_decay"  : float(cfg.pop("weight_decay", config.WEIGHT_DECAY if name == "adamw" else 0.0)),
        "new_module_lr_multiplier": float(cfg.pop("new_module_lr_multiplier", 1.0)),
        "no_weight_decay_1d": bool(cfg.pop("no_weight_decay_1d", False)),
    }
    if name in ("adam", "adamw"):
        result["betas"] = [float(b) for b in cfg.pop("betas", config.ADAM_BETAS)]
        result["eps"] = float(cfg.pop("eps", config.ADAM_EPSILON))
        result["fused"] = bool(cfg.pop("fused", config.ADAM_FUSED))
        if len(result["betas"]) != 2 or not all(0 <= b < 1 for b in result["betas"]) or result["eps"] <= 0:
            raise ValueError("Adam: betas hoặc eps không hợp lệ")
    elif name == "sgd":
        result["momentum"] = float(cfg.pop("momentum", config.SGD_MOMENTUM))
        result["nesterov"] = bool(cfg.pop("nesterov", config.SGD_NESTEROV))
    else:
        raise ValueError(f"Optimizer '{name}' không được hỗ trợ (adam | adamw | sgd)")

    if cfg:
        raise ValueError(f"Optimizer '{name}' không nhận các khóa {sorted(cfg)}")
    if not math.isfinite(result["lr"]) or result["lr"] <= 0:
        raise ValueError("Learning rate phải là số dương")
    if not math.isfinite(result["weight_decay"]) or result["weight_decay"] < 0:
        raise ValueError("Weight decay phải >= 0")
    if not math.isfinite(result["new_module_lr_multiplier"]) or result["new_module_lr_multiplier"] <= 0:
        raise ValueError("new_module_lr_multiplier phải là số dương")
    return result


# ============================================================
# Scheduler: điền mặc định, kiểm tra khóa và bắt buộc theo dõi cùng chỉ số với tiêu chí chọn mô hình
def _complete_scheduler_config(scheduler_config, selection_metric):
    cfg = deepcopy(dict(scheduler_config or {}))
    name = str(cfg.pop("name", config.SCHEDULER_NAME)).lower()

    if name == "none":
        if cfg:
            raise ValueError(f"Scheduler 'none' không nhận các khóa {sorted(cfg)}")
        return {"name": "none"}
    if name == "warmup_cosine":
        result = {
            "name"          : name,
            "warmup_epochs" : float(cfg.pop("warmup_epochs", config.SCHEDULER_WARMUP_EPOCHS)),
            "min_lr_ratio"  : float(cfg.pop("min_lr_ratio", 0.0)),
        }
        if cfg:
            raise ValueError(f"Scheduler 'warmup_cosine' không nhận các khóa {sorted(cfg)}")
        if not math.isfinite(result["warmup_epochs"]) or result["warmup_epochs"] < 0 or not 0 <= result["min_lr_ratio"] < 1:
            raise ValueError("warmup_cosine: warmup_epochs phải >= 0, min_lr_ratio thuộc [0, 1)")
        return result
    if name != "reduce_on_plateau":
        raise ValueError(f"Scheduler '{name}' không được hỗ trợ (reduce_on_plateau | warmup_cosine | none)")

    expected_mode = "min" if selection_metric == "validation_loss" else "max"
    result = {
        "name"      : name,
        "mode"      : str(cfg.pop("mode", expected_mode)),
        "factor"    : float(cfg.pop("factor", config.SCHEDULER_FACTOR)),
        "patience"  : int(cfg.pop("patience", config.SCHEDULER_PATIENCE)),
        "threshold" : float(cfg.pop("threshold", config.SCHEDULER_THRESHOLD)),
    }
    if cfg:
        raise ValueError(f"ReduceLROnPlateau không nhận các khóa {sorted(cfg)}")
    if result["mode"] != expected_mode:
        raise ValueError(f"Scheduler mode '{result['mode']}' không khớp selection_metric "
                         f"'{selection_metric}' (cần '{expected_mode}')")
    if not 0 < result["factor"] < 1 or result["patience"] < 0 or result["threshold"] < 0:
        raise ValueError("ReduceLROnPlateau: factor, patience hoặc threshold không hợp lệ")
    return result


# ============================================================
# Nhóm tham số theo learning rate: backbone tiền huấn luyện (các khối trong model.PRETRAINED_MODULES) dùng lr;
# lớp mới khởi tạo ngẫu nhiên (neck, khối trộn token, đầu phân loại) và tham số học được của Loss dùng lr x multiplier.
# multiplier = 1.0 giữ một nhóm duy nhất như trước (cùng thứ tự tham số, kết quả không đổi).
# no_weight_decay_1d = True: tách thêm mỗi nhóm thành phần có / không weight decay; không decay tham số 1 chiều (bias, norm,
# layer scale) và các tên trong model.no_weight_decay() (quy ước timm, ví dụ mã hóa vị trí, A_log / D / dt_bias của SSM)
def parameter_groups(model, criterion, lr, multiplier, no_weight_decay_1d = False):
    loss_parameters = [p for p in criterion.parameters() if p.requires_grad]
    if multiplier == 1.0:
        groups = [{"params": list(model.parameters()) + loss_parameters}]
    else:
        names = getattr(model, "PRETRAINED_MODULES", None)
        if not names:
            raise ValueError(f"{type(model).__name__} chưa khai báo PRETRAINED_MODULES (tên các khối dùng trọng số tiền huấn luyện) "
                             "nên không tách được learning rate backbone / lớp mới")
        pretrained = {id(p) for name in names for p in getattr(model, name).parameters()}
        backbone = [p for p in model.parameters() if id(p) in pretrained]
        new = [p for p in model.parameters() if id(p) not in pretrained] + loss_parameters
        if not backbone or not new:
            raise ValueError(f"{type(model).__name__}: nhóm backbone ({len(backbone)} tham số) hoặc lớp mới ({len(new)} tham số) rỗng")
        groups = [{"params": backbone, "lr": lr}, {"params": new, "lr": lr * multiplier}]
    if not no_weight_decay_1d:
        return groups

    named = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    exempt = {id(p) for name, p in model.named_parameters() if p.ndim <= 1 or name in named} | {id(p) for p in loss_parameters if p.ndim <= 1}
    split = []
    for group in groups:
        decay = [p for p in group["params"] if id(p) not in exempt]
        no_decay = [p for p in group["params"] if id(p) in exempt]
        split += ([{**group, "params": decay}] if decay else []) + ([{**group, "params": no_decay, "weight_decay": 0.0}] if no_decay else [])
    return split


# ============================================================
# Khởi tạo optimizer từ cấu hình đã hoàn chỉnh
def make_optimizer(model, criterion, optimizer_config):
    cfg = dict(optimizer_config)
    name = cfg.pop("name")
    groups = parameter_groups(model, criterion, cfg["lr"], cfg.pop("new_module_lr_multiplier"), cfg.pop("no_weight_decay_1d"))
    if name == "sgd":
        return torch.optim.SGD(groups, **cfg)
    cfg["betas"] = tuple(cfg["betas"])
    return (torch.optim.Adam if name == "adam" else torch.optim.AdamW)(groups, **cfg)


# ============================================================
# Khởi tạo scheduler từ cấu hình đã hoàn chỉnh (None nếu không dùng).
# warmup_cosine cần số bước tối ưu mỗi epoch: hệ số LR tăng tuyến tính tới 1 sau warmup_epochs, rồi giảm theo cosine
# về min_lr_ratio ở cuối epoch thứ `epochs`
def make_scheduler(optimizer, scheduler_config, steps_per_epoch = None, epochs = None):
    cfg = dict(scheduler_config)
    name = cfg.pop("name")
    if name == "none":
        return None
    if name == "warmup_cosine":
        warmup = round(cfg["warmup_epochs"] * int(steps_per_epoch))
        total = max(int(epochs) * int(steps_per_epoch), warmup + 1)
        floor = cfg["min_lr_ratio"]

        def factor(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = min((step - warmup) / max(total - warmup, 1), 1.0)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg)
