import math
from copy import deepcopy

import torch

import config

SELECTION_METRICS = ("macro_auroc", "validation_loss")


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
    }
    if name in ("adam", "adamw"):
        result["betas"] = [float(b) for b in cfg.pop("betas", config.ADAM_BETAS)]
        result["eps"] = float(cfg.pop("eps", config.ADAM_EPSILON))
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
    if name != "reduce_on_plateau":
        raise ValueError(f"Scheduler '{name}' không được hỗ trợ (reduce_on_plateau | none)")

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
# Khởi tạo optimizer từ cấu hình đã hoàn chỉnh
def make_optimizer(parameters, optimizer_config):
    cfg = dict(optimizer_config)
    name = cfg.pop("name")
    if name == "sgd":
        return torch.optim.SGD(parameters, **cfg)
    cfg["betas"] = tuple(cfg["betas"])
    return (torch.optim.Adam if name == "adam" else torch.optim.AdamW)(parameters, **cfg)


# ============================================================
# Khởi tạo scheduler từ cấu hình đã hoàn chỉnh (None nếu không dùng)
def make_scheduler(optimizer, scheduler_config):
    cfg = dict(scheduler_config)
    if cfg.pop("name") == "none":
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg)
