# ============================================================
# Cấu hình huấn luyện, optimizer và scheduler (Optimization Setting)
# Mục tiêu: Dựng MỘT dict cấu hình huấn luyện đầy đủ (điền mặc định từ config.py, kiểm tra từng giá trị), rồi khởi tạo optimizer / scheduler từ chính dict đó.
# ** Dict này vừa dùng để huấn luyện, vừa được in ra log và đưa vào chữ ký, nên ba nơi luôn khớp nhau.
# ============================================================

import math
from copy import deepcopy

import torch

import config

# Tiêu chí chọn mô hình được hỗ trợ
SELECTION_METRICS = ("macro_auroc", "validation_loss")

# ============================================================
# Dựng cấu hình huấn luyện
# Trả về: training_config (batch_size, epochs, early stopping, gradient, EMA, tiêu chí chọn mô hình, optimizer, scheduler)
def build_training_config(optimizer_config = None, scheduler_config = None, selection_metric = None):

    # Lấy dữ liệu
    selection_metric = config.SELECTION_METRIC if selection_metric is None else selection_metric

    # Kiểm tra tiêu chí chọn mô hình có được hỗ trợ
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"[build_training_config]: selection_metric phải là một trong {SELECTION_METRICS}.\n"
            f"Nhận được: selection_metric = {selection_metric!r}"
        )

    # Dựng cấu hình
    training_config = {
        # Huấn luyện
        "batch_size"            : int(config.BATCH_SIZE),
        "epochs"                : int(config.EPOCHS),

        # Early stopping
        "early_stop_patience"   : int(config.PATIENCE),
        "min_delta"             : float(config.MIN_DELTA),

        # Gradient
        "grad_accum_steps"      : int(config.GRAD_ACCUM_STEPS),
        "grad_clip_norm"        : None if config.GRAD_CLIP_NORM is None else float(config.GRAD_CLIP_NORM),

        # EMA và tiêu chí chọn mô hình
        "ema_decay"             : None if config.EMA_DECAY is None else float(config.EMA_DECAY),
        "selection_metric"      : selection_metric,

        # Optimizer và scheduler (đã điền mặc định và kiểm tra)
        "optimizer"             : complete_optimizer_config(optimizer_config),
        "scheduler"             : _complete_scheduler_config(scheduler_config, selection_metric),
    }

    # Kiểm tra giá trị
    ## Giá trị BATCH_SIZE và EPOCHS là số nguyên dương
    if training_config["batch_size"] < 1 or training_config["epochs"] < 1:
        raise ValueError(
            "[build_training_config]: BATCH_SIZE và EPOCHS phải là số nguyên dương.\n"
            f"Nhận được: BATCH_SIZE = {training_config['batch_size']}, EPOCHS = {training_config['epochs']}"
        )

    ## Giá trị PATIENCE và GRAD_ACCUM_STEPS là số nguyên dương
    if training_config["early_stop_patience"] < 1 or training_config["grad_accum_steps"] < 1:
        raise ValueError(
            "[build_training_config]: PATIENCE và GRAD_ACCUM_STEPS phải là số nguyên dương.\n"
            f"Nhận được: PATIENCE = {training_config['early_stop_patience']}, GRAD_ACCUM_STEPS = {training_config['grad_accum_steps']}"
        )

    ## Giá trị MIN_DELTA không âm
    if training_config["min_delta"] < 0:
        raise ValueError(
            "[build_training_config]: MIN_DELTA phải lớn hơn hoặc bằng 0.\n"
            f"Nhận được: MIN_DELTA = {training_config['min_delta']}"
        )

    ## Giá trị GRAD_CLIP_NORM là số dương hữu hạn hoặc None
    grad_clip_norm = training_config["grad_clip_norm"]

    if grad_clip_norm is not None and (not math.isfinite(grad_clip_norm) or grad_clip_norm <= 0):
        raise ValueError(
            "[build_training_config]: GRAD_CLIP_NORM phải là số dương hữu hạn hoặc None.\n"
            f"Nhận được: GRAD_CLIP_NORM = {grad_clip_norm}"
        )

    ## Giá trị EMA_DECAY nằm trong khoảng (0, 1) hoặc None
    ema_decay = training_config["ema_decay"]

    if ema_decay is not None and not 0 < ema_decay < 1:
        raise ValueError(
            "[build_training_config]: EMA_DECAY phải nằm trong khoảng (0, 1) hoặc None.\n"
            f"Nhận được: EMA_DECAY = {ema_decay}"
        )

    return training_config

# ============================================================
# Hoàn thiện cấu hình optimizer
# Trả về: dict cấu hình optimizer đầy đủ
def complete_optimizer_config(optimizer_config):

    # Lấy dữ liệu: Bản sao để pop từng khóa, khóa còn sót lại là khóa lạ
    remaining_config = deepcopy(dict(optimizer_config or {}))
    name = str(remaining_config.pop("name", config.OPTIMIZER_NAME)).lower()

    # Tham số chung
    result = {
        "name"          : name,
        "learning_rate" : float(remaining_config.pop("learning_rate", config.LEARNING_RATE)),
        "weight_decay"  : float(remaining_config.pop("weight_decay", config.WEIGHT_DECAY if name == "adamw" else 0.0)),
    }

    # Tham số riêng của Adam / AdamW (Weight decay mặc định chỉ áp dụng cho AdamW. Adam / SGD mặc định không decay)
    if name in ("adam", "adamw"):
        result["betas"] = [float(beta) for beta in remaining_config.pop("betas", config.ADAM_BETAS)]
        result["eps"] = float(remaining_config.pop("eps", config.ADAM_EPSILON))
        result["fused"] = bool(remaining_config.pop("fused", config.ADAM_FUSED))

        ## Giá trị betas gồm hai hệ số trong khoảng [0, 1), eps dương
        if len(result["betas"]) != 2 or not all(0 <= beta < 1 for beta in result["betas"]) or result["eps"] <= 0:
            raise ValueError(
                "[complete_optimizer_config]: Adam cần hai hệ số betas trong khoảng [0, 1) và eps > 0.\n"
                f"Nhận được: betas = {result['betas']}, eps = {result['eps']}"
            )

    # Tham số riêng của SGD
    elif name == "sgd":
        result["momentum"] = float(remaining_config.pop("momentum", config.SGD_MOMENTUM))
        result["nesterov"] = bool(remaining_config.pop("nesterov", config.SGD_NESTEROV))

    # Optimizer không được hỗ trợ
    else:
        raise ValueError(
            "[complete_optimizer_config]: Optimizer phải là một trong adam | adamw | sgd.\n"
            f"Nhận được: name = {name!r}"
        )

    # Kiểm tra giá trị
    ## Không có khóa lạ (gõ sai tên khóa)
    if remaining_config:
        raise ValueError(
            f"[complete_optimizer_config]: Optimizer '{name}' không nhận các khóa này.\n"
            f"Khóa lạ: {sorted(remaining_config)}"
        )

    ## Giá trị learning rate là số dương hữu hạn
    if not math.isfinite(result["learning_rate"]) or result["learning_rate"] <= 0:
        raise ValueError(
            "[complete_optimizer_config]: Learning rate phải là số dương hữu hạn.\n"
            f"Nhận được: learning_rate = {result['learning_rate']}"
        )

    ## Giá trị weight decay không âm, hữu hạn
    if not math.isfinite(result["weight_decay"]) or result["weight_decay"] < 0:
        raise ValueError(
            "[complete_optimizer_config]: Weight decay phải lớn hơn hoặc bằng 0.\n"
            f"Nhận được: weight_decay = {result['weight_decay']}"
        )

    return result

# ============================================================
# Hoàn thiện cấu hình scheduler
# Trả về: dict cấu hình scheduler đầy đủ
def _complete_scheduler_config(scheduler_config, selection_metric):

    # Lấy dữ liệu: Bản sao để pop từng khóa, khóa còn sót lại là khóa lạ
    remaining_config = deepcopy(dict(scheduler_config or {}))
    name = str(remaining_config.pop("name", config.SCHEDULER_NAME)).lower()

    # Không dùng scheduler
    if name == "none":

        ## Không có khóa lạ
        if remaining_config:
            raise ValueError(
                "[_complete_scheduler_config]: Scheduler 'none' không nhận các khóa này.\n"
                f"Khóa lạ: {sorted(remaining_config)}"
            )
        return {"name": "none"}

    # Kiểm tra scheduler
    ## Scheduler phải được hỗ trợ
    if name != "reduce_on_plateau":
        raise ValueError(
            "[_complete_scheduler_config]: Scheduler phải là một trong reduce_on_plateau | none.\n"
            f"Nhận được: name = {name!r}"
        )

    # ReduceLROnPlateau (mode mặc định theo tiêu chí chọn mô hình)
    expected_mode = "min" if selection_metric == "validation_loss" else "max"
    result = {
        "name"      : name,
        "mode"      : str(remaining_config.pop("mode", expected_mode)),                    # min: Theo dõi Loss, max: Theo dõi AUROC
        "factor"    : float(remaining_config.pop("factor", config.SCHEDULER_FACTOR)),      # Hệ số nhân learning rate khi giảm
        "patience"  : int(remaining_config.pop("patience", config.SCHEDULER_PATIENCE)),    # Số epoch không cải thiện trước khi giảm
        "threshold" : float(remaining_config.pop("threshold", config.SCHEDULER_THRESHOLD)),# Mức cải thiện tối thiểu
    }

    # Kiểm tra giá trị
    ## Không có khóa lạ (gõ sai tên khóa)
    if remaining_config:
        raise ValueError(
            "[_complete_scheduler_config]: ReduceLROnPlateau không nhận các khóa này.\n"
            f"Khóa lạ: {sorted(remaining_config)}"
        )

    ## Giá trị mode phải khớp tiêu chí chọn mô hình
    if result["mode"] != expected_mode:
        raise ValueError(
            f"[_complete_scheduler_config]: Scheduler mode phải khớp selection_metric '{selection_metric}' (cần '{expected_mode}').\n"
            f"Nhận được: mode = {result['mode']!r}"
        )

    ## Giá trị factor trong khoảng (0, 1), patience và threshold không âm
    if not 0 < result["factor"] < 1 or result["patience"] < 0 or result["threshold"] < 0:
        raise ValueError(
            "[_complete_scheduler_config]: ReduceLROnPlateau cần factor trong khoảng (0, 1), patience >= 0 và threshold >= 0.\n"
            f"Nhận được: factor = {result['factor']}, patience = {result['patience']}, threshold = {result['threshold']}"
        )

    return result

# ============================================================
# Khởi tạo optimizer từ cấu hình đã hoàn chỉnh (kết quả của complete_optimizer_config)
# Trả về: Optimizer (SGD / Adam / AdamW)
def make_optimizer(parameters, optimizer_config):

    # Tách tên optimizer và các tham số còn lại
    optimizer_arguments = dict(optimizer_config)
    name = optimizer_arguments.pop("name")

    # Đổi learning_rate --> lr (tên tham số của torch.optim)
    optimizer_arguments["lr"] = optimizer_arguments.pop("learning_rate")

    # SGD
    if name == "sgd":
        return torch.optim.SGD(parameters, **optimizer_arguments)

    # Adam / AdamW (betas lưu dạng danh sách trong JSON --> tuple)
    optimizer_arguments["betas"] = tuple(optimizer_arguments["betas"])
    
    return (torch.optim.Adam if name == "adam" else torch.optim.AdamW)(parameters, **optimizer_arguments)

# ============================================================
# Khởi tạo scheduler từ cấu hình đã hoàn chỉnh
# Trả về: ReduceLROnPlateau, hoặc None nếu không dùng scheduler
def make_scheduler(optimizer, scheduler_config):

    # Sao chép cấu hình (giữ nguyên bản gốc)
    scheduler_arguments = dict(scheduler_config)

    # Không dùng scheduler
    if scheduler_arguments.pop("name") == "none":
        return None

    # ReduceLROnPlateau
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_arguments)
