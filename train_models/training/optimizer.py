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
STEP_SCHEDULERS = ("warmup_cosine",)                                       # Scheduler cập nhật sau mỗi bước tối ưu (các scheduler khác: mỗi epoch)

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

        # Nhóm tham số (parameter_groups)
        "new_module_lr_multiplier"  : float(remaining_config.pop("new_module_lr_multiplier", 1.0)),     # learning_rate lớp mới = learning_rate x hệ số; 1.0 = một nhóm chung
        "no_weight_decay_1d"        : bool(remaining_config.pop("no_weight_decay_1d", False)),          # Không decay tham số 1 chiều và model.no_weight_decay()
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

    ## Giá trị new_module_lr_multiplier là số dương hữu hạn
    if not math.isfinite(result["new_module_lr_multiplier"]) or result["new_module_lr_multiplier"] <= 0:
        raise ValueError(
            "[complete_optimizer_config]: new_module_lr_multiplier phải là số dương hữu hạn.\n"
            f"Nhận được: new_module_lr_multiplier = {result['new_module_lr_multiplier']}"
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

    # Warmup tuyến tính rồi giảm theo cosine (cập nhật sau mỗi bước tối ưu)
    if name == "warmup_cosine":
        result = {
            "name"          : name,
            "warmup_epochs" : float(remaining_config.pop("warmup_epochs", config.SCHEDULER_WARMUP_EPOCHS)),  # Số epoch warmup (có thể lẻ)
            "min_lr_ratio"  : float(remaining_config.pop("min_lr_ratio", 0.0)),                              # learning_rate cuối / learning_rate đỉnh
        }

        ## Không có khóa lạ
        if remaining_config:
            raise ValueError(
                "[_complete_scheduler_config]: Scheduler 'warmup_cosine' không nhận các khóa này.\n"
                f"Khóa lạ: {sorted(remaining_config)}"
            )

        ## Giá trị warmup_epochs không âm, min_lr_ratio trong khoảng [0, 1)
        if not math.isfinite(result["warmup_epochs"]) or result["warmup_epochs"] < 0 or not 0 <= result["min_lr_ratio"] < 1:
            raise ValueError(
                "[_complete_scheduler_config]: warmup_cosine cần warmup_epochs >= 0 và min_lr_ratio trong khoảng [0, 1).\n"
                f"Nhận được: warmup_epochs = {result['warmup_epochs']}, min_lr_ratio = {result['min_lr_ratio']}"
            )
        return result

    # Kiểm tra scheduler
    ## Scheduler phải được hỗ trợ
    if name != "reduce_on_plateau":
        raise ValueError(
            "[_complete_scheduler_config]: Scheduler phải là một trong reduce_on_plateau | warmup_cosine | none.\n"
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
# Nhóm tham số theo learning_rate và weight decay
# Trả về: Danh sách param group cho torch.optim
def parameter_groups(model, criterion, learning_rate, multiplier, no_weight_decay_1d = False):

    # Lấy dữ liệu: Tham số học được của Loss (đi cùng nhóm lớp mới)
    loss_parameters = [p for p in criterion.parameters() if p.requires_grad]

    # Một nhóm chung (multiplier = 1.0: cùng thứ tự tham số như trước, kết quả không đổi)
    if multiplier == 1.0:
        groups = [{"params": list(model.parameters()) + loss_parameters}]

    # Hai nhóm: Backbone tiền huấn luyện (các khối trong model.PRETRAINED_MODULES) dùng learning_rate,
    # lớp mới khởi tạo ngẫu nhiên (neck, khối trộn token, đầu phân loại) và tham số của Loss dùng learning_rate x multiplier
    else:
        names = getattr(model, "PRETRAINED_MODULES", None)

        ## Mô hình phải khai báo PRETRAINED_MODULES
        if not names:
            raise ValueError(
                f"[parameter_groups]: {type(model).__name__} phải khai báo PRETRAINED_MODULES (tên các khối dùng trọng số tiền huấn luyện) để tách learning_rate backbone / lớp mới.\n"
                f"Nhận được: PRETRAINED_MODULES = {names!r}"
            )

        # Tách tham số backbone / lớp mới
        pretrained = {id(p) for name in names for p in getattr(model, name).parameters()}
        backbone = [p for p in model.parameters() if id(p) in pretrained]
        new = [p for p in model.parameters() if id(p) not in pretrained] + loss_parameters

        ## Cả hai nhóm đều phải có tham số
        if not backbone or not new:
            raise ValueError(
                f"[parameter_groups]: {type(model).__name__} phải có tham số ở cả nhóm backbone và nhóm lớp mới.\n"
                f"Nhận được: backbone = {len(backbone)} tham số, lớp mới = {len(new)} tham số"
            )
        groups = [{"params": backbone, "lr": learning_rate}, {"params": new, "lr": learning_rate * multiplier}]

    # Giữ weight decay cho mọi tham số
    if not no_weight_decay_1d:
        return groups

    # Tách mỗi nhóm thành phần có / không weight decay: Không decay tham số 1 chiều (bias, norm, layer scale)
    # và các tên trong model.no_weight_decay() (quy ước timm, ví dụ mã hóa vị trí, A_log / D / dt_bias của SSM)
    named = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    exempt = {id(p) for name, p in model.named_parameters() if p.ndim <= 1 or name in named} | {id(p) for p in loss_parameters if p.ndim <= 1}
    split = []
    for group in groups:
        decay = [p for p in group["params"] if id(p) not in exempt]
        no_decay = [p for p in group["params"] if id(p) in exempt]
        split += ([{**group, "params": decay}] if decay else []) + ([{**group, "params": no_decay, "weight_decay": 0.0}] if no_decay else [])

    return split

# ============================================================
# Khởi tạo optimizer từ cấu hình đã hoàn chỉnh (kết quả của complete_optimizer_config)
# Trả về: Optimizer (SGD / Adam / AdamW)
def make_optimizer(model, criterion, optimizer_config):

    # Tách tên optimizer và các tham số còn lại
    optimizer_arguments = dict(optimizer_config)
    name = optimizer_arguments.pop("name")

    # Đổi learning_rate --> lr (tên tham số của torch.optim)
    optimizer_arguments["lr"] = optimizer_arguments.pop("learning_rate")

    # Nhóm tham số (bỏ hai khóa không phải tham số của torch.optim)
    parameters = parameter_groups(model, criterion, optimizer_arguments["lr"], optimizer_arguments.pop("new_module_lr_multiplier"), optimizer_arguments.pop("no_weight_decay_1d"))

    # SGD
    if name == "sgd":
        return torch.optim.SGD(parameters, **optimizer_arguments)

    # Adam / AdamW (betas lưu dạng danh sách trong JSON --> tuple)
    optimizer_arguments["betas"] = tuple(optimizer_arguments["betas"])
    
    return (torch.optim.Adam if name == "adam" else torch.optim.AdamW)(parameters, **optimizer_arguments)

# ============================================================
# Khởi tạo scheduler từ cấu hình đã hoàn chỉnh
# Trả về: LambdaLR (warmup_cosine), ReduceLROnPlateau, hoặc None nếu không dùng scheduler
def make_scheduler(optimizer, scheduler_config, steps_per_epoch = None, epochs = None):

    # Sao chép cấu hình (giữ nguyên bản gốc)
    scheduler_arguments = dict(scheduler_config)
    name = scheduler_arguments.pop("name")

    # Không dùng scheduler
    if name == "none":
        return None

    # Warmup cosine (cần số bước tối ưu mỗi epoch): Hệ số learning_rate tăng tuyến tính tới 1 sau warmup_epochs,
    # rồi giảm theo cosine về min_lr_ratio ở cuối epoch thứ `epochs`
    if name == "warmup_cosine":
        warmup = round(scheduler_arguments["warmup_epochs"] * int(steps_per_epoch))
        total = max(int(epochs) * int(steps_per_epoch), warmup + 1)
        floor = scheduler_arguments["min_lr_ratio"]

        # Hệ số learning_rate tại bước tối ưu thứ step
        def factor(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = min((step - warmup) / max(total - warmup, 1), 1.0)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    # ReduceLROnPlateau
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_arguments)
