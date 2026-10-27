import math
import torch
import config
from utilities.runtime import runtime


def normalize_training_config(raw=None):
    x = dict(raw or {})
    allowed = {"batch_size", "augmentation", "optimizer", "scheduler", "selection_metric", "early_stop_patience"}
    if set(x) - allowed:
        raise ValueError(f"Unknown training config keys: {sorted(set(x) - allowed)}")
    x.setdefault("batch_size", int(config.BATCH_SIZE))
    x.setdefault("augmentation", "default")
    x.setdefault("optimizer", {})
    x.setdefault("scheduler", {})
    x.setdefault("selection_metric", "macro_auroc")
    x.setdefault("early_stop_patience", int(runtime.PATIENCE))
    x["optimizer"] = dict(x["optimizer"] or {})
    x["scheduler"] = dict(x["scheduler"] or {})
    if x["scheduler"].get("name", "reduce_on_plateau") != "none":
        x["scheduler"].setdefault(
            "mode", "min" if x["selection_metric"] == "validation_loss" else "max"
        )
    if not isinstance(x["batch_size"], int) or x["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")
    if x["augmentation"] not in ("default", "chexnet"):
        raise ValueError("augmentation must be 'default' or 'chexnet'")
    if x["selection_metric"] not in ("macro_auroc", "validation_loss"):
        raise ValueError("selection_metric must be macro_auroc or validation_loss")
    if not isinstance(x["early_stop_patience"], int) or x["early_stop_patience"] < 1:
        raise ValueError("early_stop_patience must be a positive integer")
    return x


def make_optimizer(parameters, optimizer_config=None, optimizer_factory=None):
    cfg = dict(optimizer_config or {})
    if optimizer_factory is not None:
        if not callable(optimizer_factory):
            raise TypeError("optimizer_factory must be callable(parameters) -> torch.optim.Optimizer")
        if cfg:
            raise ValueError("Specify either optimizer_config or optimizer_factory, not both")
        result = optimizer_factory(parameters)
        if not isinstance(result, torch.optim.Optimizer):
            raise TypeError("optimizer_factory must return a torch.optim.Optimizer")
        return result

    name = str(cfg.pop("name", "adamw")).lower()
    lr = float(cfg.pop("lr", runtime.LEARNING_RATE))
    weight_decay = float(cfg.pop("weight_decay", runtime.WEIGHT_DECAY if name == "adamw" else 0.0))
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Invalid lr/weight_decay")
    if name in ("adam", "adamw"):
        betas = tuple(cfg.pop("betas", (0.9, 0.999)))
        eps = float(cfg.pop("eps", 1e-8))
        if len(betas) != 2 or not all(0 <= b < 1 for b in betas) or eps <= 0:
            raise ValueError("Invalid Adam betas/eps")
        kw = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        if cfg: raise ValueError(f"Unknown {name} options: {sorted(cfg)}")
        return (torch.optim.Adam if name == "adam" else torch.optim.AdamW)(parameters, **kw)
    if name == "sgd":
        momentum = float(cfg.pop("momentum", 0.0))
        nesterov = bool(cfg.pop("nesterov", False))
        if cfg: raise ValueError(f"Unknown SGD options: {sorted(cfg)}")
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay,
                               momentum=momentum, nesterov=nesterov)
    raise ValueError(f"Unsupported optimizer: {name}")


def make_scheduler(optimizer, scheduler_config=None):
    cfg = dict(scheduler_config or {})
    name = str(cfg.pop("name", "reduce_on_plateau")).lower()
    if name == "none":
        if cfg: raise ValueError(f"Scheduler none has extra options: {sorted(cfg)}")
        return None
    if name == "reduce_on_plateau":
        mode = str(cfg.pop("mode", "max"))
        factor = float(cfg.pop("factor", getattr(config, "SCHEDULER_FACTOR", 0.5)))
        patience = int(cfg.pop("patience", getattr(config, "SCHEDULER_PATIENCE", 3)))
        threshold = float(cfg.pop("threshold", 1e-4))
        if cfg: raise ValueError(f"Unknown ReduceLROnPlateau options: {sorted(cfg)}")
        if mode not in ("min", "max") or not 0 < factor < 1 or patience < 0 or threshold < 0:
            raise ValueError("Invalid ReduceLROnPlateau options")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold)
    raise ValueError(f"Unsupported scheduler: {name}")
