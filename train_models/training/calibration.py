import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

_EPS = 1e-6

def fit_temperatures(labels, probabilities, min_positive=20, min_negative=20):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("Labels/probabilities should both have shape [N,C]")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected binary labels and valid probabilities")
    logits = logit(np.clip(p, _EPS, 1 - _EPS))
    fitted = []
    for k in range(p.shape[1]):
        if y[:, k].sum() < min_positive or (len(y) - y[:, k].sum()) < min_negative:
            fitted.append(1.0)
            continue
        # optimize log-temperature for positivity and numerical stability
        def objective(log_temp):
            z = logits[:, k] * np.exp(-log_temp)
            return float(np.mean(np.logaddexp(0., z) - y[:, k] * z))
        result = minimize_scalar(objective, bounds=(np.log(0.3), np.log(10.0)), method="bounded")
        fitted.append(float(np.exp(result.x)) if result.success else 1.0)
    return np.asarray(fitted, dtype=np.float64)


def apply_temperatures(probabilities, temperatures):
    p = np.asarray(probabilities, dtype=np.float64)
    temperatures = np.asarray(temperatures, dtype=np.float64)
    if p.ndim != 2 or temperatures.shape != (p.shape[1],) or np.any(temperatures <= 0):
        raise ValueError("Incompatible positive temperature vector")
    return expit(logit(np.clip(p, _EPS, 1 - _EPS)) / temperatures[None, :])
