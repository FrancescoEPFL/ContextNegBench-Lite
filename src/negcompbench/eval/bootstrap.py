from __future__ import annotations

import numpy as np


def bootstrap_accuracy_ci(correct: list[bool], n_bootstrap: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    if not correct:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(correct, dtype=float)
    estimates = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        estimates.append(sample.mean())
    lower = float(np.quantile(estimates, alpha / 2))
    upper = float(np.quantile(estimates, 1 - alpha / 2))
    return lower, upper
