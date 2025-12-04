"""Evaluation metrics."""

from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("y_prob must be between 0 and 1")
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Simple ECE with equal-width bins over [0,1]."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        m = int(np.sum(mask))
        if m == 0:
            continue

        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += (m / n) * abs(acc - conf)

    return float(ece)
