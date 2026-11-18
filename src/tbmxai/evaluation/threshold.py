"""Decision-threshold selection on validation data (binary tasks)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def select_threshold(
    y_true: np.ndarray,
    p_positive: np.ndarray,
    method: str = "youden",
    target_sensitivity: float = 0.9,
    target_specificity: float = 0.7,
) -> float:
    """Positive-class probability threshold chosen on (validation) predictions.

    Args:
        y_true: Binary labels (1 = positive).
        p_positive: Predicted positive-class probabilities.
        method: ``youden`` — maximise sensitivity + specificity − 1;
            ``sensitivity`` — the highest threshold whose sensitivity reaches
            ``target_sensitivity`` (most specific operating point at that sensitivity);
            ``specificity`` — the lowest threshold whose specificity reaches
            ``target_specificity`` (most sensitive operating point at that specificity).
        target_sensitivity: Target for ``method="sensitivity"``.
        target_specificity: Target for ``method="specificity"``.

    Returns:
        A threshold in [0, 1]; predictions ``p >= threshold`` are positive.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_positive, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("Threshold selection needs both classes in the validation data.")
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.clip(thr, 0.0, 1.0)  # roc_curve's first threshold is +inf
    if method == "youden":
        best = int(np.argmax(tpr - fpr))
    elif method == "sensitivity":
        ok = np.flatnonzero(tpr >= target_sensitivity)
        best = int(ok[0])  # thresholds decrease along the curve: first = highest
    elif method == "specificity":
        ok = np.flatnonzero(1.0 - fpr >= target_specificity - 1e-12)
        best = int(ok[-1])  # specificity falls along the curve: last = lowest threshold
    else:
        raise ValueError(f"Unknown threshold selection method '{method}'")
    return float(thr[best])
