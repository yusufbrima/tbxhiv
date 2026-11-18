"""Validation-chosen decision thresholds."""

import numpy as np
import pytest

from tbmxai.evaluation.threshold import select_threshold


def test_youden_and_sensitivity_thresholds() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.05, 0.10, 0.20, 0.30, 0.25, 0.35, 0.40, 0.45])
    thr = select_threshold(y, p, "youden")
    assert thr == pytest.approx(0.35)  # ≥0.35 → sens 0.75, spec 1.0 (J = 0.75, the maximum)
    thr_s = select_threshold(y, p, "sensitivity", target_sensitivity=1.0)
    assert thr_s == pytest.approx(0.25) and ((p >= thr_s)[y == 1]).all()
    with pytest.raises(ValueError):
        select_threshold(np.zeros(4), p[:4])
