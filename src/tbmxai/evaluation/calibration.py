"""Post-hoc probability calibration, fitted on validation predictions.

A calibrator maps a model's predicted probabilities to better-calibrated ones
without retraining the model. It is fitted on the **validation** split only and
then applied to every split's predictions (and so to thresholds, subgroups,
ablation, saved predictions and ``tbmxai compare``).

* ``temperature`` (Guo et al., 2017): one parameter ``T`` dividing the logits;
  binary or multiclass. It never changes the ranking, so AUROC is unchanged.
* ``platt`` (Platt, 1999): logistic regression on the positive-class logit
  (slope and intercept). Binary only; ranking unchanged.
* ``isotonic`` (Zadrozny & Elkan, 2002): monotone step function. Binary only;
  can create ties and so change AUROC slightly; needs a few hundred
  validation samples to be stable.

Calibration fitted on one site does not correct a probability shift at
another site (e.g. external validation): it fixes the model's
over/under-confidence on the development distribution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np

from tbmxai.core.registry import Registry

_EPS = 1e-7


def _log_probs(y_prob: np.ndarray) -> np.ndarray:
    return np.log(np.clip(y_prob, _EPS, 1.0))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


def negative_log_likelihood(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean negative log-likelihood (log loss) of the true classes."""
    y = np.asarray(y_true).astype(int)
    return float(-np.mean(_log_probs(y_prob)[np.arange(len(y)), y]))


class Calibrator(ABC):
    """Maps probabilities ``(n, C)`` to calibrated probabilities ``(n, C)``."""

    #: Whether the calibrator handles more than two classes.
    multiclass: bool = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> Self:
        """Fit on (validation) labels and probabilities."""
        y = np.asarray(y_true).astype(int)
        p = np.asarray(y_prob, dtype=float)
        if p.ndim != 2 or len(p) != len(y):
            raise ValueError(f"y_prob shape {p.shape} incompatible with {len(y)} labels.")
        if p.shape[1] > 2 and not self.multiclass:
            raise ValueError(f"{type(self).__name__} supports binary tasks only.")
        if np.unique(y).size < 2:
            raise ValueError("Calibration needs at least two classes in the validation data.")
        self._fit(y, p)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Calibrated probabilities (rows sum to 1)."""
        return self._transform(np.asarray(y_prob, dtype=float))

    @abstractmethod
    def _fit(self, y: np.ndarray, p: np.ndarray) -> None: ...

    @abstractmethod
    def _transform(self, p: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Fitted parameters (JSON-friendly)."""


CALIBRATORS: Registry[Calibrator] = Registry("calibrator")


@CALIBRATORS.register("temperature")
class TemperatureScaling(Calibrator):
    """Softmax of ``log p / T``, with ``T`` minimising validation log loss."""

    multiclass = True

    def __init__(self, bounds: tuple[float, float] = (0.05, 20.0)) -> None:
        self.bounds = bounds
        self.temperature_: float = 1.0

    def _fit(self, y: np.ndarray, p: np.ndarray) -> None:
        from scipy.optimize import minimize_scalar

        logits = _log_probs(p)
        rows = np.arange(len(y))

        def nll(log_t: float) -> float:
            z = logits / np.exp(log_t)
            z = z - z.max(axis=1, keepdims=True)
            return float(-np.mean(z[rows, y] - np.log(np.exp(z).sum(axis=1))))

        lo, hi = np.log(self.bounds[0]), np.log(self.bounds[1])
        res = minimize_scalar(nll, bounds=(lo, hi), method="bounded")
        self.temperature_ = float(np.exp(res.x))

    def _transform(self, p: np.ndarray) -> np.ndarray:
        return _softmax(_log_probs(p) / self.temperature_)

    def params(self) -> dict[str, Any]:
        return {"temperature": self.temperature_}


@CALIBRATORS.register("platt")
class PlattScaling(Calibrator):
    """``sigmoid(a · logit(p) + b)`` fitted by (lightly regularised) logistic regression."""

    def __init__(self) -> None:
        self.slope_: float = 1.0
        self.intercept_: float = 0.0

    def _fit(self, y: np.ndarray, p: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e4, max_iter=1000)
        lr.fit(_logit(p[:, 1]).reshape(-1, 1), y)
        self.slope_ = float(lr.coef_[0, 0])
        self.intercept_ = float(lr.intercept_[0])

    def _transform(self, p: np.ndarray) -> np.ndarray:
        q = 1.0 / (1.0 + np.exp(-(self.slope_ * _logit(p[:, 1]) + self.intercept_)))
        return np.column_stack([1.0 - q, q])

    def params(self) -> dict[str, Any]:
        return {"slope": self.slope_, "intercept": self.intercept_}


@CALIBRATORS.register("isotonic")
class IsotonicCalibration(Calibrator):
    """Monotone, piecewise-constant map of the positive-class probability."""

    def __init__(self) -> None:
        self.iso_: Any = None

    def _fit(self, y: np.ndarray, p: np.ndarray) -> None:
        from sklearn.isotonic import IsotonicRegression

        self.iso_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso_.fit(p[:, 1], y)

    def _transform(self, p: np.ndarray) -> np.ndarray:
        q = np.asarray(self.iso_.predict(p[:, 1]), dtype=float)
        return np.column_stack([1.0 - q, q])

    def params(self) -> dict[str, Any]:
        if self.iso_ is None:
            return {}
        return {
            "x_thresholds": self.iso_.X_thresholds_.tolist(),
            "y_thresholds": self.iso_.y_thresholds_.tolist(),
        }
