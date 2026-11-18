"""Classification metrics.

Every metric is a :class:`Metric` subclass registered in :data:`METRICS`, so
the suite can be extended without touching the evaluator::

    @METRICS.register("youden_j")
    class YoudenJ(Metric):
        def compute(self, y_true, y_prob, y_pred):
            ...

Conventions:

* ``y_true`` — int labels ``(n,)``; ``y_prob`` — probabilities ``(n, C)``;
  ``y_pred`` — hard labels ``(n,)`` (thresholded by the evaluator).
* Binary tasks report the **positive class (index 1)** for sensitivity,
  specificity, PPV and NPV; multiclass tasks report the macro average of the
  one-vs-rest values.
* Undefined values (e.g. AUROC with one class present) are ``nan``.
* Operating-point metrics take a target in their spec: ``"spec_at_sens@0.95"``
  builds :class:`SpecificityAtSensitivity` with ``target=0.95`` (reported as
  ``spec_at_sens95``); the bare name uses the default target.
* Confusion-matrix metrics are pure NumPy so bootstrap CIs stay fast.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from tbmxai.core.registry import Registry


class Metric(ABC):
    """A scalar performance metric."""

    #: Whether larger values indicate better performance.
    higher_is_better: ClassVar[bool] = True

    @property
    def name(self) -> str:
        """Display name (registry key by default)."""
        return getattr(self, "_registry_name", type(self).__name__.lower())

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the metric value."""

    def __call__(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
        return float(self.compute(y_true, y_prob, y_pred))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class _MetricRegistry(Registry[Metric]):
    """Registry that also stamps the key onto the class as its display name."""

    def register(self, name: str | None = None, *, override: bool = False):  # type: ignore[override]
        inner = super().register(name, override=override)

        def decorator(cls: type[Metric]) -> type[Metric]:
            cls._registry_name = (name or cls.__name__).lower()  # type: ignore[attr-defined]
            return inner(cls)

        return decorator

    def build(self, name: str, /, **kwargs: object) -> Metric:  # type: ignore[override]
        """Build a metric; ``"name@value"`` passes ``target=value`` (operating-point metrics)."""
        base, sep, target = name.partition("@")
        if sep:
            try:
                kwargs = {**kwargs, "target": float(target)}
            except ValueError:
                msg = f"Bad metric target in '{name}': expected e.g. 'name@0.9'"
                raise ValueError(msg) from None
        return super().build(base, **kwargs)


METRICS: Registry[Metric] = _MetricRegistry("metric")


# --------------------------------------------------------------------- helpers


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """``C[i, j]`` = number of samples with true class ``i`` predicted as ``j``."""
    flat = y_true.astype(int) * n_classes + y_pred.astype(int)
    return np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def one_vs_rest_counts(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-class (TP, FP, FN, TN) arrays."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    return tp, fp, fn, tn


class _OneVsRestMetric(Metric):
    """Positive-class value for binary tasks, macro average for multiclass."""

    @abstractmethod
    def per_class(
        self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray
    ) -> np.ndarray:
        """Vectorised per-class metric."""

    def compute(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
        n_classes = y_prob.shape[1]
        values = self.per_class(*one_vs_rest_counts(y_true, y_pred, n_classes))
        if n_classes == 2:
            return float(values[1])
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


# ------------------------------------------------------------- discrimination


@METRICS.register("accuracy")
class Accuracy(Metric):
    """Fraction of correct predictions."""

    def compute(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")


@METRICS.register("balanced_accuracy")
class BalancedAccuracy(Metric):
    """Mean per-class recall (robust to class imbalance)."""

    def compute(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
        tp, _, fn, _ = one_vs_rest_counts(y_true, y_pred, y_prob.shape[1])
        recall = _safe_div(tp, tp + fn)
        return float(np.nanmean(recall)) if np.isfinite(recall).any() else float("nan")


@METRICS.register("sensitivity")
class Sensitivity(_OneVsRestMetric):
    """True-positive rate / recall: TP / (TP + FN)."""

    def per_class(self, tp, fp, fn, tn):  # type: ignore[no-untyped-def]
        return _safe_div(tp, tp + fn)


@METRICS.register("specificity")
class Specificity(_OneVsRestMetric):
    """True-negative rate: TN / (TN + FP)."""

    def per_class(self, tp, fp, fn, tn):  # type: ignore[no-untyped-def]
        return _safe_div(tn, tn + fp)


@METRICS.register("ppv")
class PositivePredictiveValue(_OneVsRestMetric):
    """Precision: TP / (TP + FP)."""

    def per_class(self, tp, fp, fn, tn):  # type: ignore[no-untyped-def]
        return _safe_div(tp, tp + fp)


@METRICS.register("npv")
class NegativePredictiveValue(_OneVsRestMetric):
    """TN / (TN + FN)."""

    def per_class(self, tp, fp, fn, tn):  # type: ignore[no-untyped-def]
        return _safe_div(tn, tn + fn)


@METRICS.register("f1_macro")
class F1Macro(Metric):
    """Unweighted mean of per-class F1 (classes with no support/predictions count as 0)."""

    def compute(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
        tp, fp, fn, _ = one_vs_rest_counts(y_true, y_pred, y_prob.shape[1])
        f1 = np.nan_to_num(_safe_div(2 * tp, 2 * tp + fp + fn), nan=0.0)
        return float(f1.mean())


@METRICS.register("f1_micro")
class F1Micro(Metric):
    """F1 from pooled counts (equals accuracy for single-label classification)."""

    def compute(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
        tp, fp, fn, _ = one_vs_rest_counts(y_true, y_pred, y_prob.shape[1])
        return float(_safe_div(2 * tp.sum(), 2 * tp.sum() + fp.sum() + fn.sum()))


@METRICS.register("auroc")
class AUROC(Metric):
    """Area under the ROC curve (macro one-vs-rest for multiclass)."""

    def compute(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
        n_classes = y_prob.shape[1]
        if np.unique(y_true).size < 2:
            return float("nan")
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_prob[:, 1]))
        if np.unique(y_true).size < n_classes:
            return float("nan")  # OvR macro undefined when a class is absent
        return float(
            roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro", labels=range(n_classes)
            )
        )


# ------------------------------------------------- fixed operating points (binary)


def _binary_roc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Empirical ROC (FPR, TPR) at every distinct threshold, or ``None`` if undefined."""
    if y_prob.shape[1] != 2 or np.unique(y_true).size < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], drop_intermediate=False)
    return fpr, tpr


def _percent(target: float) -> str:
    return f"{target * 100:g}".replace(".", "_")


class _OperatingPointMetric(Metric):
    """Threshold-free metric read off the test ROC curve at a fixed target.

    The operating point is chosen on the evaluated data itself (the best
    achievable trade-off), so the value is an optimistic, threshold-independent
    summary. To report performance at a threshold fixed in advance, use
    ``evaluation.threshold_selection: specificity | sensitivity`` (chosen on
    validation) and read ``sensitivity`` / ``specificity``.

    Args:
        target: Required specificity or sensitivity, in (0, 1].
    """

    prefix: ClassVar[str]
    default_target: ClassVar[float]

    def __init__(self, target: float | None = None) -> None:
        target = self.default_target if target is None else float(target)
        if not 0.0 < target <= 1.0:
            raise ValueError(f"target must be in (0, 1], got {target}")
        self.target = target

    @property
    def name(self) -> str:
        return f"{self.prefix}{_percent(self.target)}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(target={self.target})"


@METRICS.register("sens_at_spec")
class SensitivityAtSpecificity(_OperatingPointMetric):
    """Highest sensitivity at specificity >= ``target`` (default 0.70). Binary only.

    The default matches the WHO target product profile for TB triage tests
    (sensitivity >= 90 %, specificity >= 70 %).
    """

    prefix = "sens_at_spec"
    default_target = 0.70

    def compute(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
        roc = _binary_roc(y_true, y_prob)
        if roc is None:
            return float("nan")
        fpr, tpr = roc
        return float(tpr[(1.0 - fpr) >= self.target - 1e-12].max())


@METRICS.register("spec_at_sens")
class SpecificityAtSensitivity(_OperatingPointMetric):
    """Highest specificity at sensitivity >= ``target`` (default 0.90). Binary only.

    The default matches the WHO target product profile for TB triage tests.
    """

    prefix = "spec_at_sens"
    default_target = 0.90

    def compute(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
        roc = _binary_roc(y_true, y_prob)
        if roc is None:
            return float("nan")
        fpr, tpr = roc
        return float((1.0 - fpr[tpr >= self.target - 1e-12]).max())


# ----------------------------------------------------------------- calibration


@METRICS.register("brier")
class BrierScore(Metric):
    """Mean squared error of probabilities.

    Binary: ``mean((p_1 - y)^2)`` (the conventional definition, range [0, 1]).
    Multiclass: ``mean(sum_k (p_k - 1[y=k])^2)`` (range [0, 2]).
    """

    higher_is_better = False

    def compute(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
        if not len(y_true):
            return float("nan")
        if y_prob.shape[1] == 2:
            return float(np.mean((y_prob[:, 1] - y_true) ** 2))
        onehot = np.eye(y_prob.shape[1])[y_true]
        return float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))


@METRICS.register("ece")
class ExpectedCalibrationError(Metric):
    """Top-label expected calibration error with equal-width bins (Guo et al., 2017).

    Args:
        n_bins: Number of confidence bins.
    """

    higher_is_better = False

    def __init__(self, n_bins: int = 15) -> None:
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        self.n_bins = n_bins

    def compute(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
        if not len(y_true):
            return float("nan")
        confidence = y_prob.max(axis=1)
        correct = (y_prob.argmax(axis=1) == y_true).astype(float)
        # Bin index in [0, n_bins-1]; confidence == 1.0 goes into the last bin.
        bins = np.minimum((confidence * self.n_bins).astype(int), self.n_bins - 1)
        conf_sum = np.bincount(bins, weights=confidence, minlength=self.n_bins)
        acc_sum = np.bincount(bins, weights=correct, minlength=self.n_bins)
        return float(np.abs(acc_sum - conf_sum).sum() / len(y_true))

    def __repr__(self) -> str:
        return f"ExpectedCalibrationError(n_bins={self.n_bins})"
