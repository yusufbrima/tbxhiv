"""Evaluator: run a metric suite (with bootstrap CIs) on :class:`Predictions`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import pandas as pd

from tbmxai.core.config import EvaluationConfig
from tbmxai.core.types import Predictions
from tbmxai.evaluation.bootstrap import BootstrapCI, MetricResult
from tbmxai.evaluation.metrics import METRICS, Metric

DEFAULT_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "f1_micro",
    "f1_macro",
    "auroc",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "spec_at_sens",
    "sens_at_spec",
    "brier",
    "ece",
)


@dataclass
class EvaluationResult:
    """Metric results for one set of predictions.

    Attributes:
        metrics: Name → :class:`MetricResult`.
        n_samples: Number of evaluated samples.
        class_counts: Samples per class name.
        confidence: CI level used (``None`` if CIs disabled).
        ci_unit: Bootstrap resampling unit (``patient`` or ``sample``).
        n_clusters: Number of distinct patients (if known).
    """

    metrics: dict[str, MetricResult]
    n_samples: int
    class_counts: dict[str, int] = field(default_factory=dict)
    confidence: float | None = None
    ci_unit: str | None = None
    n_clusters: int | None = None

    def __getitem__(self, name: str) -> MetricResult:
        return self.metrics[name]

    def to_frame(self) -> pd.DataFrame:
        """One row per metric: ``metric, value, lower, upper``."""
        return pd.DataFrame(
            [
                {"metric": r.name, "value": r.value, "lower": r.lower, "upper": r.upper}
                for r in self.metrics.values()
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "n_samples": self.n_samples,
            "class_counts": self.class_counts,
            "confidence": self.confidence,
            "ci_unit": self.ci_unit,
            "n_patients": self.n_clusters,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }


class Evaluator:
    """Compute a configurable metric suite with optional bootstrap CIs.

    Args:
        metrics: Metric names (from :data:`METRICS`) and/or :class:`Metric`
            instances. ``None`` uses :data:`DEFAULT_METRICS`.
        threshold: Positive-class probability cut-off for binary hard labels.
        bootstrap: CI estimator; ``None`` disables CIs.

    Example:
        >>> evaluator = Evaluator(["auroc", "sensitivity"], bootstrap=BootstrapCI(500))
        >>> result = evaluator.evaluate(predictions)
        >>> print(result["auroc"])
        0.861 [0.812, 0.905]
    """

    def __init__(
        self,
        metrics: Sequence[str | Metric] | None = None,
        threshold: float = 0.5,
        bootstrap: BootstrapCI | None = None,
    ) -> None:
        specs = DEFAULT_METRICS if metrics is None else metrics
        self.metrics: list[Metric] = [METRICS.build(m) if isinstance(m, str) else m for m in specs]
        names = [m.name for m in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate metric names: {names}")
        self.threshold = threshold
        self.bootstrap = bootstrap

    @classmethod
    def from_config(cls, config: EvaluationConfig, seed: int = 0) -> Self:
        """Create from an :class:`EvaluationConfig`."""
        ci = (
            BootstrapCI(
                config.n_bootstrap,
                config.confidence,
                seed=seed,
                cluster=config.bootstrap_unit == "patient",
            )
            if config.n_bootstrap > 0
            else None
        )
        return cls(config.metrics, threshold=config.threshold, bootstrap=ci)

    def evaluate(self, predictions: Predictions) -> EvaluationResult:
        """Evaluate ``predictions``; returns point estimates and CIs."""
        y_true, y_prob = predictions.y_true, predictions.y_prob
        y_pred = predictions.predicted_labels(self.threshold)

        points = {m.name: m(y_true, y_prob, y_pred) for m in self.metrics}
        results = {name: MetricResult(name, value) for name, value in points.items()}

        ci_unit = None
        if self.bootstrap is not None and len(y_true) > 1:
            clusters = predictions.group_ids if self.bootstrap.cluster else None
            ci_unit = "patient" if clusters is not None else "sample"
            samples = {m.name: np.empty(self.bootstrap.n_resamples) for m in self.metrics}
            for b, idx in enumerate(self.bootstrap.resample_indices(y_true, clusters)):
                yt, yp, yd = y_true[idx], y_prob[idx], y_pred[idx]
                for m in self.metrics:
                    samples[m.name][b] = m(yt, yp, yd)
            for name, values in samples.items():
                lower, upper, n_valid = self.bootstrap.interval(values)
                results[name] = MetricResult(name, points[name], lower, upper, n_valid)

        assert predictions.class_names is not None
        counts = np.bincount(y_true, minlength=predictions.n_classes)
        return EvaluationResult(
            metrics=results,
            n_samples=len(y_true),
            class_counts={n: int(c) for n, c in zip(predictions.class_names, counts, strict=True)},
            confidence=None if self.bootstrap is None else self.bootstrap.confidence,
            ci_unit=ci_unit,
            n_clusters=None
            if predictions.group_ids is None
            else int(pd.unique(predictions.group_ids).size),
        )
