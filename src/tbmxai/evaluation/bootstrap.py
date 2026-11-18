"""Non-parametric bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricResult:
    """A point estimate with an optional confidence interval.

    Attributes:
        name: Metric name.
        value: Point estimate on the full sample.
        lower: Lower CI bound (``nan`` if CIs disabled/undefined).
        upper: Upper CI bound.
        n_valid_resamples: Resamples where the metric was defined.
    """

    name: str
    value: float
    lower: float = float("nan")
    upper: float = float("nan")
    n_valid_resamples: int = 0

    def to_dict(self) -> dict[str, float | str | int]:
        """Plain-dict representation."""
        return asdict(self)

    def __str__(self) -> str:
        if np.isnan(self.lower):
            return f"{self.value:.3f}"
        return f"{self.value:.3f} [{self.lower:.3f}, {self.upper:.3f}]"


class BootstrapCI:
    """Percentile bootstrap over samples or, preferably, patients (cluster bootstrap).

    Args:
        n_resamples: Number of bootstrap resamples.
        confidence: Two-sided confidence level (e.g. 0.95).
        stratified: Resample within each class, preserving class counts. This
            keeps rank metrics (AUROC) defined in every resample, which matters
            for small or imbalanced test sets.
        cluster: Resample whole clusters (patients) when cluster ids are available
            (see :meth:`resample_indices`). This is the correct unit when patients
            contribute several images/encounters.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_resamples: int = 1000,
        confidence: float = 0.95,
        stratified: bool = True,
        seed: int | None = 0,
        cluster: bool = True,
    ) -> None:
        if n_resamples < 1:
            raise ValueError("n_resamples must be >= 1")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        self.n_resamples = n_resamples
        self.confidence = confidence
        self.stratified = stratified
        self.seed = seed
        self.cluster = cluster

    def resample_indices(
        self, y_true: np.ndarray, clusters: np.ndarray | None = None
    ) -> Iterator[np.ndarray]:
        """Yield ``n_resamples`` bootstrap index arrays.

        Args:
            y_true: Labels (for stratification).
            clusters: Optional cluster id per sample (e.g. patient id). When given,
                **clusters** are resampled with replacement and all their samples are
                kept together, so correlated samples of one patient never act as
                independent observations. Stratification then uses each cluster's
                maximum label ("ever positive").
        """
        rng = np.random.default_rng(self.seed)
        if clusters is None:
            yield from self._sample_level(rng, y_true)
            return
        codes, uniques = pd.factorize(np.asarray(clusters), sort=True)
        n_clusters = len(uniques)
        order = np.argsort(codes, kind="stable")
        counts = np.bincount(codes, minlength=n_clusters)
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        cluster_label = pd.Series(y_true).groupby(codes).max().to_numpy()
        strata = (
            [np.flatnonzero(cluster_label == c) for c in np.unique(cluster_label)]
            if self.stratified
            else [np.arange(n_clusters)]
        )
        for _ in range(self.n_resamples):
            chosen = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in strata])
            lengths = counts[chosen]
            starts = np.repeat(offsets[chosen], lengths)
            within = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
            yield order[starts + within]

    def _sample_level(self, rng: np.random.Generator, y_true: np.ndarray) -> Iterator[np.ndarray]:
        n = len(y_true)
        if not self.stratified:
            for _ in range(self.n_resamples):
                yield rng.integers(0, n, n)
            return
        groups = [np.flatnonzero(y_true == c) for c in np.unique(y_true)]
        for _ in range(self.n_resamples):
            yield np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])

    def interval(self, values: np.ndarray) -> tuple[float, float, int]:
        """Percentile interval of bootstrap ``values`` (``nan`` entries ignored).

        Returns:
            ``(lower, upper, n_valid)``.
        """
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("nan"), float("nan"), 0
        alpha = (1.0 - self.confidence) / 2.0
        lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
        return float(lower), float(upper), int(finite.size)
