"""Subgroup (stratified) performance analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np
import pandas as pd

from tbmxai.core.config import SubgroupConfig
from tbmxai.core.types import Predictions
from tbmxai.core.utils import get_logger
from tbmxai.evaluation.evaluator import Evaluator

logger = get_logger(__name__)

OVERALL = "__overall__"


class SubgroupAnalyzer:
    """Evaluate predictions separately for each value of selected metadata columns.

    Numeric columns are binned: with explicit edges from ``bins`` or, when a
    column has more than ``max_categories`` distinct values, into quartiles.

    Args:
        evaluator: Evaluator used for every subgroup (its CI settings apply).
        columns: Metadata columns defining subgroups (e.g. ``sex``, ``hiv_status``, ``age``).
        min_size: Subgroups smaller than this are skipped (and logged).
        bins: Explicit bin edges per numeric column, e.g. ``{"age": [0, 30, 50, 120]}``.
        max_categories: Numeric columns with more distinct values get quartile bins.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        columns: Sequence[str],
        min_size: int = 20,
        bins: Mapping[str, Sequence[float]] | None = None,
        max_categories: int = 10,
    ) -> None:
        self.evaluator = evaluator
        self.columns = list(columns)
        self.min_size = min_size
        self.bins = {k: list(v) for k, v in (bins or {}).items()}
        self.max_categories = max_categories

    @classmethod
    def from_config(cls, config: SubgroupConfig, evaluator: Evaluator) -> Self:
        """Create from a :class:`SubgroupConfig`."""
        return cls(evaluator, config.columns, min_size=config.min_size, bins=config.bins)

    def group_labels(self, values: pd.Series) -> pd.Series:
        """Map raw column values to subgroup labels (binning numerics if needed)."""
        name = str(values.name)
        if name in self.bins:
            return pd.cut(values, bins=self.bins[name], include_lowest=True).astype(str)
        if pd.api.types.is_numeric_dtype(values) and values.nunique() > self.max_categories:
            return pd.qcut(values, q=4, duplicates="drop").astype(str)
        return values.astype(str)

    def analyze(self, predictions: Predictions) -> pd.DataFrame:
        """Return a long table: ``column, group, n, metric, value, lower, upper``.

        The first rows (``column == "__overall__"``) hold whole-population results.

        Raises:
            KeyError: If a subgroup column is missing from the prediction metadata.
        """
        meta = predictions.metadata if predictions.metadata is not None else pd.DataFrame()
        if missing := [c for c in self.columns if c not in meta.columns]:
            raise KeyError(
                f"Subgroup columns {missing} not in prediction metadata; add them to "
                "`view.metadata_columns`."
            )

        rows: list[dict[str, object]] = []
        self._append(rows, OVERALL, "all", predictions)
        for column in self.columns:
            labels = self.group_labels(meta[column])
            for group in sorted(labels.dropna().unique()):
                mask = (labels == group).to_numpy()
                if group == "nan" or mask.sum() < self.min_size:
                    logger.info("Skipping subgroup %s=%s (n=%d)", column, group, mask.sum())
                    continue
                self._append(rows, column, group, predictions.subset(np.flatnonzero(mask)))
        return pd.DataFrame(rows)

    def _append(
        self, rows: list[dict[str, object]], column: str, group: str, preds: Predictions
    ) -> None:
        result = self.evaluator.evaluate(preds)
        for metric in result.metrics.values():
            rows.append(
                {
                    "column": column,
                    "group": group,
                    "n": result.n_samples,
                    "metric": metric.name,
                    "value": metric.value,
                    "lower": metric.lower,
                    "upper": metric.upper,
                }
            )
