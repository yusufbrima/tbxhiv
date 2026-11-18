"""Run results and the report-writer contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from tbmxai.core.config import ExperimentConfig
from tbmxai.core.registry import Registry
from tbmxai.core.types import Predictions
from tbmxai.evaluation.evaluator import EvaluationResult
from tbmxai.explain.base import Explanation


@dataclass
class RunResult:
    """Everything a pipeline run produced; the input to report writers.

    Attributes:
        config: The validated (resolved) experiment config.
        output_dir: Directory the run writes to.
        cohort_summary: :meth:`Cohort.summary` output.
        split_sizes: Samples per split.
        cohort_table: Per split × class sample/patient counts.
        predictions: Split name → predictions.
        evaluations: Split name → evaluation result.
        subgroups: Long-format subgroup table (test split) or ``None``.
        explanations: Explanations produced on the test split.
        training_history: Per-epoch log for neural models, else ``None``.
        tuning_trials: Optuna trials table when tuning was enabled.
        tuning_best: Best params / value / trial when tuning was enabled.
        modality_ablation: Test metrics per available-modality subset (if enabled).
        calibration: Post-hoc calibration method, parameters and validation fit (if enabled).
    """

    config: ExperimentConfig
    output_dir: Path
    cohort_summary: dict[str, Any]
    split_sizes: dict[str, int]
    cohort_table: pd.DataFrame | None = None
    predictions: dict[str, Predictions] = field(default_factory=dict)
    evaluations: dict[str, EvaluationResult] = field(default_factory=dict)
    subgroups: pd.DataFrame | None = None
    explanations: list[Explanation] = field(default_factory=list)
    training_history: pd.DataFrame | None = None
    tuning_trials: pd.DataFrame | None = None
    tuning_best: dict[str, Any] | None = None
    decision_threshold: float | None = None
    modality_ablation: pd.DataFrame | None = None
    calibration: dict[str, Any] | None = None


class ReportWriter(ABC):
    """Renders a :class:`RunResult` to one or more files."""

    @abstractmethod
    def write(self, result: RunResult, out_dir: Path) -> Path:
        """Write the report into ``out_dir`` and return the main file/dir path."""


REPORTERS: Registry[ReportWriter] = Registry("report_writer")
