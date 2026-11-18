"""Repeated experiments: patient-level K-fold cross-validation and/or multiple seeds.

Each fold × seed is an ordinary :class:`~tbmxai.pipeline.Pipeline` run in its own
sub-directory (with all its outputs). The driver then writes aggregate tables:

* ``cv_runs.csv`` — every run's metrics (long format);
* ``cv_summary.csv`` — mean / SD / min / max of each metric across runs;
* ``oof_metrics.csv`` — **out-of-fold** evaluation: the K disjoint test folds of one
  seed are pooled and evaluated once with patient-level bootstrap CIs (each patient
  is predicted exactly once, by a model that never saw them);
* ``oof_subgroups.csv`` — subgroup analysis on the pooled out-of-fold predictions;
* ``oof_predictions_seed<s>.csv`` and ``cv_report.md``.

If the config enables tuning, every fold tunes on its own train/val data, which
gives nested cross-validation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tbmxai.core.config import ComponentConfig, ExperimentConfig
from tbmxai.core.types import Predictions
from tbmxai.core.utils import get_logger
from tbmxai.evaluation.evaluator import Evaluator
from tbmxai.pipeline import Pipeline
from tbmxai.reporting.writers import finite, markdown_table
from tbmxai.subgroups.analysis import SubgroupAnalyzer

logger = get_logger(__name__)


@dataclass
class RepeatedResult:
    """Aggregated outputs of a repeated experiment."""

    output_dir: Path
    runs: pd.DataFrame
    summary: pd.DataFrame
    oof_metrics: pd.DataFrame | None
    oof_subgroups: pd.DataFrame | None


class RepeatedExperiment:
    """Run a config over folds and/or seeds and aggregate the results.

    Args:
        config: Base experiment config.
        folds: Number of patient-level folds (switches the splitter to ``kfold``).
            ``None`` keeps the configured splitter (seed repetition only).
        seeds: Seeds to repeat with (default: the config seed). The seed controls
            fold assignment, model initialisation and bootstrap resampling.
    """

    def __init__(
        self, config: ExperimentConfig, folds: int | None = None, seeds: Sequence[int] | None = None
    ) -> None:
        if folds is not None and folds < 2:
            raise ValueError("folds must be >= 2")
        self.config = config
        self.folds = folds
        self.seeds = list(seeds) if seeds else [config.seed]

    def variants(self) -> list[tuple[str, int | None, int, ExperimentConfig]]:
        """``(run_name, fold, seed, config)`` for every run."""
        out = []
        for seed in self.seeds:
            for fold in range(self.folds) if self.folds else [None]:
                update: dict[str, object] = {"seed": seed}
                if fold is not None:
                    params = (
                        dict(self.config.split.params) if self.config.split.name == "kfold" else {}
                    )
                    val = params.get("val_size", self.config.split.params.get("val_size", 0.15))
                    params.update({"n_splits": self.folds, "fold": fold, "val_size": val})
                    update["split"] = ComponentConfig(name="kfold", params=params)
                    # A protocol still fixes the sample set; folds and per-fold explanation sets
                    # come from the CV seed (identical across models on identical sample sets).
                    update["protocol"] = self.config.protocol.model_copy(
                        update={"use_split": False, "use_xai_samples": False}
                    )
                name = f"seed{seed}" + (f"_fold{fold}" if fold is not None else "")
                out.append((name, fold, seed, self.config.model_copy(update=update)))
        return out

    def run(self, output_dir: str | Path | None = None) -> RepeatedResult:
        """Execute all runs and write aggregate tables to ``output_dir``."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = (
            Path(output_dir).expanduser()
            if output_dir
            else self.config.output_dir.expanduser() / self.config.name / f"cv-{stamp}"
        )
        base.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(base / "config.yaml")

        rows, test_preds = [], {}
        for name, fold, seed, cfg in self.variants():
            logger.info("=== Run %s ===", name)
            result = Pipeline(cfg).run(base / name)
            for split, ev in result.evaluations.items():
                for m in ev.metrics.values():
                    rows.append(
                        {
                            "run": name,
                            "fold": fold,
                            "seed": seed,
                            "split": split,
                            "metric": m.name,
                            "value": m.value,
                            "lower": m.lower,
                            "upper": m.upper,
                            "n": ev.n_samples,
                        }
                    )
            if "test" in result.predictions:
                test_preds[(seed, fold)] = result.predictions["test"]

        runs = pd.DataFrame(rows)
        summary = (
            runs.groupby(["split", "metric"])["value"]
            .agg(mean="mean", sd="std", min="min", max="max", n_runs="count")
            .reset_index()
        )
        runs.to_csv(base / "cv_runs.csv", index=False)
        summary.to_csv(base / "cv_summary.csv", index=False)

        oof, oof_sub = self._out_of_fold(test_preds, base) if self.folds else (None, None)
        self._report(base, summary, oof)
        (base / "cv_info.json").write_text(
            json.dumps(
                finite(
                    {
                        "folds": self.folds,
                        "seeds": self.seeds,
                        "n_runs": int(runs["run"].nunique()) if len(runs) else 0,
                        "nested_tuning": bool(self.config.tuning.enabled),
                    }
                ),
                indent=2,
            )
        )
        logger.info("Repeated experiment done → %s", base)
        return RepeatedResult(base, runs, summary, oof, oof_sub)

    def _out_of_fold(
        self, preds: dict[tuple[int, int | None], Predictions], base: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        evaluator = Evaluator.from_config(self.config.evaluation, seed=self.config.seed)
        metric_rows, sub_frames = [], []
        for seed in self.seeds:
            parts = [
                p for (s, _), p in sorted(preds.items(), key=lambda kv: str(kv[0])) if s == seed
            ]
            if not parts:
                continue
            pooled = _concat(parts)
            pooled.to_frame().to_csv(base / f"oof_predictions_seed{seed}.csv", index=False)
            ev = evaluator.evaluate(pooled)
            for m in ev.metrics.values():
                metric_rows.append(
                    {
                        "seed": seed,
                        "metric": m.name,
                        "value": m.value,
                        "lower": m.lower,
                        "upper": m.upper,
                        "n": ev.n_samples,
                        "n_patients": ev.n_clusters,
                        "ci_unit": ev.ci_unit,
                    }
                )
            if self.config.subgroups.columns:
                sub = SubgroupAnalyzer.from_config(self.config.subgroups, evaluator).analyze(pooled)
                sub_frames.append(sub.assign(seed=seed))
        oof = pd.DataFrame(metric_rows)
        oof.to_csv(base / "oof_metrics.csv", index=False)
        sub = pd.concat(sub_frames, ignore_index=True) if sub_frames else None
        if sub is not None:
            sub.to_csv(base / "oof_subgroups.csv", index=False)
        return oof, sub

    def _report(self, base: Path, summary: pd.DataFrame, oof: pd.DataFrame | None) -> None:
        lines = [
            f"# {self.config.name} — repeated evaluation",
            "",
            f"- Folds: {self.folds or '–'} (patient-level)  ·  Seeds: {self.seeds}",
            "",
        ]
        if oof is not None and not oof.empty:
            cell = [
                f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else f"{v:.3f}"
                for v, lo, hi in oof[["value", "lower", "upper"]].itertuples(index=False)
            ]
            wide = (
                oof.assign(cell=cell)
                .pivot(index="metric", columns="seed", values="cell")
                .reset_index()
            )
            lines += [
                "## Out-of-fold performance (pooled test folds, patient-level bootstrap CI)",
                "",
                markdown_table(wide),
                "",
            ]
        test = summary[summary["split"] == "test"].round(4)
        lines += ["## Per-run test metrics across runs", "", markdown_table(test), ""]
        (base / "cv_report.md").write_text("\n".join(lines))


def _concat(parts: list[Predictions]) -> Predictions:
    meta = [p.metadata for p in parts]
    return Predictions(
        y_true=np.concatenate([p.y_true for p in parts]),
        y_prob=np.concatenate([p.y_prob for p in parts]),
        sample_ids=np.concatenate([p.sample_ids for p in parts])
        if all(p.sample_ids is not None for p in parts)
        else None,
        class_names=parts[0].class_names,
        metadata=pd.concat(meta, ignore_index=True) if all(m is not None for m in meta) else None,
        group_ids=np.concatenate([p.group_ids for p in parts])
        if all(p.group_ids is not None for p in parts)
        else None,
    )
