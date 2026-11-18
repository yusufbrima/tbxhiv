"""Built-in report writers.

* ``json`` — one nested ``metrics.json`` (config, cohort, metrics, subgroups);
* ``tables`` — tidy, long-format tables under ``tables/`` (CSV and/or Parquet)
  ready for pandas / seaborn / plotly / R / ggplot / BI tools;
* ``markdown`` — human-readable ``report.md``.

The schemas of the tidy tables are documented in :data:`TABLE_SCHEMAS` and
the README; they are versioned by :data:`SCHEMA_VERSION`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from tbmxai.data.quality import ASSUMED_IDS_WARNING
from tbmxai.reporting.base import REPORTERS, ReportWriter, RunResult

SCHEMA_VERSION = "1.0"

#: Column descriptions of every tidy table written by :class:`TablesReportWriter`.
TABLE_SCHEMAS: dict[str, dict[str, str]] = {
    "metrics": {
        "experiment": "Experiment name",
        "split": "train | val | test",
        "metric": "Metric name",
        "value": "Point estimate",
        "lower": "Lower CI bound (NaN if CIs disabled)",
        "upper": "Upper CI bound",
        "confidence": "CI level, e.g. 0.95",
        "ci_unit": "Bootstrap resampling unit: patient (cluster) or sample",
        "n": "Number of samples evaluated",
        "n_patients": "Number of distinct patients evaluated",
    },
    "subgroups": {
        "experiment": "Experiment name",
        "split": "Always test",
        "column": "Subgroup variable (__overall__ = whole population)",
        "group": "Subgroup value / bin",
        "n": "Samples in subgroup",
        "metric": "Metric name",
        "value": "Point estimate",
        "lower": "Lower CI bound",
        "upper": "Upper CI bound",
    },
    "cohort": {
        "experiment": "Experiment name",
        "split": "all | train | val | test",
        "class": "Class name",
        "n_samples": "Samples of this class",
        "n_patients": "Distinct patients with ≥1 sample of this class",
    },
    "explanations": {
        "experiment": "Experiment name",
        "method": "Explainer name",
        "modality": "Explained modality",
        "feature": "Feature name",
        "target_class": "Class whose output is explained (all = class-agnostic method)",
        "sample_class": "True class of the samples summarised (all = every sample)",
        "n": "Samples summarised",
        "importance_mean": "Mean importance (mean |SHAP| for SHAP)",
        "importance_std": "Std of importance",
        "mean_shap": "Mean signed SHAP value (direction of effect; SHAP only)",
    },
    "modality_ablation": {
        "experiment": "Experiment name",
        "available": "Modalities given to the model (others withheld for all samples)",
        "withheld": "Modalities withheld",
        "n": "Test samples with at least one available modality",
        "metric": "Metric name",
        "value": "Point estimate",
        "lower": "Lower CI bound",
        "upper": "Upper CI bound",
    },
    "tuning_trials": {
        "experiment": "Experiment name",
        "metric": "Tuning metric (validation split)",
        "number": "Trial number",
        "value": "Validation metric of the trial",
        "state": "COMPLETE | PRUNED | FAIL",
        "duration": "Trial duration (s)",
        "param:<path>": "Sampled value of each searched config path",
    },
    "training_history": {
        "epoch": "0-based epoch",
        "train_loss": "Mean training loss",
        "val_*": "Validation loss / metrics",
        "lr": "Learning rate at epoch end",
    },
}


@REPORTERS.register("json")
class JSONReportWriter(ReportWriter):
    """Write ``metrics.json`` with config, cohort summary, metrics and subgroups."""

    def write(self, result: RunResult, out_dir: Path) -> Path:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": result.config.name,
            "config": result.config.model_dump(mode="json"),
            "cohort": result.cohort_summary,
            "split_sizes": result.split_sizes,
            "evaluation": {k: v.to_dict() for k, v in result.evaluations.items()},
            "subgroups": (
                None if result.subgroups is None else result.subgroups.to_dict(orient="records")
            ),
            "explanations": [
                {"method": e.method, "modality": str(e.modality), "scope": e.scope}
                for e in result.explanations
            ],
            "tuning": result.tuning_best,
        }
        path = out_dir / "metrics.json"
        path.write_text(json.dumps(finite(payload), indent=2, default=str))
        return path


@REPORTERS.register("tables")
class TablesReportWriter(ReportWriter):
    """Write tidy long-format tables to ``tables/`` (see :data:`TABLE_SCHEMAS`).

    Args:
        format: ``csv``, ``parquet`` or ``both``. When run from the pipeline,
            ``reporting.table_format`` is used.
    """

    def __init__(self, format: str | None = None) -> None:
        self.format = format

    def write(self, result: RunResult, out_dir: Path) -> Path:
        table_dir = out_dir / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        fmt = self.format or result.config.reporting.table_format
        name = result.config.name
        tables: dict[str, pd.DataFrame] = {}

        rows = []
        for split, evaluation in result.evaluations.items():
            for m in evaluation.metrics.values():
                rows.append(
                    {
                        "experiment": name,
                        "split": split,
                        "metric": m.name,
                        "value": m.value,
                        "lower": m.lower,
                        "upper": m.upper,
                        "confidence": evaluation.confidence,
                        "ci_unit": evaluation.ci_unit,
                        "n_patients": evaluation.n_clusters,
                        "n": evaluation.n_samples,
                    }
                )
        tables["metrics"] = pd.DataFrame(rows, columns=list(TABLE_SCHEMAS["metrics"]))

        if result.subgroups is not None:
            tables["subgroups"] = result.subgroups.assign(experiment=name, split="test")[
                list(TABLE_SCHEMAS["subgroups"])
            ]
        if result.cohort_table is not None:
            tables["cohort"] = result.cohort_table.assign(experiment=name)[
                list(TABLE_SCHEMAS["cohort"])
            ]
        exp_frames = []
        for e in result.explanations:
            frame = None
            if isinstance(e.extra.get("global_long"), pd.DataFrame):
                frame = e.extra["global_long"]
            elif isinstance(e.attributions, pd.DataFrame) and e.scope == "global":
                frame = e.attributions.reset_index().assign(target_class="all", sample_class="all")
            if frame is not None:
                exp_frames.append(
                    frame.assign(experiment=name, method=e.method, modality=str(e.modality))
                )
        if exp_frames:
            cols = list(TABLE_SCHEMAS["explanations"])
            tables["explanations"] = pd.concat(exp_frames, ignore_index=True).reindex(columns=cols)
        if result.modality_ablation is not None and not result.modality_ablation.empty:
            tables["modality_ablation"] = result.modality_ablation.assign(experiment=name)[
                list(TABLE_SCHEMAS["modality_ablation"])
            ]
        if result.tuning_trials is not None and not result.tuning_trials.empty:
            tables["tuning_trials"] = result.tuning_trials
        if result.training_history is not None and not result.training_history.empty:
            tables["training_history"] = result.training_history

        for table_name, frame in tables.items():
            if fmt in {"csv", "both"}:
                frame.to_csv(table_dir / f"{table_name}.csv", index=False)
            if fmt in {"parquet", "both"}:
                frame.to_parquet(table_dir / f"{table_name}.parquet", index=False)
        (table_dir / "schema.json").write_text(
            json.dumps({"schema_version": SCHEMA_VERSION, "tables": TABLE_SCHEMAS}, indent=2)
        )
        return table_dir


@REPORTERS.register("markdown")
class MarkdownReportWriter(ReportWriter):
    """Write ``report.md`` with metric tables (value [CI])."""

    def write(self, result: RunResult, out_dir: Path) -> Path:
        cfg = result.config
        cohort = result.cohort_summary
        lines = [
            f"# {cfg.name}",
            "",
            f"- **Model:** `{cfg.model.name}`",
            f"- **Modalities:** {', '.join(str(m) for m in cfg.view.modalities)}",
            f"- **Cohort:** {cohort.get('n_samples')} samples / "
            f"{cohort.get('n_patients')} patients; labels {cohort.get('label_counts')}",
            f"- **Splits (patient-level, samples):** {result.split_sizes}",
            f"- **Seed:** {cfg.seed}",
            f"- **Decision threshold:** {_threshold_text(result)}",
            f"- **Calibration:** {_calibration_text(result)}",
            "",
        ]
        if cohort.get("patient_ids_assumed"):
            lines += [
                f"> **Warning:** {ASSUMED_IDS_WARNING}",
                "",
            ]
        if result.tuning_best is not None:
            best = result.tuning_best
            n = 0 if result.tuning_trials is None else len(result.tuning_trials)
            lines += [
                "## Hyper-parameter tuning (Optuna)",
                "",
                f"Best validation {cfg.tuning.metric or 'metric'} = {best['value']:.4f} "
                f"(trial {best['trial']} of {n}, {best['direction']}).",
                "",
                markdown_table(
                    pd.DataFrame(
                        {"parameter": list(best["params"]), "value": list(best["params"].values())}
                    )
                ),
                "",
            ]
        for split, evaluation in result.evaluations.items():
            ci = (
                f" ({evaluation.confidence:.0%} {evaluation.ci_unit}-level bootstrap CI)"
                if evaluation.confidence
                else ""
            )
            lines += [f"## {split.capitalize()} performance{ci}", ""]
            table = pd.DataFrame(
                {
                    "metric": list(evaluation.metrics),
                    "value": [str(m) for m in evaluation.metrics.values()],
                }
            )
            lines += [markdown_table(table), ""]

        if result.modality_ablation is not None and not result.modality_ablation.empty:
            ab = result.modality_ablation.assign(
                cell=[
                    _fmt(v, lo, hi)
                    for v, lo, hi in result.modality_ablation[
                        ["value", "lower", "upper"]
                    ].itertuples(index=False)
                ]
            )
            wide = ab.pivot_table(
                index=["available", "withheld", "n"],
                columns="metric",
                values="cell",
                aggfunc="first",
            ).reset_index()
            lines += ["## Missing-modality robustness (test)", "", markdown_table(wide), ""]

        if result.subgroups is not None and not result.subgroups.empty:
            sg = result.subgroups.assign(
                cell=[
                    _fmt(v, lo, hi)
                    for v, lo, hi in result.subgroups[["value", "lower", "upper"]].itertuples(
                        index=False
                    )
                ]
            )
            wide = sg.pivot_table(
                index=["column", "group", "n"], columns="metric", values="cell", aggfunc="first"
            ).reset_index()
            lines += ["## Subgroup analysis (test)", "", markdown_table(wide), ""]

        for exp in result.explanations:
            if isinstance(exp.attributions, pd.DataFrame) and exp.scope == "global":
                top = exp.attributions.head(20).reset_index().round(4)
                lines += [f"## {exp.method} ({exp.modality})", "", markdown_table(top), ""]

        path = out_dir / "report.md"
        path.write_text("\n".join(lines))
        return path


def _fmt(value: float, lower: float, upper: float) -> str:
    if math.isnan(value):
        return "–"
    if math.isnan(lower):
        return f"{value:.3f}"
    return f"{value:.3f} [{lower:.3f}, {upper:.3f}]"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra deps)."""
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def finite(obj: Any) -> Any:
    """Replace NaN/inf with ``None`` recursively so the JSON is strictly valid."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [finite(v) for v in obj]
    return obj


def _threshold_text(result: RunResult) -> str:
    """How hard predictions were made (fixed or validation-chosen threshold)."""
    ev = result.config.evaluation
    if result.decision_threshold is None:
        return f"{ev.threshold:.3f} (fixed)"
    return f"{result.decision_threshold:.3f} ({ev.threshold_selection}, chosen on validation)"


def _calibration_text(result: RunResult) -> str:
    """Post-hoc calibration summary (method, parameters, validation log loss before → after)."""
    cal = result.calibration
    if not cal:
        return "none"
    params = ", ".join(
        f"{k}={v:.3g}" for k, v in cal.get("params", {}).items() if isinstance(v, int | float)
    )
    before, after = cal["val_before"]["nll"], cal["val_after"]["nll"]
    return (
        f"{cal['method']}{f' ({params})' if params else ''}, fitted on validation; "
        f"val log loss {before:.3f} → {after:.3f}"
    )
