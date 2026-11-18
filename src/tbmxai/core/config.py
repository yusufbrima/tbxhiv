"""Experiment configuration schema.

A single :class:`ExperimentConfig` describes a full run (data → model →
training → evaluation → reporting). It validates on load and round-trips to
YAML, so the same file drives both the CLI and the Python API.

Data and outputs may live anywhere on disk (the package never assumes the
repository layout). YAML strings support interpolation:

* ``${NAME}`` / ``${NAME:-default}`` — a key from the ``paths:`` section, else
  an environment variable, else ``default``;
* a leading ``~`` is expanded to the home directory.

Example::

    paths:
      data_root: ${TBMXAI_DATA_ROOT:-/mnt/data/tb}
      output_root: ${TBMXAI_OUTPUT_ROOT:-~/tbmxai_runs}
    output_dir: ${output_root}
    data:
      sources:
        - name: tabular_file
          params: {path: ${data_root}/ehr.csv}
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tbmxai.core.types import Modality

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def interpolate(value: Any, variables: Mapping[str, str]) -> Any:
    """Recursively expand ``${NAME[:-default]}`` and leading ``~`` in strings.

    Lookup order: ``variables`` → environment → default. Unresolved names
    without a default raise ``KeyError`` (fail fast rather than read a wrong path).
    """
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in variables:
                return str(variables[name])
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise KeyError(f"Unresolved config variable '${{{name}}}' (not in paths: or env).")

        expanded = _VAR.sub(repl, value)
        return os.path.expanduser(expanded) if expanded.startswith("~") else expanded
    if isinstance(value, dict):
        return {k: interpolate(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v, variables) for v in value]
    return value


def resolve_raw_config(
    raw: dict[str, Any], overrides: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Interpolate a raw (YAML-loaded) config dict.

    ``paths`` entries are resolved first (against env + ``overrides``), then
    used as variables for the rest of the document.
    """
    raw = dict(raw)
    paths = {**dict(raw.get("paths") or {}), **dict(overrides or {})}
    paths = {k: interpolate(v, {}) for k, v in paths.items()}
    raw["paths"] = paths
    return interpolate(raw, paths)


class _Strict(BaseModel):
    """Base for config sections: unknown keys are errors (catches typos)."""

    model_config = ConfigDict(extra="forbid")


class ComponentConfig(_Strict):
    """A registry entry plus constructor kwargs, e.g. ``{name: xgboost, params: {...}}``."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------- data


class DataConfig(_Strict):
    """Which data to ingest and how to link it (at patient level).

    Either point to a dataset in the standard format (``dataset``; sources, id /
    link columns and class names come from its ``dataset.yaml``) or list the
    ``sources`` explicitly.
    """

    dataset: Path | None = Field(
        default=None, description="Folder with dataset.yaml (tbmxai standard dataset format)."
    )
    use: list[Literal["tabular", "images", "reports"]] | None = Field(
        default=None,
        description="With `dataset`: which optional files to load (default: all present). "
        "Loading images also makes one sample per image, even for EHR-only models.",
    )
    sources: list[ComponentConfig] = Field(default_factory=list)
    patient_column: str = Field(
        default="patient_id",
        description="Canonical patient identifier; every source key is renamed to it.",
    )
    label_column: str | None = Field(
        default=None, description="Target column (default: 'label', or the dataset's only label)."
    )
    link_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Extra join keys shared by several sources (e.g. study_id for per-study EHR "
            "features + per-image manifests). Splits remain patient-level."
        ),
    )
    join: Literal["inner", "left", "outer"] = "inner"
    class_names: list[str] | None = Field(
        default=None,
        description="Ordered label values; index = encoded class. Inferred if omitted.",
    )

    @field_validator("dataset", mode="before")
    @classmethod
    def _empty_dataset(cls, value: Any) -> Any:
        return None if value in ("", None) else value

    @model_validator(mode="after")
    def _sources_or_dataset(self) -> DataConfig:
        if self.dataset is None and not self.sources:
            raise ValueError("data needs `sources` or a `dataset` folder")
        return self


class ViewConfig(_Strict):
    """Which modalities to harmonise into the model input."""

    modalities: list[Modality] = Field(min_length=1)
    metadata_columns: list[str] = Field(
        default_factory=list,
        description="Cohort columns carried along for splitting/subgroups (not used as inputs).",
    )
    tabular_columns: list[str] | None = Field(
        default=None, description="Restrict tabular features; default = all tabular columns."
    )
    tabular_exclude: list[str] = Field(
        default_factory=list,
        description="Glob patterns (fnmatch) of tabular features to leave out, applied after "
        "`tabular_columns`, e.g. `[micro_*, med_*]` for a feature-group ablation.",
    )
    missing_modalities: Literal["drop", "keep"] = Field(
        default="drop",
        description="Samples lacking one of the selected modalities: `drop` them (complete-case) "
        "or `keep` them with an availability mask (needs a model that handles missing "
        "modalities, e.g. `neural`). Samples with none of the selected modalities are "
        "always dropped.",
    )
    index_policy: Literal["all", "first", "last", "random"] = Field(
        default="all",
        description="Prediction timepoints per patient: every sample, or one index sample "
        "(first / last / random by `index_time_column`).",
    )
    index_time_column: str | None = Field(
        default=None, description="Cohort column ordering a patient's samples (index timepoint)."
    )

    @model_validator(mode="after")
    def _check_index(self) -> ViewConfig:
        if self.index_policy in ("first", "last") and not self.index_time_column:
            raise ValueError("index_policy first/last needs index_time_column")
        return self


# ----------------------------------------------------------------- training


class OptimizerConfig(_Strict):
    """Optimiser settings."""

    name: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-2, ge=0)
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = Field(default=0.9, ge=0, description="SGD only.")
    nesterov: bool = True
    encoder_lr_mult: float = Field(
        default=1.0,
        gt=0,
        description="LR multiplier for encoder parameters (e.g. 0.1 for pretrained backbones).",
    )


class SchedulerConfig(_Strict):
    """Learning-rate schedule with optional linear warm-up.

    ``cosine`` / ``linear`` decay per step after warm-up down to
    ``lr * min_lr_ratio``; ``step`` multiplies by ``gamma`` every
    ``step_size`` epochs; ``plateau`` multiplies by ``gamma`` after
    ``patience`` epochs without improvement of the monitored metric.
    """

    name: Literal["cosine", "linear", "step", "plateau", "constant"] = "cosine"
    warmup_epochs: float = Field(default=0.0, ge=0, description="Fractional epochs allowed.")
    warmup_start_factor: float = Field(default=0.01, gt=0, le=1)
    min_lr_ratio: float = Field(default=0.0, ge=0, le=1)
    step_size: int = Field(default=10, ge=1)
    gamma: float = Field(default=0.1, gt=0, le=1)
    patience: int = Field(default=3, ge=0, description="plateau only.")


class EarlyStoppingConfig(_Strict):
    """Stop when ``monitor`` stops improving."""

    enabled: bool = True
    monitor: str = Field(
        default="val_loss", description="val_loss or val_<metric>, e.g. val_auroc."
    )
    mode: Literal["min", "max", "auto"] = "auto"
    patience: int = Field(default=10, ge=1)
    min_delta: float = Field(default=0.0, ge=0)


class CheckpointConfig(_Strict):
    """Model checkpointing."""

    enabled: bool = True
    monitor: str | None = Field(default=None, description="Default: early_stopping.monitor.")
    mode: Literal["min", "max", "auto"] = "auto"
    save_top_k: int = Field(default=1, ge=0)
    save_last: bool = True
    restore_best: bool = Field(default=True, description="Load best weights after training.")


class TrainingConfig(_Strict):
    """Configuration for neural-network training (ignored by classical models)."""

    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=32, ge=1)
    eval_batch_size: int | None = None
    num_workers: int = Field(default=4, ge=0)
    device: str = Field(default="auto", description="auto | cpu | cuda | cuda:1 | mps")
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "32"
    accumulate_grad_batches: int = Field(default=1, ge=1)
    grad_clip_norm: float | None = Field(default=1.0, gt=0)
    class_weights: Literal["balanced"] | list[float] | None = "balanced"
    label_smoothing: float = Field(default=0.0, ge=0, lt=1)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    resume_from: Path | None = Field(default=None, description="Checkpoint to resume from.")
    deterministic: bool = Field(default=False, description="cuDNN deterministic algorithms.")
    log_every_n_steps: int = Field(default=50, ge=1)


# --------------------------------------------------------------- evaluation


class EvaluationConfig(_Strict):
    """Metric suite and confidence-interval settings."""

    metrics: list[str] | None = Field(default=None, description="None = default suite.")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    threshold_selection: Literal["fixed", "youden", "sensitivity", "specificity"] = Field(
        default="fixed",
        description="Binary tasks: choose the decision threshold on the VALIDATION predictions "
        "(youden = max sensitivity + specificity − 1; sensitivity = highest threshold reaching "
        "target_sensitivity; specificity = lowest threshold reaching target_specificity) and "
        "apply it to the test split. 'fixed' uses `threshold`.",
    )
    target_sensitivity: float = Field(default=0.9, gt=0.0, le=1.0)
    target_specificity: float = Field(default=0.7, gt=0.0, le=1.0)
    calibration: Literal["none", "temperature", "platt", "isotonic"] = Field(
        default="none",
        description="Post-hoc probability calibration fitted on the VALIDATION predictions and "
        "applied to every split (before threshold selection). temperature: binary or "
        "multiclass; platt / isotonic: binary only.",
    )
    n_bootstrap: int = Field(default=1000, ge=0, description="0 disables CIs.")
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    bootstrap_unit: Literal["patient", "sample"] = Field(
        default="patient",
        description="Resample patients (cluster bootstrap; correct with multiple samples per "
        "patient) or individual samples.",
    )
    modality_ablation: bool = Field(
        default=False,
        description="Also evaluate the test split with each subset of modalities withheld "
        "(models that support missing modalities only).",
    )
    splits: list[Literal["train", "val", "test"]] = Field(default_factory=lambda: ["val", "test"])


class SubgroupConfig(_Strict):
    """Subgroup analysis settings (evaluated on the test split)."""

    columns: list[str] = Field(default_factory=list)
    min_size: int = Field(default=20, ge=1)
    bins: dict[str, list[float]] = Field(
        default_factory=dict, description="Explicit bin edges for numeric columns, e.g. age."
    )


class ExplainSamplesConfig(_Strict):
    """Which samples local explanations (saliency, SHAP) are computed for.

    Samples are chosen from **ground truth only** (never from model outputs), so
    every model and modality is explained on the same patients: ``n_per_class``
    samples of **each** class, one per patient, among samples that have all
    selected modalities, ordered by a seeded hash of the patient id. Global
    methods (e.g. permutation importance) use the whole split.
    """

    split: Literal["train", "val", "test"] = "test"
    n_per_class: int = Field(default=8, ge=1)
    seed: int | None = Field(default=None, description="Default: the experiment seed.")
    file: Path | None = Field(
        default=None, description="CSV with a `sample_id` column fixing the samples explicitly."
    )


class ExplainConfig(_Strict):
    """Explainability methods to run on the fitted model."""

    methods: list[ComponentConfig] = Field(default_factory=list)
    samples: ExplainSamplesConfig = Field(default_factory=ExplainSamplesConfig)
    target_classes: Literal["all", "positive"] = Field(
        default="all",
        description="Explain the output of every class (default) or only the positive class.",
    )


class ProtocolConfig(_Strict):
    """Shared experimental protocol (see ``tbmxai protocol``).

    Fixes the sample set, the patient-level split and the explanation samples so
    that every experiment referencing it is trained, tested, analysed by subgroup
    and explained on exactly the same samples.
    """

    path: Path | None = Field(default=None, description="Protocol directory (samples.csv).")
    use_split: bool = Field(
        default=True, description="Use the protocol's train/val/test assignment."
    )
    use_xai_samples: bool = Field(
        default=True, description="Use the protocol's explanation samples."
    )

    @field_validator("path", mode="before")
    @classmethod
    def _empty_is_none(cls, value: Any) -> Any:
        return None if value in ("", None) else value


class ReportingConfig(_Strict):
    """Report writers to run at the end of the pipeline."""

    writers: list[str] = Field(default_factory=lambda: ["json", "tables", "markdown", "html"])
    save_predictions: bool = True
    table_format: Literal["csv", "parquet", "both"] = "csv"


# ------------------------------------------------------------------- tuning


class SearchParam(_Strict):
    """One dimension of the hyper-parameter search space.

    Examples::

        {type: float, low: 1.0e-5, high: 1.0e-3, log: true}
        {type: int, low: 2, high: 8}
        {type: categorical, choices: [concat, gated, attention]}
    """

    type: Literal["float", "int", "categorical"]
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def _check(self) -> SearchParam:
        if self.type == "categorical":
            if not self.choices:
                raise ValueError("categorical params need non-empty `choices`")
        elif self.low is None or self.high is None or self.low > self.high:
            raise ValueError(f"{self.type} params need low <= high")
        return self


class TuningConfig(_Strict):
    """Hyper-parameter optimisation with Optuna.

    Every trial trains on the **train** split and is scored on the **validation**
    split; the test split is used once, after refitting with the best parameters.
    ``search_space`` keys are dotted paths into this experiment config, e.g.
    ``model.params.max_depth``, ``training.optimizer.lr``,
    ``model.params.fusion.name``, ``model.params.encoders.imaging.params.backbone``.
    """

    enabled: bool = False
    n_trials: int = Field(default=30, ge=1)
    timeout_minutes: float | None = Field(default=None, gt=0)
    metric: str = Field(default="auroc", description="Any registered metric, on the val split.")
    direction: Literal["maximize", "minimize", "auto"] = "auto"
    sampler: Literal["tpe", "random", "qmc", "gp", "cmaes"] = "tpe"
    sampler_params: dict[str, Any] = Field(default_factory=dict)
    pruner: Literal["none", "median", "hyperband", "successive_halving"] = "median"
    pruner_params: dict[str, Any] = Field(default_factory=dict)
    storage: str | None = Field(
        default=None,
        description="Optuna storage URL; default sqlite in the run dir (resumable). "
        "Use a shared DB URL for parallel workers.",
    )
    study_name: str | None = None
    search_space: dict[str, SearchParam] = Field(default_factory=dict)
    n_jobs: int = Field(default=1, ge=1, description="Parallel trials in this process.")

    @model_validator(mode="after")
    def _check(self) -> TuningConfig:
        if self.enabled and not self.search_space:
            raise ValueError("tuning.enabled requires a non-empty search_space")
        return self


# --------------------------------------------------------------- top level


class InterpolatedConfig(_Strict):
    """Base for top-level YAML configs: ``paths:`` section + ``${VAR}`` interpolation."""

    paths: dict[str, str] = Field(
        default_factory=dict,
        description="Named roots (data_root, output_root, …) usable as ${name} elsewhere.",
    )

    @model_validator(mode="before")
    @classmethod
    def _interpolate(cls, data: Any) -> Any:
        return resolve_raw_config(data) if isinstance(data, dict) else data

    @classmethod
    def from_yaml(cls, path: str | Path, path_overrides: Mapping[str, str] | None = None) -> Self:
        """Load, interpolate and validate a YAML config file.

        Args:
            path: YAML file (anywhere on disk).
            path_overrides: Values that override entries of the ``paths:``
                section (e.g. ``{"data_root": "/scratch/tb"}``).
        """
        with Path(path).expanduser().open() as fh:
            raw = yaml.safe_load(fh) or {}
        if path_overrides:
            raw["paths"] = {**(raw.get("paths") or {}), **path_overrides}
        return cls.model_validate(raw)  # interpolation happens in the validator

    def to_yaml(self, path: str | Path) -> None:
        """Write this (fully resolved) config to YAML."""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)


class ExperimentConfig(InterpolatedConfig):
    """Top-level configuration for one pipeline run."""

    name: str
    seed: int = 42
    output_dir: Path = Path("outputs")
    data: DataConfig
    view: ViewConfig
    split: ComponentConfig = Field(default_factory=lambda: ComponentConfig(name="holdout"))
    model: ComponentConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    subgroups: SubgroupConfig = Field(default_factory=SubgroupConfig)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)

    @field_validator("name")
    @classmethod
    def _name_is_path_safe(cls, value: str) -> str:
        if not value or any(ch in value for ch in "/\\"):
            raise ValueError("name must be non-empty and contain no path separators")
        return value
