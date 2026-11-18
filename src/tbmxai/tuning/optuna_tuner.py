"""Hyper-parameter optimisation with Optuna.

Any value in an :class:`~tbmxai.core.config.ExperimentConfig` can be searched,
addressed by a dotted path (``model.params.max_depth``, ``training.optimizer.lr``,
``model.params.fusion.name`` …). Each trial:

1. samples the search space and writes the values into a copy of the config;
2. builds and fits the model on the **train** split (neural models use the
   validation split for early stopping and report every epoch to Optuna, so
   unpromising trials are pruned);
3. returns the configured metric on the **validation** split.

The test split is never seen during tuning.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tbmxai.core.config import ExperimentConfig, SearchParam, TuningConfig
from tbmxai.core.utils import get_logger, seed_everything
from tbmxai.data.splits import DataSplits
from tbmxai.evaluation.metrics import METRICS
from tbmxai.models.base import MODELS

logger = get_logger(__name__)


def set_by_path(tree: dict[str, Any], path: str, value: Any) -> None:
    """Set ``tree[a][b][c] = value`` for ``path == "a.b.c"`` (creating dicts as needed)."""
    keys = path.split(".")
    node = tree
    for key in keys[:-1]:
        nxt = node.get(key)
        if nxt is None:
            nxt = node[key] = {}
        if not isinstance(nxt, dict):
            raise KeyError(f"Cannot descend into non-mapping at '{key}' in path '{path}'.")
        node = nxt
    node[keys[-1]] = value


@dataclass
class TuningResult:
    """Outcome of a tuning study."""

    best_params: dict[str, Any]
    best_value: float
    best_trial: int
    best_config: ExperimentConfig
    trials: pd.DataFrame
    direction: str


class HyperparameterTuner:
    """Run an Optuna study over an experiment's search space.

    Args:
        config: Experiment config with a populated ``tuning`` section.
        run_dir: Run directory; artefacts go to ``<run_dir>/tuning/``.
        keep_trial_artifacts: Keep per-trial checkpoints (disk-heavy for neural models).
    """

    def __init__(
        self, config: ExperimentConfig, run_dir: str | Path, keep_trial_artifacts: bool = False
    ) -> None:
        if not config.tuning.search_space:
            raise ValueError("tuning.search_space is empty.")
        self.config = config
        self.tcfg: TuningConfig = config.tuning
        self.dir = Path(run_dir).expanduser() / "tuning"
        self.keep_trial_artifacts = keep_trial_artifacts
        self.metric = METRICS.build(self.tcfg.metric)
        self.direction = self._direction()

    # ---------------------------------------------------------------- optuna

    def _direction(self) -> str:
        if self.tcfg.direction != "auto":
            return self.tcfg.direction
        return "maximize" if self.metric.higher_is_better else "minimize"

    def _sampler(self) -> Any:
        import optuna

        params = {"seed": self.config.seed, **self.tcfg.sampler_params}
        samplers = {
            "tpe": optuna.samplers.TPESampler,
            "random": optuna.samplers.RandomSampler,
            "qmc": optuna.samplers.QMCSampler,
            "gp": optuna.samplers.GPSampler,
            "cmaes": optuna.samplers.CmaEsSampler,
        }
        return samplers[self.tcfg.sampler](**params)

    def _pruner(self) -> Any:
        import optuna

        pruners = {
            "none": optuna.pruners.NopPruner,
            "median": optuna.pruners.MedianPruner,
            "hyperband": optuna.pruners.HyperbandPruner,
            "successive_halving": optuna.pruners.SuccessiveHalvingPruner,
        }
        return pruners[self.tcfg.pruner](**self.tcfg.pruner_params)

    @staticmethod
    def suggest(trial: Any, name: str, spec: SearchParam) -> Any:
        """Sample one parameter from ``trial``."""
        if spec.type == "categorical":
            return trial.suggest_categorical(name, spec.choices)
        if spec.type == "int":
            step = int(spec.step) if spec.step else 1
            return trial.suggest_int(name, int(spec.low), int(spec.high), step=step, log=spec.log)  # type: ignore[arg-type]
        return trial.suggest_float(name, spec.low, spec.high, step=spec.step, log=spec.log)  # type: ignore[arg-type]

    def config_with(self, params: dict[str, Any]) -> ExperimentConfig:
        """Copy of the experiment config with ``params`` (dotted paths) applied; tuning disabled."""
        tree = self.config.model_dump(mode="json")
        tree["tuning"]["enabled"] = False
        for path, value in params.items():
            set_by_path(tree, path, value)
        return ExperimentConfig.model_validate(tree)

    # ------------------------------------------------------------- objective

    def _objective(self, trial: Any, splits: DataSplits) -> float:
        import optuna

        params = {
            name: self.suggest(trial, name, spec) for name, spec in self.tcfg.search_space.items()
        }
        cfg = self.config_with(params)
        seed_everything(cfg.seed)
        trial_dir = self.dir / "trials" / f"trial_{trial.number:04d}"
        model = MODELS.build(cfg.model.name, **cfg.model.params)
        model.setup(trial_dir, cfg.training)

        monitor = f"val_{self.tcfg.metric}"

        def report(epoch: int, logs: dict[str, float]) -> None:
            value = logs.get(monitor)
            if value is None or not math.isfinite(value):
                return
            trial.report(value, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch} ({monitor}={value:.4f})")

        if hasattr(model, "epoch_callback"):
            model.epoch_callback = report
        try:
            model.fit(splits.train, splits.val)
            preds = model.predict(splits.val)  # type: ignore[arg-type]
            y_pred = preds.predicted_labels(cfg.evaluation.threshold)
            value = float(self.metric(preds.y_true, preds.y_prob, y_pred))
        finally:
            if not self.keep_trial_artifacts:
                shutil.rmtree(trial_dir / "checkpoints", ignore_errors=True)
        logger.info("Trial %d: %s=%.4f %s", trial.number, self.tcfg.metric, value, params)
        return value if math.isfinite(value) else float("nan")

    # ------------------------------------------------------------------- run

    def tune(self, splits: DataSplits) -> TuningResult:
        """Run the study and write ``trials.csv``, ``best_params.json``, ``best_config.yaml``."""
        import optuna

        if splits.val is None or len(splits.val) == 0:
            raise ValueError("Tuning needs a validation split (set split.params.val_size > 0).")
        self.dir.mkdir(parents=True, exist_ok=True)
        storage = self.tcfg.storage or f"sqlite:///{(self.dir / 'optuna.db').resolve()}"
        study = optuna.create_study(
            study_name=self.tcfg.study_name or self.config.name,
            storage=storage,
            load_if_exists=True,
            direction=self.direction,
            sampler=self._sampler(),
            pruner=self._pruner(),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            lambda t: self._objective(t, splits),
            n_trials=self.tcfg.n_trials,
            timeout=None if self.tcfg.timeout_minutes is None else self.tcfg.timeout_minutes * 60,
            n_jobs=self.tcfg.n_jobs,
            catch=(RuntimeError, ValueError),
            gc_after_trial=True,
        )
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            raise RuntimeError("No tuning trial completed; check the logs for failures.")
        best = study.best_trial
        best_cfg = self.config_with(best.params)

        trials = study.trials_dataframe()
        for col in trials.columns:
            if pd.api.types.is_timedelta64_dtype(trials[col]):
                trials[col] = trials[col].dt.total_seconds()
        trials = trials.rename(columns=lambda c: c.replace("params_", "param:"))
        trials.insert(0, "experiment", self.config.name)
        trials.insert(1, "metric", self.tcfg.metric)
        trials.to_csv(self.dir / "trials.csv", index=False)
        summary = {
            "metric": self.tcfg.metric,
            "direction": self.direction,
            "best_value": best.value,
            "best_trial": best.number,
            "best_params": best.params,
            "n_trials": len(study.trials),
            "states": pd.Series([t.state.name for t in study.trials]).value_counts().to_dict(),
            "sampler": self.tcfg.sampler,
            "pruner": self.tcfg.pruner,
            "storage": storage,
        }
        (self.dir / "best_params.json").write_text(json.dumps(summary, indent=2, default=_json))
        best_cfg.to_yaml(self.dir / "best_config.yaml")
        logger.info(
            "Best trial %d: %s=%.4f %s", best.number, self.tcfg.metric, best.value, best.params
        )
        return TuningResult(
            best.params, float(best.value), best.number, best_cfg, trials, self.direction
        )


def _json(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)
