"""High-level orchestration: config → data → model → evaluation → reports.

Each stage is a public method so it can be called (or overridden) on its own::

    pipe = Pipeline.from_yaml("experiment.yaml", path_overrides={"data_root": "/data/tb"})
    cohort = pipe.ingest()
    view = pipe.harmonize(cohort)
    splits = pipe.split(view)            # patient-level, leakage-checked
    ...
    result = pipe.run()                  # or everything at once
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd

from tbmxai.core.config import ExperimentConfig
from tbmxai.core.types import Predictions
from tbmxai.core.utils import get_logger, seed_everything
from tbmxai.data.harmonize import Harmonizer, MultimodalView
from tbmxai.data.ingest import Cohort, Ingestor
from tbmxai.data.quality import DataQualityReport
from tbmxai.data.splits import SPLITTERS, DataSplits
from tbmxai.evaluation.calibration import Calibrator
from tbmxai.evaluation.evaluator import Evaluator
from tbmxai.evaluation.metrics import METRICS
from tbmxai.explain.base import EXPLAINERS, Explanation
from tbmxai.explain.selection import select_explanation_samples
from tbmxai.models.base import MODELS, Classifier
from tbmxai.protocol import Protocol, ProtocolError
from tbmxai.reporting.artifacts import save_explanations
from tbmxai.reporting.base import REPORTERS, RunResult
from tbmxai.reporting.provenance import collect_run_info, write_manifest
from tbmxai.reporting.writers import finite
from tbmxai.subgroups.analysis import SubgroupAnalyzer

logger = get_logger(__name__)


class Pipeline:
    """End-to-end experiment runner driven by an :class:`ExperimentConfig`.

    Args:
        config: Validated experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.calibrator: Calibrator | None = None

    @classmethod
    def from_yaml(cls, path: str | Path, path_overrides: Mapping[str, str] | None = None) -> Self:
        """Load a pipeline from a YAML config (anywhere on disk).

        Args:
            path: Config file.
            path_overrides: Override ``paths:`` entries, e.g. ``{"data_root": ...}``.
        """
        return cls(ExperimentConfig.from_yaml(path, path_overrides))

    # ----------------------------------------------------------------- stages

    def ingest(self) -> Cohort:
        """Load and link all configured data sources at patient level."""
        return Ingestor.from_config(self.config.data).run()

    def harmonize(self, cohort: Cohort) -> MultimodalView:
        """Select the configured modalities into a model-ready view."""
        return Harmonizer.from_config(self.config.view, seed=self.config.seed).build(cohort)

    def split(self, view: MultimodalView) -> DataSplits:
        """Partition the view at patient level; raises on any patient leakage."""
        splitter = SPLITTERS.build(self.config.split.name, **self.config.split.params)
        splits = splitter.split(view, seed=self.config.seed)
        splits.check_patient_disjoint()
        return splits

    def build_model(self) -> Classifier:
        """Instantiate the configured (unfitted) model."""
        return MODELS.build(self.config.model.name, **self.config.model.params)

    def build_evaluator(self) -> Evaluator:
        """Instantiate the configured evaluator."""
        return Evaluator.from_config(self.config.evaluation, seed=self.config.seed)

    def explanation_samples(
        self, view: MultimodalView, protocol: Protocol | None = None
    ) -> MultimodalView:
        """The shared explanation set within ``view`` (ground-truth based, every class).

        Priority: protocol flags → ``explain.samples.file`` → deterministic
        class-stratified selection (:func:`select_explanation_samples`).
        """
        xcfg = self.config.explain.samples
        ids = view.sample_ids.astype(str)
        if protocol is not None and self.config.protocol.use_xai_samples:
            wanted = protocol.xai_ids()
        elif xcfg.file is not None:
            wanted = set(
                pd.read_csv(Path(xcfg.file).expanduser(), dtype={"sample_id": str})["sample_id"]
            )
        else:
            seed = self.config.seed if xcfg.seed is None else xcfg.seed
            return view.subset(select_explanation_samples(view, xcfg.n_per_class, seed))
        pos = np.flatnonzero(np.isin(ids, list(wanted)))
        if len(pos) != len(wanted):
            raise ProtocolError(
                f"{len(wanted) - len(pos)} explanation samples are not in the '{xcfg.split}' split."
            )
        return view.subset(pos)

    def explain(
        self, model: Classifier, view: MultimodalView, xai_view: MultimodalView | None = None
    ) -> list[Explanation]:
        """Run every configured explainer.

        Explainers with ``uses_xai_samples`` (saliency) get the shared explanation
        set ``xai_view``; dataset-level ones (permutation importance, SHAP) get
        the whole split ``view``. Both are identical for every model under a protocol.
        """
        explanations = []
        target = self.config.explain.target_classes
        for spec in self.config.explain.methods:
            cls = EXPLAINERS.get(spec.name)
            params = dict(spec.params)
            if getattr(cls, "uses_xai_samples", False):
                params.setdefault("target_classes", target)
            explainer = cls(model=model, **params)
            data = xai_view if (cls.uses_xai_samples and xai_view is not None) else view
            explanations.append(explainer.explain(data))
        return explanations

    # -------------------------------------------------------------------- run

    def run(self, output_dir: str | Path | None = None) -> RunResult:
        """Execute every stage and write reports.

        Args:
            output_dir: Override for the run directory. Defaults to
                ``<config.output_dir>/<name>/<timestamp>``.

        Returns:
            The :class:`RunResult` (also rendered by the configured writers).
        """
        cfg = self.config
        seed_everything(cfg.seed)
        out = Path(output_dir).expanduser() if output_dir else self._default_output_dir()
        out.mkdir(parents=True, exist_ok=True)
        cfg.to_yaml(out / "config.yaml")
        (out / "run_info.json").write_text(json.dumps(collect_run_info(cfg), indent=2, default=str))
        logger.info("Run '%s' → %s", cfg.name, out)

        cohort = self.ingest()
        view = self.harmonize(cohort)
        protocol = None
        if cfg.protocol.path is not None:
            protocol = Protocol.load(cfg.protocol.path)
            view = protocol.restrict(view)
            (out / "protocol.json").write_text(
                json.dumps(
                    {
                        **protocol.info,
                        "path": str(cfg.protocol.path),
                        "fingerprint": protocol.fingerprint,
                        "use_split": cfg.protocol.use_split,
                        "use_xai_samples": cfg.protocol.use_xai_samples,
                    },
                    indent=2,
                    default=str,
                )
            )
            logger.info(
                "Protocol %s: %d samples (fingerprint %s…)",
                cfg.protocol.path,
                len(view),
                protocol.fingerprint[:12],
            )
        DataQualityReport.from_view(view).write(out)
        splits = protocol.splits(view) if protocol and cfg.protocol.use_split else self.split(view)
        splits.assignments().to_csv(out / "splits.csv", index=False)
        logger.info(
            "Patient-level split sizes (samples): %s", {k: len(v) for k, v in splits.items()}
        )

        tuning = None
        if cfg.tuning.enabled:
            from tbmxai.tuning import HyperparameterTuner

            tuning = HyperparameterTuner(cfg, out).tune(splits)
            # Refit with the best parameters (train + val for early stopping), test once.
            self.config = cfg = tuning.best_config
            cfg.to_yaml(out / "config_best.yaml")

        model = self.build_model()
        model.setup(out, cfg.training)
        model.fit(splits.train, splits.val)
        model.save(out / "model.pkl")

        result = RunResult(
            config=cfg,
            output_dir=out,
            cohort_summary=cohort.summary(),
            split_sizes={k: len(v) for k, v in splits.items()},
            cohort_table=_cohort_table(view, splits),
            training_history=getattr(getattr(model, "training_result_", None), "history", None),
            tuning_trials=None if tuning is None else tuning.trials,
            tuning_best=None
            if tuning is None
            else {
                "params": tuning.best_params,
                "value": tuning.best_value,
                "trial": tuning.best_trial,
                "direction": tuning.direction,
            },
        )
        evaluator = self.build_evaluator()
        available = dict(splits.items())
        result.calibration = self._fit_calibration(model, splits, out)
        result.decision_threshold = self._select_threshold(model, splits, evaluator, out)
        for split_name in cfg.evaluation.splits:
            if split_name not in available:
                continue
            preds = self.predict(model, available[split_name])
            result.predictions[split_name] = preds
            result.evaluations[split_name] = evaluator.evaluate(preds)
            if cfg.reporting.save_predictions:
                preds.to_frame().to_csv(out / f"predictions_{split_name}.csv", index=False)

        if cfg.evaluation.modality_ablation:
            result.modality_ablation = self.modality_ablation(model, splits.test, evaluator)

        if cfg.subgroups.columns:
            test_preds = result.predictions.get("test") or self.predict(model, splits.test)
            analyzer = SubgroupAnalyzer.from_config(cfg.subgroups, evaluator)
            result.subgroups = analyzer.analyze(test_preds)

        if cfg.explain.methods:
            explain_split = dict(splits.items())[cfg.explain.samples.split]
            xai_view = self.explanation_samples(explain_split, protocol)
            pd.DataFrame(
                {
                    "sample_id": xai_view.sample_ids,
                    "patient_id": xai_view.patient_ids,
                    "y_true": xai_view.y,
                    "class_name": [xai_view.class_names[k] for k in xai_view.y],
                }
            ).to_csv(out / "xai_samples.csv", index=False)
            result.explanations = self.explain(model, explain_split, xai_view)
            save_explanations(result.explanations, out)

        for writer_name in cfg.reporting.writers:
            path = REPORTERS.build(writer_name).write(result, out)
            logger.info("Wrote %s", path)
        (out / "cohort_summary.json").write_text(
            json.dumps(finite(result.cohort_summary), indent=2)
        )
        write_manifest(out)
        return result

    def predict(self, model: Classifier, view: MultimodalView) -> Predictions:
        """``model.predict`` followed by the fitted post-hoc calibrator (if any)."""
        preds = model.predict(view)
        if self.calibrator is not None:
            preds.y_prob = self.calibrator.transform(preds.y_prob)
        return preds

    def _fit_calibration(
        self, model: Classifier, splits: DataSplits, out: Path
    ) -> dict[str, Any] | None:
        """Fit the configured post-hoc calibrator on validation predictions.

        Saved to ``calibration.json`` with the validation log loss, Brier score and
        ECE before and after (in-sample on validation, so optimistic).
        """
        self.calibrator = None
        method = self.config.evaluation.calibration
        if method == "none":
            return None
        if splits.val is None or len(splits.val) == 0:
            logger.warning("calibration=%s needs a validation split; skipped", method)
            return None
        from tbmxai.evaluation.calibration import CALIBRATORS, negative_log_likelihood

        val = model.predict(splits.val)
        calibrator = CALIBRATORS.build(method)
        try:
            calibrator.fit(val.y_true, val.y_prob)
        except ValueError as err:
            logger.warning("calibration=%s skipped: %s", method, err)
            return None
        calibrated = calibrator.transform(val.y_prob)
        brier, ece = METRICS.build("brier"), METRICS.build("ece")

        def fit_stats(prob: np.ndarray) -> dict[str, float]:
            pred = prob.argmax(axis=1)
            return {
                "nll": negative_log_likelihood(val.y_true, prob),
                "brier": brier(val.y_true, prob, pred),
                "ece": ece(val.y_true, prob, pred),
            }

        info = {
            "method": method,
            "params": calibrator.params(),
            "fitted_on": "val",
            "n_val": len(val),
            "val_before": fit_stats(val.y_prob),
            "val_after": fit_stats(calibrated),
        }
        (out / "calibration.json").write_text(json.dumps(info, indent=2))
        self.calibrator = calibrator
        logger.info(
            "Calibration %s %s: val log loss %.4f → %.4f",
            method,
            {k: v for k, v in info["params"].items() if not isinstance(v, list)},
            info["val_before"]["nll"],
            info["val_after"]["nll"],
        )
        return info

    def _select_threshold(
        self, model: Classifier, splits: DataSplits, evaluator: Evaluator, out: Path
    ) -> float | None:
        """Choose the binary decision threshold on validation predictions (if configured).

        The threshold is fitted on the validation split only, then used for every
        split's hard predictions (test, subgroups, ablation) and saved to
        ``threshold.json`` so that ``tbmxai compare`` applies the same operating point.
        """
        ev = self.config.evaluation
        if ev.threshold_selection == "fixed":
            return None
        if splits.val is None or len(splits.val.class_names) != 2:
            logger.warning(
                "threshold_selection=%s needs a validation split and a binary task; using %.2f",
                ev.threshold_selection,
                ev.threshold,
            )
            return None
        from tbmxai.evaluation.threshold import select_threshold

        val = self.predict(model, splits.val)
        thr = select_threshold(
            val.y_true,
            val.y_prob[:, 1],
            ev.threshold_selection,
            ev.target_sensitivity,
            ev.target_specificity,
        )
        evaluator.threshold = thr
        info = {
            "threshold": thr,
            "method": ev.threshold_selection,
            "target_sensitivity": ev.target_sensitivity,
            "target_specificity": ev.target_specificity,
            "calibrated": self.calibrator is not None,
            "fitted_on": "val",
            "n_val": len(val),
        }
        (out / "threshold.json").write_text(json.dumps(info, indent=2))
        logger.info("Decision threshold %.4f (%s on validation)", thr, ev.threshold_selection)
        return thr

    def modality_ablation(
        self, model: Classifier, view: MultimodalView, evaluator: Evaluator
    ) -> pd.DataFrame | None:
        """Test-set performance with every non-empty subset of modalities available.

        For each subset ``S`` the other modalities are withheld for all samples
        (:meth:`MultimodalView.withhold`), and the model is evaluated on samples
        that naturally have at least one modality of ``S``. Requires a model with
        missing-modality support and ≥ 2 modalities.
        """
        mods = sorted(view.modalities, key=str)
        if len(mods) < 2 or not model.supports_missing_modalities:
            logger.info(
                "Modality ablation skipped (needs ≥2 modalities and a missing-capable model)"
            )
            return None
        rows = []
        for k in range(len(mods), 0, -1):
            for kept in combinations(mods, k):
                withheld = [m for m in mods if m not in kept]
                has_any = np.logical_or.reduce([view.availability[m] for m in kept])
                sub = view.subset(np.flatnonzero(has_any)).withhold(withheld)
                if len(sub) == 0:
                    continue
                res = evaluator.evaluate(self.predict(model, sub))
                for metric in res.metrics.values():
                    rows.append(
                        {
                            "available": "+".join(map(str, kept)),
                            "withheld": "+".join(map(str, withheld)) or "none",
                            "n": res.n_samples,
                            "metric": metric.name,
                            "value": metric.value,
                            "lower": metric.lower,
                            "upper": metric.upper,
                        }
                    )
        return pd.DataFrame(rows)

    def _default_output_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return self.config.output_dir.expanduser() / self.config.name / stamp


def _cohort_table(view: MultimodalView, splits: DataSplits) -> pd.DataFrame:
    """Samples and patients per split × class (tidy)."""
    rows = []
    for split_name, v in [("all", view), *splits.items()]:
        for k, cls in enumerate(v.class_names):
            mask = v.y == k
            rows.append(
                {
                    "split": split_name,
                    "class": cls,
                    "n_samples": int(mask.sum()),
                    "n_patients": int(np.unique(v.patient_ids[mask]).size),
                }
            )
    return pd.DataFrame(rows)
