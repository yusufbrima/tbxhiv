"""Interpretability for tabular models."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from tbmxai.core.types import Modality
from tbmxai.data.harmonize import MultimodalView
from tbmxai.explain.base import EXPLAINERS, Explainer, Explanation
from tbmxai.models.base import Classifier
from tbmxai.models.tabular import SklearnTabularClassifier


@EXPLAINERS.register("permutation_importance")
class PermutationImportanceExplainer(Explainer):
    """Model-agnostic global importance: drop in AUROC when a raw feature is shuffled.

    Operates on *raw* (pre-processing) columns, so a categorical feature is
    scored as a whole rather than per one-hot level.

    Args:
        model: Fitted tabular classifier.
        n_repeats: Shuffles per feature.
        seed: RNG seed.
    """

    modality = Modality.TABULAR
    scope = "global"

    def __init__(self, model: Classifier, n_repeats: int = 10, seed: int = 0) -> None:
        super().__init__(model)
        self.n_repeats = n_repeats
        self.seed = seed

    def explain(self, view: MultimodalView) -> Explanation:
        assert view.tabular is not None
        model = self.model

        def score(_: object, X: pd.DataFrame, y: np.ndarray) -> float:
            proba = model.predict_proba(_with_tabular(view, X))
            if proba.shape[1] == 2:
                return float(roc_auc_score(y, proba[:, 1]))
            return float(roc_auc_score(y, proba, multi_class="ovr"))

        result = permutation_importance(
            _Passthrough(),
            view.tabular,
            view.y,
            scoring=score,
            n_repeats=self.n_repeats,
            random_state=self.seed,
        )
        table = pd.DataFrame(
            {"importance_mean": result.importances_mean, "importance_std": result.importances_std},
            index=pd.Index(view.tabular.columns, name="feature"),
        ).sort_values("importance_mean", ascending=False)
        return Explanation("permutation_importance", self.modality, self.scope, table)


@EXPLAINERS.register("shap")
class ShapTabularExplainer(Explainer):
    """SHAP values for tabular models, per **raw** feature and for **every class**.

    * XGBoost: exact TreeSHAP via the booster (``pred_contribs``), log-odds;
    * random forest: ``shap.TreeExplainer`` (probability scale);
    * logistic regression: ``shap.LinearExplainer`` (log-odds);
    * other sklearn estimators: model-agnostic ``shap.Explainer`` (slow).

    One-hot columns of a categorical feature are summed back to that feature.
    The whole evaluation split is explained (identical across models under a
    protocol). If it exceeds ``max_samples``, a class-stratified subset is taken by
    a seeded hash of the sample ids, so every model explains the same samples.

    Outputs (``Explanation.extra``):
        ``by_class``: target class → local values (sample × feature);
        ``global_long``: mean |SHAP| per target class × true-class group × feature;
        ``global``: positive-class / all-samples summary (for plots).

    Args:
        model: Fitted ``SklearnTabularClassifier``.
        max_samples: Upper bound on explained samples.
        background_size: Background rows for linear / model-agnostic explainers.
        seed: Seed of the deterministic subsampling.
    """

    modality = Modality.TABULAR
    scope = "local"
    uses_xai_samples = False

    def __init__(
        self, model: Classifier, max_samples: int = 5000, background_size: int = 200, seed: int = 0
    ) -> None:
        super().__init__(model)
        self.max_samples = max_samples
        self.background_size = background_size
        self.seed = seed

    @classmethod
    def supports(cls, model: Classifier) -> bool:
        return isinstance(model, SklearnTabularClassifier)

    def _subset(self, view: MultimodalView) -> np.ndarray:
        if len(view) <= self.max_samples:
            return np.arange(len(view))
        rank = pd.Series(view.sample_ids.astype(str)).map(
            lambda s: hashlib.sha1(f"{self.seed}|{s}".encode()).hexdigest()
        )
        per = int(np.ceil(self.max_samples / view.n_classes))
        frame = pd.DataFrame({"pos": np.arange(len(view)), "y": view.y, "rank": rank})
        chosen = frame.sort_values("rank").groupby("y").head(per)["pos"]
        return np.sort(chosen.to_numpy())

    def explain(self, view: MultimodalView) -> Explanation:
        assert view.tabular is not None
        model: Any = self.model
        pre, est = (
            model.pipeline_.named_steps["preprocess"],
            model.pipeline_.named_steps["estimator"],
        )
        idx = self._subset(view)
        raw = view.tabular.iloc[idx].reset_index(drop=True)
        X = np.asarray(pre.transform(raw), dtype=float)
        owners = raw_feature_owners(pre)
        ids = pd.Index(view.sample_ids[idx], name="sample_id")
        y = view.y[idx]
        by_class: dict[str, pd.DataFrame] = {}
        rows = []
        per_class, scale = self._shap_values(est, X, view.n_classes)
        for k, values in per_class.items():
            if len(owners) != values.shape[1]:
                raise RuntimeError("Could not map transformed columns back to raw features.")
            local = pd.DataFrame(values, columns=owners).T.groupby(level=0, sort=False).sum().T
            local.index = ids
            name = view.class_names[k]
            by_class[name] = local
            groups = [("all", np.ones(len(local), bool))] + [
                (view.class_names[c], y == c) for c in range(view.n_classes)
            ]
            for group, mask in groups:
                if not mask.any():
                    continue
                part = local[mask]
                rows.append(
                    pd.DataFrame(
                        {
                            "target_class": name,
                            "sample_class": group,
                            "n": int(mask.sum()),
                            "feature": local.columns,
                            "importance_mean": part.abs().mean().to_numpy(),
                            "importance_std": part.abs().std().to_numpy(),
                            "mean_shap": part.mean().to_numpy(),
                        }
                    )
                )
        long = pd.concat(rows, ignore_index=True)
        positive = view.class_names[1 if view.n_classes == 2 else view.n_classes - 1]
        summary = (
            long[(long["target_class"] == positive) & (long["sample_class"] == "all")]
            .set_index("feature")[["importance_mean", "importance_std", "mean_shap"]]
            .sort_values("importance_mean", ascending=False)
        )
        return Explanation(
            method="shap",
            modality=self.modality,
            scope=self.scope,
            attributions=by_class[positive],
            sample_ids=ids.to_numpy(),
            extra={
                "by_class": by_class,
                "global": summary,
                "global_long": long,
                "feature_values": raw.set_index(ids),
                "y_true": y,
                "class_names": view.class_names,
                "scale": scale,
                "patient_ids": view.patient_ids[idx],
            },
        )

    def _shap_values(
        self, est: Any, X: np.ndarray, n_classes: int
    ) -> tuple[dict[int, np.ndarray], str]:
        """SHAP values for every class output: ``{class: (n, F)}``."""
        name = type(est).__name__
        if name == "XGBClassifier":
            import xgboost as xgb

            contrib = est.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
            if contrib.ndim == 3:  # multiclass: (n, C, F+1)
                return {k: contrib[:, k, :-1] for k in range(n_classes)}, "log-odds"
            pos = contrib[:, :-1]  # binary margin = log-odds of class 1; class 0 is its negation
            return {0: -pos, 1: pos}, "log-odds"
        import shap

        rng = np.random.default_rng(self.seed)
        background = X[rng.choice(len(X), min(self.background_size, len(X)), replace=False)]
        if name in {"RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier"}:
            raw, scale = (
                shap.TreeExplainer(est).shap_values(X, check_additivity=False),
                "probability",
            )
        elif name == "LogisticRegression":
            raw, scale = shap.LinearExplainer(est, background).shap_values(X), "log-odds"
            if n_classes == 2 and not isinstance(raw, list) and np.asarray(raw).ndim == 2:
                pos = np.asarray(raw)
                return {0: -pos, 1: pos}, scale
        else:
            out = {
                k: np.asarray(
                    shap.Explainer(lambda z, k=k: est.predict_proba(z)[:, k], background)(X).values
                )
                for k in range(n_classes)
            }
            return out, "probability"
        if isinstance(raw, list):
            return {k: np.asarray(raw[k]) for k in range(n_classes)}, scale
        raw = np.asarray(raw)
        if raw.ndim == 3:
            return {k: raw[:, :, k] for k in range(n_classes)}, scale
        return {0: -raw, 1: raw}, scale


def raw_feature_owners(pre: Any) -> list[str]:
    """Raw input feature behind each output column of a fitted ``ColumnTransformer``."""
    owners: list[str] = []
    for name, transformer, columns in pre.transformers_:
        if name == "remainder" or transformer == "drop" or len(columns) == 0:
            continue
        if name == "categorical":
            onehot = transformer.named_steps["onehot"]
            for col, cats in zip(columns, onehot.categories_, strict=True):
                owners += [col] * len(cats)
        else:
            owners += list(columns)
    return owners


# ---------------------------------------------------------------- internals


class _Passthrough:
    """Dummy estimator: permutation_importance needs one, our scorer ignores it."""

    def fit(self, X: object, y: object) -> _Passthrough:  # pragma: no cover - never called
        return self


def _with_tabular(view: MultimodalView, tabular: pd.DataFrame) -> MultimodalView:
    """Copy of ``view`` with replaced tabular features (other modalities kept)."""
    from dataclasses import replace

    return replace(view, tabular=pd.DataFrame(tabular, columns=view.tabular.columns))  # type: ignore[union-attr]
