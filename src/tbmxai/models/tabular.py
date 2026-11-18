"""Classical tabular classifiers (EHR + demographics)."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tbmxai.core.types import Modality
from tbmxai.data.harmonize import MultimodalView
from tbmxai.data.preprocessing import build_tabular_preprocessor
from tbmxai.models.base import MODELS, Classifier


class SklearnTabularClassifier(Classifier):
    """Wrap any sklearn-compatible estimator behind a fitted preprocessor.

    Subclasses only implement :meth:`make_estimator`. The preprocessing
    pipeline (impute → scale → one-hot) is fit on the training split only.

    Args:
        **params: Estimator hyper-parameters (merged over :attr:`default_params`).
    """

    required_modalities: frozenset[Modality] = frozenset({Modality.TABULAR})
    supports_missing_modalities = True  # fully-missing rows are imputed
    default_params: ClassVar[dict[str, Any]] = {}
    scale_numeric: ClassVar[bool] = True

    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params: dict[str, Any] = {**self.default_params, **params}
        self.pipeline_: Pipeline | None = None

    @abstractmethod
    def make_estimator(self) -> ClassifierMixin:
        """Return a fresh, unfitted estimator configured with :attr:`params`."""

    @property
    def feature_names_(self) -> list[str]:
        """Feature names after preprocessing (e.g. one-hot expanded)."""
        if self.pipeline_ is None:
            return []
        return list(self.pipeline_.named_steps["preprocess"].get_feature_names_out())

    def _fit(self, train: MultimodalView, val: MultimodalView | None) -> None:
        assert train.tabular is not None
        self.pipeline_ = Pipeline(
            [
                ("preprocess", build_tabular_preprocessor(train.tabular, self.scale_numeric)),
                ("estimator", self.make_estimator()),
            ]
        )
        self.pipeline_.fit(train.tabular, train.y)

    def _predict_proba(self, view: MultimodalView) -> np.ndarray:
        assert self.pipeline_ is not None and view.tabular is not None
        raw = self.pipeline_.predict_proba(view.tabular)
        # Align to all classes in case some were absent from the training split.
        proba = np.zeros((len(view), int(self.n_classes_ or raw.shape[1])))
        proba[:, self.pipeline_.classes_.astype(int)] = raw
        return proba


@MODELS.register("logistic_regression")
class LogisticRegressionClassifier(SklearnTabularClassifier):
    """L2-regularised logistic regression with balanced class weights."""

    default_params: ClassVar[dict[str, Any]] = {"max_iter": 2000, "class_weight": "balanced"}

    def make_estimator(self) -> ClassifierMixin:
        return LogisticRegression(**self.params)


@MODELS.register("random_forest")
class RandomForestTabularClassifier(SklearnTabularClassifier):
    """Random forest (no scaling needed)."""

    default_params: ClassVar[dict[str, Any]] = {
        "n_estimators": 500,
        "class_weight": "balanced",
        "n_jobs": -1,
    }
    scale_numeric = False

    def make_estimator(self) -> ClassifierMixin:
        return RandomForestClassifier(**self.params)


@MODELS.register("xgboost")
class XGBoostTabularClassifier(SklearnTabularClassifier):
    """Gradient-boosted trees via XGBoost."""

    default_params: ClassVar[dict[str, Any]] = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    scale_numeric = False

    def make_estimator(self) -> ClassifierMixin:
        from xgboost import XGBClassifier

        return XGBClassifier(**self.params)
