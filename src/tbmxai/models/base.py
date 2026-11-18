"""Model contract shared by tabular, imaging and fusion classifiers."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np

from tbmxai.core.registry import Registry
from tbmxai.core.types import Modality, Predictions
from tbmxai.data.harmonize import MultimodalView

if TYPE_CHECKING:
    from tbmxai.core.config import TrainingConfig


class NotFittedError(RuntimeError):
    """Raised when predicting with an unfitted model."""


class Classifier(ABC):
    """Base class for every model in the pipeline.

    Subclasses declare :attr:`required_modalities` and implement :meth:`_fit`
    and :meth:`_predict_proba`. The public methods enforce the modality
    contract and fitted state, so the pipeline can swap models freely.

    A model owns all *fitted* preprocessing (imputation, scaling, image
    normalisation statistics…) to guarantee it only ever sees training data.
    """

    #: Modalities the model consumes. Usually a class attribute; models whose
    #: inputs are configurable (e.g. ``NeuralClassifier``) set it per instance.
    required_modalities: frozenset[Modality]
    #: Whether samples may lack some of the model's modalities (availability masks).
    supports_missing_modalities: bool = False

    def __init__(self) -> None:
        self._fitted = False
        self.n_classes_: int | None = None

    def setup(self, run_dir: Path, training: TrainingConfig | None = None) -> None:  # noqa: B027
        """Hook called by the pipeline before :meth:`fit`.

        Receives the run directory (for checkpoints/logs) and the experiment's
        training config. Classical models ignore it.
        """

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed."""
        return self._fitted

    def fit(self, train: MultimodalView, val: MultimodalView | None = None) -> Self:
        """Fit on ``train`` (``val`` may be used for early stopping/tuning).

        Returns:
            ``self``, to allow chaining.
        """
        for view in (train, val):
            if view is not None:
                view.require(self.required_modalities)
                self._check_missing(view)
        self.n_classes_ = train.n_classes
        self._fit(train, val)
        self._fitted = True
        return self

    def predict_proba(self, view: MultimodalView) -> np.ndarray:
        """Class probabilities of shape ``(n, n_classes)``."""
        if not self._fitted:
            raise NotFittedError(f"{type(self).__name__} must be fitted before prediction.")
        view.require(self.required_modalities)
        self._check_missing(view)
        proba = np.asarray(self._predict_proba(view), dtype=float)
        if proba.shape != (len(view), self.n_classes_):
            raise RuntimeError(
                f"{type(self).__name__} returned probabilities of shape {proba.shape}, "
                f"expected {(len(view), self.n_classes_)}."
            )
        return proba

    def predict(self, view: MultimodalView) -> Predictions:
        """Predict and package results (with metadata) for evaluation."""
        return Predictions(
            y_true=view.y,
            y_prob=self.predict_proba(view),
            sample_ids=view.sample_ids,
            class_names=view.class_names,
            metadata=view.metadata,
            group_ids=view.patient_ids,
        )

    def _check_missing(self, view: MultimodalView) -> None:
        if self.supports_missing_modalities:
            return
        incomplete = [
            str(m)
            for m in self.required_modalities
            if m in view.availability and not view.availability[m].all()
        ]
        if incomplete:
            raise ValueError(
                f"{type(self).__name__} cannot handle missing {incomplete}; use "
                "view.missing_modalities: drop, or a model with missing-modality support (neural)."
            )

    def save(self, path: str | Path) -> Path:
        """Persist the model (pickle by default; torch models override)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a model saved with :meth:`save`. Only load trusted files."""
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}.")
        return obj

    @abstractmethod
    def _fit(self, train: MultimodalView, val: MultimodalView | None) -> None:
        """Model-specific training."""

    @abstractmethod
    def _predict_proba(self, view: MultimodalView) -> np.ndarray:
        """Model-specific inference."""


MODELS: Registry[Classifier] = Registry("model")
