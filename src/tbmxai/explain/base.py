"""Explainer contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from tbmxai.core.registry import Registry
from tbmxai.core.types import Modality
from tbmxai.data.harmonize import MultimodalView
from tbmxai.models.base import Classifier


@dataclass
class Explanation:
    """Output of an explainer.

    Attributes:
        method: Explainer name.
        modality: Modality being explained.
        scope: ``"global"`` (dataset-level importances) or ``"local"`` (per sample).
        attributions: Global: DataFrame indexed by feature. Local tabular:
            ``(n, n_features)`` DataFrame (positive class). Local imaging:
            ``(n, n_target_classes, H, W)`` array.
        sample_ids: Explained samples (local explanations only).
        extra: Method-specific outputs.
    """

    method: str
    modality: Modality
    scope: str
    attributions: pd.DataFrame | np.ndarray
    sample_ids: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Explainer(ABC):
    """Explains a fitted :class:`Classifier`.

    Subclasses declare which modalities they can explain and which model
    families they support (checked in :meth:`supports`).

    Args:
        model: A fitted classifier.
    """

    modality: ClassVar[Modality]
    scope: ClassVar[str]
    #: True: explain the shared explanation sample set (class-stratified, same for every
    #: model). False: explain the whole evaluation split (e.g. global importances).
    uses_xai_samples: ClassVar[bool] = False

    def __init__(self, model: Classifier) -> None:
        if not model.is_fitted:
            raise ValueError("Explainers require a fitted model.")
        if not self.supports(model):
            raise TypeError(f"{type(self).__name__} does not support {type(model).__name__}.")
        self.model = model

    @classmethod
    def supports(cls, model: Classifier) -> bool:
        """Whether this explainer can handle ``model`` (default: modality check)."""
        return cls.modality in model.required_modalities

    @abstractmethod
    def explain(self, view: MultimodalView) -> Explanation:
        """Compute an explanation on ``view`` (typically the test split)."""


EXPLAINERS: Registry[Explainer] = Registry("explainer")
