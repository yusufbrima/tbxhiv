"""Shared types exchanged between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

import numpy as np
import pandas as pd


class Modality(StrEnum):
    """Data modalities understood by the pipeline."""

    TABULAR = "tabular"  # EHR, demographics, lab values
    IMAGING = "imaging"  # chest X-rays
    TEXT = "text"  # radiology / clinical reports


@dataclass
class Predictions:
    """Model outputs aligned with ground truth.

    This is the common currency between models, the evaluator, subgroup
    analysis and reporting: anything that can produce a ``Predictions`` object
    (including an external CSV) can be evaluated.

    Attributes:
        y_true: Integer class labels, shape ``(n,)``.
        y_prob: Class probabilities, shape ``(n, n_classes)``. A 1-D array is
            interpreted as the positive-class probability of a binary task.
        sample_ids: Optional identifiers, shape ``(n,)``.
        class_names: Optional names for the probability columns.
        metadata: Optional per-sample attributes (e.g. sex, age, HIV status)
            used for subgroup analysis. Row order must match ``y_true``.
        group_ids: Optional patient id per sample; enables patient-level
            (cluster) bootstrap confidence intervals.
    """

    y_true: np.ndarray
    y_prob: np.ndarray
    sample_ids: np.ndarray | None = None
    class_names: list[str] | None = None
    metadata: pd.DataFrame | None = field(default=None, repr=False)
    group_ids: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.y_true = np.asarray(self.y_true).astype(int).ravel()
        prob = np.asarray(self.y_prob, dtype=float)
        if prob.ndim == 1:
            prob = np.column_stack([1.0 - prob, prob])
        if prob.ndim != 2 or prob.shape[0] != self.y_true.shape[0]:
            raise ValueError(
                f"y_prob shape {prob.shape} incompatible with y_true length {len(self.y_true)}."
            )
        self.y_prob = prob
        if self.y_true.size and (self.y_true.min() < 0 or self.y_true.max() >= self.n_classes):
            raise ValueError(f"y_true contains labels outside [0, {self.n_classes}).")
        if self.sample_ids is not None:
            self.sample_ids = np.asarray(self.sample_ids)
        if self.group_ids is not None:
            self.group_ids = np.asarray(self.group_ids)
            if len(self.group_ids) != len(self):
                raise ValueError("group_ids must have one entry per prediction.")
        if self.class_names is None:
            self.class_names = [str(i) for i in range(self.n_classes)]
        elif len(self.class_names) != self.n_classes:
            raise ValueError("class_names length must equal the number of probability columns.")
        if self.metadata is not None:
            if len(self.metadata) != len(self):
                raise ValueError("metadata must have one row per prediction.")
            self.metadata = self.metadata.reset_index(drop=True)

    def __len__(self) -> int:
        return int(self.y_true.shape[0])

    @property
    def n_classes(self) -> int:
        """Number of classes (probability columns)."""
        return int(self.y_prob.shape[1])

    @property
    def is_binary(self) -> bool:
        """Whether this is a two-class problem."""
        return self.n_classes == 2

    def predicted_labels(self, threshold: float | None = None) -> np.ndarray:
        """Hard labels.

        Args:
            threshold: For binary tasks, positive-class probability cut-off.
                Ignored for multiclass (argmax is used). ``None`` means argmax.
        """
        if self.is_binary and threshold is not None:
            return (self.y_prob[:, 1] >= threshold).astype(int)
        return self.y_prob.argmax(axis=1)

    def subset(self, mask: np.ndarray) -> Predictions:
        """Return predictions restricted to a boolean mask or index array."""
        idx = np.asarray(mask)
        return Predictions(
            y_true=self.y_true[idx],
            y_prob=self.y_prob[idx],
            sample_ids=None if self.sample_ids is None else self.sample_ids[idx],
            class_names=self.class_names,
            metadata=None if self.metadata is None else self.metadata.iloc[idx],
            group_ids=None if self.group_ids is None else self.group_ids[idx],
        )

    def to_frame(self, prob_prefix: str = "prob_") -> pd.DataFrame:
        """Flatten to a DataFrame (``sample_id``, ``y_true``, ``prob_<class>``…, metadata…)."""
        assert self.class_names is not None
        frame = pd.DataFrame({"y_true": self.y_true})
        if self.sample_ids is not None:
            frame.insert(0, "sample_id", self.sample_ids)
        if self.group_ids is not None:
            frame.insert(1 if self.sample_ids is not None else 0, "patient_id", self.group_ids)
        for k, name in enumerate(self.class_names):
            frame[f"{prob_prefix}{name}"] = self.y_prob[:, k]
        if self.metadata is not None:
            extra = self.metadata.drop(columns=frame.columns, errors="ignore")
            frame = pd.concat([frame, extra.reset_index(drop=True)], axis=1)
        return frame

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        label_column: str = "y_true",
        prob_prefix: str = "prob_",
        id_column: str | None = "sample_id",
        group_column: str | None = "patient_id",
    ) -> Self:
        """Inverse of :meth:`to_frame`; non-label/prob/id columns become metadata.

        ``group_column`` (if present) provides patient ids for cluster bootstrap CIs.
        """
        prob_cols = [c for c in frame.columns if c.startswith(prob_prefix)]
        if not prob_cols:
            raise ValueError(f"No probability columns with prefix '{prob_prefix}' found.")
        used = {label_column, *prob_cols}
        ids = None
        if id_column and id_column in frame.columns:
            ids = frame[id_column].to_numpy()
            used.add(id_column)
        groups = None
        if group_column and group_column in frame.columns:
            groups = frame[group_column].to_numpy()
            used.add(group_column)
        meta_cols = [c for c in frame.columns if c not in used]
        return cls(
            y_true=frame[label_column].to_numpy(),
            y_prob=frame[prob_cols].to_numpy(),
            sample_ids=ids,
            class_names=[c[len(prob_prefix) :] for c in prob_cols],
            metadata=frame[meta_cols] if meta_cols else None,
            group_ids=groups,
        )
