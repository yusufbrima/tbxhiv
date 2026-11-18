"""Patient-level dataset splitting.

**Every split in tbmxai is patient-level**: all samples (CXRs, visits…) of a
patient land in the same partition. Built-in splitters assign *patients*, not
samples, and :meth:`DataSplits.check_patient_disjoint` is enforced by the
pipeline for every splitter (including user-defined and predefined ones), so a
leaking split fails loudly instead of inflating results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, train_test_split

from tbmxai.core.registry import Registry
from tbmxai.data.harmonize import MultimodalView


class PatientLeakageError(ValueError):
    """Raised when a patient appears in more than one split."""


@dataclass
class DataSplits:
    """Train / validation / test views."""

    train: MultimodalView
    test: MultimodalView
    val: MultimodalView | None = None

    def items(self) -> list[tuple[str, MultimodalView]]:
        """(name, view) pairs for the populated splits."""
        pairs = [("train", self.train), ("val", self.val), ("test", self.test)]
        return [(name, view) for name, view in pairs if view is not None]

    def check_patient_disjoint(self) -> None:
        """Raise :class:`PatientLeakageError` if any patient is in two splits."""
        seen: dict[str, str] = {}
        for name, view in self.items():
            for pid in np.unique(view.patient_ids.astype(str)):
                if pid in seen:
                    raise PatientLeakageError(
                        f"Patient '{pid}' appears in both '{seen[pid]}' and '{name}' splits."
                    )
                seen[pid] = name

    def assignments(self) -> pd.DataFrame:
        """``sample_id, patient_id, split`` table (saved per run for reproducibility)."""
        return pd.concat(
            [
                pd.DataFrame({"sample_id": v.sample_ids, "patient_id": v.patient_ids, "split": n})
                for n, v in self.items()
            ],
            ignore_index=True,
        )


class Splitter(ABC):
    """Strategy that partitions a view into patient-disjoint :class:`DataSplits`."""

    @abstractmethod
    def split(self, view: MultimodalView, seed: int = 0) -> DataSplits:
        """Partition ``view``; must be deterministic given ``seed``."""


SPLITTERS: Registry[Splitter] = Registry("splitter")


def _patient_table(view: MultimodalView) -> pd.Series:
    """One label per patient (max label, i.e. 'ever positive') indexed by patient id."""
    return pd.Series(view.y).groupby(view.patient_ids).max()


def _select(view: MultimodalView, patients: np.ndarray) -> MultimodalView:
    return view.subset(np.flatnonzero(np.isin(view.patient_ids, patients)))


def _split_patients(
    ids: np.ndarray, labels: np.ndarray | None, size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    a, b = train_test_split(ids, test_size=size, random_state=seed, stratify=labels)
    return np.asarray(a), np.asarray(b)


@SPLITTERS.register("holdout")
class HoldoutSplitter(Splitter):
    """Random patient-level train/val/test split, stratified by patient label.

    Args:
        val_size: Fraction of *patients* for validation (0 disables).
        test_size: Fraction of *patients* for testing.
        stratify: Stratify by per-patient label (max over the patient's samples).
    """

    def __init__(self, val_size: float = 0.15, test_size: float = 0.15, stratify: bool = True):
        if not 0 < test_size < 1 or not 0 <= val_size < 1 or val_size + test_size >= 1:
            raise ValueError("Need 0 < test_size, 0 <= val_size, val_size + test_size < 1.")
        self.val_size = val_size
        self.test_size = test_size
        self.stratify = stratify

    def split(self, view: MultimodalView, seed: int = 0) -> DataSplits:
        patients = _patient_table(view)
        ids, labels = patients.index.to_numpy(), patients.to_numpy()
        rest, test = _split_patients(ids, labels if self.stratify else None, self.test_size, seed)
        train, val = rest, np.array([], dtype=ids.dtype)
        if self.val_size > 0:
            rel = self.val_size / (1.0 - self.test_size)
            rest_labels = patients.loc[rest].to_numpy() if self.stratify else None
            train, val = _split_patients(rest, rest_labels, rel, seed + 1)
        return DataSplits(
            train=_select(view, train),
            val=_select(view, val) if len(val) else None,
            test=_select(view, test),
        )


@SPLITTERS.register("kfold")
class PatientKFoldSplitter(Splitter):
    """Patient-grouped (optionally stratified) K-fold; returns one fold.

    Fold ``fold`` is the test set; a validation set is carved from the
    remaining patients. Run the pipeline once per fold (``fold=0..n_splits-1``)
    for cross-validation.

    Args:
        n_splits: Number of folds.
        fold: Which fold to hold out as test.
        val_size: Fraction of the remaining patients used for validation.
        stratify: Use ``StratifiedGroupKFold`` (else ``GroupKFold``).
    """

    def __init__(
        self, n_splits: int = 5, fold: int = 0, val_size: float = 0.15, stratify: bool = True
    ) -> None:
        if not 0 <= fold < n_splits:
            raise ValueError(f"fold must be in [0, {n_splits})")
        self.n_splits = n_splits
        self.fold = fold
        self.val_size = val_size
        self.stratify = stratify

    def split(self, view: MultimodalView, seed: int = 0) -> DataSplits:
        patients = _patient_table(view)
        ids, labels = patients.index.to_numpy(), patients.to_numpy()
        cv = (
            StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=seed)
            if self.stratify
            else GroupKFold(self.n_splits)
        )
        rest_idx, test_idx = list(cv.split(ids, labels, groups=ids))[self.fold]
        rest, test = ids[rest_idx], ids[test_idx]
        train, val = rest, np.array([], dtype=ids.dtype)
        if self.val_size > 0:
            strat = labels[rest_idx] if self.stratify else None
            train, val = _split_patients(rest, strat, self.val_size, seed + 1)
        return DataSplits(
            train=_select(view, train),
            val=_select(view, val) if len(val) else None,
            test=_select(view, test),
        )


@SPLITTERS.register("external")
class ExternalSplitter(Splitter):
    """External validation: test on one site / source, train and validate on the others.

    Samples whose ``column`` value is in ``test_values`` (e.g. ``site: montgomery``) form
    the test set; the remaining patients are split into train / validation
    (patient-level, stratified). The column must be in ``view.metadata_columns``.

    Args:
        column: Metadata column identifying the site / source.
        test_values: Value(s) held out as the external test set.
        val_size: Fraction of the remaining *patients* used for validation (0 disables).
        stratify: Stratify the train/val split by per-patient label.
        internal_test_size: Fraction of the source patients held out as an internal
            (in-distribution) test set and added to the test split, so a single run reports
            both internal and external performance (0 disables). Taken before the train/val
            split, with its own seed offset.
        split_seed: If set, the internal-test and train/val splits use this seed instead of the
            run seed, so every seed of an arm shares one split and only initialisation and data
            order vary (a seed ensemble then has genuine validation and internal-test scores).
    """

    def __init__(
        self,
        column: str = "site",
        test_values: str | list[str] = "external",
        val_size: float = 0.15,
        stratify: bool = True,
        internal_test_size: float = 0.0,
        split_seed: int | None = None,
    ) -> None:
        if not 0 <= val_size < 1 or not 0 <= internal_test_size < 1:
            raise ValueError("Need 0 <= val_size < 1 and 0 <= internal_test_size < 1.")
        if val_size + internal_test_size >= 1:
            raise ValueError("val_size + internal_test_size must be < 1.")
        self.column = column
        self.test_values = [test_values] if isinstance(test_values, str) else list(test_values)
        self.val_size = val_size
        self.stratify = stratify
        self.internal_test_size = internal_test_size
        self.split_seed = split_seed

    def split(self, view: MultimodalView, seed: int = 0) -> DataSplits:
        if self.column not in view.metadata.columns:
            raise KeyError(
                f"Site column '{self.column}' not in view metadata; add it to "
                "`view.metadata_columns`."
            )
        if self.split_seed is not None:
            seed = self.split_seed
        is_test = view.metadata[self.column].astype(str).isin(self.test_values).to_numpy()
        if not is_test.any() or is_test.all():
            present = sorted(view.metadata[self.column].astype(str).unique())
            raise ValueError(
                f"External split on {self.column} ∈ {self.test_values} leaves no test or no "
                f"training data (values present: {present})."
            )
        if self.internal_test_size > 0:
            src = _patient_table(view.subset(np.flatnonzero(~is_test)))
            _, internal = _split_patients(
                src.index.to_numpy(),
                src.to_numpy() if self.stratify else None,
                self.internal_test_size,
                seed + 7919,
            )
            is_test = is_test | np.isin(np.asarray(view.patient_ids), internal)
        test = view.subset(np.flatnonzero(is_test))
        dev = view.subset(np.flatnonzero(~is_test))
        patients = _patient_table(dev)
        ids, labels = patients.index.to_numpy(), patients.to_numpy()
        train, val = ids, np.array([], dtype=ids.dtype)
        if self.val_size > 0:
            rel = self.val_size / (1.0 - self.internal_test_size)
            train, val = _split_patients(ids, labels if self.stratify else None, rel, seed)
        splits = DataSplits(
            train=_select(dev, train), val=_select(dev, val) if len(val) else None, test=test
        )
        splits.check_patient_disjoint()
        return splits


@SPLITTERS.register("predefined")
class PredefinedSplitter(Splitter):
    """Use an existing split column (e.g. an official dataset split).

    The column must be listed in ``view.metadata_columns``. Patient
    disjointness is still verified by the pipeline.

    Args:
        column: Metadata column with split assignments.
        train_value / val_value / test_value: Values denoting each split.
    """

    def __init__(
        self,
        column: str = "split",
        train_value: str = "train",
        val_value: str = "val",
        test_value: str = "test",
    ) -> None:
        self.column = column
        self.values = {"train": train_value, "val": val_value, "test": test_value}

    def split(self, view: MultimodalView, seed: int = 0) -> DataSplits:
        if self.column not in view.metadata.columns:
            raise KeyError(
                f"Split column '{self.column}' not in view metadata; add it to "
                "`view.metadata_columns`."
            )
        assignment = view.metadata[self.column].astype(str).to_numpy()
        parts = {
            name: view.subset(np.flatnonzero(assignment == value))
            for name, value in self.values.items()
        }
        if len(parts["train"]) == 0 or len(parts["test"]) == 0:
            raise ValueError("Predefined split produced an empty train or test set.")
        splits = DataSplits(
            train=parts["train"],
            val=parts["val"] if len(parts["val"]) else None,
            test=parts["test"],
        )
        splits.check_patient_disjoint()
        return splits


@SPLITTERS.register("temporal")
class TemporalSplitter(Splitter):
    """Patient-level **temporal** validation: train on the past, test on the future.

    Each patient is assigned to a period by the time of their **first** sample
    (``column``, any orderable metadata: datetimes, years, or sortable strings
    such as MIMIC's ``anchor_year_group``). Patients whose first sample is at or
    after ``test_start`` form the test set; those at/after ``val_start`` (and
    before ``test_start``) the validation set; the rest train.

    Instead of explicit cut-offs, ``test_size`` / ``val_size`` take the most
    recent fractions of patients (ties broken at random with ``seed``).

    Args:
        column: Metadata column holding the time of each sample.
        test_start: First value belonging to the test period.
        val_start: First value belonging to the validation period.
        test_size: Fraction of most recent patients for test (if no ``test_start``).
        val_size: Fraction of the next most recent patients for validation
            (if no ``val_start``; 0 disables).
        strict: Drop samples of train/val patients that fall in a *later*
            period (e.g. a training patient's future CXRs), so no information
            from the test period is used for training.
    """

    def __init__(
        self,
        column: str,
        test_start: str | float | None = None,
        val_start: str | float | None = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        strict: bool = True,
    ) -> None:
        self.column = column
        self.test_start = test_start
        self.val_start = val_start
        self.test_size = test_size
        self.val_size = val_size
        self.strict = strict

    @staticmethod
    def _as_comparable(values: pd.Series, ref: object) -> object:
        if pd.api.types.is_datetime64_any_dtype(values):
            return pd.Timestamp(ref)  # type: ignore[arg-type]
        if pd.api.types.is_numeric_dtype(values):
            return float(ref)  # type: ignore[arg-type]
        return str(ref)

    def split(self, view: MultimodalView, seed: int = 0) -> DataSplits:
        if self.column not in view.metadata.columns:
            raise KeyError(
                f"Time column '{self.column}' not in view metadata; add it to "
                "`view.metadata_columns`."
            )
        times = view.metadata[self.column]
        if not pd.api.types.is_numeric_dtype(times) and not pd.api.types.is_datetime64_any_dtype(
            times
        ):
            times = times.astype(str)
        first = times.groupby(view.patient_ids).min()
        if first.isna().any():
            raise ValueError(f"Missing values in time column '{self.column}'.")

        rng = np.random.default_rng(seed)
        order = pd.DataFrame({"t": first, "tie": rng.random(len(first))}).sort_values(["t", "tie"])
        n = len(order)
        if self.test_start is not None:
            cut = self._as_comparable(first, self.test_start)
            test = order.index[order["t"] >= cut]
        else:
            test = order.index[n - max(1, round(self.test_size * n)) :]
        rest = order.drop(index=test)
        if self.val_start is not None:
            cut = self._as_comparable(first, self.val_start)
            val = rest.index[rest["t"] >= cut]
        elif self.val_size > 0:
            val = rest.index[len(rest) - max(1, round(self.val_size * n)) :]
        else:
            val = rest.index[:0]
        train = rest.index.difference(val)
        if len(train) == 0 or len(test) == 0:
            raise ValueError("Temporal split produced an empty train or test set.")

        parts = {
            "train": view.subset(np.flatnonzero(np.isin(view.patient_ids, train))),
            "val": view.subset(np.flatnonzero(np.isin(view.patient_ids, val))),
            "test": view.subset(np.flatnonzero(np.isin(view.patient_ids, test))),
        }
        if self.strict:
            test_from = order.loc[test, "t"].min()
            val_from = order.loc[val, "t"].min() if len(val) else test_from
            parts["train"] = self._before(parts["train"], min(val_from, test_from))
            if len(val):
                parts["val"] = self._before(parts["val"], test_from)
        return DataSplits(
            train=parts["train"],
            val=parts["val"] if len(parts["val"]) else None,
            test=parts["test"],
        )

    def _before(self, view: MultimodalView, cutoff: object) -> MultimodalView:
        times = view.metadata[self.column]
        if not pd.api.types.is_numeric_dtype(times) and not pd.api.types.is_datetime64_any_dtype(
            times
        ):
            times = times.astype(str)
        return view.subset(np.flatnonzero((times < cutoff).to_numpy()))
