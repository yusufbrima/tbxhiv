"""Harmonisation: turn a :class:`Cohort` into a modality-specific model input."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from typing import Self

import numpy as np
import pandas as pd

from tbmxai.core.config import ViewConfig
from tbmxai.core.types import Modality
from tbmxai.core.utils import get_logger
from tbmxai.data.ingest import SAMPLE_ID_COLUMN, Cohort
from tbmxai.data.sources import IMAGE_PATH_COLUMN, REPORT_TEXT_COLUMN

logger = get_logger(__name__)


#: Metadata column added to every view: which selected modalities each sample has.
MODALITY_PATTERN_COLUMN = "modality_pattern"


@dataclass
class MultimodalView:
    """Aligned model inputs for any subset of modalities.

    The modalities in :attr:`modalities` are populated; the others are ``None``.
    Unimodal tasks are simply views with a single modality.

    **Missing modalities.** A selected modality may be absent for some samples
    (e.g. a patient with EHR but no CXR). :attr:`availability` holds one boolean
    mask per selected modality; models that support missing modalities use it,
    and :meth:`withhold` simulates missingness at inference time.

    Attributes:
        sample_ids: Sample identifiers, shape ``(n,)``.
        patient_ids: Patient identifiers (for grouped splitting), shape ``(n,)``.
        y: Integer labels, shape ``(n,)``.
        class_names: Label names indexed by class.
        tabular: Raw (unprocessed) tabular features.
        image_paths: Image file paths (``""`` where missing), shape ``(n,)``.
        text: Report texts (``""`` where missing), shape ``(n,)``.
        metadata: Non-input attributes for splitting / subgroup analysis
            (always includes ``modality_pattern``).
        availability: Modality → boolean mask of samples where it is present.
    """

    sample_ids: np.ndarray
    patient_ids: np.ndarray
    y: np.ndarray
    class_names: list[str]
    tabular: pd.DataFrame | None = None
    image_paths: np.ndarray | None = None
    text: np.ndarray | None = None
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    availability: dict[Modality, np.ndarray] = field(default_factory=dict)
    #: True when patient ids were synthesised (one patient per image); see ``ImageFolderSource``.
    patient_ids_assumed: bool = False

    def __post_init__(self) -> None:
        n = len(self)
        for m in self.modalities:
            mask = self.availability.get(m)
            self.availability[m] = (
                np.ones(n, dtype=bool) if mask is None else np.asarray(mask, bool)
            )
        self.availability = {m: v for m, v in self.availability.items() if m in self.modalities}

    def __len__(self) -> int:
        return int(self.y.shape[0])

    @property
    def modalities(self) -> frozenset[Modality]:
        """Modalities selected for this view (possibly missing for some samples)."""
        present = {
            Modality.TABULAR: self.tabular is not None,
            Modality.IMAGING: self.image_paths is not None,
            Modality.TEXT: self.text is not None,
        }
        return frozenset(m for m, ok in present.items() if ok)

    @property
    def n_classes(self) -> int:
        """Number of target classes."""
        return len(self.class_names)

    @property
    def is_complete(self) -> bool:
        """Whether every sample has every selected modality."""
        return all(mask.all() for mask in self.availability.values())

    def modality_patterns(self) -> np.ndarray:
        """Per-sample string such as ``"imaging+tabular"`` (selected & present modalities)."""
        mods = sorted(self.availability, key=str)
        parts = np.array([[str(m) if ok else "" for ok in self.availability[m]] for m in mods]).T
        return (
            np.array(["+".join(p for p in row if p) or "none" for row in parts])
            if mods
            else np.array([])
        )

    def require(self, modalities: frozenset[Modality] | set[Modality]) -> None:
        """Raise ``ValueError`` if any of ``modalities`` is not selected in this view."""
        if missing := set(modalities) - self.modalities:
            raise ValueError(
                f"View lacks required modalities {sorted(map(str, missing))}; "
                f"has {sorted(map(str, self.modalities))}."
            )

    def subset(self, indices: Sequence[int] | np.ndarray) -> MultimodalView:
        """Return a new view with rows at positional ``indices``."""
        idx = np.asarray(indices)
        return replace(
            self,
            sample_ids=self.sample_ids[idx],
            patient_ids=self.patient_ids[idx],
            y=self.y[idx],
            tabular=None if self.tabular is None else self.tabular.iloc[idx].reset_index(drop=True),
            image_paths=None if self.image_paths is None else self.image_paths[idx],
            text=None if self.text is None else self.text[idx],
            metadata=self.metadata.iloc[idx].reset_index(drop=True),
            availability={m: v[idx] for m, v in self.availability.items()},
        )

    def withhold(self, modalities: Sequence[Modality | str]) -> MultimodalView:
        """Copy with ``modalities`` marked missing for every sample (inference ablation)."""
        drop = {Modality(m) for m in modalities}
        availability = {
            m: (np.zeros_like(v) if m in drop else v.copy()) for m, v in self.availability.items()
        }
        view = replace(self, availability=availability, metadata=self.metadata.copy())
        view.metadata[MODALITY_PATTERN_COLUMN] = view.modality_patterns()
        return view


class Harmonizer:
    """Select modalities from a cohort and package them as a :class:`MultimodalView`.

    Harmonisation is stateless (nothing is fitted) so it is safe to run before
    splitting; fitted preprocessing belongs to the model. Steps:

    1. **Modality selection** (experiment design): keep only ``modalities``.
    2. **Naturally missing modalities**: a modality is missing for a sample when
       its image path / report is empty, or all its tabular features are missing.
       ``missing_modalities="drop"`` keeps complete cases only; ``"keep"`` keeps
       them with an availability mask. Samples with *no* selected modality are
       always dropped.
    3. **Index timepoint**: ``index_policy`` keeps every sample or one index
       sample per patient (``first`` / ``last`` by ``index_time_column``, or ``random``).

    Args:
        modalities: Modalities to include (e.g. ``[tabular]``, ``[imaging, tabular]``).
        metadata_columns: Cohort columns to carry along for splitting/subgroups.
            May include tabular feature columns (e.g. ``sex``, ``age``).
        tabular_columns: Restrict tabular features; ``None`` keeps all.
        tabular_exclude: Glob patterns of tabular features to leave out (after
            ``tabular_columns``), e.g. ``["micro_*", "med_*"]`` for a feature-group ablation.
        missing_modalities: ``drop`` or ``keep`` (see above).
        index_policy: ``all``, ``first``, ``last`` or ``random``.
        index_time_column: Column ordering each patient's samples.
        seed: Seed for ``index_policy="random"``.
    """

    def __init__(
        self,
        modalities: Sequence[Modality | str],
        metadata_columns: Sequence[str] = (),
        tabular_columns: Sequence[str] | None = None,
        tabular_exclude: Sequence[str] = (),
        missing_modalities: str = "drop",
        index_policy: str = "all",
        index_time_column: str | None = None,
        seed: int = 0,
    ) -> None:
        if not modalities:
            raise ValueError("At least one modality is required.")
        if missing_modalities not in ("drop", "keep"):
            raise ValueError("missing_modalities must be 'drop' or 'keep'")
        if index_policy not in ("all", "first", "last", "random"):
            raise ValueError("index_policy must be all | first | last | random")
        if index_policy in ("first", "last") and not index_time_column:
            raise ValueError("index_policy first/last needs index_time_column")
        self.modalities = frozenset(Modality(m) for m in modalities)
        self.metadata_columns = list(metadata_columns)
        self.tabular_columns = None if tabular_columns is None else list(tabular_columns)
        self.tabular_exclude = list(tabular_exclude)
        self.missing_modalities = missing_modalities
        self.index_policy = index_policy
        self.index_time_column = index_time_column
        self.seed = seed

    @classmethod
    def from_config(cls, config: ViewConfig, seed: int = 0) -> Self:
        """Create from a :class:`ViewConfig`."""
        return cls(
            modalities=config.modalities,
            metadata_columns=config.metadata_columns,
            tabular_columns=config.tabular_columns,
            tabular_exclude=config.tabular_exclude,
            missing_modalities=config.missing_modalities,
            index_policy=config.index_policy,
            index_time_column=config.index_time_column,
            seed=seed,
        )

    def build(self, cohort: Cohort) -> MultimodalView:
        """Harmonise ``cohort`` into a view.

        Raises:
            ValueError: If a requested modality or column is not in the cohort.
        """
        if missing := self.modalities - cohort.modalities:
            raise ValueError(
                f"Cohort has no data for {sorted(map(str, missing))}; "
                f"available: {sorted(map(str, cohort.modalities))}."
            )
        needed = [
            *self.metadata_columns,
            *([self.index_time_column] if self.index_time_column else []),
        ]
        unknown = [c for c in needed if c not in cohort.frame.columns]
        if unknown:
            raise KeyError(f"Columns not in cohort: {unknown}")

        tab_cols: list[str] = []
        if Modality.TABULAR in self.modalities:
            tab_cols = self.tabular_columns or cohort.modality_columns[Modality.TABULAR]
            absent = [c for c in tab_cols if c not in cohort.frame.columns]
            if absent:
                raise KeyError(f"Tabular columns not in cohort: {absent}")
            if self.tabular_exclude:
                before = len(tab_cols)
                tab_cols = [
                    c for c in tab_cols if not any(fnmatchcase(c, g) for g in self.tabular_exclude)
                ]
                if not tab_cols:
                    raise ValueError(
                        f"tabular_exclude {self.tabular_exclude} removes every feature"
                    )
                logger.info(
                    "Harmonizer: tabular_exclude %s dropped %d of %d features",
                    self.tabular_exclude,
                    before - len(tab_cols),
                    before,
                )

        frame = cohort.frame
        avail = self._availability(frame, tab_cols)
        complete = np.logical_and.reduce(list(avail.values()))
        anything = np.logical_or.reduce(list(avail.values()))
        keep = complete if self.missing_modalities == "drop" else anything
        if (n := int((~keep).sum())) > 0:
            reason = (
                "incomplete" if self.missing_modalities == "drop" else "with no selected modality"
            )
            logger.info("Harmonizer: dropping %d samples %s", n, reason)
        frame = frame.loc[keep]
        avail = {m: v[keep] for m, v in avail.items()}

        if self.index_policy != "all":
            pos = self._index_positions(frame, cohort.patient_column)
            frame = frame.iloc[pos]
            avail = {m: v[pos] for m, v in avail.items()}

        def column(name: str) -> np.ndarray:
            return frame[name].fillna("").astype(str).to_numpy()

        view = MultimodalView(
            sample_ids=frame[SAMPLE_ID_COLUMN].to_numpy(),
            patient_ids=frame[cohort.patient_column].to_numpy(),
            y=frame[cohort.label_column].to_numpy(dtype=int),
            class_names=list(cohort.class_names),
            tabular=frame[tab_cols].reset_index(drop=True) if tab_cols else None,
            image_paths=column(IMAGE_PATH_COLUMN) if Modality.IMAGING in self.modalities else None,
            text=column(REPORT_TEXT_COLUMN) if Modality.TEXT in self.modalities else None,
            metadata=frame[list(dict.fromkeys(needed))].reset_index(drop=True),
            availability=avail,
            patient_ids_assumed=cohort.patient_ids_assumed,
        )
        view.metadata[MODALITY_PATTERN_COLUMN] = view.modality_patterns()
        if not view.is_complete:
            logger.info(
                "Modality patterns: %s",
                view.metadata[MODALITY_PATTERN_COLUMN].value_counts().to_dict(),
            )
        return view

    def _availability(self, frame: pd.DataFrame, tab_cols: list[str]) -> dict[Modality, np.ndarray]:
        avail: dict[Modality, np.ndarray] = {}

        def nonempty(col: str) -> np.ndarray:
            values = frame[col]
            return (values.notna() & (values.astype(str).str.strip() != "")).to_numpy()

        if Modality.TABULAR in self.modalities:
            avail[Modality.TABULAR] = frame[tab_cols].notna().any(axis=1).to_numpy()
        if Modality.IMAGING in self.modalities:
            avail[Modality.IMAGING] = nonempty(IMAGE_PATH_COLUMN)
        if Modality.TEXT in self.modalities:
            avail[Modality.TEXT] = nonempty(REPORT_TEXT_COLUMN)
        return avail

    def _index_positions(self, frame: pd.DataFrame, patient_column: str) -> np.ndarray:
        """Positional indices of one index sample per patient."""
        work = pd.DataFrame(
            {"patient": frame[patient_column].to_numpy(), "pos": np.arange(len(frame))}
        )
        if self.index_policy == "random":
            rng = np.random.default_rng(self.seed)
            work["key"] = rng.random(len(work))
            ascending = True
        else:
            work["key"] = frame[self.index_time_column].to_numpy()  # type: ignore[index]
            ascending = self.index_policy == "first"
        work["tie"] = frame[SAMPLE_ID_COLUMN].to_numpy()
        chosen = work.sort_values(["key", "tie"], ascending=[ascending, True]).drop_duplicates(
            "patient"
        )
        logger.info(
            "Index policy '%s': %d → %d samples (one per patient)",
            self.index_policy,
            len(frame),
            len(chosen),
        )
        return np.sort(chosen["pos"].to_numpy())
