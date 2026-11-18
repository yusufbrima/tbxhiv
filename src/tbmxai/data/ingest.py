"""Ingestion: link heterogeneous sources into a single :class:`Cohort`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Any, Literal, Self

import pandas as pd

from tbmxai.core.config import DataConfig
from tbmxai.core.types import Modality
from tbmxai.core.utils import get_logger
from tbmxai.data.sources import SOURCES, DataSource, SourceTable

logger = get_logger(__name__)

SAMPLE_ID_COLUMN = "sample_id"


@dataclass
class Cohort:
    """Linked multimodal record table — one row per *sample*.

    A sample is one unit of prediction (typically one CXR study). Patients may
    contribute several samples when a source has multiple rows per patient;
    splitters group by :attr:`patient_column` to prevent leakage.

    Attributes:
        frame: Linked table with ``sample_id``, patient, label, payload and
            metadata columns. Labels are integer-encoded.
        patient_column: Patient identifier column.
        label_column: Integer-encoded label column.
        class_names: ``class_names[k]`` is the original label of class ``k``.
        modality_columns: Payload columns contributed per modality.
        metadata_columns: Non-input columns carried along.
    """

    frame: pd.DataFrame
    patient_column: str
    label_column: str
    class_names: list[str]
    modality_columns: dict[Modality, list[str]]
    metadata_columns: list[str] = field(default_factory=list)
    patient_ids_assumed: bool = False

    @property
    def modalities(self) -> set[Modality]:
        """Modalities with at least one payload column."""
        return {m for m, cols in self.modality_columns.items() if cols}

    def __len__(self) -> int:
        return len(self.frame)

    def summary(self) -> dict[str, Any]:
        """Counts, label distribution and per-modality completeness."""
        completeness = {
            str(m): float(self.frame[cols].notna().any(axis=1).mean())
            for m, cols in self.modality_columns.items()
            if cols
        }
        counts = self.frame[self.label_column].value_counts().sort_index()
        return {
            "n_samples": len(self.frame),
            "n_patients": int(self.frame[self.patient_column].nunique()),
            "modalities": sorted(str(m) for m in self.modalities),
            "n_columns": {str(m): len(c) for m, c in self.modality_columns.items()},
            "label_counts": {self.class_names[int(k)]: int(v) for k, v in counts.items()},
            "modality_completeness": completeness,
            "patient_ids_assumed": self.patient_ids_assumed,
        }


class Ingestor:
    """Load sources and join them on the patient identifier.

    Args:
        sources: Data sources to ingest (any mix of modalities).
        patient_column: Canonical patient id name; each source's key is renamed to it.
        label_column: Column holding the target; must appear in exactly one source.
        link_columns: Extra join keys (e.g. ``study_id``). Each pair of tables is
            joined on the patient column plus the link columns both contain.
            Link columns are never model inputs.
        join: How to join sources (``inner`` keeps patients present in all).
        class_names: Ordered label values. Inferred (sorted) when ``None``.

    Raises:
        ValueError: On column collisions, a missing/duplicated label column, or
            a many-to-many join (both sides have duplicate keys, which would
            produce a Cartesian product).
    """

    def __init__(
        self,
        sources: Sequence[DataSource],
        patient_column: str = "patient_id",
        label_column: str = "label",
        join: Literal["inner", "left", "outer"] = "inner",
        class_names: Sequence[str] | None = None,
        link_columns: Sequence[str] = (),
        patient_ids_assumed: bool = False,
    ) -> None:
        if not sources:
            raise ValueError("At least one data source is required.")
        self.sources = list(sources)
        self.patient_column = patient_column
        self.label_column = label_column
        self.join = join
        self.link_columns = list(link_columns)
        self.patient_ids_assumed = patient_ids_assumed
        self.class_names = None if class_names is None else [str(c) for c in class_names]

    @classmethod
    def from_config(cls, config: DataConfig) -> Self:
        """Build sources from the registry (or a standard dataset) and create an ingestor."""
        specs = list(config.sources)
        label = config.label_column or "label"
        link = list(config.link_columns)
        classes = config.class_names
        patient_column = config.patient_column
        assumed = False
        if config.dataset is not None:
            from tbmxai.data.dataset_format import DatasetSpec

            ds = DatasetSpec.load(config.dataset)
            label = config.label_column or ds.default_label()
            specs = ds.sources(label, use=config.use) + specs
            link = link or ds.link_columns
            classes = classes or ds.labels[label].classes
            assumed = ds.patient_ids == "assumed"
            patient_column = "patient_id"
            logger.info(
                "Dataset '%s' (%s): label '%s', modalities %s",
                ds.name,
                ds.root,
                label,
                ds.modalities,
            )
        sources = [SOURCES.build(s.name, **s.params) for s in specs]
        return cls(
            sources,
            patient_column=patient_column,
            label_column=label,
            join=config.join,
            class_names=classes,
            link_columns=link,
            patient_ids_assumed=assumed,
        )

    def run(self) -> Cohort:
        """Load, validate, join and label-encode all sources."""
        keep = [self.label_column, *self.link_columns]
        tables = [self._canonicalise(src.load(keep=keep)) for src in self.sources]
        self._check_tables(tables)
        frame = reduce(self._merge, (t.frame for t in tables))
        frame = self._encode_labels(frame).reset_index(drop=True)
        frame.insert(0, SAMPLE_ID_COLUMN, self._sample_keys(frame))

        modality_columns: dict[Modality, list[str]] = {m: [] for m in Modality}
        metadata: list[str] = []
        not_inputs = {self.label_column, *self.link_columns}
        for table in tables:
            modality_columns[table.modality] += [
                c for c in table.payload_columns if c not in not_inputs
            ]
            metadata += [c for c in table.metadata_columns if c not in not_inputs]
        metadata += [c for c in self.link_columns if c in frame.columns]

        cohort = Cohort(
            frame=frame.reset_index(drop=True),
            patient_column=self.patient_column,
            label_column=self.label_column,
            class_names=self._fitted_class_names,
            modality_columns=modality_columns,
            metadata_columns=list(dict.fromkeys(metadata)),
            patient_ids_assumed=self.patient_ids_assumed
            or any(t.patient_ids_assumed for t in tables),
        )
        logger.info("Ingested cohort: %s", cohort.summary())
        return cohort

    # ------------------------------------------------------------------ helpers

    def _canonicalise(self, table: SourceTable) -> SourceTable:
        """Rename the source key to the canonical patient column."""
        if table.key_column != self.patient_column:
            if self.patient_column in table.frame.columns:
                raise ValueError(
                    f"Source has both key '{table.key_column}' and a "
                    f"'{self.patient_column}' column."
                )
            table.frame = table.frame.rename(columns={table.key_column: self.patient_column})
            table.key_column = self.patient_column
        return table

    def _sample_keys(self, frame: pd.DataFrame) -> pd.Series:
        """Stable, content-derived sample ids: ``patient|link…|image file``.

        Identical across experiments that share the same underlying samples
        (independent of row order, sources joined, or modalities selected), so
        samples can be matched across models and protocols. Remaining duplicates
        get a ``#k`` suffix in a deterministic order.
        """
        from tbmxai.data.sources import IMAGE_PATH_COLUMN

        parts = [frame[self.patient_column].astype(str)]
        parts += [frame[c].astype(str) for c in self.link_columns if c in frame]
        if IMAGE_PATH_COLUMN in frame:
            parts.append(
                frame[IMAGE_PATH_COLUMN].fillna("").astype(str).map(lambda p: Path(p).name)
            )
        key = parts[0].str.cat(parts[1:], sep="|") if len(parts) > 1 else parts[0]
        dup = key.groupby(key).cumcount()
        return key.where(dup == 0, key + "#" + dup.astype(str)).reset_index(drop=True)

    def _merge(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        """Join on the patient column + shared link columns; refuse many-to-many joins."""
        on = [self.patient_column, *(c for c in self.link_columns if c in left and c in right)]
        if left.duplicated(on).any() and right.duplicated(on).any():
            raise ValueError(
                f"Both tables have duplicate keys on {on}; joining them would produce a "
                "Cartesian product. Add a link column (data.link_columns, e.g. study_id) "
                "or aggregate one table beforehand."
            )
        if len(on) > 1:  # align dtypes of link keys (e.g. int vs str across files)
            left = left.astype({c: str for c in on[1:]})
            right = right.astype({c: str for c in on[1:]})
        return left.merge(right, on=on, how=self.join)

    def _check_tables(self, tables: list[SourceTable]) -> None:
        seen: dict[str, int] = {}
        shared = {self.patient_column, *self.link_columns}
        for i, table in enumerate(tables):
            for col in table.frame.columns:
                if col in shared:
                    continue
                if col in seen:
                    raise ValueError(
                        f"Column '{col}' appears in sources {seen[col]} and {i}; "
                        "use `rename` on one of them."
                    )
                seen[col] = i
        if self.label_column not in seen:
            raise ValueError(f"Label column '{self.label_column}' not found in any source.")

    def _encode_labels(self, frame: pd.DataFrame) -> pd.DataFrame:
        n_missing = int(frame[self.label_column].isna().sum())
        if n_missing:
            logger.warning("Dropping %d samples with missing labels", n_missing)
            frame = frame.dropna(subset=[self.label_column])
        # integer-valued labels read as floats ("1.0") must match classes such as "1"
        raw = frame[self.label_column].astype(str).str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)
        names = self.class_names or sorted(raw.unique(), key=_natural_key)
        unknown = set(raw.unique()) - set(names)
        if unknown:
            raise ValueError(f"Labels {sorted(unknown)} not in class_names {names}.")
        self._fitted_class_names = list(names)
        frame = frame.copy()
        frame[self.label_column] = raw.map({n: i for i, n in enumerate(names)}).astype(int)
        return frame


def _natural_key(value: str) -> tuple[int, float | str]:
    """Sort numeric-looking labels numerically (``"0" < "1" < "10"``), others lexically."""
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)
