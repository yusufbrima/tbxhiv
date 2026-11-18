"""Abstract data source and shared file-reading helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pandas as pd

from tbmxai.core.registry import Registry
from tbmxai.core.types import Modality

#: Canonical column names produced by sources for non-tabular payloads.
IMAGE_PATH_COLUMN = "image_path"
REPORT_TEXT_COLUMN = "report_text"


@dataclass
class SourceTable:
    """Output of :meth:`DataSource.load`.

    Attributes:
        frame: Table containing ``key_column`` plus payload/metadata columns.
        modality: Modality of the payload columns.
        key_column: Name of the patient identifier column in ``frame``.
        payload_columns: Columns that are model inputs for ``modality``.
        metadata_columns: Columns carried along but not used as model inputs.
        patient_ids_assumed: True if the source had no real patient ids (each
            item was treated as its own patient); patient-level splitting then
            cannot be guaranteed and every report says so.
    """

    frame: pd.DataFrame
    modality: Modality
    key_column: str
    payload_columns: list[str]
    metadata_columns: list[str] = field(default_factory=list)
    patient_ids_assumed: bool = False


class DataSource(ABC):
    """A single raw input (e.g. an EHR export, a CXR manifest, a report dump).

    Subclasses implement :meth:`_read` and declare :attr:`modality`. The base
    class handles key validation, renaming and column selection so that every
    source returns a consistent :class:`SourceTable`.

    Args:
        key_column: Patient identifier column as named *in this source*.
            The :class:`~tbmxai.data.ingest.Ingestor` renames it to the
            canonical patient column.
        rename: Optional column renames applied right after reading.
        metadata_columns: Columns to keep as metadata (not model inputs).
    """

    modality: ClassVar[Modality]
    #: Set by sources that synthesise patient ids (see :class:`SourceTable`).
    patient_ids_assumed: bool = False

    def __init__(
        self,
        key_column: str = "patient_id",
        rename: dict[str, str] | None = None,
        metadata_columns: Sequence[str] = (),
    ) -> None:
        self.key_column = key_column
        self.rename = dict(rename or {})
        self.metadata_columns = list(metadata_columns)

    @abstractmethod
    def _read(self) -> pd.DataFrame:
        """Read the raw table (after which renames are applied)."""

    @abstractmethod
    def _payload_columns(self, frame: pd.DataFrame) -> list[str]:
        """Return the model-input columns of ``frame`` (excluding key and metadata)."""

    def load(self, keep: Sequence[str] = ()) -> SourceTable:
        """Read, validate and package this source.

        Args:
            keep: Extra columns to retain if present (e.g. the label column),
                regardless of the feature/metadata selection.

        Raises:
            KeyError: If the key or a requested column is missing.
        """
        frame = self._read().rename(columns=self.rename)
        missing = [c for c in (self.key_column, *self.metadata_columns) if c not in frame.columns]
        if missing:
            raise KeyError(f"{type(self).__name__}: missing columns {missing}")
        frame = frame.dropna(subset=[self.key_column])
        frame[self.key_column] = frame[self.key_column].astype(str)
        payload = self._payload_columns(frame)
        extra = [c for c in keep if c in frame.columns]
        keep = [self.key_column, *payload, *self.metadata_columns, *extra]
        return SourceTable(
            frame=frame.loc[:, list(dict.fromkeys(keep))].reset_index(drop=True),
            modality=self.modality,
            key_column=self.key_column,
            payload_columns=payload,
            metadata_columns=self.metadata_columns,
            patient_ids_assumed=self.patient_ids_assumed,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key_column={self.key_column!r})"


SOURCES: Registry[DataSource] = Registry("data_source")


def read_table(path: str | Path, **kwargs: object) -> pd.DataFrame:
    """Read a CSV/TSV/Parquet/Excel file based on its suffix."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, **kwargs)  # type: ignore[arg-type]
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", **kwargs)  # type: ignore[arg-type]
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, **kwargs)  # type: ignore[arg-type]
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported table format: {path}")
