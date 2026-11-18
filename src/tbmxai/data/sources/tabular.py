"""Tabular sources: EHR, demographics, labs, label files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from tbmxai.core.types import Modality
from tbmxai.data.sources.base import SOURCES, DataSource, read_table


@SOURCES.register("tabular_file")
class TabularFileSource(DataSource):
    """A CSV/TSV/Parquet/Excel table with one or more rows per patient.

    Args:
        path: File to read.
        feature_columns: Columns to use as tabular model inputs. ``None`` means
            every column except the key and ``metadata_columns``. Pass ``[]``
            for a pure label/metadata file.
        exclude_columns: Columns never used as features when ``feature_columns`` is ``None``
            (e.g. label columns of other targets, timestamps, identifiers).
        read_kwargs: Extra kwargs forwarded to the pandas reader.
        **kwargs: Forwarded to :class:`DataSource`.

    Note:
        The label column may live in any tabular source; the ingestor removes
        it from the feature set automatically.
    """

    modality = Modality.TABULAR

    def __init__(
        self,
        path: str | Path,
        feature_columns: Sequence[str] | None = None,
        exclude_columns: Sequence[str] = (),
        read_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.path = Path(path)
        self.feature_columns = None if feature_columns is None else list(feature_columns)
        self.exclude_columns = list(exclude_columns)
        self.read_kwargs = dict(read_kwargs or {})

    def _read(self) -> pd.DataFrame:
        return read_table(self.path, **self.read_kwargs)

    def _payload_columns(self, frame: pd.DataFrame) -> list[str]:
        if self.feature_columns is not None:
            missing = [c for c in self.feature_columns if c not in frame.columns]
            if missing:
                raise KeyError(f"{self.path.name}: feature columns not found: {missing}")
            return self.feature_columns
        excluded = {self.key_column, *self.metadata_columns, *self.exclude_columns}
        return [c for c in frame.columns if c not in excluded]
