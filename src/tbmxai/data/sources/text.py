"""Text sources: radiology and clinical reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tbmxai.core.types import Modality
from tbmxai.data.sources.base import REPORT_TEXT_COLUMN, SOURCES, DataSource, read_table


@SOURCES.register("report_file")
class ReportFileSource(DataSource):
    """A table of free-text reports keyed by patient.

    Args:
        path: File to read.
        text_column: Column holding the report text.
        **kwargs: Forwarded to :class:`DataSource`.
    """

    modality = Modality.TEXT

    def __init__(self, path: str | Path, text_column: str = "report", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.path = Path(path)
        self.text_column = text_column

    def _read(self) -> pd.DataFrame:
        frame = read_table(self.path)
        if self.text_column not in frame.columns:
            raise KeyError(f"{self.path.name}: text column '{self.text_column}' not found")
        frame[REPORT_TEXT_COLUMN] = frame[self.text_column].fillna("").astype(str)
        return frame

    def _payload_columns(self, frame: pd.DataFrame) -> list[str]:
        return [REPORT_TEXT_COLUMN]
