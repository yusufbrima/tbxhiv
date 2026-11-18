"""Data sources. Importing this package registers all built-in sources."""

from tbmxai.data.sources.base import (
    IMAGE_PATH_COLUMN,
    REPORT_TEXT_COLUMN,
    SOURCES,
    DataSource,
    SourceTable,
    read_table,
)
from tbmxai.data.sources.imaging import CXRDirectorySource, CXRManifestSource, ImageFolderSource
from tbmxai.data.sources.tabular import TabularFileSource
from tbmxai.data.sources.text import ReportFileSource

__all__ = [
    "IMAGE_PATH_COLUMN",
    "REPORT_TEXT_COLUMN",
    "SOURCES",
    "CXRDirectorySource",
    "CXRManifestSource",
    "DataSource",
    "ImageFolderSource",
    "ReportFileSource",
    "SourceTable",
    "TabularFileSource",
    "read_table",
]
