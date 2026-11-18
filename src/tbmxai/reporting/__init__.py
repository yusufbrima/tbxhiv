"""Reporting: run results, report writers, provenance and run loading."""

from tbmxai.reporting.artifacts import save_explanations
from tbmxai.reporting.base import REPORTERS, ReportWriter, RunResult
from tbmxai.reporting.html import HTMLReportWriter
from tbmxai.reporting.provenance import collect_run_info, load_run, write_manifest
from tbmxai.reporting.writers import (
    SCHEMA_VERSION,
    TABLE_SCHEMAS,
    JSONReportWriter,
    MarkdownReportWriter,
    TablesReportWriter,
)

__all__ = [
    "REPORTERS",
    "SCHEMA_VERSION",
    "TABLE_SCHEMAS",
    "HTMLReportWriter",
    "JSONReportWriter",
    "MarkdownReportWriter",
    "ReportWriter",
    "RunResult",
    "TablesReportWriter",
    "collect_run_info",
    "load_run",
    "save_explanations",
    "write_manifest",
]
