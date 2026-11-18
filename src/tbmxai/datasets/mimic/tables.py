"""Access to raw MIMIC tables (compressed CSVs) through DuckDB.

Large tables (``labevents``, ``chartevents``…) are never loaded whole: queries
filter by the selected patients inside DuckDB, streaming the ``.csv.gz`` files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from tbmxai.core.utils import get_logger
from tbmxai.datasets.mimic.config import MimicSourcesConfig

logger = get_logger(__name__)


@dataclass(frozen=True)
class TableSpec:
    """Location and typed columns of one raw table."""

    root: str  # attribute of MimicSourcesConfig
    relpath: str
    types: dict[str, str]


#: Raw tables used by the adapter. Columns not listed are read as VARCHAR.
TABLES: dict[str, TableSpec] = {
    # MIMIC-IV hosp
    "patients": TableSpec(
        "mimic_iv",
        "hosp/patients.csv.gz",
        {"subject_id": "BIGINT", "anchor_age": "INTEGER", "anchor_year": "INTEGER"},
    ),
    "admissions": TableSpec(
        "mimic_iv",
        "hosp/admissions.csv.gz",
        {
            "subject_id": "BIGINT",
            "hadm_id": "BIGINT",
            "admittime": "TIMESTAMP",
            "dischtime": "TIMESTAMP",
            "edregtime": "TIMESTAMP",
        },
    ),
    "diagnoses_icd": TableSpec(
        "mimic_iv",
        "hosp/diagnoses_icd.csv.gz",
        {
            "subject_id": "BIGINT",
            "hadm_id": "BIGINT",
            "seq_num": "INTEGER",
            "icd_version": "INTEGER",
        },
    ),
    "labevents": TableSpec(
        "mimic_iv",
        "hosp/labevents.csv.gz",
        {
            "subject_id": "BIGINT",
            "itemid": "BIGINT",
            "charttime": "TIMESTAMP",
            "valuenum": "DOUBLE",
        },
    ),
    "d_labitems": TableSpec("mimic_iv", "hosp/d_labitems.csv.gz", {"itemid": "BIGINT"}),
    "omr": TableSpec("mimic_iv", "hosp/omr.csv.gz", {"subject_id": "BIGINT", "chartdate": "DATE"}),
    "prescriptions": TableSpec(
        "mimic_iv",
        "hosp/prescriptions.csv.gz",
        {"subject_id": "BIGINT", "starttime": "TIMESTAMP", "stoptime": "TIMESTAMP"},
    ),
    "microbiologyevents": TableSpec(
        "mimic_iv",
        "hosp/microbiologyevents.csv.gz",
        {"subject_id": "BIGINT", "chartdate": "TIMESTAMP", "charttime": "TIMESTAMP"},
    ),
    "transfers": TableSpec(
        "mimic_iv",
        "hosp/transfers.csv.gz",
        {
            "subject_id": "BIGINT",
            "hadm_id": "BIGINT",
            "intime": "TIMESTAMP",
            "outtime": "TIMESTAMP",
        },
    ),
    "services": TableSpec(
        "mimic_iv",
        "hosp/services.csv.gz",
        {"subject_id": "BIGINT", "hadm_id": "BIGINT", "transfertime": "TIMESTAMP"},
    ),
    "procedures_icd": TableSpec(
        "mimic_iv",
        "hosp/procedures_icd.csv.gz",
        {
            "subject_id": "BIGINT",
            "hadm_id": "BIGINT",
            "seq_num": "INTEGER",
            "chartdate": "DATE",
            "icd_version": "INTEGER",
        },
    ),
    # MIMIC-IV icu
    "icustays": TableSpec(
        "mimic_iv",
        "icu/icustays.csv.gz",
        {
            "subject_id": "BIGINT",
            "hadm_id": "BIGINT",
            "stay_id": "BIGINT",
            "intime": "TIMESTAMP",
            "outtime": "TIMESTAMP",
        },
    ),
    "chartevents": TableSpec(
        "mimic_iv",
        "icu/chartevents.csv.gz",
        {
            "subject_id": "BIGINT",
            "itemid": "BIGINT",
            "charttime": "TIMESTAMP",
            "valuenum": "DOUBLE",
        },
    ),
    "d_items": TableSpec("mimic_iv", "icu/d_items.csv.gz", {"itemid": "BIGINT"}),
    # MIMIC-IV-ED
    "edstays": TableSpec(
        "mimic_iv_ed",
        "ed/edstays.csv.gz",
        {
            "subject_id": "BIGINT",
            "hadm_id": "BIGINT",
            "stay_id": "BIGINT",
            "intime": "TIMESTAMP",
            "outtime": "TIMESTAMP",
        },
    ),
    "triage": TableSpec(
        "mimic_iv_ed",
        "ed/triage.csv.gz",
        {
            "subject_id": "BIGINT",
            "stay_id": "BIGINT",
            "temperature": "DOUBLE",
            "heartrate": "DOUBLE",
            "resprate": "DOUBLE",
            "o2sat": "DOUBLE",
            "sbp": "DOUBLE",
            "dbp": "DOUBLE",
            "acuity": "DOUBLE",
        },
    ),
    "vitalsign": TableSpec(
        "mimic_iv_ed",
        "ed/vitalsign.csv.gz",
        {
            "subject_id": "BIGINT",
            "stay_id": "BIGINT",
            "charttime": "TIMESTAMP",
            "temperature": "DOUBLE",
            "heartrate": "DOUBLE",
            "resprate": "DOUBLE",
            "o2sat": "DOUBLE",
            "sbp": "DOUBLE",
            "dbp": "DOUBLE",
        },
    ),
    "ed_diagnosis": TableSpec(
        "mimic_iv_ed",
        "ed/diagnosis.csv.gz",
        {
            "subject_id": "BIGINT",
            "stay_id": "BIGINT",
            "seq_num": "INTEGER",
            "icd_version": "INTEGER",
        },
    ),
    # MIMIC-CXR-JPG
    "cxr_metadata": TableSpec(
        "mimic_cxr_jpg",
        "mimic-cxr-2.0.0-metadata.csv.gz",
        {
            "subject_id": "BIGINT",
            "study_id": "BIGINT",
            "StudyDate": "VARCHAR",
            "StudyTime": "VARCHAR",
        },
    ),
    "cxr_split": TableSpec(
        "mimic_cxr_jpg",
        "mimic-cxr-2.0.0-split.csv.gz",
        {"subject_id": "BIGINT", "study_id": "BIGINT"},
    ),
    "cxr_chexpert": TableSpec(
        "mimic_cxr_jpg",
        "mimic-cxr-2.0.0-chexpert.csv.gz",
        {"subject_id": "BIGINT", "study_id": "BIGINT"},
    ),
}

#: Tables without which nothing can be built.
REQUIRED_TABLES = ("patients", "admissions", "diagnoses_icd", "cxr_metadata")


class MimicTables:
    """Resolve and query raw MIMIC tables.

    Args:
        sources: Roots of the PhysioNet downloads.
        memory_limit: DuckDB memory limit (e.g. ``"16GB"``).
        threads: DuckDB threads (``None`` = all cores).
    """

    def __init__(
        self,
        sources: MimicSourcesConfig,
        memory_limit: str | None = None,
        threads: int | None = None,
        temp_directory: str | Path | None = None,
    ) -> None:
        self.sources = sources
        self.con = duckdb.connect()
        if temp_directory:  # where joins spill when exceeding memory_limit
            tmp = Path(temp_directory).expanduser()
            tmp.mkdir(parents=True, exist_ok=True)
            self.con.execute(f"SET temp_directory='{tmp}'")
            self.con.execute("SET preserve_insertion_order=false")
        if memory_limit:
            self.con.execute(f"SET memory_limit='{memory_limit}'")
        if threads:
            self.con.execute(f"SET threads={int(threads)}")

    def path(self, key: str) -> Path | None:
        """Absolute path of table ``key`` (``None`` if its root is not configured)."""
        spec = TABLES[key]
        root = getattr(self.sources, spec.root)
        if root is None:
            return None
        base = Path(root).expanduser() / spec.relpath
        if base.exists():
            return base
        plain = base.with_suffix("")  # also accept uncompressed .csv
        return plain if plain.exists() else base

    def available(self, key: str) -> bool:
        """Whether table ``key`` exists on disk."""
        p = self.path(key)
        return p is not None and p.exists()

    def status(self) -> pd.DataFrame:
        """Availability of every known table (for ``tbmxai mimic status``)."""
        rows = []
        for key, spec in TABLES.items():
            p = self.path(key)
            rows.append(
                {
                    "table": key,
                    "required": key in REQUIRED_TABLES,
                    "present": bool(p and p.exists()),
                    "size_mb": round(p.stat().st_size / 2**20, 1) if p and p.exists() else None,
                    "path": str(p) if p else f"<{spec.root} not configured>",
                }
            )
        return pd.DataFrame(rows)

    def check_required(self) -> None:
        """Raise ``FileNotFoundError`` listing missing required tables."""
        missing = [f"{k} ({self.path(k)})" for k in REQUIRED_TABLES if not self.available(k)]
        if missing:
            raise FileNotFoundError("Missing required MIMIC tables: " + ", ".join(missing))

    def scan(self, key: str) -> str:
        """SQL table expression reading ``key`` with typed columns."""
        p = self.path(key)
        if p is None or not p.exists():
            raise FileNotFoundError(f"MIMIC table '{key}' not found at {p}")
        types = ", ".join(f"'{c}': '{t}'" for c, t in TABLES[key].types.items())
        path = str(p).replace("'", "''")
        return (
            f"read_csv('{path}', header=true, delim=',', quote='\"', escape='\"', "
            f"all_varchar=true, types={{{types}}})"
        )

    def query(self, sql: str, **frames: pd.DataFrame) -> pd.DataFrame:
        """Run ``sql`` with ``frames`` registered as DuckDB views; return a DataFrame."""
        for name, frame in frames.items():
            self.con.register(name, frame)
        try:
            return self.con.execute(sql).df()
        finally:
            for name in frames:
                self.con.unregister(name)

    def read(self, key: str, where: str = "", **frames: pd.DataFrame) -> pd.DataFrame:
        """``SELECT * FROM key [WHERE ...]``."""
        clause = f" WHERE {where}" if where else ""
        return self.query(f"SELECT * FROM {self.scan(key)}{clause}", **frames)
