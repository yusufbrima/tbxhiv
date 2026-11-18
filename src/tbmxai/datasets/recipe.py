"""Declarative dataset recipes: raw files and tables → tbmxai standard dataset, no code.

A recipe is a YAML file with five parts::

    name: nlm_tb
    description: Shenzhen + Montgomery TB chest X-ray sets
    patient_ids: assumed               # provided | assumed (no real patient ids)
    archives: ["*.zip"]                # optional: extracted once into RAW/<zip stem>/

    records:                           # what one sample is
      files: "**/CXR_png/*.png"        # one record per matching file (relative to RAW) …
      # table: meta.csv                # … or one record per row of a table
      dedupe: patient_id               # optional: keep the first record per value
      sort_by: patient_id

    fields:                            # derived columns, evaluated in order
      patient_id: {from: filename, regex: '(?P<id>CHNCXR_\\d+)'}
      tb:         {from: filename, regex: '_(\\d)\\.png$', map: {"0": normal, "1": tb}}
      reading_text: {from: file, glob: "**/ClinicalReadings/*.txt", lookup_by: stem}
      parsed:     {from: field:reading_text, parse: nlm_reading}  # dict → fields sex, age, …
      age_group:  {from: field:age, bins: [0, 18, 65, 200], labels: ["<18", "18-64", "65+"]}

    tables:                            # which fields go where
      labels: [tb, sex, age_group]     # + patient_id (and study_id if defined)
      tabular: [age]
      images: []                       # image_path is added automatically for file records
      extra: {readings: [reading]}     # written, never loaded (reference only)

    labels:                            # → dataset.yaml
      tb: {classes: [normal, tb], description: TB on the radiograph}

Field sources (``from``):

============== =================================================================
``filename``   file name (``CHNCXR_0001_0.png``)
``stem``       file name without suffix
``path``       path relative to RAW (posix)
``parent``     name of the containing folder; ``parent:2`` = two levels up
``column:X``   column ``X`` of a table record
``field:X``    a field defined earlier
``const``      the literal ``value``
``file``       contents of a side-car text file: files matching ``glob`` are indexed
               by stem; the record's ``lookup_by`` value (default ``stem``) is looked up
``table``      a column of another table: rows of ``path`` indexed by ``key``; the
               record's ``lookup_by`` value is looked up and ``column`` returned
``list``       membership in list files (``lists: {train: lists/train.txt, …}``);
               a line matches when the record path ends with it
============== =================================================================

Transforms, applied in this order: ``regex`` (+ ``group``), ``parse`` (a registered
parser, see :data:`~tbmxai.datasets.adapters.PARSERS`), ``map`` (+ ``unmapped``:
``error`` | ``missing`` | ``keep``), ``format`` (e.g. ``"site_{}"``), ``cast`` (``str`` |
``float`` | ``int``),
``bins`` / ``labels`` (left-closed intervals), ``default`` (fills missing values).

YAML note: quote map keys that YAML would otherwise convert (``"1"``, ``"yes"``,
``"no"``, ``"on"``, ``"off"``, ``"null"``); unquoted they become numbers, booleans or null.

The build writes the tables, ``dataset.yaml`` and ``build_info.json`` (recipe hash,
extracted archives, counts, missing values per field). Run
``tbmxai dataset validate OUT`` (``tbmxai dataset build`` does it for you).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
import yaml
from pydantic import ConfigDict, Field, model_validator

from tbmxai.core.config import _Strict
from tbmxai.core.utils import get_logger

logger = get_logger(__name__)

Role = Literal["labels", "tabular", "images", "reports"]
_FILE_SOURCES = {"filename", "stem", "path", "parent"}


class FieldSpec(_Strict):
    """How one field is derived for every record."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source: str = Field(alias="from", description="Where the raw value comes from.")
    value: Any = Field(default=None, description="from: const")
    glob: str | None = Field(default=None, description="from: file — side-car files to index.")
    path: str | None = Field(default=None, description="from: table — table file (csv/tsv/xlsx).")
    key: str | None = Field(default=None, description="from: table — key column of the table.")
    column: str | None = Field(default=None, description="from: table — column to return.")
    lists: dict[str, str] | None = Field(default=None, description="from: list — value → file.")
    lookup_by: str = Field(default="stem", description="Record value used for file/table lookups.")
    regex: str | None = None
    group: str | int | None = Field(
        default=None, description="Regex group (default: 'id' if present, else 1, else 0)."
    )
    parse: str | None = None
    map: dict[str, Any] | None = None
    unmapped: Literal["error", "missing", "keep"] = "error"
    format: str | None = Field(
        default=None, description="Python format string applied to the value, e.g. 'tbx11k_{}'."
    )
    cast: Literal["str", "float", "int"] | None = None
    bins: list[float] | None = None
    labels: list[str] | None = None
    default: Any = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        kind = self.source.split(":", 1)[0]
        known = _FILE_SOURCES | {"column", "field", "const", "file", "table", "list"}
        if kind not in known:
            raise ValueError(f"unknown source '{self.source}' (one of {sorted(known)})")
        if kind == "file" and not self.glob:
            raise ValueError("from: file needs `glob`")
        if kind == "table" and not (self.path and self.key and self.column):
            raise ValueError("from: table needs `path`, `key` and `column`")
        if kind == "list" and not self.lists:
            raise ValueError("from: list needs `lists`")
        if (self.bins is None) != (self.labels is None):
            raise ValueError("`bins` and `labels` go together")
        if self.bins is not None and len(self.labels or []) != len(self.bins) - 1:
            raise ValueError("`labels` needs one entry fewer than `bins`")
        return self


class RecordsSpec(_Strict):
    """What one sample (record) is: a file, or a row of a table."""

    files: str | list[str] | None = Field(default=None, description="Glob(s) relative to RAW.")
    table: str | None = Field(default=None, description="Table relative to RAW (one row each).")
    exclude: list[str] = Field(
        default_factory=lambda: ["__MACOSX/*", "*/__MACOSX/*"],
        description="Globs (relative paths) to skip; hidden files are always skipped.",
    )
    dedupe: str | None = Field(default=None, description="Keep the first record per value.")
    drop_duplicate_images: bool = Field(
        default=False,
        description="Fingerprint every image (difference hash) and keep one record per "
        "near-identical image; groups whose labels disagree are dropped entirely. "
        "Dropped records are listed in build_info.json.",
    )
    sort_by: str | None = Field(default=None, description="Field to sort the output by.")

    @model_validator(mode="after")
    def _one_kind(self) -> Self:
        if (self.files is None) == (self.table is None):
            raise ValueError("records needs exactly one of `files` or `table`")
        return self


class LabelSpec(_Strict):
    classes: list[str] = Field(min_length=2)
    description: str = ""


class TablesSpec(_Strict):
    labels: list[str] = Field(min_length=1)
    tabular: list[str] | None = None
    images: list[str] | None = None
    reports: list[str] | None = None
    extra: dict[str, list[str]] = Field(default_factory=dict)


class Recipe(_Strict):
    """A parsed recipe file."""

    name: str
    description: str = ""
    patient_ids: Literal["provided", "assumed"] = "provided"
    archives: list[str] = Field(default_factory=list)
    records: RecordsSpec
    fields: dict[str, FieldSpec]
    tables: TablesSpec
    labels: dict[str, LabelSpec] = Field(min_length=1)
    image_root: Literal["raw", "out"] = Field(
        default="raw", description="raw: image paths relative to RAW (images stay in place)."
    )
    time_column: str | None = None

    _source: Path | None = None
    _sha256: str = ""

    @classmethod
    def load(cls, path: str | Path) -> Self:
        p = Path(path).expanduser()
        text = p.read_text()
        recipe = cls.model_validate(yaml.safe_load(text) or {})
        recipe._source = p.resolve()
        recipe._sha256 = hashlib.sha256(text.encode()).hexdigest()
        return recipe

    @model_validator(mode="after")
    def _check_tables(self) -> Self:
        if "patient_id" not in self.fields:
            raise ValueError("fields must define patient_id")
        for name in self.labels:
            if name not in self.tables.labels:
                raise ValueError(f"label '{name}' must be listed in tables.labels")
        if self.tables.reports is not None and "report_text" not in self.fields:
            raise ValueError("tables.reports needs a field named report_text")
        return self

    # ------------------------------------------------------------------ build

    def build(self, raw_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
        """Run the recipe on ``raw_dir`` and write the standard dataset into ``out_dir``."""
        from tbmxai.data.dataset_format import DatasetSpec
        from tbmxai.data.dataset_format import LabelSpec as DatasetLabel

        raw = Path(raw_dir).expanduser().resolve()
        out = Path(out_dir).expanduser()
        if not raw.is_dir():
            raise FileNotFoundError(f"Raw folder {raw} not found")
        extracted = _extract_archives(raw, self.archives)
        records = self._records(raw)
        if records.empty:
            raise ValueError(f"No records found in {raw} for {self.records}")

        frame = _Evaluator(raw, records, self.fields).run()
        frame = self._finalise(frame)
        dropped: list[dict[str, Any]] = []
        if self.records.drop_duplicate_images:
            frame, dropped = self._drop_duplicate_images(frame, raw)
        out.mkdir(parents=True, exist_ok=True)
        written = self._write_tables(frame, out)
        DatasetSpec(
            name=self.name,
            description=self.description,
            patient_ids=self.patient_ids,
            labels={
                k: DatasetLabel(classes=v.classes, description=v.description)
                for k, v in self.labels.items()
            },
            time_column=self.time_column,
            image_root=str(raw) if self.image_root == "raw" and "image_path" in frame else None,
        ).save(out)

        info = {
            "created_utc": datetime.now(UTC).isoformat(),
            "recipe": str(self._source) if self._source else None,
            "recipe_sha256": self._sha256,
            "raw_dir": str(raw),
            "archives_extracted": [p.name for p in extracted],
            "n_records": len(frame),
            "n_patients": int(frame["patient_id"].nunique()),
            "label_counts": {
                k: frame[k].value_counts(dropna=False).sort_index().to_dict() for k in self.labels
            },
            "missing_per_field": {
                k: int(frame[k].isna().sum()) for k in frame if frame[k].isna().any()
            },
            "tables": written,
            "duplicate_images_dropped": dropped,
        }
        (out / "build_info.json").write_text(json.dumps(info, indent=2, default=str))
        logger.info("Recipe '%s': %d records → %s", self.name, len(frame), out)
        return info

    def _records(self, raw: Path) -> pd.DataFrame:
        spec = self.records
        if spec.table is not None:
            table = _read_table(raw / spec.table)
            return table.rename(columns=lambda c: f"column:{c}")
        globs = [spec.files] if isinstance(spec.files, str) else list(spec.files or [])
        found: dict[str, Path] = {}
        for g in globs:
            for f in raw.glob(g):
                rel = f.relative_to(raw).as_posix()
                if not f.is_file() or any(
                    part.startswith(".") for part in f.relative_to(raw).parts
                ):
                    continue
                if any(fnmatch.fnmatch(rel, pat) for pat in spec.exclude):
                    continue
                found.setdefault(rel, f)
        rows = [
            {
                "path": rel,
                "filename": f.name,
                "stem": f.stem,
                "_abs": f,
            }
            for rel, f in sorted(found.items())
        ]
        return pd.DataFrame(rows)

    def _finalise(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame[frame["patient_id"].notna()].copy()
        frame["patient_id"] = frame["patient_id"].astype(str)
        if self.records.dedupe:
            frame = frame.drop_duplicates(self.records.dedupe)
        if self.records.sort_by:
            frame = frame.sort_values(self.records.sort_by, kind="stable")
        frame = frame.reset_index(drop=True)
        for name, spec in self.labels.items():
            bad = sorted(set(frame[name].dropna().astype(str)) - set(spec.classes))
            if bad:
                raise ValueError(f"label '{name}' has values {bad} not in classes {spec.classes}")
        return frame

    def _drop_duplicate_images(
        self, frame: pd.DataFrame, raw: Path
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        """Keep the first record of each near-identical image group (by image_path order)."""
        from tbmxai.data.dataset_format import _dhash

        if "image_path" not in frame:
            raise ValueError("drop_duplicate_images needs file records (image_path)")
        hashes = frame["image_path"].map(lambda p: _dhash(raw / p))
        order = frame.assign(_h=hashes).sort_values("image_path", kind="stable")
        drop: set[int] = set()
        dropped = []
        for h, group in order[order["_h"].notna()].groupby("_h", sort=False):
            if len(group) < 2:
                continue
            conflict = any(group[k].nunique(dropna=False) > 1 for k in self.labels)
            gone = group.index if conflict else group.index[1:]
            drop.update(gone)
            dropped.append(
                {
                    "hash": h,
                    "kept": None if conflict else str(group["image_path"].iloc[0]),
                    "dropped": [str(frame.at[i, "image_path"]) for i in gone],
                    "reason": "label conflict" if conflict else "duplicate",
                }
            )
        if dropped:
            logger.info("Dropped %d records in %d duplicate-image groups", len(drop), len(dropped))
        return frame.drop(index=sorted(drop)).reset_index(drop=True), dropped

    def _write_tables(self, frame: pd.DataFrame, out: Path) -> dict[str, list[str]]:
        keys = ["patient_id"] + (["study_id"] if "study_id" in frame else [])
        roles: dict[str, list[str]] = {"labels": self.tables.labels}
        for role in ("tabular", "images", "reports"):
            cols = getattr(self.tables, role)
            if role == "images" and cols is None and "image_path" in frame:
                cols = []
            if cols is not None:
                roles[role] = list(cols)
        if "images" in roles and "image_path" in frame:
            roles["images"] = ["image_path", *[c for c in roles["images"] if c != "image_path"]]
        if "reports" in roles:
            roles["reports"] = ["report_text", *[c for c in roles["reports"] if c != "report_text"]]
        roles.update(self.tables.extra)
        written = {}
        for role, cols in roles.items():
            missing = [c for c in cols if c not in frame]
            if missing:
                raise ValueError(f"tables.{role}: unknown fields {missing}")
            table = frame[keys + [c for c in cols if c not in keys]]
            table.to_csv(out / f"{role}.csv", index=False)
            written[role] = list(table.columns)
        return written


# ------------------------------------------------------------------ evaluation


class _Evaluator:
    """Computes the fields of every record, in recipe order."""

    def __init__(self, raw: Path, records: pd.DataFrame, fields: dict[str, FieldSpec]) -> None:
        self.raw = raw
        self.records = records
        self.fields = fields
        self.values: dict[str, pd.Series] = {}
        self._file_index: dict[str, dict[str, str]] = {}
        self._tables: dict[tuple[str, str], pd.DataFrame] = {}

    def run(self) -> pd.DataFrame:
        for name, spec in self.fields.items():
            result = self._field(name, spec)
            if isinstance(result, pd.DataFrame):  # a parser returned dicts: expand
                for col in result:
                    self.values[col] = result[col]
            else:
                self.values[name] = result
        frame = pd.DataFrame(self.values, index=self.records.index)
        if "_abs" in self.records and "image_path" not in frame:
            frame.insert(0, "image_path", self.records["path"])
        return frame

    def _attribute(self, which: str) -> pd.Series:
        if which in _FILE_SOURCES - {"parent"} and which in self.records:
            return self.records[which]
        if which.startswith("parent"):
            levels = int(which.split(":", 1)[1]) if ":" in which else 1
            return self.records["_abs"].map(lambda p: p.parents[levels - 1].name)
        if which.startswith("field:"):
            name = which.split(":", 1)[1]
            if name not in self.values:
                raise ValueError(f"field '{name}' used before it is defined")
            return self.values[name]
        if which.startswith("column:"):
            if which not in self.records:
                raise ValueError(f"table records have no column '{which[7:]}'")
            return self.records[which]
        raise ValueError(f"'{which}' is not available for these records")

    def _field(self, name: str, spec: FieldSpec) -> pd.Series | pd.DataFrame:
        kind = spec.source.split(":", 1)[0]
        idx = self.records.index
        if kind == "const":
            s = pd.Series([spec.value] * len(idx), index=idx, dtype=object)
        elif kind == "file":
            index = self._files(spec.glob or "")
            s = self._attribute(spec.lookup_by).map(lambda k: index.get(str(k)))
        elif kind == "table":
            table = self._table(spec.path or "", spec.key or "")
            col = table[spec.column]
            s = self._attribute(spec.lookup_by).map(lambda k: col.get(str(k)))
        elif kind == "list":
            s = self._membership(spec.lists or {})
        else:
            s = self._attribute(spec.source)
        s = s.astype(object).where(s.notna(), None)
        if spec.regex:
            s = _apply_regex(s, spec.regex, spec.group)
        if spec.parse:
            from tbmxai.datasets.adapters import PARSERS

            fn = PARSERS.get(spec.parse.lower())
            if fn is None:
                raise ValueError(f"unknown parser '{spec.parse}' (registered: {sorted(PARSERS)})")
            parsed = [fn("" if _missing(v) else str(v)) for v in s]
            if parsed and all(isinstance(p, dict) for p in parsed):
                return pd.DataFrame(parsed, index=idx)
            s = pd.Series(parsed, index=idx, dtype=object)
        if spec.map is not None:
            s = _apply_map(name, s, spec.map, spec.unmapped)
        if spec.format:
            fmt = spec.format
            s = s.map(lambda v: None if _missing(v) else fmt.format(v)).astype(object)
        if spec.cast:
            s = _apply_cast(name, s, spec.cast)
        if spec.bins is not None:
            values = pd.to_numeric(s, errors="coerce")
            binned = pd.cut(values, spec.bins, labels=spec.labels, right=False)
            s = binned.astype(object).where(binned.notna(), None)
        if spec.default is not None:
            s = s.where(s.notna(), spec.default)
        return s

    def _files(self, pattern: str) -> dict[str, str]:
        if pattern not in self._file_index:
            self._file_index[pattern] = {
                p.stem: p.read_text(errors="replace")
                for p in sorted(self.raw.glob(pattern))
                if p.is_file() and not p.name.startswith(".")
            }
        return self._file_index[pattern]

    def _table(self, path: str, key: str) -> pd.DataFrame:
        if (path, key) not in self._tables:
            t = _read_table(self.raw / path)
            if key not in t:
                raise ValueError(f"{path} has no column '{key}'")
            t[key] = t[key].astype(str)
            if t[key].duplicated().any():
                raise ValueError(f"{path}: key column '{key}' is not unique")
            self._tables[(path, key)] = t.set_index(key)
        return self._tables[(path, key)]

    def _membership(self, lists: dict[str, str]) -> pd.Series:
        entries: dict[str, str] = {}
        for value, rel in lists.items():
            for line in (self.raw / rel).read_text().splitlines():
                line = line.strip().split()[0] if line.strip() else ""
                if line:
                    entries.setdefault(line.lstrip("./"), value)

        def lookup(path: str) -> str | None:
            parts = path.split("/")
            for k in range(len(parts)):
                hit = entries.get("/".join(parts[k:]))
                if hit is not None:
                    return hit
            return None

        if "path" not in self.records:
            raise ValueError("from: list needs file records")
        return self.records["path"].map(lookup)


def _missing(v: Any) -> bool:
    """None or NaN (pandas turns None into NaN in some operations)."""
    return v is None or (isinstance(v, float) and np.isnan(v))


def _apply_regex(s: pd.Series, pattern: str, group: str | int | None) -> pd.Series:
    rx = re.compile(pattern)
    if group is None:
        group = "id" if "id" in rx.groupindex else (1 if rx.groups else 0)

    def one(v: Any) -> Any:
        if _missing(v):
            return None
        m = rx.search(str(v))
        return m.group(group) if m else None

    return s.map(one)


def _apply_map(name: str, s: pd.Series, mapping: dict[str, Any], unmapped: str) -> pd.Series:
    table = {str(k): v for k, v in mapping.items()}

    def one(v: Any) -> Any:
        if _missing(v):
            return None
        key = str(v)
        if key in table:
            return table[key]
        if unmapped == "keep":
            return v
        if unmapped == "missing":
            return None
        raise ValueError(f"field '{name}': value {v!r} not in map {sorted(table)}")

    return s.map(one).astype(object)


def _apply_cast(name: str, s: pd.Series, kind: str) -> pd.Series:
    if kind == "str":
        return s.map(lambda v: None if _missing(v) else str(v)).astype(object)
    values = pd.to_numeric(s, errors="coerce")
    bad = s.notna() & values.isna()
    if bad.any():
        raise ValueError(f"field '{name}': cannot cast {s[bad].iloc[0]!r} to {kind}")
    if kind == "int":
        return values.astype("Int64")
    return values.astype(float)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    sep = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def _extract_archives(raw: Path, patterns: list[str]) -> list[Path]:
    """Extract matching zip archives in ``raw`` once each into ``raw/<stem>/``."""
    done = []
    for pattern in patterns:
        for zf in sorted(raw.glob(pattern)):
            marker = raw / f".{zf.name}.extracted"
            if marker.exists() or not zipfile.is_zipfile(zf):
                continue
            logger.info("Extracting %s", zf)
            with zipfile.ZipFile(zf) as z:
                z.extractall(raw / zf.stem)
            marker.touch()
            done.append(zf)
    return done


__all__ = ["FieldSpec", "Recipe", "RecordsSpec"]
