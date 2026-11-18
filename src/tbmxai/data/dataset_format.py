"""The tbmxai standard dataset format: one CSV per modality, linked by a key.

A dataset is a folder of plain tables with fixed names and fixed key columns::

    my_dataset/
    ├── dataset.yaml   # name, label columns + classes (a few lines)
    ├── labels.csv     # patient_id, [study_id], <label columns>, <metadata: sex, age, …>
    ├── tabular.csv    # patient_id, [study_id], <features>                  (optional)
    ├── images.csv     # patient_id, [study_id], image_path, [<image metadata>] (optional)
    └── reports.csv    # patient_id, [study_id], report_text                 (optional)

Rules:

* ``patient_id`` is in every file; it is the unit of patient-level splitting.
* ``study_id`` (optional) is a finer key (study, visit, encounter). Two files are
  linked on ``patient_id`` + ``study_id`` when both have it, otherwise on
  ``patient_id`` alone (patient-level rows then apply to all of a patient's studies).
* Every other column has a fixed role decided by the file it is in: labels.csv →
  targets (declared in ``dataset.yaml``) and metadata (subgroups, splits);
  tabular.csv → model inputs; images.csv → image path + image metadata;
  reports.csv → ``report_text``.
* Each file may be ``.csv`` or ``.parquet`` (for large tables).

Data that is not yet in this shape is converted once by a small script, or for
image folders by :func:`index_images` (``tbmxai dataset index-images``), which
takes the patient id from file names, folder paths, DICOM headers or a lookup
table and writes ``images.csv``. Experiments use a dataset with
``data: {dataset: path/to/my_dataset, label_column: tb_status}``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
import yaml
from pydantic import Field, PrivateAttr

from tbmxai.core.config import ComponentConfig, _Strict, interpolate
from tbmxai.core.utils import get_logger

logger = get_logger(__name__)

DATASET_FILE = "dataset.yaml"
FORMAT_VERSION = "2.0"

#: Fixed column names.
PATIENT_ID = "patient_id"
STUDY_ID = "study_id"
IMAGE_PATH = "image_path"
REPORT_TEXT = "report_text"
FEATURE_TIME = "feature_time"  # optional tabular column: when the features were recorded

#: File stems per role; ``.csv`` or ``.parquet``.
ROLES = ("labels", "tabular", "images", "reports")
SUFFIXES = (".csv", ".parquet")
Use = Literal["tabular", "images", "reports"]


class LabelSpec(_Strict):
    """A target column of ``labels.csv``."""

    classes: list[str] = Field(
        min_length=2, description="Values in encoding order (positive last)."
    )
    description: str = ""


class DatasetSpec(_Strict):
    """Schema of ``dataset.yaml``."""

    format_version: str = FORMAT_VERSION
    name: str
    description: str = ""
    patient_ids: Literal["provided", "assumed"] = Field(
        default="provided",
        description="'assumed' when patient_id is not a real patient id (e.g. one per image).",
    )
    labels: dict[str, LabelSpec] = Field(min_length=1, description="Target columns of labels.csv.")
    time_column: str | None = Field(
        default=None, description="Prediction time column in labels.csv (optional)."
    )
    image_root: str | None = Field(
        default=None, description="Base folder for relative image paths (default: dataset folder)."
    )

    _root: Path = PrivateAttr(default=Path("."))

    # ------------------------------------------------------------------ io

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load ``dataset.yaml`` from a dataset folder (or the file itself)."""
        p = Path(path).expanduser()
        file = p / DATASET_FILE if p.is_dir() else p
        if not file.exists():
            raise FileNotFoundError(f"{file} not found (create one with `tbmxai dataset init`).")
        raw = interpolate(yaml.safe_load(file.read_text()) or {}, {})
        spec = cls.model_validate(raw)
        spec._root = file.parent.resolve()
        return spec

    def save(self, folder: str | Path) -> Path:
        """Write ``dataset.yaml`` into ``folder``."""
        d = Path(folder).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json", exclude_none=True)
        (d / DATASET_FILE).write_text(yaml.safe_dump(data, sort_keys=False))
        self._root = d.resolve()
        return d / DATASET_FILE

    @property
    def root(self) -> Path:
        """Folder containing ``dataset.yaml``."""
        return self._root

    @property
    def images_root(self) -> Path:
        """Base folder for relative ``image_path`` values."""
        if self.image_root is None:
            return self._root
        p = Path(self.image_root).expanduser()
        return p if p.is_absolute() else self._root / p

    def files(self) -> dict[str, Path]:
        """The data files present, by role (``labels``, ``tabular``, ``images``, ``reports``)."""
        out = {}
        for role in ROLES:
            found = [
                self._root / f"{role}{s}" for s in SUFFIXES if (self._root / f"{role}{s}").exists()
            ]
            if len(found) > 1:
                raise ValueError(f"Both {found[0].name} and {found[1].name} exist; keep one.")
            if found:
                out[role] = found[0]
        return out

    def file(self, role: str) -> Path:
        """Path of one data file (raises if absent)."""
        files = self.files()
        if role not in files:
            raise FileNotFoundError(f"{self._root}: no {role}.csv / {role}.parquet")
        return files[role]

    @property
    def modalities(self) -> list[str]:
        """Modalities present: ``tabular``, ``imaging``, ``text``."""
        files = self.files()
        return [
            m
            for role, m in (("tabular", "tabular"), ("images", "imaging"), ("reports", "text"))
            if role in files
        ]

    @property
    def link_columns(self) -> list[str]:
        """``[study_id]`` if any file has it."""
        return [STUDY_ID] if any(STUDY_ID in columns(p) for p in self.files().values()) else []

    def default_label(self) -> str:
        """The label column when an experiment does not name one."""
        if len(self.labels) == 1:
            return next(iter(self.labels))
        raise ValueError(
            f"Dataset '{self.name}' has several labels {sorted(self.labels)}; "
            "set data.label_column."
        )

    # -------------------------------------------------------------- sources

    def roles(self) -> dict[str, list[str]]:
        """Role of every non-key column: ``features``, ``metadata``, ``image_metadata``.

        A column in several files is used once: tabular features take precedence
        over labels-file metadata (``age`` can be both an input and a subgroup),
        which takes precedence over image metadata.
        """
        files = self.files()
        keys = {PATIENT_ID, STUDY_ID}
        feats = (
            [c for c in columns(files["tabular"]) if c not in keys | {FEATURE_TIME}]
            if "tabular" in files
            else []
        )
        meta = (
            [c for c in columns(files["labels"]) if c not in keys | set(self.labels) | set(feats)]
            if "labels" in files
            else []
        )
        img = (
            [
                c
                for c in columns(files["images"])
                if c not in keys | {IMAGE_PATH} | set(self.labels) | set(feats) | set(meta)
            ]
            if "images" in files
            else []
        )
        return {"features": feats, "metadata": meta, "image_metadata": img}

    def sources(self, label_column: str, use: Sequence[str] | None = None) -> list[ComponentConfig]:
        """Source configs for ingestion (``use`` restricts the optional files)."""
        if label_column not in self.labels:
            raise KeyError(f"Label '{label_column}' not in dataset labels {sorted(self.labels)}")
        wanted = set(use) if use is not None else {"tabular", "images", "reports"}
        files = self.files()
        if "labels" not in files:
            raise FileNotFoundError(f"{self._root}: labels.csv is required")
        roles = self.roles()
        if (
            "tabular" not in wanted
        ):  # labels-file columns shadowed by features become metadata again
            roles = {
                **roles,
                "metadata": [
                    c
                    for c in columns(files["labels"])
                    if c not in {PATIENT_ID, STUDY_ID, *self.labels}
                ],
            }
        key = {"key_column": PATIENT_ID}
        out = [
            ComponentConfig(
                name="tabular_file",
                params={
                    "path": str(files["labels"]),
                    **key,
                    "feature_columns": [],
                    "metadata_columns": roles["metadata"],
                },
            )
        ]
        if "tabular" in files and "tabular" in wanted:
            out.append(
                ComponentConfig(
                    name="tabular_file",
                    params={
                        "path": str(files["tabular"]),
                        **key,
                        "feature_columns": roles["features"],
                    },
                )
            )
        if "images" in files and "images" in wanted:
            out.append(
                ComponentConfig(
                    name="cxr_manifest",
                    params={
                        "path": str(files["images"]),
                        **key,
                        "path_column": IMAGE_PATH,
                        "image_root": str(self.images_root),
                        "metadata_columns": roles["image_metadata"],
                    },
                )
            )
        if "reports" in files and "reports" in wanted:
            out.append(
                ComponentConfig(
                    name="report_file",
                    params={"path": str(files["reports"]), **key, "text_column": REPORT_TEXT},
                )
            )
        return out


def columns(path: Path) -> list[str]:
    """Column names of a table without reading its rows."""
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    return list(pd.read_csv(path, nrows=0).columns)


def _read(path: Path) -> pd.DataFrame:
    from tbmxai.data.sources import read_table

    frame = read_table(path)
    for k in (PATIENT_ID, STUDY_ID):  # ids are strings everywhere
        if k in frame:
            frame[k] = frame[k].where(frame[k].isna(), frame[k].astype(str))
    return frame


# ------------------------------------------------------------------ validation


@dataclass
class ValidationReport:
    """Result of :func:`validate_dataset`."""

    issues: list[dict[str, str]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def add(self, level: str, check: str, message: str) -> None:
        self.issues.append({"level": level, "check": check, "message": message})

    @property
    def errors(self) -> list[dict[str, str]]:
        return [i for i in self.issues if i["level"] == "error"]

    @property
    def warnings(self) -> list[dict[str, str]]:
        return [i for i in self.issues if i["level"] == "warning"]

    @property
    def ok(self) -> bool:
        """No errors (warnings allowed)."""
        return not self.errors

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.issues, columns=["level", "check", "message"])


def _dhash(path: Path, size: int = 16) -> str | None:
    """Difference hash of an image.

    Robust to resizing and compression; an exact match means a near-duplicate.
    """
    from PIL import Image

    from tbmxai.data.torch_datasets import load_image

    try:
        img = load_image(path).resize((size + 1, size), Image.BILINEAR)
    except Exception:
        return None
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return hashlib.sha1(np.packbits(bits).tobytes()).hexdigest()


def _keys(frame: pd.DataFrame) -> list[str]:
    return [k for k in (PATIENT_ID, STUDY_ID) if k in frame.columns]


def validate_dataset(
    path: str | Path, check_images: int | None = 500, seed: int = 0
) -> ValidationReport:
    """Check a dataset folder before use.

    Args:
        path: Dataset folder (or ``dataset.yaml``).
        check_images: Images to open and fingerprint (random sample); ``None`` = all.
            Existence is checked for every image.
        seed: Sampling seed.
    """
    rep = ValidationReport()
    try:
        spec = DatasetSpec.load(path)
        files = spec.files()
    except Exception as exc:
        rep.add("error", "dataset.yaml", str(exc))
        return rep
    rep.info.update(
        name=spec.name,
        files={r: p.name for r, p in files.items()},
        labels=sorted(spec.labels),
        patient_ids=spec.patient_ids,
    )
    if "labels" not in files:
        rep.add("error", "files", "labels.csv (or labels.parquet) is required")
    if not {"tabular", "images", "reports"} & set(files):
        rep.add("error", "files", "add at least one of tabular.csv, images.csv, reports.csv")
    tables: dict[str, pd.DataFrame] = {}
    for role, file in files.items():
        try:
            tables[role] = _read(file)
        except Exception as exc:
            rep.add("error", f"{role} file", f"cannot read {file.name}: {exc}")
    if rep.errors:
        return rep

    required = {"images": IMAGE_PATH, "reports": REPORT_TEXT}
    for role, frame in tables.items():
        rep.info[f"{role}_rows"] = len(frame)
        if PATIENT_ID not in frame.columns:
            rep.add("error", "patient_id", f"{role}: column '{PATIENT_ID}' missing")
        elif frame[PATIENT_ID].isna().any():
            rep.add(
                "error",
                "patient_id",
                f"{role}: {int(frame[PATIENT_ID].isna().sum())} rows without a patient_id",
            )
        if role in required and required[role] not in frame.columns:
            rep.add("error", f"{role} columns", f"{role}: column '{required[role]}' missing")
        if role != "labels" and (clash := [c for c in spec.labels if c in frame.columns]):
            rep.add(
                "error",
                "label outside labels file",
                f"{role} contains label column(s) {clash}; labels belong only in labels.csv",
            )
    if rep.errors:
        return rep

    labels = tables["labels"]
    _validate_labels(rep, spec, labels)
    _validate_links(rep, tables)
    roles = spec.roles()
    rep.info["metadata_columns"] = roles["metadata"]
    if "tabular" in tables:
        _validate_tabular(rep, spec, labels, tables["tabular"], roles["features"])
    if "images" in tables:
        _validate_images(rep, spec, tables["images"], check_images, seed)
    if "reports" in tables:
        texts = tables["reports"][REPORT_TEXT].fillna("").astype(str).str.strip()
        if blank := int(texts.eq("").sum()):
            rep.add("info", "empty reports", f"{blank} empty reports (treated as missing text)")
        leak = [n for n in spec.labels if re.search(r"chexpert|report|nlp", n, re.IGNORECASE)]
        if leak:
            rep.add(
                "warning",
                "report-derived labels",
                f"labels {leak} look report-derived; do not use reports as input for them",
            )
    if spec.patient_ids == "assumed":
        rep.add(
            "warning",
            "patient ids",
            "patient_ids: assumed — patient-level splitting cannot be "
            "guaranteed (results will be flagged)",
        )
    return rep


def _validate_labels(rep: ValidationReport, spec: DatasetSpec, labels: pd.DataFrame) -> None:
    keys = _keys(labels)
    if dups := int(labels.duplicated(keys).sum()):
        rep.add("error", "labels keys", f"{dups} duplicate rows on {keys} in labels")
    for name, lab in spec.labels.items():
        if name not in labels.columns:
            rep.add("error", "label columns", f"label '{name}' not in labels")
            continue
        values = labels[name].dropna().astype(str).str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)
        if unknown := sorted(set(values) - set(lab.classes)):
            rep.add(
                "error",
                "label values",
                f"'{name}' has values {unknown[:10]} not in classes {lab.classes}",
            )
        counts = values.value_counts()
        rep.info[f"label_counts[{name}]"] = counts.to_dict()
        if n_na := int(labels[name].isna().sum()):
            rep.add(
                "info", "label missing", f"'{name}': {n_na} unlabelled rows (dropped at ingestion)"
            )
        if len(counts) and counts.min() < 20:
            rep.add(
                "warning", "class size", f"'{name}': smallest class has {int(counts.min())} samples"
            )
        if STUDY_ID in labels.columns and (
            mixed := int((labels.groupby(PATIENT_ID)[name].nunique() > 1).sum())
        ):
            rep.add(
                "info",
                "labels per patient",
                f"'{name}': {mixed} patients have different labels across studies",
            )
    if spec.time_column and spec.time_column not in labels.columns:
        rep.add("error", "time column", f"time_column '{spec.time_column}' not in labels")
    rep.info["patients"] = int(labels[PATIENT_ID].nunique())


def _validate_links(rep: ValidationReport, tables: dict[str, pd.DataFrame]) -> None:
    """Every modality row must link to a label; links must not multiply rows."""
    labels = tables["labels"]
    for role, frame in tables.items():
        if role == "labels":
            continue
        on = [k for k in _keys(frame) if k in labels.columns]
        if labels.duplicated(on).any() and frame.duplicated(on).any():
            rep.add(
                "error",
                "links",
                f"labels and {role} both have several rows per {on}: joining them "
                f"multiplies rows. Add {STUDY_ID} to both, or one row per {on} in one of them",
            )
        linked = frame[on].merge(labels[on].drop_duplicates(), on=on, how="left", indicator=True)
        if orphan := int((linked["_merge"] == "left_only").sum()):
            rep.add(
                "warning",
                "unlinked rows",
                f"{orphan} {role} rows have no matching labels row on {on}",
            )
        if role == "tabular" and frame.duplicated(_keys(frame)).any():
            rep.add("error", "tabular keys", f"duplicate rows on {_keys(frame)} in tabular")


def _validate_tabular(
    rep: ValidationReport,
    spec: DatasetSpec,
    labels: pd.DataFrame,
    tab: pd.DataFrame,
    feats: list[str],
) -> None:
    rep.info["tabular_features"] = len(feats)
    empty = [c for c in feats if tab[c].isna().all()]
    if empty:
        rep.add(
            "warning",
            "empty features",
            f"{len(empty)} features are entirely missing (e.g. {empty[:5]})",
        )
    if const := [c for c in feats if tab[c].nunique(dropna=True) == 1]:
        rep.add(
            "info", "constant features", f"{len(const)} features are constant (e.g. {const[:5]})"
        )
    idlike = [c for c in feats if tab[c].dtype == object and tab[c].nunique() > 0.9 * len(tab) > 50]
    if idlike:
        rep.add(
            "warning",
            "id-like features",
            f"{idlike[:5]} look like identifiers (almost unique per row)",
        )
    on = [k for k in _keys(tab) if k in labels.columns]
    merged = labels.merge(tab, on=on, how="inner", suffixes=("", "__tab"))
    _leakage_scan(rep, spec, merged, feats)
    if spec.time_column and FEATURE_TIME in tab.columns and spec.time_column in merged:
        t_pred = pd.to_datetime(merged[spec.time_column], errors="coerce")
        t_feat = pd.to_datetime(merged[FEATURE_TIME], errors="coerce")
        if late := int((t_feat > t_pred).sum()):
            rep.add(
                "error",
                "feature timing",
                f"{late} rows have features recorded after the prediction time (leakage)",
            )


def _leakage_scan(
    rep: ValidationReport, spec: DatasetSpec, merged: pd.DataFrame, feats: list[str]
) -> None:
    """Flag numeric features that almost perfectly separate a binary label."""
    from sklearn.metrics import roc_auc_score

    for name, lab in spec.labels.items():
        if name not in merged or len(lab.classes) != 2:
            continue
        y = merged[name].astype(str).str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)
        mask = y.isin(lab.classes)
        if mask.sum() < 30:
            continue
        yb = (y[mask] == lab.classes[1]).astype(int)
        if yb.nunique() < 2:
            continue
        suspicious = []
        for c in feats:
            col = merged.loc[mask, c] if c in merged else None
            if col is None or not pd.api.types.is_numeric_dtype(col):
                continue
            ok = col.notna()
            if ok.sum() < 30 or yb[ok].nunique() < 2:
                continue
            auc = roc_auc_score(yb[ok], col[ok])
            if max(auc, 1 - auc) >= 0.99:
                suspicious.append(c)
        if suspicious:
            rep.add(
                "warning",
                "possible leakage",
                f"features {suspicious[:5]} alone separate '{name}' almost perfectly "
                "(AUROC ≥ 0.99)",
            )


def _validate_images(
    rep: ValidationReport,
    spec: DatasetSpec,
    imgs: pd.DataFrame,
    check_images: int | None,
    seed: int,
) -> None:
    root = spec.images_root
    paths = (
        imgs[IMAGE_PATH].astype(str).map(lambda p: Path(p) if Path(p).is_absolute() else root / p)
    )
    patients = imgs[PATIENT_ID].astype(str)
    if imgs.duplicated(IMAGE_PATH).any():
        rep.add("warning", "duplicate image rows", "the same image_path is listed more than once")
    exists = paths.map(lambda p: p.exists())
    rep.info["images"] = len(paths)
    if (~exists).any():
        rep.add(
            "error" if exists.mean() < 0.5 else "warning",
            "image files",
            f"{int((~exists).sum())} of {len(paths)} image files are missing "
            "(missing ones are skipped)",
        )
    present = np.flatnonzero(exists.to_numpy())
    if not len(present):
        return
    rng = np.random.default_rng(seed)
    sample = (
        present
        if check_images is None or len(present) <= check_images
        else rng.choice(present, check_images, replace=False)
    )
    hashes: dict[str, set[str]] = {}
    unreadable = 0
    for i in sample:
        h = _dhash(paths.iloc[i])
        if h is None:
            unreadable += 1
        else:
            hashes.setdefault(h, set()).add(patients.iloc[i])
    rep.info["images_checked"] = len(sample)
    if unreadable:
        rep.add(
            "error",
            "image decoding",
            f"{unreadable} of {len(sample)} sampled images cannot be read",
        )
    if cross := sum(1 for owners in hashes.values() if len(owners) > 1):
        rep.add(
            "warning",
            "duplicate images",
            f"{cross} near-identical images belong to different patients "
            "(possible duplicate patients or relabelled copies → leakage across splits)",
        )


# --------------------------------------------------------------- image indexing


def index_images(
    image_dir: str | Path,
    dataset: str | Path,
    patient_id_from: str = "filename",
    id_pattern: str = r"(?P<id>.+)",
    study_pattern: str | None = None,
    lookup: str | Path | None = None,
    lookup_image_column: str = "image",
    lookup_patient_column: str = "patient_id",
    label_from_folder: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Scan an image folder and write the dataset's ``images.csv`` (and optionally labels).

    Use this when images are files on disk without a table. The patient id is
    taken from wherever the collection keeps it (``patient_id_from``):
    ``filename`` / ``path`` (regex ``id_pattern``, group ``id``), ``dicom``
    (``PatientID``; ``StudyInstanceUID`` → ``study_id``), ``table`` (a ``lookup``
    file mapping image file → patient), or ``none`` (each image is its own
    patient; ``dataset.yaml`` must then say ``patient_ids: assumed``).

    Args:
        image_dir: Folder scanned recursively.
        dataset: Dataset folder to write into.
        label_from_folder: If set, the image's parent folder name is the label;
            ``labels.csv`` is written with this column name (one row per image,
            ``study_id`` = image identifier).
        overwrite: Replace existing ``images.csv`` / ``labels.csv``.

    Returns:
        The files written, by role.
    """
    from tbmxai.data.sources import IMAGE_PATH_COLUMN, ImageFolderSource

    d = Path(dataset).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    src = ImageFolderSource(
        image_dir,
        patient_id_from=patient_id_from,
        id_pattern=id_pattern,
        study_pattern=study_pattern,
        metadata=lookup,
        metadata_image_column=lookup_image_column,
        metadata_patient_column=lookup_patient_column,
        label_from_folder=label_from_folder is not None,
        label_column=label_from_folder or "label",
    )
    frame = src._read()
    root = Path(image_dir).expanduser().resolve()
    absolute = frame[IMAGE_PATH_COLUMN].map(lambda p: Path(p).resolve())
    try:  # relative to the dataset folder when the images live inside it
        frame[IMAGE_PATH] = absolute.map(lambda p: p.relative_to(d.resolve()).as_posix())
    except ValueError:
        frame[IMAGE_PATH] = absolute.map(str)
    if label_from_folder and STUDY_ID not in frame and patient_id_from != "none":
        frame[STUDY_ID] = absolute.map(lambda p: p.relative_to(root).with_suffix("").as_posix())
    keys = [k for k in (PATIENT_ID, STUDY_ID) if k in frame]
    written: dict[str, Path] = {}
    targets = {"images": d / "images.csv"}
    if label_from_folder:
        targets["labels"] = d / "labels.csv"
    for path in targets.values():
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists (pass overwrite=True / --overwrite)")
    frame[[*keys, IMAGE_PATH]].to_csv(targets["images"], index=False)
    written["images"] = targets["images"]
    if label_from_folder:
        frame[[*keys, label_from_folder]].drop_duplicates().to_csv(targets["labels"], index=False)
        written["labels"] = targets["labels"]
    if not (d / DATASET_FILE).exists() and label_from_folder:
        classes = sorted(frame[label_from_folder].astype(str).unique())
        DatasetSpec(
            name=d.name,
            labels={label_from_folder: LabelSpec(classes=classes)},
            patient_ids="assumed" if patient_id_from == "none" else "provided",
        ).save(d)
        written["dataset.yaml"] = d / DATASET_FILE
    logger.info("Indexed %d images of %d patients → %s", len(frame), frame[PATIENT_ID].nunique(), d)
    if patient_id_from == "none":
        logger.warning("patient_id_from=none: set `patient_ids: assumed` in %s", d / DATASET_FILE)
    return written


# ------------------------------------------------------------------------ init


def init_dataset(
    folder: str | Path,
    name: str | None = None,
    modalities: Sequence[str] = ("tabular", "images", "reports"),
) -> list[Path]:
    """Write a template ``dataset.yaml`` and empty CSVs with the right headers.

    Existing files are never overwritten.
    """
    d = Path(folder).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def write(rel: str, content: str) -> None:
        p = d / rel
        if p.exists():
            logger.info("Keeping existing %s", p)
            return
        p.write_text(content)
        written.append(p)

    mods = set(modalities)
    write(
        DATASET_FILE,
        "\n".join(
            [
                f"# tbmxai dataset (format {FORMAT_VERSION}). Fill the CSVs, then:",
                f"#   tbmxai dataset validate {d}",
                f"format_version: '{FORMAT_VERSION}'",
                f"name: {name or d.name}",
                "description: ''",
                "labels:                        # target columns of labels.csv",
                "  label: {classes: ['0', '1']} # values in encoding order; positive class last",
                "time_column: null              # prediction-time column in labels.csv (optional)",
                "patient_ids: provided          # 'assumed' if patient_id is not a real patient id",
                "# image_root: /data/images     # base for relative image_path (default: here)",
            ]
        )
        + "\n",
    )
    write("labels.csv", "patient_id,label,sex,age\n")
    if "tabular" in mods:
        write("tabular.csv", "patient_id,feature_1,feature_2\n")
    if "images" in mods:
        write("images.csv", "patient_id,image_path\n")
    if "reports" in mods:
        write("reports.csv", "patient_id,report_text\n")
    return written
