"""Imaging sources: chest X-ray manifests and image directories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from tbmxai.core.types import Modality
from tbmxai.core.utils import get_logger
from tbmxai.data.sources.base import IMAGE_PATH_COLUMN, SOURCES, DataSource, read_table

logger = get_logger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".dcm", ".tif", ".tiff")


class _ImagingSource(DataSource):
    """Shared logic: resolve paths, optionally drop missing files."""

    modality = Modality.IMAGING

    def __init__(
        self, image_root: str | Path | None = None, check_exists: bool = True, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.image_root = Path(image_root) if image_root else None
        self.check_exists = check_exists

    def _resolve(self, paths: pd.Series) -> pd.Series:
        root = self.image_root
        resolved = paths.astype(str).map(
            lambda p: str(root / p) if root and not Path(p).is_absolute() else p
        )
        return resolved

    def _drop_missing(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.check_exists:
            return frame
        exists = frame[IMAGE_PATH_COLUMN].map(lambda p: Path(p).exists())
        if (n_missing := int((~exists).sum())) > 0:
            logger.warning("%s: dropping %d rows with missing image files", self, n_missing)
        return frame.loc[exists]

    def _payload_columns(self, frame: pd.DataFrame) -> list[str]:
        return [IMAGE_PATH_COLUMN]


@SOURCES.register("cxr_manifest")
class CXRManifestSource(_ImagingSource):
    """A table listing image files per patient (multiple images per patient allowed).

    Args:
        path: Manifest file (CSV/Parquet/...).
        path_column: Column holding image paths (absolute or relative to ``image_root``).
        image_root: Base directory for relative paths.
        check_exists: Drop rows whose file does not exist.
        **kwargs: Forwarded to :class:`DataSource` (``key_column``, ``metadata_columns``…).
    """

    def __init__(self, path: str | Path, path_column: str = "path", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.path = Path(path)
        self.path_column = path_column

    def _read(self) -> pd.DataFrame:
        frame = read_table(self.path)
        if self.path_column not in frame.columns:
            raise KeyError(f"{self.path.name}: path column '{self.path_column}' not found")
        frame[IMAGE_PATH_COLUMN] = self._resolve(frame[self.path_column])
        return self._drop_missing(frame)


ID_SOURCES = ("filename", "path", "dicom", "table", "none")


@SOURCES.register("image_folder")
@SOURCES.register("cxr_directory")
class ImageFolderSource(_ImagingSource):
    """Images discovered on disk, with the patient id taken from wherever the dataset keeps it.

    ``patient_id_from`` selects where the id comes from:

    * ``filename`` — regex ``id_pattern`` on the file name without extension
      (``CHNCXR_0001_0.png`` with ``(?P<id>CHNCXR_\\d+)`` → ``CHNCXR_0001``);
    * ``path`` — regex ``id_pattern`` on the path relative to ``root``
      (``p10/p10000032/s5041/x.jpg`` with ``p(?P<id>\\d{8})/`` → ``10000032``);
    * ``dicom`` — the DICOM ``PatientID`` tag (and ``StudyInstanceUID`` as study id);
    * ``table`` — a metadata table mapping images to patients
      (``metadata``: file; ``metadata_image_column``: file name or stem;
      ``metadata_patient_column``: patient id);
    * ``none`` — no patient information: every image is its own patient. Patient-level
      splitting can then **not** be guaranteed; the cohort is flagged and every report warns.

    Labels can come from the folder name (``label_from_folder: true`` → column
    ``label_column`` = name of the image's parent folder, e.g. ``Normal/``, ``Tuberculosis/``).

    Args:
        root: Directory to scan (recursively).
        patient_id_from: One of ``filename``, ``path``, ``dicom``, ``table``, ``none``.
        id_pattern: Regex with a named group ``id`` (or one group) for ``filename`` / ``path``.
        study_pattern: Optional regex (group ``id``) on the relative path giving a study id.
        study_column: Name of the produced study-id column (if any).
        metadata: Metadata table for ``patient_id_from: table``.
        metadata_image_column: Column in ``metadata`` holding the image file name or stem.
        metadata_patient_column: Column in ``metadata`` holding the patient id.
        metadata_columns_from_table: Extra metadata-table columns to keep (e.g. labels, sex).
        label_from_folder: Take the label from the parent folder name.
        label_column: Name of the produced label column.
        suffixes: Accepted file extensions.
        **kwargs: Forwarded to :class:`DataSource` (``key_column``, ``metadata_columns``…).
    """

    def __init__(
        self,
        root: str | Path,
        patient_id_from: str = "filename",
        id_pattern: str = r"(?P<id>.+)",
        study_pattern: str | None = None,
        study_column: str = "study_id",
        metadata: str | Path | None = None,
        metadata_image_column: str = "image",
        metadata_patient_column: str = "patient_id",
        metadata_columns_from_table: list[str] | None = None,
        label_from_folder: bool = False,
        label_column: str = "label",
        suffixes: tuple[str, ...] = IMAGE_SUFFIXES,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("key_column", "patient_id")
        super().__init__(image_root=None, check_exists=False, **kwargs)
        if patient_id_from not in ID_SOURCES:
            raise ValueError(f"patient_id_from must be one of {ID_SOURCES}")
        if patient_id_from == "table" and metadata is None:
            raise ValueError("patient_id_from: table needs `metadata`")
        self.root = Path(root).expanduser()
        self.patient_id_from = patient_id_from
        self.id_pattern = re.compile(id_pattern)
        self.study_pattern = re.compile(study_pattern) if study_pattern else None
        self.study_column = study_column
        self.metadata = Path(metadata).expanduser() if metadata else None
        self.metadata_image_column = metadata_image_column
        self.metadata_patient_column = metadata_patient_column
        self.metadata_columns_from_table = list(metadata_columns_from_table or [])
        self.label_from_folder = label_from_folder
        self.label_column = label_column
        self.suffixes = tuple(s.lower() for s in suffixes)
        self.patient_ids_assumed = patient_id_from == "none"

    @staticmethod
    def _match(pattern: re.Pattern[str], text: str) -> str | None:
        m = pattern.search(text)
        if m is None:
            return None
        return m.group("id") if "id" in m.groupdict() else m.group(1 if m.groups() else 0)

    def _files(self) -> list[Path]:
        files = [f for f in sorted(self.root.rglob("*")) if f.suffix.lower() in self.suffixes]
        if not files:
            raise FileNotFoundError(f"No images with suffixes {self.suffixes} under {self.root}")
        return files

    def _read(self) -> pd.DataFrame:
        rows: list[dict[str, str | None]] = []
        skipped = 0
        for file in self._files():
            rel = file.relative_to(self.root).as_posix()
            row: dict[str, str | None] = {
                IMAGE_PATH_COLUMN: str(file),
                "_file": file.name,
                "_stem": file.stem,
            }
            if self.patient_id_from == "filename":
                row[self.key_column] = self._match(self.id_pattern, file.stem)
            elif self.patient_id_from == "path":
                row[self.key_column] = self._match(self.id_pattern, rel)
            elif self.patient_id_from == "dicom":
                pid, study = _dicom_ids(file)
                row[self.key_column] = pid
                if study:
                    row[self.study_column] = study
            elif self.patient_id_from == "none":
                row[self.key_column] = Path(rel).with_suffix("").as_posix()  # unique per image
            if self.study_pattern is not None:
                row[self.study_column] = self._match(self.study_pattern, rel)
            if self.label_from_folder:
                row[self.label_column] = file.parent.name
            if self.patient_id_from != "table" and row.get(self.key_column) is None:
                skipped += 1
                continue
            rows.append(row)
        if skipped:
            logger.warning("%s: %d images without a patient id were skipped", self, skipped)
        frame = pd.DataFrame(rows)
        if self.patient_id_from == "table":
            frame = self._join_table(frame)
        if self.patient_ids_assumed:
            logger.warning(
                "%s: no patient ids — each image is treated as its own patient; patient-level "
                "splitting cannot be guaranteed (results are flagged).",
                self,
            )
        return frame.drop(columns=["_file", "_stem"], errors="ignore")

    def _join_table(self, frame: pd.DataFrame) -> pd.DataFrame:
        meta = read_table(self.metadata)  # type: ignore[arg-type]
        for col in (self.metadata_image_column, self.metadata_patient_column):
            if col not in meta.columns:
                raise KeyError(f"{self.metadata}: column '{col}' not found")
        keep = [
            self.metadata_image_column,
            self.metadata_patient_column,
            *self.metadata_columns_from_table,
        ]
        meta = meta[keep].copy()
        key = meta[self.metadata_image_column].astype(str)
        # match on full file name if it has an extension, else on the stem
        on = "_file" if key.str.contains(r"\.[A-Za-z0-9]+$", regex=True).all() else "_stem"
        meta["_join"] = key
        merged = frame.merge(meta, left_on=on, right_on="_join", how="inner")
        if (n := len(frame) - len(merged)) > 0:
            logger.warning("%s: %d images not found in the metadata table were skipped", self, n)
        merged = merged.drop(columns=["_join", self.metadata_image_column])
        if self.metadata_patient_column != self.key_column:
            merged = merged.rename(columns={self.metadata_patient_column: self.key_column})
        return merged


#: Backwards-compatible name.
CXRDirectorySource = ImageFolderSource


def _dicom_ids(path: Path) -> tuple[str | None, str | None]:
    """``(PatientID, StudyInstanceUID)`` from a DICOM header (pixels are not read)."""
    import pydicom

    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        return None, None
    pid = str(getattr(ds, "PatientID", "") or "").strip() or None
    study = str(getattr(ds, "StudyInstanceUID", "") or "").strip() or None
    return pid, study
