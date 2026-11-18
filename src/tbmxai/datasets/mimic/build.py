"""Two-step builder: ``select`` a patient subset, then ``build`` tbmxai-ready tables.

Step 1 — :meth:`MimicBuilder.select` (needs only metadata + EHR tables):
    ``<output_dir>/selection/`` ← ``studies.parquet``, ``images.parquet``,
    ``image_urls.txt`` (PhysioNet URLs of just the selected JPGs), ``selection_info.json``.

Step 2 — :meth:`MimicBuilder.build` (after the JPGs are downloaded):
    ``<output_dir>/`` ← a tbmxai standard dataset: ``dataset.yaml``,
    ``labels.parquet`` (per study: all targets + subgroup metadata),
    ``tabular.parquet`` (per study: EHR features), ``images.parquet`` (per image:
    ``image_path`` relative to the MIMIC-CXR-JPG root), ``reports.parquet`` (per
    study: ``report_text`` = FINDINGS + IMPRESSION), plus ``feature_dictionary.csv``
    and ``build_info.json``. All keyed by ``patient_id`` + ``study_id``.
"""

from __future__ import annotations

import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tbmxai.core.utils import get_logger
from tbmxai.datasets.mimic.cohort import (
    derive_labels,
    limit_studies,
    link_encounters,
    load_images,
    make_dev_split,
    pick_images,
    report_relpath,
    select_patients,
    studies_from_images,
)
from tbmxai.datasets.mimic.config import MimicBuildConfig
from tbmxai.datasets.mimic.features import FEATURE_GROUPS, FeatureGroup, HistoryFeatures
from tbmxai.datasets.mimic.tables import MimicTables
from tbmxai.reporting.writers import finite

logger = get_logger(__name__)

PATIENT = "patient_id"
STUDY = "study_id"

#: Columns computed by feature groups that are context, not model inputs.
NON_INPUT_FEATURES = ("demo_anchor_year_group",)


class MimicBuilder:
    """Build a MIMIC-IV + MIMIC-CXR development subset.

    Args:
        config: Build configuration.
        memory_limit: Optional DuckDB memory limit (e.g. ``"24GB"``).
    """

    def __init__(self, config: MimicBuildConfig, memory_limit: str | None = None) -> None:
        self.config = config
        self.tables = MimicTables(
            config.sources,
            memory_limit=memory_limit,
            temp_directory=config.output_dir.expanduser() / ".duckdb_tmp",
        )
        self.out = config.output_dir.expanduser()

    # ------------------------------------------------------------------ select

    def select(self) -> dict[str, Any]:
        """Sample patients and write the image download list."""
        cfg = self.config
        self.tables.check_required()
        sel_dir = cfg.selection_dir
        sel_dir.mkdir(parents=True, exist_ok=True)

        images = load_images(self.tables, cfg.selection.views)
        studies = studies_from_images(images)
        logger.info(
            "CXR: %d frontal images, %d studies, %d patients",
            len(images),
            len(studies),
            studies["subject_id"].nunique(),
        )
        studies = link_encounters(studies, self.tables)
        studies = derive_labels(studies, self.tables, cfg.labels)
        if cfg.selection.require_encounter:
            # Episode labels do not need an encounter: keep their positive studies.
            keep = studies["encounter_type"] != "none"
            for name in cfg.selection.enrich_labels:
                if cfg.labels[name].source == "episode":
                    keep |= studies[f"label_{name}"].fillna(0).eq(1)
            studies = studies[keep]

        population = _prevalence(studies, cfg.labels)
        patients = select_patients(studies, cfg.selection)
        studies = studies[studies["subject_id"].isin(patients)]
        studies = limit_studies(studies, cfg.selection.max_studies_per_patient, cfg.selection.seed)
        images = pick_images(
            images[images["study_id"].isin(studies["study_id"])], cfg.selection.one_image_per_study
        )

        studies.to_parquet(sel_dir / "studies.parquet", index=False)
        images.to_parquet(sel_dir / "images.parquet", index=False)
        dev = make_dev_split(studies, cfg.selection.dev_split)
        dev.to_csv(sel_dir / "dev_split.csv", index=False)
        base = f"{cfg.physionet_base_url.rstrip('/')}/{cfg.cxr_jpg_project.strip('/')}"
        (sel_dir / "image_urls.txt").write_text(
            "\n".join(f"{base}/{p}" for p in images["path"]) + "\n"
        )
        info = {
            "created_utc": datetime.now(UTC).isoformat(),
            "n_patients": int(studies["subject_id"].nunique()),
            "n_studies": len(studies),
            "n_images": len(images),
            "encounter_types": studies["encounter_type"].value_counts().to_dict(),
            "prevalence_source_population": population,
            "prevalence_subset": _prevalence(studies, cfg.labels),
            "enriched_labels": cfg.selection.enrich_labels,
            "dev_split_patients": dev["dev_split"].value_counts().to_dict(),
            "selection": cfg.selection.model_dump(mode="json"),
        }
        (sel_dir / "selection_info.json").write_text(
            json.dumps(finite(info), indent=2, default=str)
        )
        logger.info(
            "Selected %d patients / %d studies / %d images → %s",
            info["n_patients"],
            info["n_studies"],
            info["n_images"],
            sel_dir,
        )
        return info

    def ensure_dev_split(self, overwrite: bool = False) -> pd.DataFrame:
        """``selection/dev_split.csv`` for an existing selection (created if missing)."""
        sel_dir = self.config.selection_dir
        path = sel_dir / "dev_split.csv"
        if path.exists() and not overwrite:
            return pd.read_csv(path)
        studies = pd.read_parquet(sel_dir / "studies.parquet")
        dev = make_dev_split(studies, self.config.selection.dev_split)
        dev.to_csv(path, index=False)
        logger.info("Dev split → %s: %s", path, dev["dev_split"].value_counts().to_dict())
        return dev

    # ------------------------------------------------------------------- build

    def build(self) -> dict[str, Any]:
        """Compute features/labels/manifests for the selected subset."""
        cfg = self.config
        sel_dir = cfg.selection_dir
        if not (sel_dir / "studies.parquet").exists():
            raise FileNotFoundError(f"Run `select` first ({sel_dir / 'studies.parquet'} missing).")
        studies = pd.read_parquet(sel_dir / "studies.parquet")
        images = pd.read_parquet(sel_dir / "images.parquet")
        self.out.mkdir(parents=True, exist_ok=True)

        features, dictionary = self._features(studies)
        labels = self._labels(studies, features)
        labels.to_parquet(self.out / "labels.parquet", index=False)

        # Context-only columns go to labels.parquet (metadata), never to model inputs.
        context_only = [c for c in NON_INPUT_FEATURES if c in features.columns]
        features = features.drop(columns=context_only)
        dictionary = dictionary[~dictionary["column"].isin(context_only)]
        features.insert(
            0, PATIENT, studies.set_index(STUDY).loc[features[STUDY], "subject_id"].to_numpy()
        )
        features.to_parquet(self.out / "tabular.parquet", index=False)
        dictionary.to_csv(self.out / "feature_dictionary.csv", index=False)

        manifest = self._manifest(images)
        manifest.to_parquet(self.out / "images.parquet", index=False)

        n_reports = 0
        reports = self._reports(studies)
        if reports is not None:
            reports.to_parquet(self.out / "reports.parquet", index=False)
            n_reports = int(reports["report_text"].str.len().gt(0).sum())

        info = {
            "created_utc": datetime.now(UTC).isoformat(),
            "n_patients": int(studies["subject_id"].nunique()),
            "n_studies": len(studies),
            "n_images": len(manifest),
            "n_images_present": int(manifest["present"].sum()),
            "n_reports": n_reports,
            "n_features": len(dictionary),
            "features_per_group": dictionary.groupby("group").size().to_dict(),
            "label_prevalence": _prevalence(studies, cfg.labels),
            "config": cfg.model_dump(mode="json"),
        }
        (self.out / "build_info.json").write_text(json.dumps(finite(info), indent=2, default=str))
        self.write_dataset_spec(labels)
        if info["n_images_present"] < info["n_images"]:
            logger.warning(
                "%d of %d selected images are not downloaded yet",
                info["n_images"] - info["n_images_present"],
                info["n_images"],
            )
        logger.info(
            "Built %d features for %d studies → %s", info["n_features"], len(studies), self.out
        )
        return info

    def write_dataset_spec(self, labels: pd.DataFrame) -> Path:
        """Describe the build output as a tbmxai standard dataset (``dataset.yaml``)."""
        from tbmxai.data.dataset_format import DatasetSpec, LabelSpec

        label_cols = [c for c in labels.columns if c.startswith("label_")]
        spec = DatasetSpec(
            name=f"mimic_{self.out.name}",
            description="MIMIC-IV + MIMIC-IV-ED + MIMIC-CXR-JPG subset built by tbmxai "
            "(EHR features use only data recorded before each CXR).",
            labels={
                c: LabelSpec(classes=["0", "1"], description=c.removeprefix("label_"))
                for c in label_cols
            },
            time_column="study_time",
            image_root=str(self.config.sources.mimic_cxr_jpg.expanduser().resolve()),
        )
        return spec.save(self.out)

    # --------------------------------------------------------------- internals

    def _groups(self) -> list[FeatureGroup]:
        groups = []
        for name, params in self.config.features.items():
            cls = FEATURE_GROUPS.get(name)
            kwargs = dict(params or {})
            if issubclass(cls, HistoryFeatures):
                kwargs.setdefault("label_definitions", self.config.labels)
            groups.append(cls(**kwargs))
        return groups

    def _features(self, studies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame = studies[[STUDY]].copy()
        rows = []
        for group in self._groups():
            missing = group.available(self.tables)
            if missing:
                logger.warning(
                    "Skipping feature group '%s': missing tables %s", group.prefix, missing
                )
                continue
            logger.info("Extracting feature group '%s' …", group.prefix)
            part = group.extract(studies, self.tables)
            part = part.drop_duplicates(STUDY)
            empty = [c for c in part.columns if c != STUDY and part[c].isna().all()]
            if empty:
                logger.info("%s: dropping %d never-observed columns", group.prefix, len(empty))
                part = part.drop(columns=empty)
            frame = frame.merge(part, on=STUDY, how="left")
            for col in part.columns.drop(STUDY):
                values = frame[col]
                rows.append(
                    {
                        "column": col,
                        "group": group.prefix,
                        "dtype": "numeric"
                        if pd.api.types.is_numeric_dtype(values)
                        else "categorical",
                        "coverage": float(values.notna().mean()),
                        "description": group.descriptions.get(col, ""),
                    }
                )
        return frame, pd.DataFrame(
            rows, columns=["column", "group", "dtype", "coverage", "description"]
        )

    def _labels(self, studies: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        label_cols = [c for c in studies.columns if c.startswith("label_")]
        out = studies[["subject_id", STUDY, *label_cols, "encounter_type", "n_images"]].rename(
            columns={"subject_id": PATIENT}
        )
        # Prediction timepoint = CXR study time. MIMIC dates are shifted per patient, so this
        # orders a patient's own studies; use `year_group` for cross-patient temporal splits.
        out["study_time"] = pd.to_datetime(studies["study_time"]).to_numpy()
        if "official_split" in studies:
            out["official_split"] = studies["official_split"].to_numpy()
        dev = self.ensure_dev_split().set_index("subject_id")["dev_split"]
        out["dev_split"] = studies["subject_id"].map(dev).to_numpy()
        # Subgroup metadata (duplicated from features on purpose: metadata ≠ model input).
        meta_map = {
            "demo_sex": "sex",
            "demo_age": "age",
            "demo_race": "race",
            "enc_insurance": "insurance",
            "hist_hiv": "hiv_prior",
            "demo_anchor_year_group": "year_group",
        }
        present = {k: v for k, v in meta_map.items() if k in features.columns}
        meta = features[[STUDY, *present]].rename(columns=present)
        out = out.merge(meta, on=STUDY, how="left")
        if "age" in out:
            out["age_group"] = pd.cut(
                out["age"], [0, 40, 65, 200], right=False, labels=["<40", "40-64", "65+"]
            ).astype(str)
        return out

    def _manifest(self, images: pd.DataFrame) -> pd.DataFrame:
        root = self.config.sources.mimic_cxr_jpg.expanduser()
        out = images.rename(columns={"subject_id": PATIENT})[
            [
                PATIENT,
                STUDY,
                "dicom_id",
                "path",
                "view_position",
                *(["official_split"] if "official_split" in images else []),
            ]
        ].copy()
        out["present"] = [(root / p).exists() for p in out["path"]]
        return out.rename(columns={"path": "image_path"})

    def _reports(self, studies: pd.DataFrame) -> pd.DataFrame | None:
        src = self.config.sources.mimic_cxr_reports
        if src is None:
            return None
        src = Path(src).expanduser()
        if not src.exists():
            logger.warning("Reports source %s not found; skipping reports", src)
            return None
        rels = [
            report_relpath(s, st)
            for s, st in zip(studies["subject_id"], studies[STUDY], strict=True)
        ]
        texts: list[str] = []
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as zf:
                names = {n.split("files/", 1)[-1]: n for n in zf.namelist() if n.endswith(".txt")}
                for rel in rels:
                    member = names.get(rel.removeprefix("files/"))
                    texts.append(zf.read(member).decode("utf-8", "replace") if member else "")
        else:
            for rel in rels:
                p = src / rel
                texts.append(p.read_text(errors="replace") if p.exists() else "")
        out = pd.DataFrame(
            {
                PATIENT: studies["subject_id"].to_numpy(),
                STUDY: studies[STUDY].to_numpy(),
                "report": texts,
            }
        )
        out["report_text"] = out["report"].map(extract_sections)  # FINDINGS + IMPRESSION
        return out.rename(columns={"report": "report_full"})


_SECTION = re.compile(r"^\s*(FINDINGS|IMPRESSION)\s*:", re.IGNORECASE | re.MULTILINE)
_ANY_HEADER = re.compile(r"^\s*[A-Z][A-Z /]{2,40}:", re.MULTILINE)


def extract_sections(report: str) -> str:
    """FINDINGS + IMPRESSION text of a MIMIC-CXR report (falls back to the full report)."""
    if not report:
        return ""
    parts = []
    for match in _SECTION.finditer(report):
        start = match.end()
        nxt = _ANY_HEADER.search(report, start)
        parts.append(report[start : nxt.start() if nxt else len(report)].strip())
    text = " ".join(p for p in parts if p)
    return re.sub(r"\s+", " ", text or report).strip()


def _prevalence(studies: pd.DataFrame, labels: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    out = {}
    for name in labels:
        col = studies.get(f"label_{name}")
        if col is None:
            continue
        valid = col.dropna()
        out[name] = {
            "n_labelled": len(valid),
            "n_positive": int((valid == 1).sum()),
            "prevalence": float(np.round((valid == 1).mean(), 5)) if len(valid) else float("nan"),
            "n_positive_patients": int(studies.loc[col.fillna(0) == 1, "subject_id"].nunique()),
        }
    return out
