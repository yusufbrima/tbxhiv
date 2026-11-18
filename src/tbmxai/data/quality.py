"""Data-quality report for a harmonised view (computed before modelling).

Produces tidy tables that flag the usual problems of clinical multimodal data:
missingness per feature and per feature group, constant / near-constant
features, high-cardinality categoricals, samples per patient, modality
availability patterns, label balance and label inconsistency within patients.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tbmxai.data.harmonize import MODALITY_PATTERN_COLUMN, MultimodalView

#: Shown wherever results come from data without real patient ids.
ASSUMED_IDS_WARNING = (
    "Patient ids were not available for at least one source: every image was treated as its "
    "own patient, so patient-level splitting cannot be guaranteed (the same person may appear "
    "in train and test). Interpret results with caution."
)


@dataclass
class DataQualityReport:
    """Tables describing the quality of a :class:`MultimodalView`.

    Attributes:
        summary: One row per check: ``check, value, flag, detail``.
        features: Per tabular feature: dtype, missing fraction, unique values, flags.
        groups: Per feature group (prefix before the first ``_``): mean / max missingness.
        modality_patterns: Samples / patients / positives per availability pattern.
    """

    summary: pd.DataFrame
    features: pd.DataFrame
    groups: pd.DataFrame
    modality_patterns: pd.DataFrame

    @classmethod
    def from_view(
        cls,
        view: MultimodalView,
        high_missing: float = 0.95,
        high_cardinality: int = 50,
    ) -> DataQualityReport:
        """Compute all checks.

        Args:
            view: Harmonised view (before splitting).
            high_missing: Flag features missing in more than this fraction of samples.
            high_cardinality: Flag categoricals with more distinct values.
        """
        n = len(view)
        checks: list[dict[str, object]] = []

        def check(name: str, value: object, flag: bool = False, detail: str = "") -> None:
            checks.append({"check": name, "value": value, "flag": bool(flag), "detail": detail})

        per_patient = pd.Series(view.patient_ids).value_counts()
        check(
            "patient_ids_assumed",
            view.patient_ids_assumed,
            view.patient_ids_assumed,
            ASSUMED_IDS_WARNING if view.patient_ids_assumed else "",
        )
        check("samples", n)
        check("patients", int(per_patient.size))
        check("samples_per_patient_median", float(per_patient.median()))
        check(
            "samples_per_patient_max",
            int(per_patient.max()),
            per_patient.max() > 20,
            "many samples per patient: prefer index_policy or patient-level metrics",
        )
        counts = np.bincount(view.y, minlength=view.n_classes)
        prevalence = counts / max(n, 1)
        check(
            "minority_class_fraction",
            float(prevalence.min()),
            prevalence.min() < 0.05,
            "rare class: report PPV/NPV and CIs; consider enrichment awareness",
        )
        for k, name in enumerate(view.class_names):
            check(f"class_count[{name}]", int(counts[k]))
        if per_patient.max() > 1:
            labels_per_patient = pd.Series(view.y).groupby(view.patient_ids).nunique()
            mixed = int((labels_per_patient > 1).sum())
            check(
                "patients_with_mixed_labels",
                mixed,
                False,
                "expected for per-encounter labels; check if labels should be patient-level",
            )
        dup = pd.Series(view.sample_ids).duplicated().sum()
        check("duplicate_sample_ids", int(dup), dup > 0)

        features = pd.DataFrame(
            columns=["feature", "dtype", "missing_fraction", "n_unique", "flags"]
        )
        groups = pd.DataFrame(columns=["group", "n_features", "mean_missing", "max_missing"])
        if view.tabular is not None and view.tabular.shape[1]:
            tab = view.tabular
            missing = tab.isna().mean()
            nunique = tab.nunique(dropna=True)
            numeric = tab.select_dtypes(include=["number", "bool"]).columns
            flags = []
            for col in tab.columns:
                f = []
                if missing[col] > high_missing:
                    f.append("mostly_missing")
                if nunique[col] <= 1:
                    f.append("constant")
                if col not in numeric and nunique[col] > high_cardinality:
                    f.append("high_cardinality")
                flags.append(",".join(f))
            features = pd.DataFrame(
                {
                    "feature": tab.columns,
                    "dtype": [str(tab[c].dtype) for c in tab.columns],
                    "missing_fraction": missing.to_numpy().round(4),
                    "n_unique": nunique.to_numpy(),
                    "flags": flags,
                }
            ).sort_values("missing_fraction", ascending=False)
            prefix = pd.Series(tab.columns, index=tab.columns).str.split("_").str[0]
            groups = (
                pd.DataFrame({"group": prefix, "missing": missing})
                .groupby("group")["missing"]
                .agg(n_features="size", mean_missing="mean", max_missing="max")
                .round(4)
                .reset_index()
            )
            check("tabular_features", int(tab.shape[1]))
            n_flag = int((features["flags"] != "").sum())
            check("flagged_features", n_flag, n_flag > 0, "see data_quality_features")

        if MODALITY_PATTERN_COLUMN in view.metadata:
            pattern = view.metadata[MODALITY_PATTERN_COLUMN].to_numpy()
        else:
            pattern = view.modality_patterns()
        mp = (
            pd.DataFrame({"pattern": pattern, "patient": view.patient_ids, "y": view.y})
            .groupby("pattern")
            .agg(
                n_samples=("y", "size"),
                n_patients=("patient", "nunique"),
                positive_rate=("y", lambda s: float((s == view.n_classes - 1).mean())),
            )
            .reset_index()
        )
        if len(mp) > 1:
            rates = mp["positive_rate"]
            check(
                "modality_patterns",
                len(mp),
                rates.max() - rates.min() > 0.2,
                "availability is associated with the label (informative missingness)",
            )
        return cls(pd.DataFrame(checks), features, groups, mp)

    def write(self, out_dir: Path) -> list[Path]:
        """Write ``tables/data_quality_{summary,features,groups,modalities}.csv``."""
        table_dir = out_dir / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, frame in (
            ("summary", self.summary),
            ("features", self.features),
            ("groups", self.groups),
            ("modalities", self.modality_patterns),
        ):
            path = table_dir / f"data_quality_{name}.csv"
            frame.to_csv(path, index=False)
            paths.append(path)
        return paths
