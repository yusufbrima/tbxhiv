"""Index-radiograph cohort whose EHR features were available when the radiograph was taken.

The first build selected laboratory and ICU values by chart time, so results stored after the film
were used, and it read admission attributes for films taken before the admission. This script keeps
the cohort, labels and feature columns of OUT_SOURCE and recomputes only the affected groups:

  lab, icu   values whose result was stored before the film (``storetime``, else ``charttime``);
  enc        for a film taken before the admission time, encounter type ``ed`` and no admission
             attributes;
  demo       race and language from admissions that started before the film.

``ed_presentation`` becomes the route of presentation (an ED-only encounter, or an admission with an
ED registration); the former meaning, film taken during an ED stay, is kept as ``film_during_ed``.

Usage: python scripts/cohort/build_available_features.py SOURCE_DATASET FULL_DATASET OUT_DATASET
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

from tbmxai.datasets.mimic.config import MimicSourcesConfig
from tbmxai.datasets.mimic.features import (
    AGGREGATIONS,
    DemographicsFeatures,
    EncounterFeatures,
    ICUChartFeatures,
    LabFeatures,
    slug,
)
from tbmxai.datasets.mimic.tables import MimicTables

PHYS = Path("~/data/physionet").expanduser()


def included(prefix: str, group, tables: MimicTables, table: str, columns: list[str]) -> list[str]:
    """Item ids whose feature names are among the existing columns (same column set as before)."""
    bases = {c.rsplit("_", 1)[0] for c in columns if c.rsplit("_", 1)[-1] in AGGREGATIONS}
    ids = tables.read(table)["itemid"].astype(str).tolist()
    names = group.var_names(tables, ids)
    return [i for i in ids if f"{prefix}_{slug(names[i])}" in bases]


def main() -> None:
    src, full, out = (Path(a).expanduser() for a in sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    tables = MimicTables(
        MimicSourcesConfig.model_validate(
            {
                "mimic_iv": str(PHYS / "mimiciv/3.1"),
                "mimic_iv_ed": str(PHYS / "mimic-iv-ed/2.2"),
                "mimic_cxr_jpg": str(PHYS / "mimic-cxr-jpg/2.1.0"),
            }
        ),
        memory_limit="9GB",
        temp_directory=out / ".duckdb_tmp",
    )
    labels = pd.read_parquet(src / "labels.parquet")
    old = pd.read_parquet(src / "tabular.parquet").set_index("study_id")
    studies = pd.read_parquet(full / "selection" / "studies.parquet")
    studies = studies[studies["study_id"].isin(labels["study_id"])].copy()
    studies["study_time"] = pd.to_datetime(studies["study_time"])
    studies["encounter_start"] = pd.to_datetime(studies["encounter_start"])
    assert len(studies) == len(labels) == len(old)

    lab_cols = [c for c in old if c.startswith("lab_")]
    icu_cols = [c for c in old if c.startswith("icu_")]
    probe_lab, probe_icu = LabFeatures(), ICUChartFeatures()
    groups = {
        "lab": LabFeatures(
            lookback_hours=48, min_coverage=0, available_only=True,
            include=included("lab", probe_lab, tables, "d_labitems", lab_cols),
        ),
        "icu": ICUChartFeatures(
            lookback_hours=24, min_coverage=0, available_only=True,
            include=included("icu", probe_icu, tables, "d_items", icu_cols),
        ),
        "enc": EncounterFeatures(at_study_time=True),
        "demo": DemographicsFeatures(at_study_time=True),
    }  # fmt: skip
    new = old.copy()
    report = {}
    for prefix, group in groups.items():
        print(f"extracting {prefix} ...", flush=True)
        part = group.extract(studies, tables).drop_duplicates("study_id").set_index("study_id")
        cols = [c for c in old if c.startswith(f"{prefix}_")]
        part = part.reindex(index=old.index, columns=cols)
        counts = [c for c in cols if c.endswith("_count")]
        part[counts] = part[counts].fillna(0)
        changed = 0
        for c in cols:  # column by column: a frame of objects would not fit in memory
            a, b = part[c], old[c]
            changed += int(((a != b) & ~(a.isna() & b.isna())).sum())
        report[prefix] = {"columns": len(cols), "values_changed": changed}
        new[cols] = part[cols]
        print(prefix, report[prefix], flush=True)
    lab_counts = [c for c in lab_cols if c.endswith("_count")]
    report["lab_counts_above_first_build"] = int(
        (new[lab_counts].to_numpy() > old[lab_counts].to_numpy()).sum()
    )
    assert list(new.columns) == list(old.columns)

    adm = tables.query(f"SELECT hadm_id, edregtime FROM {tables.scan('admissions')}")
    hadm = pd.to_numeric(studies["hadm_id"], errors="coerce")
    reg = hadm.map(adm.set_index("hadm_id")["edregtime"]).set_axis(studies["study_id"])
    labels["film_during_ed"] = labels["ed_presentation"]
    route = (labels["encounter_type"] == "ed") | labels["study_id"].map(reg).notna().to_numpy()
    labels["ed_presentation"] = route.astype(int)
    report["ed_presentation"] = {
        "before": int(labels["film_during_ed"].sum()),
        "after": int(labels["ed_presentation"].sum()),
    }

    labels.to_parquet(out / "labels.parquet", index=False)
    new.reset_index().to_parquet(out / "tabular.parquet", index=False)
    for name in ("images.parquet", "reports.parquet"):
        shutil.copy2(src / name, out / name)
    spec = yaml.safe_load((src / "dataset.yaml").read_text())
    spec["name"] = f"{spec['name']}_available"
    spec["description"] += " EHR features limited to what was available at film time."
    (out / "dataset.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    (out / "build_info.json").write_text(json.dumps({"source": str(src), **report}, indent=2))
    shutil.rmtree(out / ".duckdb_tmp", ignore_errors=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
