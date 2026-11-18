"""Index-radiograph cohort for community-onset pneumonia.

Derived from a built MIMIC standard dataset (the full cohort; works on any build):

  * unit = encounter (an admission, or an ED stay without admission); one INDEX FILM per encounter:
    the first study taken no later than 24 h after presentation (encounters whose first film is
    later are dropped: those films are hospital-course films, not presentation films);
  * primary target ``label_cap``, community-onset pneumonia, less dependent on the film:
      1  pneumonia code in the PRINCIPAL position (admission: diagnoses_icd seq_num 1; ED-only
         stay: ED diagnosis seq_num 1) and, for admissions, antibiotics on >= 3 calendar days of
         the encounter;
      0  no pneumonia code anywhere on the encounter;
      NA otherwise (pneumonia only in a secondary position = hospital-onset or incidental, or a
         principal code without 3 antibiotic days); counted and reported, not modelled;
    ``label_pneumonia_any`` keeps the first draft's definition (any position) for reference;
  * pre-specified strata on the index film: ED presentation (film during an ED stay), view
    position (PA / AP), ICU at film time (transfers), hours from presentation;
  * every table of the source dataset filtered to the index studies; dev_split kept, so test
    patients stay unseen by MIMIC pretraining.

Pneumonia codes are the package's own definition (default_labels()["pneumonia"]).

Usage: python scripts/cohort/build_index_cohort.py SRC_DATASET OUT_DATASET
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from tbmxai.datasets.mimic.config import default_labels
from tbmxai.datasets.mimic.icd import match_prefixes

PHYS = Path("~/data/physionet").expanduser()
HOSP = PHYS / "mimiciv/3.1/hosp"
ED = PHYS / "mimic-iv-ed/2.2/ed"
WINDOW_H = 24
MIN_ABX_DAYS = 3
ANTIBIOTICS = (
    "ceftriaxone", "azithromycin", "levofloxacin", "moxifloxacin", "doxycycline", "piperacillin",
    "cefepime", "meropenem", "ampicillin", "amoxicillin", "cefuroxime", "ceftaroline", "ceftazidime",
    "aztreonam", "linezolid", "vancomycin", "clindamycin", "ertapenem", "imipenem", "cefpodoxime",
    "clarithromycin", "sulfamethoxazole", "cefazolin", "gentamicin", "tobramycin", "ciprofloxacin",
)  # fmt: skip


def index_films(src: Path) -> pd.DataFrame:
    labels = pd.read_parquet(src / "labels.parquet")
    studies = pd.read_parquet(src / "selection" / "studies.parquet")
    enc = studies[["study_id", "hadm_id", "ed_stay_id", "encounter_start", "encounter_end"]]
    d = labels.merge(enc, on="study_id", how="inner")
    d = d[d["encounter_type"].isin(["inpatient", "ed"])].copy()
    d["encounter_key"] = np.where(
        d["encounter_type"] == "inpatient",
        "h" + d["hadm_id"].astype("Int64").astype(str),
        "e" + d["ed_stay_id"].astype("Int64").astype(str),
    )
    d["study_time"] = pd.to_datetime(d["study_time"])
    d["encounter_start"] = pd.to_datetime(d["encounter_start"])
    d["hours_from_presentation"] = (
        d["study_time"] - d["encounter_start"]
    ).dt.total_seconds() / 3600
    first = (
        d.sort_values(["encounter_key", "study_time", "study_id"]).groupby("encounter_key").head(1)
    )
    return first[first["hours_from_presentation"] <= WINDOW_H].copy(), len(first)


def code_positions(idx: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per encounter: pneumonia code in the principal position, and anywhere."""
    pn = default_labels()["pneumonia"]
    hadm = idx.loc[idx["encounter_type"] == "inpatient", ["encounter_key", "hadm_id"]].copy()
    hadm["hadm_id"] = hadm["hadm_id"].astype("int64")
    con.register("h", hadm)
    dx = con.execute(
        f"""SELECT h.encounter_key, d.seq_num, d.icd_code, d.icd_version
        FROM h JOIN read_csv_auto('{HOSP / "diagnoses_icd.csv.gz"}') d ON d.hadm_id = h.hadm_id"""
    ).df()
    eds = idx.loc[idx["encounter_type"] == "ed", ["encounter_key", "ed_stay_id"]].copy()
    eds["ed_stay_id"] = eds["ed_stay_id"].astype("int64")
    con.register("e", eds)
    edx = con.execute(
        f"""SELECT e.encounter_key, d.seq_num, d.icd_code, d.icd_version
        FROM e JOIN read_csv_auto('{ED / "diagnosis.csv.gz"}') d ON d.stay_id = e.ed_stay_id"""
    ).df()
    allx = pd.concat([dx, edx], ignore_index=True)
    allx["pn"] = match_prefixes(
        allx["icd_code"].astype(str), allx["icd_version"], pn.icd9, pn.icd10
    )
    g = allx.groupby("encounter_key")
    out = pd.DataFrame(
        {
            "pneu_any": g["pn"].any(),
            "pneu_principal": allx[allx["seq_num"] == 1].groupby("encounter_key")["pn"].any(),
        }
    )
    return out.fillna(False).astype(bool)


def antibiotic_days(idx: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.Series:
    """Per admission: calendar days of the encounter covered by an antibiotic order."""
    adm = idx.loc[idx["encounter_type"] == "inpatient", ["encounter_key", "hadm_id",
                                                        "encounter_start", "encounter_end"]].copy()  # fmt: skip
    adm["hadm_id"] = adm["hadm_id"].astype("int64")
    con.register("a", adm)
    pattern = "|".join(ANTIBIOTICS)
    rx = con.execute(
        f"""SELECT a.encounter_key, a.encounter_start, a.encounter_end, p.starttime, p.stoptime,
                   lower(p.drug) AS drug, upper(coalesce(p.route, '')) AS route
        FROM a JOIN read_csv_auto('{HOSP / "prescriptions.csv.gz"}', all_varchar=true) p
          ON CAST(p.hadm_id AS BIGINT) = a.hadm_id
        WHERE regexp_matches(lower(p.drug), '{pattern}')"""
    ).df()
    # oral vancomycin treats C. difficile, not pneumonia
    rx = rx[~(rx["drug"].str.contains("vancomycin") & rx["route"].str.startswith("PO"))]
    for c in ("starttime", "stoptime", "encounter_start", "encounter_end"):
        rx[c] = pd.to_datetime(rx[c], errors="coerce")
    rx = rx.dropna(subset=["starttime"])
    rx["stoptime"] = rx["stoptime"].fillna(rx["starttime"])
    lo = rx[["starttime", "encounter_start"]].max(axis=1).dt.normalize()
    hi = rx[["stoptime", "encounter_end"]].min(axis=1).dt.normalize()
    rx = rx.assign(lo=lo, hi=hi)[hi >= lo]
    days = {}
    for key, g in rx.groupby("encounter_key"):
        covered = set()
        for a, b in zip(g["lo"], g["hi"], strict=True):
            covered.update(pd.date_range(a, b, freq="D"))
        days[key] = len(covered)
    return pd.Series(days, name="abx_days", dtype=float)


def icu_at_film(idx: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.Series:
    adm = idx.loc[
        idx["encounter_type"] == "inpatient", ["encounter_key", "hadm_id", "study_time"]
    ].copy()
    adm["hadm_id"] = adm["hadm_id"].astype("int64")
    con.register("f", adm)
    t = con.execute(
        f"""SELECT DISTINCT f.encounter_key
        FROM f JOIN read_csv_auto('{HOSP / "transfers.csv.gz"}') t ON t.hadm_id = f.hadm_id
        WHERE (upper(t.careunit) LIKE '%ICU%' OR upper(t.careunit) LIKE '%INTENSIVE%')
          AND f.study_time >= t.intime AND f.study_time <= coalesce(t.outtime, f.study_time)"""
    ).df()
    return pd.Series(True, index=t["encounter_key"], name="icu_at_film")


def main() -> None:
    src, out = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='5GB'")
    idx, n_encounters = index_films(src)
    idx = idx.set_index("encounter_key")
    codes = code_positions(idx.reset_index(), con)
    idx = idx.join(codes)
    idx[["pneu_any", "pneu_principal"]] = (
        idx[["pneu_any", "pneu_principal"]].fillna(False).astype(bool)
    )
    idx = idx.join(antibiotic_days(idx.reset_index(), con))
    idx["abx_days"] = idx["abx_days"].fillna(0)
    idx = idx.join(icu_at_film(idx.reset_index(), con))
    idx["icu_at_film"] = idx["icu_at_film"].fillna(False).astype(int)
    images = pd.read_parquet(src / "images.parquet")
    view = (
        images.assign(r=images["view_position"].map({"PA": 0, "AP": 1}).fillna(2))
        .sort_values(["study_id", "r"])
        .drop_duplicates("study_id")
        .set_index("study_id")["view_position"]
    )
    idx["index_view"] = idx["study_id"].map(view)
    idx["ed_presentation"] = idx["ed_stay_id"].notna().astype(int)
    abx_ok = (idx["encounter_type"] == "ed") | (idx["abx_days"] >= MIN_ABX_DAYS)
    cap = pd.Series(pd.NA, index=idx.index, dtype="Int64")
    cap[~idx["pneu_any"]] = 0
    cap[idx["pneu_principal"] & abx_ok] = 1
    idx["label_cap"] = cap
    idx["label_pneumonia_any"] = idx["pneu_any"].astype(int)
    idx["pneumonia_secondary_only"] = (idx["pneu_any"] & ~idx["pneu_principal"]).astype(int)
    keep_cols = list(pd.read_parquet(src / "labels.parquet").columns)
    extra = ["encounter_type", "label_cap", "label_pneumonia_any", "pneumonia_secondary_only",
             "abx_days", "ed_presentation", "index_view", "icu_at_film", "hours_from_presentation"]  # fmt: skip
    lab = idx.reset_index()[["encounter_key"] + [c for c in keep_cols if c not in extra] + extra]
    lab.to_parquet(out / "labels.parquet", index=False)
    keys = set(lab["study_id"])
    for name in ("tabular", "images", "reports"):
        f = src / f"{name}.parquet"
        if f.exists():
            t = pd.read_parquet(f)
            t[t["study_id"].isin(keys)].to_parquet(out / f"{name}.parquet", index=False)
    spec = yaml.safe_load((src / "dataset.yaml").read_text())
    spec["name"] = f"{spec['name']}_index"
    spec["description"] = (
        "Index-film cohort: one presentation film (<= 24 h) per encounter; label_cap = "
        "community-onset pneumonia (principal-position code, >= 3 antibiotic days for admissions). "
        f"Derived from {src}."
    )
    spec["labels"] = {
        "label_cap": {"classes": ["0", "1"], "description": "community-onset pneumonia"},
        "label_pneumonia_any": {
            "classes": ["0", "1"],
            "description": "pneumonia code, any position",
        },
        "label_chexpert_pneumonia": spec["labels"]["label_chexpert_pneumonia"],
    }
    if spec.get("image_root") and not Path(spec["image_root"]).is_absolute():
        spec["image_root"] = str((src / spec["image_root"]).resolve())
    (out / "dataset.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    lc = lab["label_cap"]
    info = {
        "source": str(src),
        "encounters_with_film": n_encounters,
        "encounters_index_film_within_24h": len(lab),
        "patients": int(lab["patient_id"].nunique()),
        "label_cap": {"positive": int((lc == 1).sum()), "negative": int((lc == 0).sum()),
                      "excluded_NA": int(lc.isna().sum())},
        "excluded_breakdown": {
            "secondary_only": int(((lc.isna()) & (lab["pneumonia_secondary_only"] == 1)).sum()),
            "principal_without_3_abx_days": int(((lc.isna()) & (lab["pneumonia_secondary_only"] == 0)).sum()),
        },
        "ed_presentation": int(lab["ed_presentation"].sum()),
        "icu_at_film": int(lab["icu_at_film"].sum()),
        "view": lab["index_view"].value_counts().to_dict(),
        "per_split_encounters": lab["dev_split"].value_counts().to_dict(),
        "positives_per_split": lab[lc == 1]["dev_split"].value_counts().to_dict(),
    }  # fmt: skip
    (out / "build_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
