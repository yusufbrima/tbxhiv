"""Study loading, encounter linkage, label derivation and patient-level subset selection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from tbmxai.core.utils import get_logger
from tbmxai.datasets.mimic.config import DevSplitConfig, LabelDefinition, SelectionConfig
from tbmxai.datasets.mimic.icd import match_prefixes, normalize
from tbmxai.datasets.mimic.tables import MimicTables

logger = get_logger(__name__)


def jpg_relpath(subject_id: int, study_id: int, dicom_id: str) -> str:
    """Relative path of a MIMIC-CXR-JPG image (``files/p10/p10000032/s5041.../<dicom>.jpg``)."""
    sid = str(int(subject_id))
    return f"files/p{sid[:2]}/p{sid}/s{int(study_id)}/{dicom_id}.jpg"


def report_relpath(subject_id: int, study_id: int) -> str:
    """Relative path of a MIMIC-CXR report (``files/p10/p10000032/s5041....txt``)."""
    sid = str(int(subject_id))
    return f"files/p{sid[:2]}/p{sid}/s{int(study_id)}.txt"


def _parse_study_time(date: pd.Series, time: pd.Series) -> pd.Series:
    """``StudyDate`` (YYYYMMDD) + ``StudyTime`` (HHMMSS[.fff], leading zeros may be lost)."""
    hhmmss = pd.to_numeric(time, errors="coerce").fillna(0).astype(int).astype(str).str.zfill(6)
    return pd.to_datetime(date.astype(str).str[:8] + hhmmss, format="%Y%m%d%H%M%S", errors="coerce")


def load_images(tables: MimicTables, views: list[str]) -> pd.DataFrame:
    """Image-level CXR metadata (frontal views by default) with study time and official split."""
    meta = tables.read("cxr_metadata")
    meta = meta[meta["ViewPosition"].isin(views)].copy()
    meta["image_time"] = _parse_study_time(meta["StudyDate"], meta["StudyTime"])
    meta = meta.dropna(subset=["image_time"])
    images = meta.rename(columns={"ViewPosition": "view_position"})[
        ["dicom_id", "subject_id", "study_id", "view_position", "image_time"]
    ]
    if tables.available("cxr_split"):
        split = tables.read("cxr_split")[["dicom_id", "split"]].rename(
            columns={"split": "official_split"}
        )
        images = images.merge(split, on="dicom_id", how="left")
    images["path"] = [
        jpg_relpath(s, st, d)
        for s, st, d in zip(
            images["subject_id"], images["study_id"], images["dicom_id"], strict=True
        )
    ]
    return images.reset_index(drop=True)


def studies_from_images(images: pd.DataFrame) -> pd.DataFrame:
    """One row per study: earliest image time is the study time."""
    agg = images.groupby(["subject_id", "study_id"], as_index=False).agg(
        study_time=("image_time", "min"), n_images=("dicom_id", "size")
    )
    if "official_split" in images:
        split = images.groupby("study_id")["official_split"].agg(
            lambda s: s.mode().iat[0] if s.notna().any() else None
        )
        agg = agg.merge(
            split.rename("official_split"), left_on="study_id", right_index=True, how="left"
        )
    return agg


def link_encounters(studies: pd.DataFrame, tables: MimicTables) -> pd.DataFrame:
    """Attach the encounter during which each study was taken.

    * ``inpatient``: ``min(edregtime, admittime) <= study_time <= dischtime`` (studies
      ordered in the ED before admission belong to that admission);
    * ``ed``: otherwise, an ED stay with ``intime <= study_time <= outtime``;
    * ``none``: no encounter (outpatient / unlinked).

    Adds ``hadm_id``, ``ed_stay_id``, ``encounter_type``, ``encounter_start``, ``encounter_end``.
    """
    s = studies[["subject_id", "study_id", "study_time"]]
    adm = tables.query(
        f"""
        SELECT s.study_id, a.hadm_id,
               LEAST(COALESCE(a.edregtime, a.admittime), a.admittime) AS encounter_start,
               a.dischtime AS encounter_end
        FROM s JOIN {tables.scan("admissions")} a ON a.subject_id = s.subject_id
        WHERE s.study_time >= LEAST(COALESCE(a.edregtime, a.admittime), a.admittime)
          AND s.study_time <= a.dischtime
        QUALIFY row_number() OVER (PARTITION BY s.study_id ORDER BY encounter_start DESC) = 1
        """,
        s=s,
    )
    out = studies.merge(adm, on="study_id", how="left")
    out["ed_stay_id"] = pd.NA
    if tables.available("edstays"):
        ed = tables.query(
            f"""
            SELECT s.study_id, e.stay_id AS ed_stay_id, e.intime, e.outtime
            FROM s JOIN {tables.scan("edstays")} e ON e.subject_id = s.subject_id
            WHERE s.study_time >= e.intime AND s.study_time <= e.outtime
            QUALIFY row_number() OVER (PARTITION BY s.study_id ORDER BY e.intime DESC) = 1
            """,
            s=s,
        )
        out = out.drop(columns="ed_stay_id").merge(ed, on="study_id", how="left")
        ed_only = out["hadm_id"].isna() & out["ed_stay_id"].notna()
        out.loc[ed_only, "encounter_start"] = out.loc[ed_only, "intime"]
        out.loc[ed_only, "encounter_end"] = out.loc[ed_only, "outtime"]
        out = out.drop(columns=["intime", "outtime"])
    out["encounter_type"] = np.select(
        [out["hadm_id"].notna(), out["ed_stay_id"].notna()], ["inpatient", "ed"], default="none"
    )
    return out


# ------------------------------------------------------------------- labels


def _encounter_codes(studies: pd.DataFrame, tables: MimicTables) -> pd.DataFrame:
    """Long table ``study_id, icd_code, icd_version`` of each study's encounter diagnoses."""
    parts = []
    hadm = studies.loc[studies["hadm_id"].notna(), ["study_id", "hadm_id"]].astype(
        {"hadm_id": "int64"}
    )
    if len(hadm):
        parts.append(
            tables.query(
                f"""SELECT h.study_id, d.icd_code, d.icd_version
                FROM h JOIN {tables.scan("diagnoses_icd")} d ON d.hadm_id = h.hadm_id""",
                h=hadm,
            )
        )
    ed = studies.loc[
        (studies["encounter_type"] == "ed") & studies["ed_stay_id"].notna(),
        ["study_id", "ed_stay_id"],
    ].astype({"ed_stay_id": "int64"})
    if len(ed) and tables.available("ed_diagnosis"):
        parts.append(
            tables.query(
                f"""SELECT e.study_id, d.icd_code, d.icd_version
                FROM e JOIN {tables.scan("ed_diagnosis")} d ON d.stay_id = e.ed_stay_id""",
                e=ed,
            )
        )
    if not parts:
        return pd.DataFrame(columns=["study_id", "icd_code", "icd_version"])
    return pd.concat(parts, ignore_index=True)


def derive_labels(
    studies: pd.DataFrame, tables: MimicTables, definitions: Mapping[str, LabelDefinition]
) -> pd.DataFrame:
    """Add ``label_<name>`` columns (nullable Int64: 1, 0 or <NA>).

    ICD labels are <NA> for studies without an encounter (no diagnoses exist).
    CheXpert labels follow the definition's ``uncertain`` policy; unmentioned
    findings are negative.
    """
    out = studies.copy()
    has_enc = out["encounter_type"] != "none"
    codes = _encounter_codes(out, tables)
    resolved: dict[str, pd.Series] = {}

    def resolve(name: str) -> pd.Series:
        if name in resolved:
            return resolved[name]
        d = definitions[name]
        if d.source == "chexpert":
            values = _chexpert_label(out, tables, d)
        elif d.source == "episode":
            values = _episode_label(out, tables, d)
            for other in d.all_of:
                values = values * resolve(other)
            values = values.astype("Int64")
        else:
            series = pd.Series(pd.NA, index=out.index, dtype="Int64")
            if d.icd9 or d.icd10:
                hit = codes.loc[
                    match_prefixes(codes["icd_code"], codes["icd_version"], d.icd9, d.icd10),
                    "study_id",
                ]
                series[has_enc] = out.loc[has_enc, "study_id"].isin(set(hit)).astype(int)
            else:
                series[has_enc] = 1
            for other in d.all_of:
                series = series * resolve(other)
            values = series.astype("Int64")
        resolved[name] = values
        return values

    for name in definitions:
        out[f"label_{name}"] = resolve(name)
    return out


def _sql(text: str) -> str:
    """Escape a value for a single-quoted SQL literal."""
    return text.replace("'", "''")


def _icd_condition(alias: str, d: LabelDefinition) -> str:
    """SQL condition: ``alias.icd_code`` starts with a version-appropriate prefix."""
    parts = []
    for version, prefixes in ((9, d.icd9), (10, d.icd10)):
        if prefixes:
            likes = " OR ".join(f"{alias}.icd_code LIKE '{_sql(normalize(p))}%'" for p in prefixes)
            parts.append(f"(CAST({alias}.icd_version AS INTEGER) = {version} AND ({likes}))")
    return " OR ".join(parts) or "FALSE"


def episode_events(
    tables: MimicTables, d: LabelDefinition, subjects: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Evidence events ``subject_id, start, end, kind`` for an episode label definition."""
    parts = []
    restrict = "JOIN subj USING (subject_id)" if subjects is not None else ""
    frames = {"subj": subjects} if subjects is not None else {}
    if d.icd9 or d.icd10:
        parts.append(
            tables.query(
                f"""SELECT DISTINCT a.subject_id, a.admittime AS start, a.dischtime AS "end",
                       'icd_admission' AS kind
                FROM {tables.scan("diagnoses_icd")} dx
                JOIN {tables.scan("admissions")} a ON a.hadm_id = dx.hadm_id
                {restrict.replace("USING (subject_id)", "ON subj.subject_id = a.subject_id")}
                WHERE {_icd_condition("dx", d)}""",
                **frames,
            )
        )
        if tables.available("ed_diagnosis") and tables.available("edstays"):
            parts.append(
                tables.query(
                    f"""SELECT DISTINCT e.subject_id, e.intime AS start, e.outtime AS "end",
                           'icd_ed' AS kind
                    FROM {tables.scan("ed_diagnosis")} dx
                    JOIN {tables.scan("edstays")} e ON e.stay_id = dx.stay_id
                    {restrict.replace("USING (subject_id)", "ON subj.subject_id = e.subject_id")}
                    WHERE {_icd_condition("dx", d)}""",
                    **frames,
                )
            )
    if d.organisms:
        if not tables.available("microbiologyevents"):
            raise FileNotFoundError("episode label needs microbiologyevents for `organisms`")
        likes = " OR ".join(f"upper(m.org_name) LIKE '%{_sql(o.upper())}%'" for o in d.organisms)
        parts.append(
            tables.query(
                f"""SELECT DISTINCT m.subject_id, COALESCE(m.charttime, m.chartdate) AS start,
                       COALESCE(m.charttime, m.chartdate) AS "end", 'microbiology' AS kind
                FROM {tables.scan("microbiologyevents")} m
                {restrict.replace("USING (subject_id)", "ON subj.subject_id = m.subject_id")}
                WHERE {likes}""",
                **frames,
            )
        )
    if d.drugs:
        if not tables.available("prescriptions"):
            raise FileNotFoundError("episode label needs prescriptions for `drugs`")
        likes = " OR ".join(f"lower(r.drug) LIKE '%{_sql(x.lower())}%'" for x in d.drugs)
        parts.append(
            tables.query(
                f"""SELECT r.subject_id, min(r.starttime) AS start, min(r.starttime) AS "end",
                       'drug' AS kind
                FROM {tables.scan("prescriptions")} r
                {restrict.replace("USING (subject_id)", "ON subj.subject_id = r.subject_id")}
                WHERE {likes} AND r.starttime IS NOT NULL
                GROUP BY r.subject_id""",
                **frames,
            )
        )
    if not parts:
        return pd.DataFrame(columns=["subject_id", "start", "end", "kind"])
    events = pd.concat(parts, ignore_index=True).dropna(subset=["start"])
    events["end"] = events["end"].fillna(events["start"])
    return events


def _episode_label(studies: pd.DataFrame, tables: MimicTables, d: LabelDefinition) -> pd.Series:
    events = episode_events(tables, d)
    out = pd.Series(0, index=studies.index, dtype="Int64")
    if events.empty:
        return out
    w = pd.Timedelta(days=d.window_days)
    s = studies[["study_id", "subject_id", "study_time"]].reset_index()
    m = s.merge(events, on="subject_id", how="inner")
    inside = (m["study_time"] >= m["start"] - w) & (m["study_time"] <= m["end"] + w)
    positive = set(m.loc[inside, "index"])
    evidence = set(m["index"])
    if d.outside_window == "missing":
        out[list(evidence - positive)] = pd.NA
    out[list(positive)] = 1
    out[studies["study_time"].isna().to_numpy() & studies.index.isin(evidence)] = pd.NA
    return out


def _chexpert_label(studies: pd.DataFrame, tables: MimicTables, d: LabelDefinition) -> pd.Series:
    if not tables.available("cxr_chexpert"):
        logger.warning("CheXpert table missing; label for %s is empty", d.chexpert_finding)
        return pd.Series(pd.NA, index=studies.index, dtype="Int64")
    chex = tables.read("cxr_chexpert")[["study_id", d.chexpert_finding]]
    raw = studies[["study_id"]].merge(chex, on="study_id", how="left")[d.chexpert_finding]
    values = pd.to_numeric(raw, errors="coerce")
    mapped = values.fillna(0.0).map(
        {1.0: 1, 0.0: 0, -1.0: {"positive": 1, "negative": 0, "drop": pd.NA}[d.uncertain]}
    )
    in_table = studies["study_id"].isin(set(chex["study_id"]))
    return pd.Series(np.where(in_table, mapped, pd.NA), index=studies.index).astype("Int64")


# ---------------------------------------------------------------- selection


def select_patients(studies: pd.DataFrame, cfg: SelectionConfig) -> np.ndarray:
    """Sample patient ids: all patients positive for ``enrich_labels`` first, then random fill."""
    rng = np.random.default_rng(cfg.seed)
    patients = np.sort(studies["subject_id"].unique())
    chosen: list[int] = []
    for name in cfg.enrich_labels:
        col = f"label_{name}"
        pos = np.sort(studies.loc[studies[col].fillna(0) == 1, "subject_id"].unique())
        new = np.setdiff1d(pos, chosen)
        room = cfg.n_patients - len(chosen)
        if len(new) > room:
            new = rng.choice(new, room, replace=False)
        chosen.extend(new.tolist())
        logger.info("Enrichment '%s': %d positive patients included", name, len(new))
    rest = np.setdiff1d(patients, chosen)
    room = max(0, cfg.n_patients - len(chosen))
    chosen.extend(rng.choice(rest, min(room, len(rest)), replace=False).tolist())
    return np.sort(np.asarray(chosen, dtype=int))


def limit_studies(studies: pd.DataFrame, max_per_patient: int | None, seed: int) -> pd.DataFrame:
    """Randomly keep at most ``max_per_patient`` studies per patient."""
    if max_per_patient is None:
        return studies
    return (
        studies.sample(frac=1.0, random_state=seed)
        .groupby("subject_id", group_keys=False)
        .head(max_per_patient)
        .sort_values(["subject_id", "study_time"])
    )


def pick_images(images: pd.DataFrame, one_per_study: bool) -> pd.DataFrame:
    """Optionally keep a single frontal image per study (PA preferred over AP)."""
    if not one_per_study:
        return images
    rank = images["view_position"].map({"PA": 0, "AP": 1}).fillna(2)
    return (
        images.assign(_rank=rank)
        .sort_values(["study_id", "_rank", "dicom_id"])
        .drop_duplicates("study_id")
        .drop(columns="_rank")
    )


def make_dev_split(studies: pd.DataFrame, cfg: DevSplitConfig) -> pd.DataFrame:
    """Patient-level ``subject_id, dev_split`` (train / val / test), stratified.

    Each patient's stratum is the first label in ``cfg.stratify`` with a positive study
    (rare labels first), else ``none``; strata too small to split go to train. With
    ``cfg.inherit_from``, patients in that file keep their assignment and only the others are
    split.
    """
    from sklearn.model_selection import train_test_split

    if cfg.inherit_from is not None:
        old = pd.read_csv(Path(cfg.inherit_from).expanduser())[["subject_id", "dev_split"]]
        present = studies["subject_id"].unique()
        old = old[old["subject_id"].isin(present)]
        new_studies = studies[~studies["subject_id"].isin(old["subject_id"])]
        parts = [old]
        if len(new_studies):
            parts.append(make_dev_split(new_studies, cfg.model_copy(update={"inherit_from": None})))
        out = pd.concat(parts, ignore_index=True).sort_values("subject_id", kind="stable")
        return out.reset_index(drop=True)

    per = studies.groupby("subject_id")
    stratum = pd.Series("none", index=sorted(studies["subject_id"].unique()), dtype=object)
    for name in reversed(cfg.stratify):
        col = f"label_{name}"
        if col in studies:
            pos = per[col].apply(lambda x: bool((x.fillna(0) == 1).any()))
            stratum[pos[pos].index] = name
    counts = stratum.value_counts()
    small = stratum.isin(counts[counts < 10].index)
    ids, strata = stratum.index[~small].to_numpy(), stratum[~small].to_numpy()
    dev, test = train_test_split(ids, test_size=cfg.test, stratify=strata, random_state=cfg.seed)
    dev_strata = stratum.loc[dev].to_numpy()
    _train, val = train_test_split(
        dev, test_size=cfg.val / (1 - cfg.test), stratify=dev_strata, random_state=cfg.seed
    )
    out = pd.Series("train", index=stratum.index, dtype=object)
    out[val] = "val"
    out[test] = "test"
    return out.rename_axis("subject_id").rename("dev_split").reset_index()
