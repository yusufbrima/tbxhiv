"""EHR feature groups for MIMIC-IV, computed per CXR study.

**Leakage rule:** every feature uses only information recorded *strictly before*
the study time (``study_time``). Discharge diagnoses of the current encounter
are never features (they define the labels); microbiology contributes *orders*
only, never results.

Each group receives the study index (``subject_id, study_id, study_time,
hadm_id, ed_stay_id, encounter_type, encounter_start``) and returns one row
per study (``study_id`` + feature columns prefixed with the group's prefix).
Groups are registered in :data:`FEATURE_GROUPS`; add your own the same way.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from tbmxai.core.registry import Registry
from tbmxai.core.utils import get_logger
from tbmxai.datasets.mimic.config import LabelDefinition
from tbmxai.datasets.mimic.icd import CHAPTERS, chapter, match_prefixes
from tbmxai.datasets.mimic.tables import MimicTables

logger = get_logger(__name__)

AGGREGATIONS = ("last", "min", "max", "mean", "count")


def slug(text: object, max_len: int = 40) -> str:
    """Lower-case snake identifier for column names."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(text).strip().lower()).strip("_")
    return s[:max_len] or "na"


class FeatureGroup(ABC):
    """A family of EHR features.

    Attributes:
        prefix: Column prefix of this group's features.
        tables: Raw tables required; the group is skipped (with a warning) if any is missing.
    """

    prefix: ClassVar[str]
    tables: ClassVar[tuple[str, ...]]

    def __init__(self, **params: Any) -> None:
        if params:
            raise TypeError(f"{type(self).__name__} got unexpected params {sorted(params)}")
        #: Human-readable descriptions of produced columns (for the feature dictionary).
        self.descriptions: dict[str, str] = {}

    def available(self, tables: MimicTables) -> list[str]:
        """Missing required tables (empty list = runnable)."""
        return [t for t in self.tables if not tables.available(t)]

    @abstractmethod
    def extract(self, studies: pd.DataFrame, tables: MimicTables) -> pd.DataFrame:
        """Return ``study_id`` + feature columns, one row per study (missing allowed)."""


FEATURE_GROUPS: Registry[FeatureGroup] = Registry("mimic_feature_group")


def _study_index(studies: pd.DataFrame, lookback_hours: float | None = None) -> pd.DataFrame:
    s = studies[["subject_id", "study_id", "study_time"]].copy()
    if lookback_hours is not None:
        s["window_start"] = s["study_time"] - pd.to_timedelta(lookback_hours, unit="h")
    return s


def _flags(long: pd.DataFrame, key: str, prefix: str, top_n: int | None) -> pd.DataFrame:
    """Binary study × value flags for the ``top_n`` most frequent values of ``key``."""
    if long.empty:
        return pd.DataFrame(columns=["study_id"])
    counts = long.drop_duplicates(["study_id", key])[key].value_counts()
    keep = counts.index[:top_n] if top_n else counts.index
    sub = long[long[key].isin(keep)].drop_duplicates(["study_id", key])
    wide = pd.crosstab(sub["study_id"], sub[key]).clip(upper=1)
    wide.columns = [f"{prefix}_{slug(c)}" for c in wide.columns]
    wide = wide.T.groupby(level=0).max().T  # merge columns whose slugs collide
    return wide.reset_index()


# ---------------------------------------------------------------- static groups


@FEATURE_GROUPS.register("demographics")
class DemographicsFeatures(FeatureGroup):
    """Sex, age at study, race group, language (patient-level, time-invariant)."""

    prefix = "demo"
    tables = ("patients", "admissions")

    RACE_GROUPS: ClassVar[dict[str, str]] = {
        "WHITE": "white",
        "BLACK": "black",
        "HISPANIC": "hispanic",
        "SOUTH AMERICAN": "hispanic",
        "ASIAN": "asian",
        "AMERICAN INDIAN": "other",
        "NATIVE HAWAIIAN": "other",
        "MULTIPLE": "other",
        "OTHER": "other",
        "PORTUGUESE": "white",
    }

    def race_group(self, race: object) -> str:
        r = str(race).upper()
        if r in {"NAN", "NONE", ""} or "UNKNOWN" in r or "UNABLE" in r or "DECLINED" in r:
            return "unknown"
        return next((g for k, g in self.RACE_GROUPS.items() if r.startswith(k)), "other")

    def extract(self, studies: pd.DataFrame, tables: MimicTables) -> pd.DataFrame:
        s = _study_index(studies)
        pat = tables.query(
            f"""SELECT p.subject_id, p.gender, p.anchor_age, p.anchor_year, p.anchor_year_group
            FROM {tables.scan("patients")} p WHERE p.subject_id IN (SELECT subject_id FROM s)""",
            s=s,
        )
        adm = tables.query(
            f"""SELECT a.subject_id, a.race, a.language, a.admittime
            FROM {tables.scan("admissions")} a WHERE a.subject_id IN (SELECT subject_id FROM s)""",
            s=s,
        )
        # Most frequent recorded race / language per patient (patient attributes).
        adm["race_group"] = adm["race"].map(self.race_group)
        known = adm[adm["race_group"] != "unknown"]
        race = known.groupby("subject_id")["race_group"].agg(lambda x: x.mode().iat[0])
        lang = (
            adm.dropna(subset=["language"])
            .groupby("subject_id")["language"]
            .agg(lambda x: x.mode().iat[0])
        )
        out = s.merge(pat, on="subject_id", how="left")
        age = out["anchor_age"] + (out["study_time"].dt.year - out["anchor_year"])
        res = pd.DataFrame(
            {
                "study_id": out["study_id"],
                "demo_sex": out["gender"],
                "demo_age": age.astype(float),
                "demo_race": out["subject_id"].map(race).fillna("unknown"),
                "demo_language_english": (out["subject_id"].map(lang).str.upper() == "ENGLISH")
                .astype(float)
                .where(out["subject_id"].map(lang).notna()),
                "demo_anchor_year_group": out["anchor_year_group"],
            }
        )
        self.descriptions = {
            "demo_sex": "Administrative sex (F/M)",
            "demo_age": "Age at study (years)",
            "demo_race": "Race group (mode over admissions)",
            "demo_language_english": "Preferred language is English",
            "demo_anchor_year_group": "De-identified year group (context, not a clinical feature)",
        }
        return res


@FEATURE_GROUPS.register("encounter")
class EncounterFeatures(FeatureGroup):
    """Encounter context at study time: type, admission attributes, timing, ICU status."""

    prefix = "enc"
    tables = ("admissions",)

    def extract(self, studies: pd.DataFrame, tables: MimicTables) -> pd.DataFrame:
        s = studies[
            [
                "study_id",
                "subject_id",
                "study_time",
                "hadm_id",
                "ed_stay_id",
                "encounter_type",
                "encounter_start",
            ]
        ].copy()
        hadm = s.dropna(subset=["hadm_id"])[["hadm_id"]].astype("int64").drop_duplicates()
        adm = tables.query(
            f"""SELECT a.hadm_id, a.admission_type, a.admission_location, a.insurance,
                       a.marital_status
            FROM {tables.scan("admissions")} a WHERE a.hadm_id IN (SELECT hadm_id FROM h)""",
            h=hadm,
        )
        s["hadm_id"] = pd.to_numeric(s["hadm_id"], errors="coerce").astype("Int64")
        adm["hadm_id"] = adm["hadm_id"].astype("Int64")
        out = s.merge(adm, on="hadm_id", how="left")
        res = pd.DataFrame(
            {
                "study_id": out["study_id"],
                "enc_type": out["encounter_type"],
                "enc_admission_type": out["admission_type"],
                "enc_admission_location": out["admission_location"],
                "enc_insurance": out["insurance"],
                "enc_marital_status": out["marital_status"],
                "enc_hours_since_start": (
                    out["study_time"] - out["encounter_start"]
                ).dt.total_seconds()
                / 3600,
            }
        )
        if tables.available("edstays"):
            ed = studies.dropna(subset=["ed_stay_id"])[["study_id", "ed_stay_id"]].astype(
                {"ed_stay_id": "int64"}
            )
            arr = tables.query(
                f"""SELECT e.study_id, x.arrival_transport FROM e
                JOIN {tables.scan("edstays")} x ON x.stay_id = e.ed_stay_id""",
                e=ed,
            )
            res = res.merge(
                arr.rename(columns={"arrival_transport": "enc_ed_arrival_transport"}),
                on="study_id",
                how="left",
            )
        if tables.available("icustays"):
            icu = tables.query(
                f"""SELECT s.study_id,
                       max(CASE WHEN i.intime <= s.study_time
                                 AND (i.outtime IS NULL OR i.outtime >= s.study_time)
                                THEN 1 ELSE 0 END) AS enc_in_icu,
                       max(CASE WHEN i.intime <= s.study_time
                                THEN date_diff('minute', i.intime, s.study_time) / 60.0 END) AS hrs
                FROM s JOIN {tables.scan("icustays")} i ON i.subject_id = s.subject_id
                   AND i.intime <= s.study_time AND (i.outtime IS NULL OR i.outtime >= s.study_time)
                GROUP BY s.study_id""",
                s=_study_index(studies),
            ).rename(columns={"hrs": "enc_hours_since_icu_admit"})
            res = res.merge(icu, on="study_id", how="left")
            res["enc_in_icu"] = res["enc_in_icu"].fillna(0)
        self.descriptions = {
            c: "Encounter context at study time" for c in res.columns if c != "study_id"
        }
        return res


@FEATURE_GROUPS.register("history")
class HistoryFeatures(FeatureGroup):
    """Prior utilisation and diagnoses from encounters that ended *before* the study.

    Args:
        label_definitions: Injected by the builder; ``hist_<label>`` flags are
            created for every ICD label (e.g. prior TB, prior HIV).
    """

    prefix = "hist"
    tables = ("admissions", "diagnoses_icd")

    def __init__(self, label_definitions: Mapping[str, LabelDefinition] | None = None) -> None:
        super().__init__()
        self.label_definitions = {
            k: v
            for k, v in (label_definitions or {}).items()
            if v.source == "icd" and (v.icd9 or v.icd10)
        }

    def extract(self, studies: pd.DataFrame, tables: MimicTables) -> pd.DataFrame:
        s = _study_index(studies)
        prior = tables.query(
            f"""SELECT s.study_id, a.hadm_id, a.dischtime
            FROM s JOIN {tables.scan("admissions")} a
              ON a.subject_id = s.subject_id AND a.dischtime < s.study_time""",
            s=s,
        )
        res = s[["study_id"]].copy()
        util = prior.groupby("study_id").agg(n=("hadm_id", "nunique"), last=("dischtime", "max"))
        res["hist_n_prior_admissions"] = res["study_id"].map(util["n"]).fillna(0)
        last = res["study_id"].map(util["last"])
        res["hist_days_since_last_discharge"] = (
            s["study_time"].to_numpy() - pd.to_datetime(last).to_numpy()
        ) / np.timedelta64(1, "D")
        if tables.available("edstays"):
            ed = tables.query(
                f"""SELECT s.study_id, count(*) AS n FROM s JOIN {tables.scan("edstays")} e
                  ON e.subject_id = s.subject_id AND e.outtime < s.study_time GROUP BY 1""",
                s=s,
            )
            res["hist_n_prior_ed_visits"] = (
                res["study_id"].map(ed.set_index("study_id")["n"]).fillna(0)
            )

        codes = tables.query(
            f"""SELECT p.study_id, d.icd_code, d.icd_version FROM p
            JOIN {tables.scan("diagnoses_icd")} d ON d.hadm_id = p.hadm_id""",
            p=prior[["study_id", "hadm_id"]],
        )
        res["hist_n_prior_dx"] = res["study_id"].map(codes.groupby("study_id").size()).fillna(0)
        if not codes.empty:
            codes["chapter"] = [
                chapter(c, v) for c, v in zip(codes["icd_code"], codes["icd_version"], strict=True)
            ]
        for ch in CHAPTERS:
            ids = set(codes.loc[codes["chapter"] == ch, "study_id"]) if not codes.empty else set()
            res[f"hist_dx_{ch}"] = res["study_id"].isin(ids).astype(float)
        for name, d in self.label_definitions.items():
            ids = (
                set(
                    codes.loc[
                        match_prefixes(codes["icd_code"], codes["icd_version"], d.icd9, d.icd10),
                        "study_id",
                    ]
                )
                if not codes.empty
                else set()
            )
            res[f"hist_{name}"] = res["study_id"].isin(ids).astype(float)
        self.descriptions = {
            c: "From encounters ending before the study" for c in res.columns if c != "study_id"
        }
        return res


# ------------------------------------------------------------- windowed numeric


class WindowedNumericGroup(FeatureGroup):
    """Aggregate time-stamped numeric events in ``[study_time - lookback, study_time)``.

    Args:
        lookback_hours: Window length.
        aggregations: Subset of ``last, min, max, mean, count``.
        min_coverage: Keep a variable if it is observed in at least this fraction of studies.
        coverage_denominator: ``all`` studies, or only studies ``with_any`` event of this group
            (sensible for ICU/ED data that most studies lack).
        include: Restrict to these variable ids (``None`` = all).
    """

    default_lookback: ClassVar[float] = 48.0
    default_denominator: ClassVar[str] = "all"

    def __init__(
        self,
        lookback_hours: float | None = None,
        aggregations: Sequence[str] = ("last", "min", "max", "mean", "count"),
        min_coverage: float = 0.02,
        coverage_denominator: str | None = None,
        include: Sequence[int | str] | None = None,
    ) -> None:
        super().__init__()
        bad = set(aggregations) - set(AGGREGATIONS)
        if bad:
            raise ValueError(f"Unknown aggregations {bad}; choose from {AGGREGATIONS}")
        self.lookback_hours = lookback_hours or self.default_lookback
        self.aggregations = list(aggregations)
        self.min_coverage = min_coverage
        self.coverage_denominator = coverage_denominator or self.default_denominator
        self.include = None if include is None else [str(i) for i in include]

    @abstractmethod
    def events_sql(self, tables: MimicTables) -> str:
        """SQL returning ``subject_id, t (TIMESTAMP), var (VARCHAR), v (DOUBLE)``."""

    def var_names(self, tables: MimicTables, variables: Sequence[str]) -> dict[str, str]:
        """Readable names for variable ids (default: the id itself)."""
        return {v: v for v in variables}

    def extract(self, studies: pd.DataFrame, tables: MimicTables) -> pd.DataFrame:
        s = _study_index(studies, self.lookback_hours)
        include = ""
        if self.include:
            quoted = ", ".join("'" + v.replace("'", "''") + "'" for v in self.include)
            include = f"AND e.var IN ({quoted})"
        long = tables.query(
            f"""
            WITH e AS ({self.events_sql(tables)})
            SELECT s.study_id, e.var,
                   arg_max(e.v, e.t) AS last, min(e.v) AS min, max(e.v) AS max,
                   avg(e.v) AS mean, count(*) AS count
            FROM s JOIN e ON e.subject_id = s.subject_id
              AND e.t < s.study_time AND e.t >= s.window_start
            WHERE e.v IS NOT NULL {include}
            GROUP BY s.study_id, e.var
            """,
            s=s,
        )
        res = s[["study_id"]].copy()
        if long.empty:
            return res
        denom = len(s) if self.coverage_denominator == "all" else long["study_id"].nunique()
        coverage = long.groupby("var")["study_id"].nunique() / max(denom, 1)
        keep = coverage[coverage >= self.min_coverage].index
        dropped = len(coverage) - len(keep)
        if dropped:
            logger.info(
                "%s: dropped %d variables below coverage %.3f",
                self.prefix,
                dropped,
                self.min_coverage,
            )
        long = long[long["var"].isin(keep)]
        names = self.var_names(tables, list(keep))
        wide = long.pivot(index="study_id", columns="var", values=self.aggregations)
        wide.columns = [
            f"{self.prefix}_{slug(names.get(var, var))}_{agg}" for agg, var in wide.columns
        ]
        wide = wide.T.groupby(level=0).first().T
        for agg, var in [(a, v) for a in self.aggregations for v in keep]:
            col = f"{self.prefix}_{slug(names.get(var, var))}_{agg}"
            self.descriptions[col] = (
                f"{agg} of {names.get(var, var)} (id {var}) "
                f"in the {self.lookback_hours:g} h before the study"
            )
        res = res.merge(wide.reset_index(), on="study_id", how="left")
        count_cols = [c for c in res.columns if c.endswith("_count")]
        res[count_cols] = res[count_cols].fillna(0)
        return res


@FEATURE_GROUPS.register("labs")
class LabFeatures(WindowedNumericGroup):
    """All numeric laboratory results (``hosp/labevents``)."""

    prefix = "lab"
    tables = ("labevents", "d_labitems")
    default_lookback = 48.0

    def events_sql(self, tables: MimicTables) -> str:
        return f"""SELECT subject_id, charttime AS t, CAST(itemid AS VARCHAR) AS var, valuenum AS v
                   FROM {tables.scan("labevents")}
                   WHERE subject_id IN (SELECT subject_id FROM s)"""

    def var_names(self, tables: MimicTables, variables: Sequence[str]) -> dict[str, str]:
        items = tables.read("d_labitems")
        items["name"] = items["label"].fillna("") + " " + items["fluid"].fillna("")
        mapping = dict(zip(items["itemid"].astype(str), items["name"].str.strip(), strict=True))
        return {v: f"{mapping.get(v, 'item')}_{v}" for v in variables}


@FEATURE_GROUPS.register("icu_charts")
class ICUChartFeatures(WindowedNumericGroup):
    """All numeric ICU chart events (vitals, ventilation, scores; ``icu/chartevents``)."""

    prefix = "icu"
    tables = ("chartevents", "d_items")
    default_lookback = 24.0
    default_denominator = "with_any"

    def events_sql(self, tables: MimicTables) -> str:
        return f"""SELECT subject_id, charttime AS t, CAST(itemid AS VARCHAR) AS var, valuenum AS v
                   FROM {tables.scan("chartevents")}
                   WHERE subject_id IN (SELECT subject_id FROM s)"""

    def var_names(self, tables: MimicTables, variables: Sequence[str]) -> dict[str, str]:
        items = tables.read("d_items")
        mapping = dict(zip(items["itemid"].astype(str), items["label"].fillna("item"), strict=True))
        return {v: f"{mapping.get(v, 'item')}_{v}" for v in variables}


_ED_VITALS = ("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp")


@FEATURE_GROUPS.register("ed_vitals")
class EDVitalFeatures(WindowedNumericGroup):
    """ED vital signs time series (``ed/vitalsign``)."""

    prefix = "edv"
    tables = ("vitalsign",)
    default_lookback = 24.0
    default_denominator = "with_any"

    def events_sql(self, tables: MimicTables) -> str:
        unions = " UNION ALL ".join(
            f"SELECT subject_id, charttime AS t, '{c}' AS var, {c} AS v FROM vs" for c in _ED_VITALS
        )
        return f"""WITH vs AS (SELECT * FROM {tables.scan("vitalsign")}
                               WHERE subject_id IN (SELECT subject_id FROM s))
                   SELECT * FROM ({unions})"""


@FEATURE_GROUPS.register("ed_triage")
class EDTriageFeatures(FeatureGroup):
    """ED triage at arrival (vitals, pain, acuity, chief-complaint flags) of the most
    recent ED stay that started before the study (within ``lookback_hours``).

    Args:
        lookback_hours: Max time between ED arrival and the study.
        complaint_top_n: Number of most frequent chief-complaint terms turned into flags.
    """

    prefix = "triage"
    tables = ("edstays", "triage")

    def __init__(self, lookback_hours: float = 72.0, complaint_top_n: int = 50) -> None:
        super().__init__()
        self.lookback_hours = lookback_hours
        self.complaint_top_n = complaint_top_n

    def extract(self, studies: pd.DataFrame, tables: MimicTables) -> pd.DataFrame:
        s = _study_index(studies, self.lookback_hours)
        tri = tables.query(
            f"""SELECT s.study_id, t.*
            FROM s JOIN {tables.scan("edstays")} e
              ON e.subject_id = s.subject_id
              AND e.intime < s.study_time AND e.intime >= s.window_start
            JOIN {tables.scan("triage")} t ON t.stay_id = e.stay_id
            QUALIFY row_number() OVER (PARTITION BY s.study_id ORDER BY e.intime DESC) = 1""",
            s=s,
        )
        res = s[["study_id"]].copy()
        if tri.empty:
            return res
        cols = {c: f"triage_{c}" for c in (*_ED_VITALS, "acuity")}
        num = tri[["study_id", *cols]].rename(columns=cols)
        num["triage_pain"] = pd.to_numeric(tri["pain"], errors="coerce").where(
            lambda x: x.between(0, 10)
        )
        res = res.merge(num, on="study_id", how="left")
        terms = (
            tri[["study_id", "chiefcomplaint"]]
            .dropna()
            .assign(term=lambda d: d["chiefcomplaint"].str.lower().str.split(r"[,/;]"))
            .explode("term")
            .assign(term=lambda d: d["term"].str.strip())
        )
        terms = terms[terms["term"].str.len() > 0]
        flags = _flags(terms, "term", "triage_cc", self.complaint_top_n)
        res = res.merge(flags, on="study_id", how="left")
        flag_cols = [c for c in res.columns if c.startswith("triage_cc_")]
        has_triage = res["study_id"].isin(set(tri["study_id"]))
        res.loc[has_triage, flag_cols] = res.loc[has_triage, flag_cols].fillna(0)
        self.descriptions = {
            c: "ED triage (most recent ED arrival before study)"
            for c in res.columns
            if c != "study_id"
        }
        return res


@FEATURE_GROUPS.register("omr")
class OMRFeatures(FeatureGroup):
    """Outpatient measurements (BMI, weight, height, blood pressure, eGFR…; ``hosp/omr``).

    Last value per measurement recorded on a *previous calendar day* (OMR has no time
    of day), within ``lookback_days``, plus days since that measurement.
    """

    prefix = "omr"
    tables = ("omr",)

    def __init__(self, lookback_days: int = 365) -> None:
        super().__init__()
        self.lookback_days = lookback_days

    def extract(self, studies: pd.DataFrame, tables: MimicTables) -> pd.DataFrame:
        s = _study_index(studies, self.lookback_days * 24)
        long = tables.query(
            f"""SELECT s.study_id, o.result_name, o.result_value, o.chartdate
            FROM s JOIN {tables.scan("omr")} o ON o.subject_id = s.subject_id
              AND o.chartdate < CAST(s.study_time AS DATE)
              AND o.chartdate >= CAST(s.window_start AS DATE)
            QUALIFY row_number() OVER (
                PARTITION BY s.study_id, o.result_name ORDER BY o.chartdate DESC, o.seq_num DESC
            ) = 1""",
            s=s,
        )
        res = s[["study_id"]].copy()
        if long.empty:
            return res
        long = long.merge(s[["study_id", "study_time"]], on="study_id")
        long["name"] = "omr_" + long["result_name"].map(slug)
        value = long["result_value"].astype(str)
        is_bp = value.str.contains("/", regex=False)
        parts = value.str.split("/", n=1, expand=True).reindex(columns=[0, 1])
        days = (long["study_time"].dt.normalize() - pd.to_datetime(long["chartdate"])).dt.days
        frames = [
            pd.DataFrame(
                {
                    "study_id": long.loc[~is_bp, "study_id"],
                    "col": long.loc[~is_bp, "name"],
                    "v": pd.to_numeric(value[~is_bp], errors="coerce"),
                }
            ),
            pd.DataFrame(
                {
                    "study_id": long.loc[is_bp, "study_id"],
                    "col": long.loc[is_bp, "name"] + "_systolic",
                    "v": pd.to_numeric(parts.loc[is_bp, 0], errors="coerce"),
                }
            ),
            pd.DataFrame(
                {
                    "study_id": long.loc[is_bp, "study_id"],
                    "col": long.loc[is_bp, "name"] + "_diastolic",
                    "v": pd.to_numeric(parts.loc[is_bp, 1], errors="coerce"),
                }
            ),
            pd.DataFrame(
                {
                    "study_id": long["study_id"],
                    "col": long["name"] + "_days_since",
                    "v": days.astype(float),
                }
            ),
        ]
        wide = pd.concat(frames, ignore_index=True).pivot_table(
            index="study_id", columns="col", values="v", aggfunc="first"
        )
        self.descriptions = {
            c: "Last outpatient (OMR) value before the study day" for c in wide.columns
        }
        return res.merge(wide.reset_index(), on="study_id", how="left")


@FEATURE_GROUPS.register("medications")
class MedicationFeatures(FeatureGroup):
    """Prescriptions started in ``[study_time - lookback, study_time)`` (``hosp/prescriptions``).

    Args:
        lookback_hours: Window length.
        top_n: Most frequent drug names turned into flags.
        drug_groups: Group name → case-insensitive substrings; a flag per group.
    """

    prefix = "med"
    tables = ("prescriptions",)

    DEFAULT_GROUPS: ClassVar[dict[str, list[str]]] = {
        "antitubercular": ["isoniazid", "rifampin", "rifabutin", "pyrazinamide", "ethambutol"],
        "antiretroviral": [
            "tenofovir",
            "emtricitabine",
            "efavirenz",
            "dolutegravir",
            "raltegravir",
            "bictegravir",
            "darunavir",
            "atazanavir",
            "ritonavir",
            "lamivudine",
            "abacavir",
            "zidovudine",
            "elvitegravir",
            "rilpivirine",
            "lopinavir",
        ],
        "pcp_agents": ["sulfamethoxazole", "trimethoprim", "atovaquone", "dapsone", "pentamidine"],
        "broad_antibiotic": [
            "vancomycin",
            "piperacillin",
            "cefepime",
            "meropenem",
            "ceftriaxone",
            "azithromycin",
            "levofloxacin",
            "moxifloxacin",
            "ceftazidime",
            "linezolid",
        ],
        "corticosteroid": ["prednisone", "methylprednisolone", "dexamethasone", "hydrocortisone"],
        "antifungal": ["fluconazole", "voriconazole", "micafungin", "amphotericin"],
        "anticoagulant": ["heparin", "enoxaparin", "warfarin", "apixaban", "rivaroxaban"],
    }

    def __init__(
        self,
        lookback_hours: float = 72.0,
        top_n: int = 100,
        drug_groups: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        super().__init__()
        self.lookback_hours = lookback_hours
        self.top_n = top_n
        self.drug_groups = {k: list(v) for k, v in (drug_groups or self.DEFAULT_GROUPS).items()}

    def extract(self, studies: pd.DataFrame, tables: MimicTables) -> pd.DataFrame:
        s = _study_index(studies, self.lookback_hours)
        long = tables.query(
            f"""SELECT DISTINCT s.study_id, lower(trim(p.drug)) AS drug
            FROM s JOIN {tables.scan("prescriptions")} p ON p.subject_id = s.subject_id
              AND p.starttime < s.study_time AND p.starttime >= s.window_start
            WHERE p.drug IS NOT NULL""",
            s=s,
        )
        res = s[["study_id"]].copy()
        res["med_n_distinct"] = res["study_id"].map(long.groupby("study_id").size()).fillna(0)
        for group, needles in self.drug_groups.items():
            pattern = "|".join(map(re.escape, (n.lower() for n in needles)))
            ids = set(
                long.loc[long["drug"].str.contains(pattern, regex=True, na=False), "study_id"]
            )
            res[f"med_group_{slug(group)}"] = res["study_id"].isin(ids).astype(float)
        flags = _flags(long, "drug", "med", self.top_n)
        res = res.merge(flags, on="study_id", how="left")
        flag_cols = [
            c
            for c in res.columns
            if c.startswith("med_") and not c.startswith(("med_group_", "med_n_"))
        ]
        res[flag_cols] = res[flag_cols].fillna(0)
        self.descriptions = {
            c: f"Prescription started in the {self.lookback_hours:g} h before the study"
            for c in res.columns
            if c != "study_id"
        }
        return res


@FEATURE_GROUPS.register("microbiology_orders")
class MicrobiologyOrderFeatures(FeatureGroup):
    """Microbiology tests *ordered* before the study (results are never used).

    Note: orders reflect clinical suspicion (e.g. AFB smears for TB). They are
    legitimately available at CXR time but can make TB tasks look easier;
    disable this group for a "CXR-triage" setting.

    Args:
        lookback_hours: Window length.
        top_n: Most frequent test / specimen names turned into flags.
    """

    prefix = "micro"
    tables = ("microbiologyevents",)
    AFB_PATTERN: ClassVar[str] = r"acid fast|afb|mycobacter|mtb|tuberculosis|quantiferon"

    def __init__(self, lookback_hours: float = 72.0, top_n: int = 50) -> None:
        super().__init__()
        self.lookback_hours = lookback_hours
        self.top_n = top_n

    def extract(self, studies: pd.DataFrame, tables: MimicTables) -> pd.DataFrame:
        s = _study_index(studies, self.lookback_hours)
        long = tables.query(
            f"""SELECT DISTINCT s.study_id, lower(m.test_name) AS test,
                   lower(m.spec_type_desc) AS specimen
            FROM s JOIN {tables.scan("microbiologyevents")} m ON m.subject_id = s.subject_id
              AND COALESCE(m.charttime, m.chartdate) < s.study_time
              AND COALESCE(m.charttime, m.chartdate) >= s.window_start""",
            s=s,
        )
        res = s[["study_id"]].copy()
        res["micro_n_orders"] = res["study_id"].map(long.groupby("study_id").size()).fillna(0)
        afb = long["test"].str.contains(self.AFB_PATTERN, regex=True, na=False)
        res["micro_afb_ordered"] = (
            res["study_id"].isin(set(long.loc[afb, "study_id"])).astype(float)
        )
        for key, pre in (("test", "micro_test"), ("specimen", "micro_spec")):
            flags = _flags(long.dropna(subset=[key]), key, pre, self.top_n)
            res = res.merge(flags, on="study_id", how="left")
        flag_cols = [c for c in res.columns if c.startswith(("micro_test_", "micro_spec_"))]
        res[flag_cols] = res[flag_cols].fillna(0)
        self.descriptions = {
            c: "Microbiology order before the study (no results)"
            for c in res.columns
            if c != "study_id"
        }
        return res


# ------------------------------------------------------------ ward / service / procedures


@FEATURE_GROUPS.register("care_unit")
class CareUnitFeatures(FeatureGroup):
    """Where the patient is at CXR time (``hosp/transfers``).

    The care unit whose stay covers the study time (``intime <= t < outtime``),
    hours spent in that unit, transfers so far since the encounter started, and
    ICU / ED flags. Only movements up to the study time are used.
    """

    prefix = "ward"
    tables = ("transfers",)
    ICU_PATTERN: ClassVar[str] = r"icu|intensive care|ccu|coronary care|neuro sicu|trauma sicu"

    def extract(self, studies: pd.DataFrame, tables: MimicTables) -> pd.DataFrame:
        s = studies[["subject_id", "study_id", "study_time", "encounter_start"]].copy()
        current = tables.query(
            f"""SELECT s.study_id, t.careunit, t.intime
            FROM s JOIN {tables.scan("transfers")} t ON t.subject_id = s.subject_id
              AND t.intime <= s.study_time AND (t.outtime IS NULL OR t.outtime > s.study_time)
              AND t.eventtype <> 'discharge'
            QUALIFY row_number() OVER (PARTITION BY s.study_id ORDER BY t.intime DESC) = 1""",
            s=s,
        )
        moves = tables.query(
            f"""SELECT s.study_id, count(*) AS n
            FROM s JOIN {tables.scan("transfers")} t ON t.subject_id = s.subject_id
              AND t.intime <= s.study_time AND t.intime >= s.encounter_start
              AND t.eventtype IN ('transfer', 'admit', 'ED')
            GROUP BY s.study_id""",
            s=s,
        )
        res = s[["study_id", "study_time"]].merge(current, on="study_id", how="left")
        unit = res["careunit"].fillna("unknown").astype(str)
        res = pd.DataFrame(
            {
                "study_id": res["study_id"],
                "ward_careunit": unit,
                "ward_hours_in_unit": (
                    res["study_time"] - pd.to_datetime(res["intime"])
                ).dt.total_seconds()
                / 3600,
                "ward_is_icu": unit.str.lower()
                .str.contains(self.ICU_PATTERN, regex=True)
                .astype(float),
                "ward_is_ed": unit.str.lower().str.contains("emergency", regex=False).astype(float),
            }
        )
        res["ward_n_moves_in_encounter"] = (
            res["study_id"].map(moves.set_index("study_id")["n"]).fillna(0)
        )
        self.descriptions = {
            "ward_careunit": "Care unit at study time",
            "ward_hours_in_unit": "Hours in the current care unit at study time",
            "ward_is_icu": "Current care unit is an ICU",
            "ward_is_ed": "Current care unit is the emergency department",
            "ward_n_moves_in_encounter": "Care-unit moves since the encounter started",
        }
        return res


@FEATURE_GROUPS.register("service")
class ServiceFeatures(FeatureGroup):
    """Clinical service responsible for the patient at CXR time (``hosp/services``).

    The latest service assignment of the study's admission with
    ``transfertime <= study_time``, and the number of service changes so far.
    """

    prefix = "svc"
    tables = ("services",)

    def extract(self, studies: pd.DataFrame, tables: MimicTables) -> pd.DataFrame:
        s = studies[["study_id", "study_time", "hadm_id"]].dropna(subset=["hadm_id"]).copy()
        s["hadm_id"] = s["hadm_id"].astype("int64")
        long = tables.query(
            f"""SELECT s.study_id, v.curr_service, v.transfertime
            FROM s JOIN {tables.scan("services")} v
              ON v.hadm_id = s.hadm_id AND v.transfertime <= s.study_time""",
            s=s,
        )
        res = studies[["study_id"]].copy()
        if long.empty:
            return res
        latest = long.sort_values("transfertime").groupby("study_id")["curr_service"].last()
        res["svc_current"] = res["study_id"].map(latest)
        res["svc_n_changes"] = (
            res["study_id"].map(long.groupby("study_id").size() - 1).clip(lower=0)
        )
        self.descriptions = {
            "svc_current": "Clinical service at study time (e.g. MED, SURG, CMED)",
            "svc_n_changes": "Service changes in the admission before the study",
        }
        return res


@FEATURE_GROUPS.register("procedure_history")
class ProcedureHistoryFeatures(FeatureGroup):
    """Procedures (ICD-9-CM / ICD-10-PCS) from **previous** admissions.

    Current-admission procedure codes are assigned at discharge, so only
    admissions discharged before the study are used (no leakage).

    Args:
        top_n: Most frequent procedure code prefixes turned into flags.
        prefix_len: Code prefix length used for grouping (4 ≈ ICD-9 subcategory /
            ICD-10-PCS body part).
    """

    prefix = "proc"
    tables = ("procedures_icd", "admissions")

    def __init__(self, top_n: int = 50, prefix_len: int = 4) -> None:
        super().__init__()
        self.top_n = top_n
        self.prefix_len = prefix_len

    def extract(self, studies: pd.DataFrame, tables: MimicTables) -> pd.DataFrame:
        s = _study_index(studies)
        long = tables.query(
            f"""SELECT s.study_id, p.hadm_id, p.icd_code, p.icd_version
            FROM s JOIN {tables.scan("admissions")} a
              ON a.subject_id = s.subject_id AND a.dischtime < s.study_time
            JOIN {tables.scan("procedures_icd")} p ON p.hadm_id = a.hadm_id""",
            s=s,
        )
        res = s[["study_id"]].copy()
        res["proc_n_prior"] = res["study_id"].map(long.groupby("study_id").size()).fillna(0)
        res["proc_n_prior_admissions"] = (
            res["study_id"].map(long.groupby("study_id")["hadm_id"].nunique()).fillna(0)
        )
        if not long.empty:
            long["code"] = (
                "icd"
                + long["icd_version"].astype(str)
                + "_"
                + long["icd_code"].astype(str).str.strip().str.upper().str[: self.prefix_len]
            )
            flags = _flags(long, "code", "proc", self.top_n)
            res = res.merge(flags, on="study_id", how="left")
            cols = [c for c in res.columns if c.startswith("proc_icd")]
            res[cols] = res[cols].fillna(0)
        self.descriptions = {
            c: "Procedure in an admission discharged before the study"
            for c in res.columns
            if c != "study_id"
        }
        return res
