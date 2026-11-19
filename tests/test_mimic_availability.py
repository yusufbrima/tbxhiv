"""Features restricted to what was available when the radiograph was taken."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tbmxai.datasets.mimic.config import MimicSourcesConfig
from tbmxai.datasets.mimic.features import (
    DemographicsFeatures,
    EDVitalFeatures,
    EncounterFeatures,
    LabFeatures,
)
from tbmxai.datasets.mimic.tables import MimicTables

FILM = pd.Timestamp("2150-01-02 12:00")


@pytest.fixture()
def tables(tmp_path: Path) -> MimicTables:
    hosp = tmp_path / "mimiciv" / "hosp"
    hosp.mkdir(parents=True)
    pd.DataFrame(
        {
            "subject_id": [1, 1, 1, 1],
            "itemid": [50000, 50000, 50000, 50000],
            "charttime": [
                "2150-01-02 09:00",
                "2150-01-02 11:00",
                "2150-01-02 11:30",
                "2150-01-02 12:30",  # charted after the film, although stored before it
            ],
            "storetime": ["2150-01-02 10:00", "2150-01-02 13:00", "", "2150-01-02 11:45"],
            "valuenum": [1.0, 2.0, 4.0, 8.0],
        }
    ).to_csv(hosp / "labevents.csv", index=False)
    pd.DataFrame({"itemid": [50000], "label": ["Sodium"], "fluid": ["Blood"]}).to_csv(
        hosp / "d_labitems.csv", index=False
    )
    pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [10, 11],
            "admittime": ["2150-01-02 15:00", "2150-06-01 08:00"],
            "dischtime": ["2150-01-05 10:00", "2150-06-03 08:00"],
            "edregtime": ["2150-01-02 10:00", ""],
            "admission_type": ["EW EMER.", "URGENT"],
            "admission_location": ["EMERGENCY ROOM", "TRANSFER"],
            "insurance": ["Medicare", "Medicare"],
            "marital_status": ["SINGLE", "SINGLE"],
            "race": ["UNKNOWN", "ASIAN"],
            "language": ["", "ENGLISH"],
        }
    ).to_csv(hosp / "admissions.csv", index=False)
    pd.DataFrame(
        {
            "subject_id": [1],
            "gender": ["F"],
            "anchor_age": [40],
            "anchor_year": [2150],
            "anchor_year_group": ["2014 - 2016"],
        }
    ).to_csv(hosp / "patients.csv", index=False)
    return MimicTables(
        MimicSourcesConfig.model_validate(
            {"mimic_iv": str(tmp_path / "mimiciv"), "mimic_cxr_jpg": str(tmp_path / "cxr")}
        )
    )


@pytest.fixture()
def studies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1],
            "study_id": [100],
            "study_time": [FILM],
            "hadm_id": [10],
            "ed_stay_id": [pd.NA],
            "encounter_type": ["inpatient"],
            "encounter_start": [pd.Timestamp("2150-01-02 10:00")],
        }
    )


def test_labs_keep_only_results_stored_before_the_film(tables, studies) -> None:
    every = LabFeatures(min_coverage=0).extract(studies, tables).iloc[0]
    known = LabFeatures(min_coverage=0, available_only=True).extract(studies, tables).iloc[0]
    assert every["lab_sodium_blood_50000_count"] == 3
    # 09:00 was stored at 10:00; 11:30 has no storetime and falls back to its chart time
    assert known["lab_sodium_blood_50000_count"] == 2
    assert known["lab_sodium_blood_50000_max"] == 4.0
    assert known["lab_sodium_blood_50000_min"] == 1.0


def test_groups_without_a_result_time_refuse_the_option() -> None:
    with pytest.raises(ValueError, match="storetime"):
        EDVitalFeatures(available_only=True)


def test_encounter_before_admission_is_an_ed_study(tables, studies) -> None:
    default = EncounterFeatures().extract(studies, tables).iloc[0]
    known = EncounterFeatures(at_study_time=True).extract(studies, tables).iloc[0]
    assert default["enc_type"] == "inpatient" and default["enc_insurance"] == "Medicare"
    assert known["enc_type"] == "ed"
    for col in ("enc_admission_type", "enc_admission_location", "enc_insurance"):
        assert pd.isna(known[col])
    assert known["enc_hours_since_start"] == pytest.approx(2.0)


def test_demographics_ignore_later_admissions(tables, studies) -> None:
    default = DemographicsFeatures().extract(studies, tables).iloc[0]
    known = DemographicsFeatures(at_study_time=True).extract(studies, tables).iloc[0]
    assert default["demo_race"] == "asian" and default["demo_language_english"] == 1.0
    assert known["demo_race"] == "unknown" and pd.isna(known["demo_language_english"])
