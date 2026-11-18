"""A tiny *fake* MIMIC-IV / MIMIC-IV-ED / MIMIC-CXR-JPG tree with the real file layout
and column schemas. Used for tests and for dry-running the adapter before the real
(credentialed) data is downloaded. Contains no real patient data.

Layout (mirrors ``wget -nH --cut-dirs=1`` of PhysioNet)::

    <root>/mimiciv/3.1/{hosp,icu}/*.csv.gz
    <root>/mimic-iv-ed/2.2/ed/*.csv.gz
    <root>/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-{metadata,split,chexpert}.csv.gz + files/…/*.jpg
    <root>/mimic-cxr/2.1.0/files/…/s<study>.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CHEXPERT = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]
LAB_ITEMS = {
    51301: ("White Blood Cells", 7.0, 6.0),
    50889: ("C-Reactive Protein", 5.0, 60.0),
    51222: ("Hemoglobin", 12.5, -1.5),
    50862: ("Albumin", 3.8, -0.6),
}
CHART_ITEMS = {
    220045: ("Heart Rate", 85.0, 15.0),
    220210: ("Respiratory Rate", 18.0, 6.0),
    220277: ("O2 saturation pulseoxymetry", 96.0, -4.0),
}


@dataclass(frozen=True)
class FakeMimicRoots:
    """Roots to plug into ``MimicSourcesConfig``."""

    root: Path
    mimic_iv: Path
    mimic_iv_ed: Path
    mimic_cxr_jpg: Path
    mimic_cxr_reports: Path


def _gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")


def _ts(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")


def make_fake_mimic(
    root: str | Path, n_patients: int = 80, seed: int = 0, image_size: int = 64
) -> FakeMimicRoots:
    """Write a fake MIMIC tree under ``root`` and return its roots."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    root = Path(root)
    roots = FakeMimicRoots(
        root=root,
        mimic_iv=root / "mimiciv/3.1",
        mimic_iv_ed=root / "mimic-iv-ed/2.2",
        mimic_cxr_jpg=root / "mimic-cxr-jpg/2.1.0",
        mimic_cxr_reports=root / "mimic-cxr/2.1.0",
    )
    T: dict[str, list[dict]] = {
        k: []
        for k in (
            "patients",
            "admissions",
            "diagnoses_icd",
            "labevents",
            "omr",
            "prescriptions",
            "micro",
            "icustays",
            "chartevents",
            "edstays",
            "triage",
            "vitalsign",
            "ed_diagnosis",
            "metadata",
            "split",
            "chexpert",
            "transfers",
            "services",
            "procedures_icd",
        )
    }
    hadm_id, stay_id, study_id, lab_id = 20_000_000, 30_000_000, 50_000_000, 1
    for i in range(n_patients):
        sid = 10_000_000 + i * 37
        hiv = rng.random() < 0.25
        anchor_year = 2150 + int(rng.integers(0, 30))
        T["patients"].append(
            {
                "subject_id": sid,
                "gender": rng.choice(["F", "M"]),
                "anchor_age": int(rng.integers(20, 85)),
                "anchor_year": anchor_year,
                "anchor_year_group": "2014 - 2016",
                "dod": "",
            }
        )
        t = pd.Timestamp(f"{anchor_year}-01-01") + pd.Timedelta(days=int(rng.integers(0, 200)))
        race = rng.choice(
            [
                "WHITE",
                "BLACK/AFRICAN AMERICAN",
                "HISPANIC/LATINO - PUERTO RICAN",
                "ASIAN",
                "UNKNOWN",
            ]
        )
        split = rng.choice(["train", "validate", "test"], p=[0.8, 0.1, 0.1])
        for _visit in range(int(rng.integers(1, 4))):
            t += pd.Timedelta(days=int(rng.integers(20, 300)))
            kind = rng.choice(["inpatient", "ed", "outpatient"], p=[0.65, 0.2, 0.15])
            pneu = rng.random() < (0.55 if hiv else 0.35)
            tb = rng.random() < (0.25 if hiv else 0.06)
            positive = pneu or tb
            codes: list[tuple[str, int]] = []
            v = 10 if t.year >= 2165 else 9
            if pneu:
                codes.append(("J189", 10) if v == 10 else ("486", 9))
            if tb:
                codes.append(("A150", 10) if v == 10 else ("0110", 9))
            if hiv:
                codes.append(("B20", 10) if v == 10 else ("042", 9))
            codes.append(("I10", 10) if v == 10 else ("4019", 9))
            ed_in = t
            ed_out = t + pd.Timedelta(hours=6)
            cur_hadm = None
            if kind in ("inpatient", "ed"):
                stay_id += 1
                T["edstays"].append(
                    {
                        "subject_id": sid,
                        "hadm_id": "",
                        "stay_id": stay_id,
                        "intime": _ts(ed_in),
                        "outtime": _ts(ed_out),
                        "gender": "F",
                        "race": race,
                        "arrival_transport": rng.choice(["AMBULANCE", "WALK IN"]),
                        "disposition": "HOME",
                    }
                )
                T["triage"].append(
                    {
                        "subject_id": sid,
                        "stay_id": stay_id,
                        "temperature": round(98.4 + 2.0 * positive + rng.normal(0, 0.6), 1),
                        "heartrate": round(80 + 15 * positive + rng.normal(0, 8)),
                        "resprate": round(16 + 5 * positive + rng.normal(0, 2)),
                        "o2sat": round(98 - 4 * positive + rng.normal(0, 1)),
                        "sbp": round(rng.normal(125, 15)),
                        "dbp": round(rng.normal(75, 10)),
                        "pain": rng.choice(["0", "5", "UA", "8"]),
                        "acuity": int(rng.integers(1, 5)),
                        "chiefcomplaint": rng.choice(
                            ["Cough, Fever", "Dyspnea", "Chest pain", "Fever"]
                        )
                        if positive
                        else rng.choice(["Chest pain", "Abd pain", "Fall"]),
                    }
                )
                for k in range(3):
                    T["vitalsign"].append(
                        {
                            "subject_id": sid,
                            "stay_id": stay_id,
                            "charttime": _ts(ed_in + pd.Timedelta(hours=k)),
                            "temperature": round(98.4 + 2.0 * positive + rng.normal(0, 0.5), 1),
                            "heartrate": round(80 + 15 * positive + rng.normal(0, 8)),
                            "resprate": 18,
                            "o2sat": 97,
                            "sbp": 120,
                            "dbp": 70,
                            "rhythm": "",
                            "pain": "0",
                        }
                    )
            if kind == "inpatient":
                hadm_id += 1
                cur_hadm = hadm_id
                T["edstays"][-1]["hadm_id"] = cur_hadm
                admit = ed_out
                disch = admit + pd.Timedelta(days=int(rng.integers(2, 10)))
                T["admissions"].append(
                    {
                        "subject_id": sid,
                        "hadm_id": cur_hadm,
                        "admittime": _ts(admit),
                        "dischtime": _ts(disch),
                        "deathtime": "",
                        "admission_type": rng.choice(["EW EMER.", "URGENT", "ELECTIVE"]),
                        "admit_provider_id": "P1",
                        "admission_location": "EMERGENCY ROOM",
                        "discharge_location": "HOME",
                        "insurance": rng.choice(["Medicare", "Medicaid", "Other"]),
                        "language": "ENGLISH",
                        "marital_status": rng.choice(["MARRIED", "SINGLE"]),
                        "race": race,
                        "edregtime": _ts(ed_in),
                        "edouttime": _ts(ed_out),
                        "hospital_expire_flag": 0,
                    }
                )
                # Ward movements (ED → ward), service and coded procedures of this admission.
                ward = rng.choice(["Medicine", "Medical Intensive Care Unit (MICU)", "Surgery"])
                T["transfers"] += [
                    {
                        "subject_id": sid,
                        "hadm_id": cur_hadm,
                        "transfer_id": len(T["transfers"]) + 1,
                        "eventtype": "ED",
                        "careunit": "Emergency Department",
                        "intime": _ts(ed_in),
                        "outtime": _ts(ed_out),
                    },
                    {
                        "subject_id": sid,
                        "hadm_id": cur_hadm,
                        "transfer_id": len(T["transfers"]) + 2,
                        "eventtype": "admit",
                        "careunit": ward,
                        "intime": _ts(admit),
                        "outtime": _ts(disch),
                    },
                    {
                        "subject_id": sid,
                        "hadm_id": cur_hadm,
                        "transfer_id": len(T["transfers"]) + 3,
                        "eventtype": "discharge",
                        "careunit": "UNKNOWN",
                        "intime": _ts(disch),
                        "outtime": "",
                    },
                ]
                T["services"].append(
                    {
                        "subject_id": sid,
                        "hadm_id": cur_hadm,
                        "transfertime": _ts(ed_in),
                        "prev_service": "",
                        "curr_service": rng.choice(["MED", "CMED", "SURG", "OMED"]),
                    }
                )
                for n, (pcode, pver) in enumerate(
                    [
                        ("9671", 9) if v == 9 else ("5A1945Z", 10),
                        ("3893", 9) if v == 9 else ("02HV33Z", 10),
                    ][: int(rng.integers(0, 3))],
                    1,
                ):
                    T["procedures_icd"].append(
                        {
                            "subject_id": sid,
                            "hadm_id": cur_hadm,
                            "seq_num": n,
                            "chartdate": admit.strftime("%Y-%m-%d"),
                            "icd_code": pcode,
                            "icd_version": pver,
                        }
                    )
                for n, (code, ver) in enumerate(codes, 1):
                    T["diagnoses_icd"].append(
                        {
                            "subject_id": sid,
                            "hadm_id": cur_hadm,
                            "seq_num": n,
                            "icd_code": code,
                            "icd_version": ver,
                        }
                    )
                if rng.random() < 0.5:
                    stay_id += 1
                    T["icustays"].append(
                        {
                            "subject_id": sid,
                            "hadm_id": cur_hadm,
                            "stay_id": stay_id,
                            "first_careunit": "MICU",
                            "last_careunit": "MICU",
                            "intime": _ts(admit),
                            "outtime": _ts(admit + pd.Timedelta(days=2)),
                            "los": 2.0,
                        }
                    )
                    for item, (_, base, eff) in CHART_ITEMS.items():
                        for k in range(2):
                            val = base + eff * positive + rng.normal(0, 3)
                            T["chartevents"].append(
                                {
                                    "subject_id": sid,
                                    "hadm_id": cur_hadm,
                                    "stay_id": stay_id,
                                    "caregiver_id": 1,
                                    "charttime": _ts(admit + pd.Timedelta(hours=1 + k)),
                                    "storetime": "",
                                    "itemid": item,
                                    "value": f"{val:.0f}",
                                    "valuenum": round(val, 1),
                                    "valueuom": "",
                                    "warning": 0,
                                }
                            )
                if hiv:
                    T["prescriptions"].append(
                        {
                            "subject_id": sid,
                            "hadm_id": cur_hadm,
                            "starttime": _ts(ed_in + pd.Timedelta(hours=1)),
                            "stoptime": _ts(disch),
                            "drug": "Emtricitabine-Tenofovir",
                            "route": "PO",
                        }
                    )
                T["prescriptions"].append(
                    {
                        "subject_id": sid,
                        "hadm_id": cur_hadm,
                        "starttime": _ts(ed_in + pd.Timedelta(hours=1)),
                        "stoptime": _ts(disch),
                        "drug": rng.choice(["Ceftriaxone", "Heparin", "Acetaminophen"]),
                        "route": "IV",
                    }
                )
                if tb or rng.random() < 0.2:
                    T["micro"].append(
                        {
                            "subject_id": sid,
                            "hadm_id": cur_hadm,
                            "chartdate": _ts(ed_in.normalize()),
                            "charttime": _ts(ed_in + pd.Timedelta(minutes=90)),
                            "spec_type_desc": "SPUTUM",
                            "test_name": "ACID FAST SMEAR" if tb else "GRAM STAIN",
                            "org_name": "",
                        }
                    )
            elif kind == "ed":
                for n, (code, ver) in enumerate(codes, 1):
                    T["ed_diagnosis"].append(
                        {
                            "subject_id": sid,
                            "stay_id": stay_id,
                            "seq_num": n,
                            "icd_code": code,
                            "icd_version": ver,
                            "icd_title": "",
                        }
                    )
            # labs before the study
            for item, (_, base, eff) in LAB_ITEMS.items():
                val = base + eff * positive + rng.normal(0, abs(eff) / 3 + 0.3)
                T["labevents"].append(
                    {
                        "labevent_id": lab_id,
                        "subject_id": sid,
                        "hadm_id": cur_hadm or "",
                        "specimen_id": lab_id,
                        "itemid": item,
                        "order_provider_id": "",
                        "charttime": _ts(ed_in + pd.Timedelta(minutes=45)),
                        "storetime": "",
                        "value": f"{val:.1f}",
                        "valuenum": round(val, 2),
                        "valueuom": "",
                        "ref_range_lower": "",
                        "ref_range_upper": "",
                        "flag": "",
                        "priority": "STAT",
                        "comments": "note, with comma",
                    }
                )
                lab_id += 1
            T["omr"].append(
                {
                    "subject_id": sid,
                    "chartdate": (t - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                    "seq_num": 1,
                    "result_name": "Blood Pressure",
                    "result_value": "128/82",
                }
            )
            T["omr"].append(
                {
                    "subject_id": sid,
                    "chartdate": (t - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                    "seq_num": 1,
                    "result_name": "BMI (kg/m2)",
                    "result_value": f"{rng.normal(26, 4):.1f}",
                }
            )
            # CXR study 2–3 h after ED arrival
            study_id += 1
            st = ed_in + pd.Timedelta(hours=2, minutes=int(rng.integers(0, 59)))
            for k, view in enumerate(["PA", "LATERAL"] if rng.random() < 0.5 else ["AP"]):
                dicom = f"{study_id:x}-{k:04d}-{i:04d}"
                T["metadata"].append(
                    {
                        "dicom_id": dicom,
                        "subject_id": sid,
                        "study_id": study_id,
                        "PerformedProcedureStepDescription": "CHEST (PA AND LAT)",
                        "ViewPosition": view,
                        "Rows": image_size,
                        "Columns": image_size,
                        "StudyDate": st.strftime("%Y%m%d"),
                        "StudyTime": f"{int(st.strftime('%H%M%S'))}.123",
                        "ProcedureCodeSequence_CodeMeaning": "",
                        "ViewCodeSequence_CodeMeaning": "",
                        "PatientOrientationCodeSequence_CodeMeaning": "",
                    }
                )
                T["split"].append(
                    {"dicom_id": dicom, "study_id": study_id, "subject_id": sid, "split": split}
                )
                img = rng.normal(90, 20, (image_size, image_size))
                if positive:
                    yy, xx = np.ogrid[:image_size, :image_size]
                    cy, cx = rng.integers(8, image_size // 2), rng.integers(8, image_size - 8)
                    img += 70 * np.exp(
                        -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (image_size / 10) ** 2)
                    )
                s = str(sid)
                p = roots.mimic_cxr_jpg / f"files/p{s[:2]}/p{s}/s{study_id}/{dicom}.jpg"
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(p, quality=90)
            chex = {c: "" for c in CHEXPERT}
            chex["Pneumonia"] = (
                1.0 if pneu else (rng.choice([-1.0, 0.0]) if rng.random() < 0.2 else "")
            )
            chex["Consolidation"] = 1.0 if positive and rng.random() < 0.6 else ""
            chex["No Finding"] = "" if positive else 1.0
            T["chexpert"].append({"subject_id": sid, "study_id": study_id, **chex})
            s = str(sid)
            rp = roots.mimic_cxr_reports / f"files/p{s[:2]}/p{s}/s{study_id}.txt"
            rp.parent.mkdir(parents=True, exist_ok=True)
            finding = "Right upper lobe consolidation." if positive else "Lungs are clear."
            rp.write_text(
                f"FINAL REPORT\n EXAMINATION: CHEST\n\n FINDINGS: {finding}\n\n IMPRESSION: "
                f"{'Findings concerning for infection.' if positive else 'No acute process.'}\n"
            )

    iv, ed, jpg = roots.mimic_iv, roots.mimic_iv_ed, roots.mimic_cxr_jpg
    _gz(pd.DataFrame(T["patients"]), iv / "hosp/patients.csv.gz")
    _gz(pd.DataFrame(T["admissions"]), iv / "hosp/admissions.csv.gz")
    _gz(pd.DataFrame(T["diagnoses_icd"]), iv / "hosp/diagnoses_icd.csv.gz")
    _gz(pd.DataFrame(T["labevents"]), iv / "hosp/labevents.csv.gz")
    _gz(
        pd.DataFrame(
            [
                {"itemid": k, "label": v[0], "fluid": "Blood", "category": "Chemistry"}
                for k, v in LAB_ITEMS.items()
            ]
        ),
        iv / "hosp/d_labitems.csv.gz",
    )
    _gz(pd.DataFrame(T["omr"]), iv / "hosp/omr.csv.gz")
    _gz(pd.DataFrame(T["prescriptions"]), iv / "hosp/prescriptions.csv.gz")
    _gz(
        pd.DataFrame(
            T["micro"],
            columns=[
                "subject_id",
                "hadm_id",
                "chartdate",
                "charttime",
                "spec_type_desc",
                "test_name",
                "org_name",
            ],
        ),
        iv / "hosp/microbiologyevents.csv.gz",
    )
    _gz(
        pd.DataFrame(
            T["transfers"],
            columns=[
                "subject_id",
                "hadm_id",
                "transfer_id",
                "eventtype",
                "careunit",
                "intime",
                "outtime",
            ],
        ),
        iv / "hosp/transfers.csv.gz",
    )
    _gz(
        pd.DataFrame(
            T["services"],
            columns=["subject_id", "hadm_id", "transfertime", "prev_service", "curr_service"],
        ),
        iv / "hosp/services.csv.gz",
    )
    _gz(
        pd.DataFrame(
            T["procedures_icd"],
            columns=["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"],
        ),
        iv / "hosp/procedures_icd.csv.gz",
    )
    _gz(pd.DataFrame(T["icustays"]), iv / "icu/icustays.csv.gz")
    _gz(pd.DataFrame(T["chartevents"]), iv / "icu/chartevents.csv.gz")
    _gz(
        pd.DataFrame(
            [
                {"itemid": k, "label": v[0], "category": "Routine Vital Signs"}
                for k, v in CHART_ITEMS.items()
            ]
        ),
        iv / "icu/d_items.csv.gz",
    )
    _gz(pd.DataFrame(T["edstays"]), ed / "ed/edstays.csv.gz")
    _gz(pd.DataFrame(T["triage"]), ed / "ed/triage.csv.gz")
    _gz(pd.DataFrame(T["vitalsign"]), ed / "ed/vitalsign.csv.gz")
    _gz(
        pd.DataFrame(
            T["ed_diagnosis"],
            columns=["subject_id", "stay_id", "seq_num", "icd_code", "icd_version", "icd_title"],
        ),
        ed / "ed/diagnosis.csv.gz",
    )
    _gz(pd.DataFrame(T["metadata"]), jpg / "mimic-cxr-2.0.0-metadata.csv.gz")
    _gz(pd.DataFrame(T["split"]), jpg / "mimic-cxr-2.0.0-split.csv.gz")
    _gz(pd.DataFrame(T["chexpert"]), jpg / "mimic-cxr-2.0.0-chexpert.csv.gz")
    # Reports also as the official zip (same internal layout: files/pXX/pY/sZ.txt).
    import zipfile

    with zipfile.ZipFile(roots.mimic_cxr_reports / "mimic-cxr-reports.zip", "w") as zf:
        for txt in sorted((roots.mimic_cxr_reports / "files").rglob("*.txt")):
            zf.write(txt, txt.relative_to(roots.mimic_cxr_reports).as_posix())
    return roots
