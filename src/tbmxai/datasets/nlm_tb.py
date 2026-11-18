"""Shenzhen + Montgomery (NLM) tuberculosis CXR sets → tbmxai standard dataset.

Sources (public, from the U.S. National Library of Medicine; Jaeger et al.,
Quant Imaging Med Surg 2014;4(6):475-477):

* ``ChinaSet_AllFiles.zip`` — Shenzhen No.3 People's Hospital: 662 frontal CXRs
  (336 TB, 326 normal), file names ``CHNCXR_0001_0.png``.
* ``NLM-MontgomeryCXRSet.zip`` — Montgomery County TB screening programme, USA:
  138 frontal CXRs (58 TB, 80 normal), file names ``MCUCXR_0001_0.png``.

The file-name suffix is the label (``_1`` = TB). Each image has a clinical
reading with sex, age and the radiologist's finding. Sex and age become model
inputs (``tabular.csv``) and subgroup metadata; the finding text is kept in
``readings.csv`` only, which is **not loaded**: it states the diagnosis and
would leak the label.

The sets provide no patient identifiers (one image id per file), so the dataset
is written with ``patient_ids: assumed`` and every report carries that warning.

Output (``out_dir``)::

    dataset.yaml   labels: {tb: [normal, tb]}, image_root = raw folder
    labels.csv     patient_id, tb, site, sex, age, age_group
    tabular.csv    patient_id, age, sex_male
    images.csv     patient_id, image_path
    readings.csv   patient_id, site, reading      (reference only; not an input)
    build_info.json
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

logger = get_logger(__name__)

SITES = {"CHNCXR": "shenzhen", "MCUCXR": "montgomery"}
ARCHIVES = {
    "montgomery": "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip",
    "shenzhen": "https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip",
}
ID_PATTERN = r"(?P<id>(?:CHN|MCU)CXR_\d+)"
AGE_BINS = [0, 18, 35, 50, 65, 200]
AGE_LABELS = ["<18", "18-34", "35-49", "50-64", "65+"]

_AGE_UNIT = {"y": 1.0, "m": 1 / 12, "d": 1 / 365, "w": 1 / 52}


def parse_reading(text: str) -> dict[str, Any]:
    """Sex, age (years) and finding text from a Shenzhen or Montgomery clinical reading.

    Shenzhen: ``"male 35yrs\\nPTB in the left upper field"``.
    Montgomery: ``"Patient's Sex: M\\nPatient's Age: 040Y\\nnormal"``.
    """
    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n") if ln.strip()]
    joined = " ".join(lines)
    # Shenzhen readings contain typos and missing spaces: "femal 32yrs", "female24yrs", "male 42"
    sex_word = r"(?<![a-z])(female|femal|male)(?![a-z])"
    sex = None
    if m := re.search(r"Sex:\s*([MF])\b", joined, re.IGNORECASE):
        sex = {"m": "M", "f": "F"}[m.group(1).lower()]
    elif m := re.search(sex_word, joined, re.IGNORECASE):
        sex = "F" if m.group(1).lower().startswith("femal") else "M"
    age = None
    m = (
        re.search(r"Age:\s*(\d+(?:\.\d+)?)\s*([YMDW])?", joined, re.IGNORECASE)
        or re.search(
            r"(\d+(?:\.\d+)?)\s*(yrs?|years?|months?|mos?|days?|weeks?|[ymdw])(?![a-z])",
            joined,
            re.IGNORECASE,
        )
        or re.search(sex_word + r"\s*(\d+(?:\.\d+)?)()", joined, re.IGNORECASE)  # "male 42"
    )
    if m:
        value, unit = m.group(m.lastindex - 1), m.group(m.lastindex) or "y"
        age = round(float(value) * _AGE_UNIT.get(unit[0].lower(), 1.0), 2)
    finding = [
        ln
        for ln in lines
        if not re.search(r"Sex:|Age:", ln, re.IGNORECASE)
        and not re.fullmatch(r"(?i)(female|femal|male)\s*\d*\s*\w*", ln)
    ]
    return {"sex": sex, "age": age, "reading": " ".join(finding)}


def extract_archives(raw_dir: str | Path) -> list[Path]:
    """Unzip any downloaded NLM archives in ``raw_dir`` (skips those already extracted)."""
    raw = Path(raw_dir).expanduser()
    done = []
    for name in (Path(u).name for u in ARCHIVES.values()):
        zf = raw / name
        marker = raw / f".{name}.extracted"
        if zf.exists() and not marker.exists():
            logger.info("Extracting %s", zf)
            with zipfile.ZipFile(zf) as z:
                z.extractall(raw / zf.stem)
            marker.touch()
            done.append(zf)
    return done


def build_nlm_tb(raw_dir: str | Path, out_dir: str | Path, extract: bool = True) -> dict[str, Any]:
    """Convert the extracted Shenzhen / Montgomery sets in ``raw_dir`` into a standard dataset.

    Either or both sets may be present. Images are found under any ``CXR_png``
    folder (lung masks elsewhere in the archives are ignored); readings under
    ``ClinicalReadings``.
    """
    from tbmxai.data.dataset_format import DatasetSpec, LabelSpec
    from tbmxai.data.sources import IMAGE_PATH_COLUMN, ImageFolderSource

    raw = Path(raw_dir).expanduser().resolve()
    out = Path(out_dir).expanduser()
    if extract:
        extract_archives(raw)
    folders = sorted({p.parent for p in raw.rglob("CXR_png/*.png")})
    if not folders:
        raise FileNotFoundError(
            f"No CXR_png/*.png under {raw}; download + extract the NLM archives first."
        )
    frames = [
        ImageFolderSource(
            f, patient_id_from="filename", id_pattern=ID_PATTERN, suffixes=(".png",)
        )._read()
        for f in folders
    ]
    images = pd.concat(frames, ignore_index=True).drop_duplicates("patient_id")
    images["file"] = images[IMAGE_PATH_COLUMN].map(lambda p: Path(p).name)
    images["site"] = images["patient_id"].str.split("_").str[0].map(SITES)
    images["tb"] = np.where(images["file"].str.contains(r"_1\.png$", case=False), "tb", "normal")

    readings = {p.stem: p.read_text(errors="replace") for p in raw.rglob("ClinicalReadings/*.txt")}
    parsed = pd.DataFrame(
        [
            {"stem": Path(f).stem, **parse_reading(readings.get(Path(f).stem, ""))}
            for f in images["file"]
        ]
    )
    df = pd.concat([images.reset_index(drop=True), parsed[["sex", "age", "reading"]]], axis=1)
    df["age_group"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    df.loc[df["age"].isna(), "age_group"] = "unknown"
    df["sex"] = df["sex"].fillna("unknown")
    df["image_path"] = df[IMAGE_PATH_COLUMN].map(
        lambda p: Path(p).resolve().relative_to(raw).as_posix()
    )
    df = df.sort_values("patient_id").reset_index(drop=True)

    out.mkdir(parents=True, exist_ok=True)
    df[["patient_id", "tb", "site", "sex", "age", "age_group"]].to_csv(
        out / "labels.csv", index=False
    )
    tab = df[["patient_id", "age"]].copy()
    tab["sex_male"] = df["sex"].map({"M": 1.0, "F": 0.0})
    tab.to_csv(out / "tabular.csv", index=False)
    df[["patient_id", "image_path"]].to_csv(out / "images.csv", index=False)
    df[["patient_id", "site", "reading"]].to_csv(out / "readings.csv", index=False)
    DatasetSpec(
        name="nlm_tb",
        description="Shenzhen (China) + Montgomery County (USA) TB chest X-ray sets, NLM "
        "(Jaeger et al. 2014). One image per id; no patient identifiers.",
        patient_ids="assumed",
        labels={
            "tb": LabelSpec(
                classes=["normal", "tb"], description="TB manifestation (file suffix _1)"
            )
        },
        image_root=str(raw),
    ).save(out)

    missing_readings = int(sum(Path(f).stem not in readings for f in df["file"]))
    info = {
        "created_utc": datetime.now(UTC).isoformat(),
        "raw_dir": str(raw),
        "n_images": len(df),
        "per_site": df.groupby(["site", "tb"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "missing_readings": missing_readings,
        "unparsed_sex": int((df["sex"] == "unknown").sum()),
        "unparsed_age": int(df["age"].isna().sum()),
    }
    (out / "build_info.json").write_text(json.dumps(info, indent=2))
    logger.info("NLM TB dataset: %s → %s", info["per_site"], out)
    return info
