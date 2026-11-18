"""Shenzhen + Montgomery converter, external (site) split and repo configs on a fake layout."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tbmxai.data.dataset_format import DatasetSpec, validate_dataset
from tbmxai.datasets.nlm_tb import build_nlm_tb, parse_reading


@pytest.mark.parametrize(
    ("text", "sex", "age"),
    [
        ("male 35yrs\nnormal\n", "M", 35.0),
        ("Female   38yrs\nPTB in the left upper field", "F", 38.0),
        ("Male 7month\nnormal", "M", 0.58),
        ("femal 32yrs\nnormal\n", "F", 32.0),
        ("female24yrs\nnormal", "F", 24.0),
        ("male35yrs  \t\nPTB in the left upper field \t", "M", 35.0),
        ("male 42\nnormal", "M", 42.0),
        ("Patient's Sex: O \nPatient's Age: 005Y\nnormal\n", None, 5.0),
        ("Patient's Sex: M\nPatient's Age: 040Y\nnormal", "M", 40.0),
        ("Patient's Sex: F\nPatient's Age: 062Y\nleft upper lobe cavity", "F", 62.0),
        ("", None, None),
    ],
)
def test_parse_reading(text, sex, age) -> None:
    parsed = parse_reading(text)
    assert parsed["sex"] == sex and parsed["age"] == age
    assert "Sex" not in parsed["reading"] and "yrs" not in parsed["reading"]


def _fake_nlm(raw, n_shenzhen=40, n_montgomery=20, seed=0):
    rng = np.random.default_rng(seed)
    sets = [
        ("ChinaSet_AllFiles/ChinaSet_AllFiles", "CHNCXR", n_shenzhen),
        ("NLM-MontgomeryCXRSet/MontgomerySet", "MCUCXR", n_montgomery),
    ]
    for folder, prefix, n in sets:
        for i in range(1, n + 1):
            tb = int(i % 2 == 0)
            stem = f"{prefix}_{i:04d}_{tb}"
            img = rng.integers(0, 80, (48, 48)).astype(np.uint8)
            if tb:
                img[8:24, 8:24] += 120  # learnable "lesion"
            for sub in (
                "CXR_png",
                "ManualMask/leftMask",
            ):  # masks share file names: must be ignored
                (raw / folder / sub).mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(raw / folder / sub / f"{stem}.png")
            (raw / folder / "ClinicalReadings").mkdir(exist_ok=True)
            age, finding = 20 + i, "PTB in the right upper field" if tb else "normal"
            reading = (
                f"male {age}yrs\n{finding}\n"
                if prefix == "CHNCXR"
                else f"Patient's Sex: F\nPatient's Age: {age:03d}Y\n{finding}\n"
            )
            (raw / folder / "ClinicalReadings" / f"{stem}.txt").write_text(reading)


def test_build_and_validate(tmp_path) -> None:
    _fake_nlm(tmp_path / "raw")
    info = build_nlm_tb(tmp_path / "raw", tmp_path / "ds")
    assert info["n_images"] == 60 and info["unparsed_age"] == 0
    assert info["per_site"] == {
        "montgomery": {"normal": 10, "tb": 10},
        "shenzhen": {"normal": 20, "tb": 20},
    }
    labels = pd.read_csv(tmp_path / "ds" / "labels.csv")
    assert set(labels["sex"]) == {"M", "F"} and "reading" not in labels
    assert "reading" not in pd.read_csv(
        tmp_path / "ds" / "tabular.csv"
    )  # finding text never an input
    images = pd.read_csv(tmp_path / "ds" / "images.csv")
    assert images["image_path"].str.contains("/CXR_png/").all()
    spec = DatasetSpec.load(tmp_path / "ds")
    assert spec.patient_ids == "assumed" and spec.roles()["features"] == ["age", "sex_male"]
    report = validate_dataset(tmp_path / "ds", check_images=None)
    assert report.ok, report.to_frame()
