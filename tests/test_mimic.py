"""MIMIC adapter on a fake MIMIC tree (real layout/schemas, no real data)."""

import numpy as np
import pandas as pd
import pytest

from tbmxai import ExperimentConfig, Pipeline
from tbmxai.datasets.mimic import MimicBuildConfig, MimicBuilder
from tbmxai.datasets.mimic.build import extract_sections
from tbmxai.datasets.mimic.fake import make_fake_mimic
from tbmxai.datasets.mimic.icd import chapter, match_prefixes


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("mimic")
    roots = make_fake_mimic(root / "physionet", n_patients=120, seed=3, image_size=32)
    cfg = MimicBuildConfig.model_validate(
        {
            "output_dir": str(root / "subset"),
            "sources": {
                "mimic_iv": str(roots.mimic_iv),
                "mimic_iv_ed": str(roots.mimic_iv_ed),
                "mimic_cxr_jpg": str(roots.mimic_cxr_jpg),
                "mimic_cxr_reports": str(roots.mimic_cxr_reports / "mimic-cxr-reports.zip"),
            },
            "selection": {"n_patients": 90, "seed": 1},
        }
    )
    builder = MimicBuilder(cfg)
    sel = builder.select()
    info = builder.build()
    return {"root": root, "roots": roots, "cfg": cfg, "select": sel, "build": info}


def test_select_is_patient_subset_with_enrichment(built) -> None:
    sel = built["select"]
    assert sel["n_patients"] == 90
    studies = pd.read_parquet(built["cfg"].selection_dir / "studies.parquet")
    assert (studies["encounter_type"] != "none").all()  # require_encounter
    # every TB-positive patient of the eligible population is included (enrichment)
    assert (
        sel["prevalence_subset"]["tb"]["n_positive_patients"]
        == sel["prevalence_source_population"]["tb"]["n_positive_patients"]
    )
    urls = (built["cfg"].selection_dir / "image_urls.txt").read_text().split()
    assert len(urls) == sel["n_images"] and all(
        u.startswith("https://physionet.org/files/mimic-cxr-jpg/") for u in urls
    )


def test_build_outputs(built) -> None:
    out = built["cfg"].output_dir
    features = pd.read_parquet(out / "tabular.parquet")
    labels = pd.read_parquet(out / "labels.parquet")
    manifest = pd.read_parquet(out / "images.parquet")
    reports = pd.read_parquet(out / "reports.parquet")
    assert features["study_id"].is_unique and len(features) == len(labels)
    assert manifest["present"].all()
    assert reports["report_text"].str.len().gt(0).all()
    groups = set(built["build"]["features_per_group"])
    assert {
        "demo",
        "enc",
        "hist",
        "lab",
        "triage",
        "edv",
        "omr",
        "med",
        "micro",
        "ward",
        "svc",
        "proc",
    } <= groups
    # extra groups: care unit / service at study time, procedures from previous admissions only
    assert set(features["ward_careunit"].dropna()) <= {
        "Emergency Department",
        "unknown",
        "Medicine",
        "Medical Intensive Care Unit (MICU)",
        "Surgery",
    }
    assert features["svc_current"].notna().any()
    assert (features["proc_n_prior"] >= 0).all()
    # leakage guard: a patient's first study has no *prior* admissions, so no procedures may count
    first = features.merge(labels[["study_id", "study_time"]], on="study_id").sort_values(
        "study_time"
    )
    first = first.drop_duplicates("patient_id")
    assert (first["proc_n_prior"] == 0).all()
    # a study taken in the ED is located in the ED, not in the later ward
    assert "Emergency Department" in set(features["ward_careunit"])
    # no label column and no context-only column leaks into the feature table
    assert not any(c.startswith("label_") for c in features.columns)
    assert "demo_anchor_year_group" not in features.columns
    # ICU chart events in the fake data happen after each CXR → excluded by the time rule
    assert not any(c.startswith("icu_") for c in features.columns)
    assert {
        "label_pneumonia",
        "label_tb",
        "label_hiv_pneumonia",
        "label_chexpert_pneumonia",
    } <= set(labels.columns)
    hp = labels.dropna(subset=["label_hiv_pneumonia"])
    assert (
        (hp["label_hiv_pneumonia"] == 1) <= ((hp["label_hiv"] == 1) & (hp["label_pneumonia"] == 1))
    ).all()


def test_build_writes_valid_standard_dataset(built) -> None:
    from tbmxai.data.dataset_format import DatasetSpec, validate_dataset

    out = built["cfg"].output_dir
    spec = DatasetSpec.load(out)
    assert spec.link_columns == ["study_id"] and spec.time_column == "study_time"
    assert {"label_pneumonia", "label_tb", "label_chexpert_pneumonia"} <= set(spec.labels)
    report = validate_dataset(out, check_images=30)
    assert report.ok, report.to_frame()


def _env(monkeypatch, built) -> None:
    monkeypatch.setenv("TBMXAI_MIMIC_OUT", str(built["cfg"].output_dir))
    monkeypatch.setenv("MIMIC_DIR", str(built["roots"].root))
    monkeypatch.setenv("TBMXAI_OUTPUT_ROOT", str(built["root"] / "runs"))


def test_repo_ehr_config_runs(built, monkeypatch, tmp_path) -> None:
    _env(monkeypatch, built)
    cfg = ExperimentConfig.from_yaml("configs/mimic/tabular_xgb.yaml", {"target": "label_tb"})
    cfg = cfg.model_copy(
        update={
            "evaluation": cfg.evaluation.model_copy(update={"n_bootstrap": 20}),
            "explain": cfg.explain.model_copy(update={"methods": []}),
            "subgroups": cfg.subgroups.model_copy(update={"min_size": 5}),
        }
    )
    result = Pipeline(cfg).run(tmp_path / "ehr")
    assert cfg.data.label_column == "label_tb"
    assert result.split_sizes["test"] > 0
    splits = pd.read_csv(tmp_path / "ehr" / "splits.csv")
    assert splits.groupby("patient_id")["split"].nunique().max() == 1


def test_icd_helpers() -> None:
    assert chapter("J189", 10) == "respiratory" and chapter("486", 9) == "respiratory"
    assert chapter("V08", 9) == "factors" and chapter("D509", 10) == "blood"
    mask = match_prefixes(
        pd.Series(["A150", "0110", "J189", "B20"]),
        pd.Series([10, 9, 10, 10]),
        ["010", "011"],
        ["A15"],
    )
    assert mask.tolist() == [True, True, False, False]


def test_extract_sections() -> None:
    text = "FINAL REPORT\n INDICATION: cough\n FINDINGS: Clear lungs.\n IMPRESSION: Normal.\n"
    assert extract_sections(text) == "Clear lungs. Normal."


def test_episode_label_windows(monkeypatch) -> None:
    """Episode labels: positive within ±window of any evidence event, <NA> outside it for
    patients with evidence, 0 for patients without evidence; works without an encounter."""
    from tbmxai.datasets.mimic import cohort
    from tbmxai.datasets.mimic.config import LabelDefinition

    t = pd.Timestamp
    events = pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "start": [t("2150-01-10"), t("2150-06-01"), t("2160-03-01")],
            "end": [t("2150-01-20"), t("2150-06-01"), t("2160-03-01")],
            "kind": ["icd_admission", "microbiology", "drug"],
        }
    )
    monkeypatch.setattr(cohort, "episode_events", lambda tables, d: events)
    studies = pd.DataFrame(
        {
            "study_id": [10, 11, 12, 13, 14, 15],
            "subject_id": [1, 1, 1, 2, 3, 2],
            "study_time": [
                t("2150-02-15"),  # 26 days after discharge → positive
                t("2150-05-05"),  # 27 days before the culture → positive
                t("2152-01-01"),  # same patient, years later → <NA>
                t("2160-02-01"),  # 29 days before first dose → positive
                t("2150-01-15"),  # patient without evidence → 0
                pd.NaT,  # unknown time → <NA>
            ],
            "encounter_type": ["none", "inpatient", "ed", "none", "inpatient", "none"],
        }
    )
    d = LabelDefinition(source="episode", drugs=["x"], window_days=30)
    got = cohort._episode_label(studies, None, d)
    assert got.tolist()[:5] == [1, 1, pd.NA, 1, 0] and pd.isna(got.iloc[5])
    neg = cohort._episode_label(studies, None, d.model_copy(update={"outside_window": "negative"}))
    assert neg.iloc[2] == 0
    with pytest.raises(ValueError, match="episode labels need"):
        LabelDefinition(source="episode")


def test_shrink_images(tmp_path) -> None:
    import numpy as np
    from PIL import Image

    from tbmxai.datasets.mimic.images import shrink_many, url_to_path

    root = tmp_path / "physionet"
    urls = [
        "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/p10/p1/s1/big.jpg",
        "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/p10/p1/s1/small.jpg",
        "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/p10/p1/s1/missing.jpg",
    ]
    rng = np.random.default_rng(0)
    for url, size in zip(urls[:2], [(2000, 1500), (800, 600)], strict=True):
        path = url_to_path(url, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, size[::-1], dtype=np.uint8)).save(path, quality=95)
    counts = shrink_many((url_to_path(u, root) for u in urls), max_side=1024, workers=1)
    assert counts == {"shrunk": 1, "kept": 1, "missing": 1}
    with Image.open(url_to_path(urls[0], root)) as im:
        assert im.size == (1024, 768) and im.mode == "L"
    assert url_to_path(urls[0], root).read_bytes()[-2:] == b"\xff\xd9"  # still "complete"
    assert shrink_many([url_to_path(urls[0], root)], max_side=1024, workers=1) == {"kept": 1}
    assert not list(root.rglob(".*.tmp"))


def test_dev_split_is_patient_level_and_stratified() -> None:
    from tbmxai.datasets.mimic.cohort import make_dev_split
    from tbmxai.datasets.mimic.config import DevSplitConfig

    rng = np.random.default_rng(0)
    n = 1000
    studies = pd.DataFrame(
        {
            "subject_id": np.repeat(np.arange(n), 3),
            "study_id": np.arange(3 * n),
            "label_pneumonia": rng.integers(0, 2, 3 * n),
            "label_hiv_pneumonia": np.repeat((np.arange(n) % 20 == 0).astype(int), 3),
        }
    )
    cfg = DevSplitConfig(val=0.1, test=0.2, stratify=["hiv_pneumonia", "pneumonia"])
    dev = make_dev_split(studies, cfg)
    assert dev["subject_id"].is_unique and len(dev) == n
    frac = dev["dev_split"].value_counts(normalize=True)
    assert frac["test"] == pytest.approx(0.2, abs=0.01) and frac["val"] == pytest.approx(
        0.1, abs=0.01
    )
    hiv = dev[dev["subject_id"] % 20 == 0]["dev_split"].value_counts()
    assert hiv["test"] == 10 and hiv["val"] == 5  # 50 HIV-pneumonia patients, stratified
    assert dev.equals(make_dev_split(studies, cfg))  # deterministic


def test_dev_split_inherits_earlier_assignments(tmp_path) -> None:
    from tbmxai.datasets.mimic.cohort import make_dev_split
    from tbmxai.datasets.mimic.config import DevSplitConfig

    rng = np.random.default_rng(1)

    def studies_for(n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "subject_id": np.repeat(np.arange(n), 2),
                "study_id": np.arange(2 * n),
                "label_pneumonia": rng.integers(0, 2, 2 * n),
            }
        )

    cfg = DevSplitConfig(val=0.1, test=0.2, stratify=["pneumonia"])
    small = make_dev_split(studies_for(300), cfg)
    small.to_csv(tmp_path / "dev_split.csv", index=False)
    big = make_dev_split(
        studies_for(1000), cfg.model_copy(update={"inherit_from": tmp_path / "dev_split.csv"})
    )
    assert big["subject_id"].is_unique and len(big) == 1000
    kept = big.merge(small, on="subject_id", suffixes=("", "_old"))
    assert len(kept) == 300 and (kept["dev_split"] == kept["dev_split_old"]).all()
    frac = big.loc[big["subject_id"] >= 300, "dev_split"].value_counts(normalize=True)
    assert frac["test"] == pytest.approx(0.2, abs=0.02) and frac["val"] == pytest.approx(
        0.1, abs=0.02
    )
