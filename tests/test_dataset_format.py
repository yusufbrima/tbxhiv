"""Standard dataset format: one file per modality linked by patient_id [+ study_id]."""

import shutil

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from tbmxai import ExperimentConfig, Pipeline
from tbmxai.data import Harmonizer, Ingestor
from tbmxai.data.dataset_format import DatasetSpec, index_images, init_dataset, validate_dataset
from tbmxai.data.quality import DataQualityReport
from tbmxai.data.sources import ImageFolderSource


def _standard_from_synthetic(synthetic, root):
    """Convert the synthetic demo data into the standard layout."""
    root.mkdir(parents=True, exist_ok=True)
    ehr = pd.read_csv(synthetic.ehr).rename(columns={"pid": "patient_id"})
    ehr[["patient_id", "tb_status", "sex", "age", "hiv_status"]].to_csv(
        root / "labels.csv", index=False
    )
    ehr.drop(columns=["tb_status", "sex", "hiv_status"]).to_csv(root / "tabular.csv", index=False)
    man = pd.read_csv(synthetic.manifest).rename(
        columns={"pid": "patient_id", "path": "image_path"}
    )
    shutil.copytree(synthetic.images, root / "images", dirs_exist_ok=True)
    man["image_path"] = "images/" + man["image_path"]
    man.to_csv(root / "images.csv", index=False)
    pd.read_csv(synthetic.reports).rename(
        columns={"pid": "patient_id", "report": "report_text"}
    ).to_csv(root / "reports.csv", index=False)
    spec = {"name": "synthetic_std", "labels": {"tb_status": {"classes": ["non-TB", "TB"]}}}
    (root / "dataset.yaml").write_text(yaml.safe_dump(spec))
    return root


def test_valid_dataset_and_experiment(synthetic, tmp_path) -> None:
    root = _standard_from_synthetic(synthetic, tmp_path / "ds")
    report = validate_dataset(root, check_images=50)
    assert report.ok, report.to_frame()
    assert report.info["images_checked"] == 50 and report.info["patients"] == 200
    spec = DatasetSpec.load(root)
    roles = spec.roles()
    assert (
        "age" in roles["features"] and "age" not in roles["metadata"]
    )  # feature + subgroup, loaded once
    assert {"sex", "hiv_status"} <= set(roles["metadata"]) and roles["image_metadata"] == ["view"]
    cfg = ExperimentConfig.model_validate(
        {
            "name": "std",
            "data": {"dataset": str(root)},  # label resolved from dataset.yaml
            "view": {"modalities": ["tabular"], "metadata_columns": ["sex", "age"]},
            "model": {"name": "logistic_regression"},
            "evaluation": {"n_bootstrap": 20},
        }
    )
    pipe = Pipeline(cfg)
    cohort = pipe.ingest()
    assert cohort.label_column == "tb_status" and cohort.class_names == ["non-TB", "TB"]
    tab = pipe.harmonize(cohort).tabular
    assert "sex" not in tab and "tb_status" not in tab
    assert len(cohort) == len(pd.read_csv(root / "images.csv"))  # one sample per image
    tab_only = Ingestor.from_config(cfg.data.model_copy(update={"use": ["tabular"]})).run()
    assert len(tab_only) == 200  # one sample per patient
    result = pipe.run(tmp_path / "run")
    assert result.evaluations["test"]["auroc"].value > 0.6
    info = (tmp_path / "run" / "run_info.json").read_text()
    assert "dataset:labels" in info and "dataset:tabular" in info  # fingerprinted


def test_validator_catches_problems(synthetic, tmp_path) -> None:
    root = _standard_from_synthetic(synthetic, tmp_path / "bad")
    labels = pd.read_csv(root / "labels.csv")
    labels.loc[0, "tb_status"] = "maybe"  # unknown class
    pd.concat([labels, labels.head(2)]).to_csv(root / "labels.csv", index=False)  # duplicate keys
    tab = pd.read_csv(root / "tabular.csv")
    tab["leaky"] = (pd.read_csv(synthetic.ehr)["tb_status"] == "TB").astype(
        int
    )  # perfect predictor
    tab.to_csv(root / "tabular.csv", index=False)
    (root / pd.read_csv(root / "images.csv")["image_path"].iat[0]).unlink()  # missing image
    issues = validate_dataset(root, check_images=20).to_frame()
    checks = set(issues["check"])
    assert {"label values", "labels keys", "possible leakage", "image files"} <= checks
    assert (issues.query("check == 'label values'")["level"] == "error").all()

    tab["tb_status"] = 1  # a label inside the feature file is an error
    tab.to_csv(root / "tabular.csv", index=False)
    assert "label outside labels file" in set(validate_dataset(root).to_frame()["check"])


def test_links_must_not_multiply_rows(synthetic, tmp_path) -> None:
    root = _standard_from_synthetic(synthetic, tmp_path / "links")
    labels = pd.read_csv(root / "labels.csv")
    pd.concat([labels.assign(study_id="a"), labels.assign(study_id="b")]).to_csv(
        root / "labels.csv", index=False
    )  # labels per study; images per patient (several each) → ambiguous
    checks = validate_dataset(root, check_images=0).to_frame()
    assert "links" in set(checks.query("level == 'error'")["check"])


def _write_images(folder, names, value=90):
    for i, rel in enumerate(names):
        p = folder / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((16, 16), value + i, dtype=np.uint8)).save(p)


def test_image_folder_id_strategies(tmp_path) -> None:
    root = tmp_path / "imgs"
    _write_images(root, ["p001/s1/a.png", "p001/s2/b.png", "p002/s3/c.png"])
    by_path = ImageFolderSource(
        root, patient_id_from="path", id_pattern=r"p(?P<id>\d+)/", study_pattern=r"/s(?P<id>\d+)/"
    ).load(keep=["study_id"])
    assert sorted(by_path.frame["patient_id"]) == ["001", "001", "002"]
    assert sorted(by_path.frame["study_id"]) == ["1", "2", "3"]

    meta = tmp_path / "meta.csv"
    pd.DataFrame({"image": ["a", "b", "c"], "subject": ["X", "X", "Y"]}).to_csv(meta, index=False)
    written = index_images(
        root, tmp_path / "ds", patient_id_from="table", lookup=meta, lookup_patient_column="subject"
    )
    images = pd.read_csv(written["images"])
    assert list(images.columns) == ["patient_id", "image_path"]
    assert dict(zip(images["image_path"].str[-5:], images["patient_id"], strict=True)) == {
        "a.png": "X",
        "b.png": "X",
        "c.png": "Y",
    }
    with pytest.raises(FileExistsError):
        index_images(root, tmp_path / "ds", patient_id_from="none")


def test_dicom_ids(tmp_path) -> None:
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    folder = tmp_path / "dcm"
    folder.mkdir()
    for i, (pid, study) in enumerate([("A1", "1.2.3"), ("A1", "1.2.3"), ("B2", "1.2.4")]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.PatientID, ds.StudyInstanceUID = pid, study
        ds.SOPClassUID, ds.SOPInstanceUID = (
            SecondaryCaptureImageStorage,
            meta.MediaStorageSOPInstanceUID,
        )
        ds.Modality, ds.Rows, ds.Columns = "CR", 8, 8
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
        ds.PixelData = np.full((8, 8), 40 * i, np.uint8).tobytes()
        ds.save_as(folder / f"img{i}.dcm", enforce_file_format=True)
    images = pd.read_csv(index_images(folder, tmp_path / "ds", patient_id_from="dicom")["images"])
    assert sorted(images["patient_id"]) == ["A1", "A1", "B2"]
    assert sorted(images["study_id"].unique()) == ["1.2.3", "1.2.4"]


def test_folder_labels_and_assumed_ids_flagged(tmp_path) -> None:
    raw = tmp_path / "kaggle_download"
    _write_images(raw, [f"Normal/n{i}.png" for i in range(12)], 60)
    _write_images(raw, [f"Tuberculosis/t{i}.png" for i in range(12)], 160)
    root = tmp_path / "kaggle_ds"
    written = index_images(raw, root, patient_id_from="none", label_from_folder="label")
    assert set(written) == {"images", "labels", "dataset.yaml"}
    spec = DatasetSpec.load(root)
    assert spec.patient_ids == "assumed" and spec.labels["label"].classes == [
        "Normal",
        "Tuberculosis",
    ]
    report = validate_dataset(root)
    assert report.ok and "patient ids" in set(report.to_frame()["check"]), report.to_frame()
    assert report.info["label_counts[label]"] == {"Normal": 12, "Tuberculosis": 12}
    data = {"dataset": str(root)}
    cohort = Ingestor.from_config(
        ExperimentConfig.model_validate(
            {
                "name": "f",
                "data": data,
                "view": {"modalities": ["imaging"]},
                "model": {"name": "xgboost"},
            }
        ).data
    ).run()
    assert (
        len(cohort) == 24 and cohort.patient_ids_assumed and cohort.summary()["patient_ids_assumed"]
    )
    view = Harmonizer(["imaging"]).build(cohort)
    quality = DataQualityReport.from_view(view).summary.set_index("check")
    assert bool(quality.loc["patient_ids_assumed", "flag"])

    # an experiment on such data carries the warning into every report
    cfg = ExperimentConfig.model_validate(
        {
            "name": "assumed",
            "data": data,
            "view": {"modalities": ["imaging"]},
            "split": {"name": "holdout", "params": {"val_size": 0.2, "test_size": 0.3}},
            "model": {
                "name": "neural",
                "params": {
                    "encoders": {
                        "imaging": {
                            "name": "timm_vision",
                            "params": {"backbone": "resnet18", "pretrained": False},
                        }
                    },
                    "image": {"size": 16},
                },
            },
            "training": {"epochs": 1, "num_workers": 0, "device": "cpu"},
            "evaluation": {"n_bootstrap": 10},
        }
    )
    Pipeline(cfg).run(tmp_path / "run")
    for report_file in ("report.md", "report.html"):
        assert (
            "patient-level splitting cannot be guaranteed"
            in (tmp_path / "run" / report_file).read_text()
        )


def test_spec_rules_and_init(tmp_path) -> None:
    written = init_dataset(tmp_path / "new")
    assert {p.name for p in written} == {
        "dataset.yaml",
        "labels.csv",
        "tabular.csv",
        "images.csv",
        "reports.csv",
    }
    spec = DatasetSpec.load(tmp_path / "new")
    assert spec.modalities == ["tabular", "imaging", "text"] and spec.link_columns == []
    assert init_dataset(tmp_path / "new") == []  # never overwrites
    (tmp_path / "new" / "labels.parquet").write_bytes(b"")
    with pytest.raises(ValueError, match="keep one"):
        spec.files()
    report = validate_dataset(tmp_path / "new")
    assert not report.ok
