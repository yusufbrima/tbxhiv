import numpy as np
import pytest

from tbmxai import ExperimentConfig
from tbmxai.core.config import interpolate
from tbmxai.data import Harmonizer, Ingestor
from tbmxai.data.sources import CXRManifestSource, TabularFileSource
from tbmxai.data.splits import DataSplits, PatientKFoldSplitter, PatientLeakageError


@pytest.fixture
def view(synthetic):
    sources = [
        TabularFileSource(synthetic.ehr, key_column="pid"),
        CXRManifestSource(synthetic.manifest, key_column="pid", image_root=synthetic.images),
    ]
    return Harmonizer(["imaging"]).build(Ingestor(sources, label_column="tb_status").run())


def test_kfold_covers_all_patients_once(view) -> None:
    test_patients = []
    for fold in range(4):
        splits = PatientKFoldSplitter(n_splits=4, fold=fold).split(view, seed=0)
        splits.check_patient_disjoint()
        test_patients.extend(np.unique(splits.test.patient_ids))
    assert sorted(test_patients) == sorted(np.unique(view.patient_ids))


def test_leakage_detected(view) -> None:
    idx = np.arange(len(view))
    leaky = DataSplits(
        train=view.subset(idx[: len(idx) // 2 + 1]), test=view.subset(idx[len(idx) // 2 :])
    )
    if len(set(leaky.train.patient_ids) & set(leaky.test.patient_ids)) == 0:
        pytest.skip("no overlap produced")
    with pytest.raises(PatientLeakageError):
        leaky.check_patient_disjoint()


def test_interpolation(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TB_DATA", "/mnt/tb")
    assert interpolate("${TB_DATA}/ehr.csv", {}) == "/mnt/tb/ehr.csv"
    assert interpolate("${MISSING:-/x}/a", {}) == "/x/a"
    assert interpolate("${data_root}/a", {"data_root": "/d"}) == "/d/a"
    assert not interpolate("~/runs", {}).startswith("~")
    with pytest.raises(KeyError):
        interpolate("${NOPE}", {})

    (tmp_path / "c.yaml").write_text(
        "name: x\npaths: {data_root: '${TB_DATA}', output_root: '/out'}\noutput_dir: ${output_root}\n"
        "data: {sources: [{name: tabular_file, params: {path: '${data_root}/ehr.csv'}}]}\n"
        "view: {modalities: [tabular]}\nmodel: {name: xgboost}\n"
    )
    cfg = ExperimentConfig.from_yaml(tmp_path / "c.yaml", {"data_root": "/scratch"})
    assert cfg.data.sources[0].params["path"] == "/scratch/ehr.csv"
    assert str(cfg.output_dir) == "/out"
