import numpy as np
import pytest

from tbmxai.core.types import Modality
from tbmxai.data import Harmonizer, Ingestor
from tbmxai.data.sources import (
    CXRDirectorySource,
    CXRManifestSource,
    ReportFileSource,
    TabularFileSource,
)
from tbmxai.data.splits import HoldoutSplitter


def _ehr(s):
    return TabularFileSource(s.ehr, key_column="pid")


def test_tabular_only_ingest(synthetic) -> None:
    cohort = Ingestor(
        [_ehr(synthetic)], label_column="tb_status", class_names=["non-TB", "TB"]
    ).run()
    assert cohort.modalities == {Modality.TABULAR}
    assert len(cohort) == 200
    assert "tb_status" not in cohort.modality_columns[Modality.TABULAR]
    assert set(cohort.frame["tb_status"].unique()) <= {0, 1}


def test_multimodal_ingest_and_harmonize(synthetic) -> None:
    sources = [
        _ehr(synthetic),
        CXRManifestSource(synthetic.manifest, key_column="pid", image_root=synthetic.images),
        ReportFileSource(synthetic.reports, key_column="pid"),
    ]
    cohort = Ingestor(sources, label_column="tb_status").run()
    assert cohort.modalities == set(Modality)
    assert len(cohort) > cohort.frame["patient_id"].nunique()  # several images per patient

    view = Harmonizer([Modality.IMAGING, Modality.TABULAR], metadata_columns=["sex"]).build(cohort)
    assert view.modalities == {Modality.IMAGING, Modality.TABULAR}
    assert view.text is None and view.image_paths is not None
    assert len(view) == len(cohort)


def test_cxr_directory_source(synthetic) -> None:
    src = CXRDirectorySource(synthetic.images, id_pattern=r"(?P<id>P\d+)_\d+", key_column="pid")
    table = src.load()
    assert table.frame["pid"].str.match(r"P\d+$").all()


def test_missing_modality_raises(synthetic) -> None:
    cohort = Ingestor([_ehr(synthetic)], label_column="tb_status").run()
    with pytest.raises(ValueError, match="no data for"):
        Harmonizer(["imaging"]).build(cohort)


def test_two_many_to_one_sources_rejected(synthetic) -> None:
    a = CXRManifestSource(synthetic.manifest, key_column="pid", image_root=synthetic.images)
    b = TabularFileSource(synthetic.manifest, key_column="pid", rename={"path": "p2", "view": "v2"})
    with pytest.raises(ValueError, match="Cartesian"):
        Ingestor([_ehr(synthetic), a, b], label_column="tb_status").run()


def test_holdout_is_patient_disjoint(synthetic) -> None:
    sources = [
        _ehr(synthetic),
        CXRManifestSource(synthetic.manifest, key_column="pid", image_root=synthetic.images),
    ]
    view = Harmonizer(["imaging"]).build(Ingestor(sources, label_column="tb_status").run())
    splits = HoldoutSplitter(0.2, 0.2).split(view, seed=3)
    assert splits.val is not None
    sets = [set(v.patient_ids) for _, v in splits.items()]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(len(v) for _, v in splits.items()) == len(view)
    assert np.isin([0, 1], splits.test.y).all()


def test_tabular_exclude_drops_feature_groups(synthetic) -> None:
    cohort = Ingestor([_ehr(synthetic)], label_column="tb_status").run()
    cols = cohort.modality_columns[Modality.TABULAR]
    first = cols[0]
    view = Harmonizer(["tabular"], tabular_exclude=[first[:3] + "*"]).build(cohort)
    kept = [c for c in cols if not c.startswith(first[:3])]
    assert view.tabular.shape[1] == len(kept) < len(cols)
    with pytest.raises(ValueError, match="removes every feature"):
        Harmonizer(["tabular"], tabular_exclude=["*"]).build(cohort)
