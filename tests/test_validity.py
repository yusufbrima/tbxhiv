"""Clinical-validity features: cluster bootstrap, temporal splits, index timepoints, missing modalities."""

import numpy as np
import pandas as pd
import pytest
import torch

from tbmxai import ExperimentConfig, Pipeline
from tbmxai.core.config import ComponentConfig
from tbmxai.core.types import Modality, Predictions
from tbmxai.data import Harmonizer, Ingestor
from tbmxai.data.harmonize import MODALITY_PATTERN_COLUMN, MultimodalView
from tbmxai.data.sources import CXRManifestSource, TabularFileSource
from tbmxai.data.splits import TemporalSplitter
from tbmxai.data.torch_datasets import MultimodalTorchDataset, build_image_transform
from tbmxai.evaluation import BootstrapCI, Evaluator
from tbmxai.models.base import Classifier
from tbmxai.models.nn import build_network

# ------------------------------------------------------------ cluster bootstrap


def test_cluster_bootstrap_wider_with_duplicated_samples() -> None:
    rng = np.random.default_rng(0)
    n, copies = 150, 6
    y = rng.integers(0, 2, n)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, n), 0, 1)
    # every patient contributes `copies` identical images
    preds = Predictions(
        np.repeat(y, copies), np.repeat(p, copies), group_ids=np.repeat(np.arange(n), copies)
    )
    ci = BootstrapCI(400, seed=1)
    clustered = Evaluator(["auroc"], bootstrap=ci).evaluate(preds)
    naive = Evaluator(["auroc"], bootstrap=BootstrapCI(400, seed=1, cluster=False)).evaluate(preds)
    single = Evaluator(["auroc"], bootstrap=ci).evaluate(Predictions(y, p, group_ids=np.arange(n)))
    width = lambda r: r["auroc"].upper - r["auroc"].lower  # noqa: E731
    assert clustered.ci_unit == "patient" and naive.ci_unit == "sample"
    assert width(clustered) > 1.8 * width(naive)  # naive CI is far too narrow
    assert width(clustered) == pytest.approx(width(single), rel=0.35)  # ≈ one-image-per-patient CI
    assert clustered.n_clusters == n


def test_cluster_indices_keep_patients_together() -> None:
    groups = np.array([0, 0, 1, 2, 2, 2])
    y = np.array([1, 1, 0, 0, 0, 0])
    for idx in BootstrapCI(20, seed=0).resample_indices(y, groups):
        counts = pd.Series(groups[idx]).value_counts()
        sizes = pd.Series(groups).value_counts()
        assert all(counts[g] % sizes[g] == 0 for g in counts.index)


# ------------------------------------------------------------- temporal split


def _toy_view(n_patients: int = 40, per: int = 3) -> MultimodalView:
    pid = np.repeat(np.arange(n_patients), per)
    first = np.repeat(np.arange(n_patients) % 10, per)  # first-visit year offset
    t = 2010 + first + np.tile(np.arange(per) * 2, n_patients)  # later visits 2 and 4 years on
    return MultimodalView(
        sample_ids=np.arange(len(pid)),
        patient_ids=pid,
        y=pid % 2,
        class_names=["0", "1"],
        tabular=pd.DataFrame({"x": np.arange(len(pid), dtype=float)}),
        metadata=pd.DataFrame({"year": t}),
    )


def test_temporal_split_future_test_and_strict() -> None:
    view = _toy_view()
    splits = TemporalSplitter("year", val_start=2016, test_start=2018, strict=True).split(view)
    splits.check_patient_disjoint()
    first = pd.Series(view.metadata["year"].to_numpy()).groupby(view.patient_ids).min()
    assert (first[np.unique(splits.test.patient_ids)] >= 2018).all()
    assert (
        splits.train.metadata["year"] < 2016
    ).all()  # strict: no training data from later periods
    assert (splits.val.metadata["year"] < 2018).all()
    loose = TemporalSplitter("year", val_start=2016, test_start=2018, strict=False).split(view)
    assert len(loose.train) > len(splits.train)


def test_temporal_split_fractions() -> None:
    splits = TemporalSplitter("year", test_size=0.25, val_size=0.25).split(_toy_view(), seed=0)
    assert splits.val is not None and len(np.unique(splits.test.patient_ids)) == 10


# ------------------------------------------------------- index + missingness


@pytest.fixture
def partial_cohort(synthetic, tmp_path):
    """EHR for every patient, CXRs for ~60% of them (left join → naturally missing images)."""
    manifest = pd.read_csv(synthetic.manifest)
    keep = manifest["pid"].drop_duplicates().sample(frac=0.6, random_state=0)
    path = tmp_path / "partial_manifest.csv"
    manifest[manifest["pid"].isin(keep)].to_csv(path, index=False)
    sources = [
        TabularFileSource(synthetic.ehr, key_column="pid"),
        CXRManifestSource(path, key_column="pid", image_root=synthetic.images),
    ]
    return Ingestor(sources, label_column="tb_status", join="left").run()


def test_missing_modalities_drop_vs_keep(partial_cohort) -> None:
    drop = Harmonizer(["imaging", "tabular"]).build(partial_cohort)
    keep = Harmonizer(["imaging", "tabular"], missing_modalities="keep").build(partial_cohort)
    assert drop.is_complete and not keep.is_complete
    assert len(keep) > len(drop)
    assert set(keep.metadata[MODALITY_PATTERN_COLUMN]) == {"imaging+tabular", "tabular"}
    assert (keep.image_paths[~keep.availability[Modality.IMAGING]] == "").all()
    withheld = keep.withhold(["tabular"])
    assert not withheld.availability[Modality.TABULAR].any()


def test_index_policy(partial_cohort) -> None:
    cohort = partial_cohort
    cohort.frame["t"] = range(len(cohort.frame))  # later row = later time
    full = Harmonizer(["imaging"], metadata_columns=["t"]).build(cohort)
    first = Harmonizer(
        ["imaging"], metadata_columns=["t"], index_policy="first", index_time_column="t"
    ).build(cohort)
    last = Harmonizer(
        ["imaging"], metadata_columns=["t"], index_policy="last", index_time_column="t"
    ).build(cohort)
    assert len(first) == len(np.unique(full.patient_ids)) == len(last)
    by_patient = full.metadata.assign(p=full.patient_ids).groupby("p")["t"]
    assert (first.metadata["t"].to_numpy() == by_patient.min()[first.patient_ids].to_numpy()).all()
    assert (last.metadata["t"].to_numpy() == by_patient.max()[last.patient_ids].to_numpy()).all()


def test_models_without_support_reject_missing(partial_cohort) -> None:
    class ImageOnly(Classifier):
        required_modalities = frozenset({Modality.IMAGING, Modality.TABULAR})

        def _fit(self, train, val):
            pass

        def _predict_proba(self, view):
            return np.full((len(view), 2), 0.5)

    keep = Harmonizer(["imaging", "tabular"], missing_modalities="keep").build(partial_cohort)
    with pytest.raises(ValueError, match="cannot handle missing"):
        ImageOnly().fit(keep)


@pytest.mark.parametrize(
    "fusion", ["concat", "gated", "bilinear", "attention", "cross_attention", "late"]
)
def test_absent_modality_input_is_ignored(fusion: str) -> None:
    torch.manual_seed(0)
    net = build_network(
        {
            "imaging": ComponentConfig(
                name="timm_vision", params={"backbone": "resnet18", "pretrained": False}
            ),
            "tabular": ComponentConfig(name="mlp_tabular", params={"hidden_dims": [8]}),
        },
        n_classes=2,
        fusion=ComponentConfig(
            name=fusion, params={"query": "tabular"} if fusion == "cross_attention" else {}
        ),
        encoder_overrides={"tabular": {"in_features": 4}, "imaging": {"in_chans": 1}},
    ).eval()
    tab = torch.randn(3, 4)
    present = {
        "imaging": torch.tensor([True, False, False]),
        "tabular": torch.tensor([True, True, True]),
    }
    a = net({"imaging": torch.randn(3, 1, 32, 32), "tabular": tab, "_present": present})
    b = net({"imaging": torch.randn(3, 1, 32, 32), "tabular": tab, "_present": present})
    assert torch.isfinite(a).all()
    assert torch.allclose(a[1:], b[1:], atol=1e-5)  # absent image content has no influence
    assert not torch.allclose(a[0], b[0])  # present image does


def test_modality_dropout_keeps_one(partial_cohort) -> None:
    view = Harmonizer(["imaging", "tabular"], missing_modalities="keep").build(partial_cohort)
    ds = MultimodalTorchDataset(
        view,
        [Modality.IMAGING, Modality.TABULAR],
        tabular=np.zeros((len(view), 3)),
        image_transform=build_image_transform(16),
        image_shape=(1, 16, 16),
        modality_dropout={"imaging": 1.0, "tabular": 1.0},
    )
    for i in range(20):
        present = ds[i]["inputs"]["_present"]
        assert sum(present.values()) == 1


def test_missing_modality_neural_pipeline_with_ablation(
    synthetic, partial_cohort, tmp_path
) -> None:
    manifest = tmp_path / "partial_manifest.csv"  # written by the partial_cohort fixture
    cfg = ExperimentConfig.model_validate(
        {
            "name": "missing_mm",
            "data": {
                "label_column": "tb_status",
                "class_names": ["non-TB", "TB"],
                "join": "left",
                "sources": [
                    {
                        "name": "tabular_file",
                        "params": {"path": str(synthetic.ehr), "key_column": "pid"},
                    },
                    {
                        "name": "cxr_manifest",
                        "params": {
                            "path": str(manifest),
                            "key_column": "pid",
                            "image_root": str(synthetic.images),
                        },
                    },
                ],
            },
            "view": {
                "modalities": ["imaging", "tabular"],
                "metadata_columns": ["sex"],
                "missing_modalities": "keep",
            },
            "model": {
                "name": "neural",
                "params": {
                    "encoders": {
                        "imaging": {
                            "name": "timm_vision",
                            "params": {"backbone": "resnet18", "pretrained": False},
                        },
                        "tabular": {"name": "mlp_tabular", "params": {"hidden_dims": [16]}},
                    },
                    "fusion": {"name": "gated", "params": {"dim": 16}},
                    "image": {"size": 32},
                    "modality_dropout": {"imaging": 0.3, "tabular": 0.3},
                },
            },
            "training": {"epochs": 2, "num_workers": 0, "device": "cpu"},
            "evaluation": {"n_bootstrap": 20, "modality_ablation": True},
            "subgroups": {"columns": ["modality_pattern"], "min_size": 5},
        }
    )
    result = Pipeline(cfg).run(tmp_path / "run")
    ab = result.modality_ablation
    assert set(ab["available"]) == {"imaging+tabular", "imaging", "tabular"}
    assert (tmp_path / "run" / "tables" / "modality_ablation.csv").exists()
    assert "tabular" in set(result.subgroups["group"])
    assert result.evaluations["test"].ci_unit == "patient"
