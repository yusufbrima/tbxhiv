"""Statistical model comparison: DeLong, McNemar, multiple-testing correction, cluster bootstrap."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score
from typer.testing import CliRunner

from tbmxai import ExperimentConfig, Pipeline
from tbmxai.cli import app
from tbmxai.core.types import Predictions
from tbmxai.evaluation.compare import ModelComparator, adjust_pvalues, delong_test, mcnemar_exact


def _scores(rng, y, signal):
    return 1 / (1 + np.exp(-(signal * (2 * y - 1) + rng.normal(0, 1, len(y)))))


def test_delong_matches_sklearn_and_bootstrap_se() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    a, b = _scores(rng, y, 1.2), _scores(rng, y, 0.6)
    res = delong_test(y, a, b)
    assert res["auc_a"] == pytest.approx(roc_auc_score(y, a))
    assert res["auc_b"] == pytest.approx(roc_auc_score(y, b))
    diffs = []
    for _ in range(600):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    assert res["se"] == pytest.approx(np.std(diffs), rel=0.2)
    assert res["p"] < 0.01
    assert delong_test(y, a, a)["p"] == 1.0


def test_delong_type_one_error() -> None:
    rng = np.random.default_rng(1)
    rejections = 0
    for _ in range(300):
        y = rng.integers(0, 2, 200)
        a, b = _scores(rng, y, 0.8), _scores(rng, y, 0.8)  # equally good, independent noise
        rejections += delong_test(y, a, b)["p"] < 0.05
    assert 0.02 <= rejections / 300 <= 0.09


def test_corrections_and_mcnemar() -> None:
    p = [0.01, 0.04, 0.03]
    assert adjust_pvalues(p, "holm") == pytest.approx([0.03, 0.06, 0.06])
    assert adjust_pvalues(p, "bh") == pytest.approx([0.03, 0.04, 0.04])
    assert np.isnan(adjust_pvalues([np.nan, 0.01], "holm")[0])
    a = np.array([1] * 10 + [0] * 2 + [1] * 30, bool)
    b = np.array([0] * 10 + [1] * 2 + [1] * 30, bool)
    res = mcnemar_exact(a, b)
    assert (res["a_only_correct"], res["b_only_correct"]) == (10, 2)
    assert res["p"] == pytest.approx(stats.binomtest(10, 12, 0.5).pvalue)


def _models(n=300, copies=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    groups = np.repeat(np.arange(n), copies)
    yy = np.repeat(y, copies)
    ids = np.array([f"p{g}|{k}" for g in range(n) for k in range(copies)])

    def mk(s):
        return Predictions(
            yy, np.repeat(_scores(rng, y, s), copies), sample_ids=ids, group_ids=groups
        )

    return {"good": mk(1.5), "medium": mk(0.8), "random": mk(0.0)}


def test_comparator_detects_differences_and_flags_clustering(tmp_path) -> None:
    res = ModelComparator(["auroc", "sensitivity", "brier"], n_bootstrap=400).compare(_models())
    pw = res.pairwise.set_index(["metric", "model_a", "model_b"])
    assert pw.loc[("auroc", "good", "random"), "significant"]
    assert pw.loc[("auroc", "good", "random"), "diff_lower"] > 0
    assert res.info["clustered"] and pw["clustered"].all()
    assert set(res.models["ci_unit"]) == {"patient"}
    assert np.isfinite(pw.loc[("auroc", "good", "random"), "p_delong_adj"])
    assert np.isfinite(pw.loc[("sensitivity", "good", "random"), "p_mcnemar"])
    res.write(tmp_path)
    for f in (
        "comparison_models.csv",
        "comparison_pairwise.csv",
        "comparison.md",
        "comparison_differences.png",
    ):
        assert (tmp_path / f).exists()


def test_cluster_bootstrap_is_wider_than_naive() -> None:
    models = _models(copies=5)
    clustered = ModelComparator(["auroc"], n_bootstrap=400).compare(models).pairwise
    naive = ModelComparator(["auroc"], n_bootstrap=400, cluster=False).compare(models).pairwise
    width = lambda t: (t["diff_upper"] - t["diff_lower"]).to_numpy()  # noqa: E731
    assert (width(clustered) > 1.5 * width(naive)).all()


def test_comparator_requires_same_samples() -> None:
    models = _models()
    models["random"] = models["random"].subset(np.arange(30, len(models["random"])))
    with pytest.raises(ValueError, match="different samples"):
        ModelComparator(["auroc"], n_bootstrap=50).compare(models)
    res = ModelComparator(["auroc"], n_bootstrap=50).compare(models, allow_subset=True)
    assert res.info["n_samples"] == len(models["random"])


def test_cli_compare_runs_and_score_cam(synthetic, tmp_path) -> None:
    base = {
        "name": "cmp",
        "seed": 3,
        "data": {
            "label_column": "tb_status",
            "class_names": ["non-TB", "TB"],
            "sources": [
                {
                    "name": "tabular_file",
                    "params": {"path": str(synthetic.ehr), "key_column": "pid"},
                },
                {
                    "name": "cxr_manifest",
                    "params": {
                        "path": str(synthetic.manifest),
                        "key_column": "pid",
                        "image_root": str(synthetic.images),
                    },
                },
            ],
        },
        "evaluation": {"n_bootstrap": 20},
    }
    runs = {
        "ehr": {
            **base,
            "view": {"modalities": ["tabular"]},
            "model": {"name": "xgboost", "params": {"n_estimators": 30}},
        },
        "cxr": {
            **base,
            "view": {"modalities": ["imaging"]},
            "model": {
                "name": "neural",
                "params": {
                    "encoders": {
                        "imaging": {
                            "name": "timm_vision",
                            "params": {"backbone": "resnet18", "pretrained": False},
                        }
                    },
                    "image": {"size": 32},
                },
            },
            "training": {"epochs": 1, "num_workers": 0, "device": "cpu"},
            "explain": {
                "methods": [
                    {"name": "score_cam", "params": {"max_channels": 16}},
                    {"name": "grad_cam"},
                ],
                "samples": {"n_per_class": 2},
            },
        },
    }
    dirs = []
    for name, spec in runs.items():
        result = Pipeline(ExperimentConfig.model_validate(spec)).run(tmp_path / name)
        dirs.append(str(tmp_path / name))
        if name == "cxr":
            sc = next(e for e in result.explanations if e.method == "score_cam")
            assert sc.attributions.shape[:2] == (4, 2) and np.isfinite(sc.attributions).all()
            assert set(sc.extra["y_true"]) == {0, 1}
            assert (tmp_path / name / "figures" / "saliency_score_cam.png").exists()
    out = tmp_path / "cmp"
    res = CliRunner().invoke(
        app,
        [
            "compare",
            *dirs,
            "-n",
            "ehr",
            "-n",
            "cxr",
            "-o",
            str(out),
            "--n-bootstrap",
            "200",
            "-m",
            "auroc",
            "-m",
            "sensitivity",
        ],
    )
    assert res.exit_code == 0, res.output
    pw = pd.read_csv(out / "comparison_pairwise.csv")
    assert set(pw["metric"]) == {"auroc", "sensitivity"} and len(pw) == 2
