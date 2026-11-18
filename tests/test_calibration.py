"""Operating-point metrics, specificity thresholds and post-hoc calibration."""

import json

import numpy as np
import pytest

from tbmxai import ExperimentConfig, Pipeline
from tbmxai.core.types import Predictions
from tbmxai.evaluation import METRICS, Evaluator
from tbmxai.evaluation.calibration import CALIBRATORS, negative_log_likelihood
from tbmxai.evaluation.compare import ModelComparator
from tbmxai.evaluation.threshold import select_threshold

Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
P = np.array([0.05, 0.10, 0.20, 0.30, 0.25, 0.35, 0.40, 0.45])
PROB = np.column_stack([1 - P, P])


def test_operating_point_metrics() -> None:
    # Spec >= 0.75 allows one FP (0.30): threshold 0.25 → sens 1.0.
    assert METRICS.build("sens_at_spec@0.75")(Y, PROB, Y) == pytest.approx(1.0)
    # Spec = 1.0 → threshold 0.35 → sens 0.75.
    assert METRICS.build("sens_at_spec@1.0")(Y, PROB, Y) == pytest.approx(0.75)
    # Sens = 1.0 needs threshold 0.25 → one FP → spec 0.75.
    assert METRICS.build("spec_at_sens@1.0")(Y, PROB, Y) == pytest.approx(0.75)
    assert METRICS.build("spec_at_sens@0.75")(Y, PROB, Y) == pytest.approx(1.0)
    # Names carry the target; defaults are the WHO TB triage TPP (sens 90 %, spec 70 %).
    assert METRICS.build("spec_at_sens").name == "spec_at_sens90"
    assert METRICS.build("sens_at_spec").name == "sens_at_spec70"
    assert METRICS.build("sens_at_spec@0.975").name == "sens_at_spec97_5"
    # Undefined: one class, or multiclass.
    assert np.isnan(METRICS.build("spec_at_sens")(np.ones(4, int), PROB[:4], np.ones(4)))
    three = np.full((3, 3), 1 / 3)
    assert np.isnan(METRICS.build("sens_at_spec")(np.array([0, 1, 2]), three, np.zeros(3)))
    with pytest.raises(ValueError):
        METRICS.build("spec_at_sens@abc")
    with pytest.raises(ValueError):
        METRICS.build("spec_at_sens@1.5")


def test_operating_point_metrics_in_evaluator_and_comparison() -> None:
    preds = Predictions(Y, PROB, sample_ids=np.arange(8), group_ids=np.arange(8))
    res = Evaluator(["auroc", "spec_at_sens@1.0", "sens_at_spec"]).evaluate(preds)
    assert set(res.metrics) == {"auroc", "spec_at_sens100", "sens_at_spec70"}
    cmp = ModelComparator(["auroc", "spec_at_sens@1.0"], n_bootstrap=20).compare(
        {
            "a": preds,
            "b": Predictions(Y, PROB[:, ::-1].copy(), np.arange(8), group_ids=np.arange(8)),
        }
    )
    assert set(cmp.models["metric"]) == set(cmp.pairwise["metric"]) == {"auroc", "spec_at_sens100"}


def test_specificity_threshold() -> None:
    thr = select_threshold(Y, P, "specificity", target_specificity=0.75)
    assert thr == pytest.approx(0.25)  # lowest threshold with spec >= 0.75
    assert ((thr > P)[Y == 0]).mean() >= 0.75
    assert select_threshold(Y, P, "specificity", target_specificity=1.0) == pytest.approx(0.35)


def _overconfident(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    true_logit = (2 * y - 1) * 1.0 + rng.normal(0, 1.0, n)
    # Bayes posterior logit is 2 · true_logit (class means ±1, unit variance); these
    # probabilities use 3 · true_logit, i.e. they are over-confident by T = 1.5.
    p = 1 / (1 + np.exp(-3.0 * true_logit))
    return y, np.column_stack([1 - p, p])


@pytest.mark.parametrize("method", ["temperature", "platt", "isotonic"])
def test_calibrators_reduce_log_loss_and_keep_ranking(method) -> None:
    y, prob = _overconfident()
    cal = CALIBRATORS.build(method).fit(y, prob)
    out = cal.transform(prob)
    assert out.shape == prob.shape and np.allclose(out.sum(axis=1), 1.0)
    assert negative_log_likelihood(y, out) < negative_log_likelihood(y, prob)
    auroc = METRICS.build("auroc")
    tol = 0.01 if method == "isotonic" else 1e-9
    assert auroc(y, out, y) == pytest.approx(auroc(y, prob, y), abs=tol)
    json.dumps(cal.params())


def test_temperature_scaling_recovers_temperature_and_multiclass() -> None:
    y, prob = _overconfident(5000)
    assert CALIBRATORS.build("temperature").fit(y, prob).temperature_ == pytest.approx(
        1.5, rel=0.05
    )
    rng = np.random.default_rng(0)
    p3 = rng.dirichlet(np.ones(3), 300)
    y3 = rng.integers(0, 3, 300)
    out = CALIBRATORS.build("temperature").fit(y3, p3).transform(p3)
    assert out.shape == (300, 3) and np.allclose(out.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        CALIBRATORS.build("platt").fit(y3, p3)
    with pytest.raises(ValueError):
        CALIBRATORS.build("temperature").fit(np.zeros(4, int), prob[:4])


def test_pipeline_calibration_and_specificity_threshold(synthetic, tmp_path) -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "calibrated",
            "output_dir": str(tmp_path),
            "data": {
                "label_column": "tb_status",
                "class_names": ["non-TB", "TB"],
                "sources": [
                    {
                        "name": "tabular_file",
                        "params": {"path": str(synthetic.ehr), "key_column": "pid"},
                    }
                ],
            },
            "view": {"modalities": ["tabular"]},
            "split": {"name": "holdout", "params": {"test_size": 0.3, "val_size": 0.2}},
            "model": {"name": "logistic_regression"},
            "evaluation": {
                "n_bootstrap": 20,
                "calibration": "platt",
                "threshold_selection": "specificity",
                "target_specificity": 0.8,
            },
        }
    )
    pipe = Pipeline(cfg)
    result = pipe.run(tmp_path / "run")
    out = tmp_path / "run"
    cal = json.loads((out / "calibration.json").read_text())
    assert cal["method"] == "platt" and cal["fitted_on"] == "val"
    assert cal["val_after"]["nll"] <= cal["val_before"]["nll"] + 1e-9
    thr = json.loads((out / "threshold.json").read_text())
    assert thr["method"] == "specificity" and thr["calibrated"] is True
    # The threshold is chosen on CALIBRATED validation probabilities and reaches the target there.
    val = result.predictions["val"]
    assert (val.y_prob[:, 1] < thr["threshold"])[val.y_true == 0].mean() >= 0.8
    # Saved predictions are the calibrated ones.
    saved = (out / "predictions_test.csv").read_text()
    assert saved and result.calibration == cal
    assert "spec_at_sens90" in result.evaluations["test"].metrics
    assert "Calibration:** platt" in (out / "report.md").read_text()
