import numpy as np
import pytest
from sklearn import metrics as skm

from tbmxai.core.types import Predictions
from tbmxai.evaluation import METRICS, BootstrapCI, Evaluator


@pytest.fixture
def binary() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p1 = np.clip(0.35 * y + rng.uniform(0, 0.65, 400), 0, 1)
    return y, np.column_stack([1 - p1, p1])


def _m(name: str, y, prob, pred) -> float:
    return METRICS.build(name)(y, prob, pred)


def test_binary_metrics_match_sklearn(binary) -> None:
    y, prob = binary
    pred = (prob[:, 1] >= 0.5).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y, pred).ravel()
    assert _m("accuracy", y, prob, pred) == pytest.approx(skm.accuracy_score(y, pred))
    assert _m("f1_macro", y, prob, pred) == pytest.approx(skm.f1_score(y, pred, average="macro"))
    assert _m("f1_micro", y, prob, pred) == pytest.approx(skm.f1_score(y, pred, average="micro"))
    assert _m("auroc", y, prob, pred) == pytest.approx(skm.roc_auc_score(y, prob[:, 1]))
    assert _m("sensitivity", y, prob, pred) == pytest.approx(tp / (tp + fn))
    assert _m("specificity", y, prob, pred) == pytest.approx(tn / (tn + fp))
    assert _m("ppv", y, prob, pred) == pytest.approx(tp / (tp + fp))
    assert _m("npv", y, prob, pred) == pytest.approx(tn / (tn + fn))
    assert _m("brier", y, prob, pred) == pytest.approx(skm.brier_score_loss(y, prob[:, 1]))
    assert _m("balanced_accuracy", y, prob, pred) == pytest.approx(
        skm.balanced_accuracy_score(y, pred)
    )


def test_multiclass_metrics_match_sklearn() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    logits = rng.normal(size=(300, 3)) + 1.5 * np.eye(3)[y]
    prob = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    pred = prob.argmax(1)
    assert _m("f1_macro", y, prob, pred) == pytest.approx(skm.f1_score(y, pred, average="macro"))
    assert _m("sensitivity", y, prob, pred) == pytest.approx(
        skm.recall_score(y, pred, average="macro")
    )
    assert _m("auroc", y, prob, pred) == pytest.approx(
        skm.roc_auc_score(y, prob, multi_class="ovr")
    )


def test_ece_perfect_and_overconfident() -> None:
    y = np.array([0, 1, 0, 1])
    perfect = np.eye(2)[y]
    assert _m("ece", y, perfect, y) == pytest.approx(0.0)
    wrong = np.eye(2)[1 - y]
    assert _m("ece", y, wrong, 1 - y) == pytest.approx(1.0)


def test_auroc_undefined_single_class() -> None:
    y = np.zeros(5, dtype=int)
    prob = np.tile([0.7, 0.3], (5, 1))
    assert np.isnan(_m("auroc", y, prob, np.zeros(5, dtype=int)))


def test_evaluator_ci_brackets_point(binary) -> None:
    y, prob = binary
    result = Evaluator(bootstrap=BootstrapCI(200, seed=0)).evaluate(Predictions(y, prob))
    for r in result.metrics.values():
        assert r.lower <= r.value <= r.upper, r
    assert result.n_samples == 400
    assert set(result.to_frame().columns) == {"metric", "value", "lower", "upper"}


def test_predictions_roundtrip(binary) -> None:
    y, prob = binary
    import pandas as pd

    meta = pd.DataFrame({"sex": ["F", "M"] * 200})
    preds = Predictions(
        y, prob, sample_ids=np.arange(400), class_names=["neg", "pos"], metadata=meta
    )
    back = Predictions.from_frame(preds.to_frame())
    np.testing.assert_allclose(back.y_prob, prob)
    assert back.class_names == ["neg", "pos"]
    assert back.metadata is not None and list(back.metadata.columns) == ["sex"]
