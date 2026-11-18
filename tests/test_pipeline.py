import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from tbmxai import ExperimentConfig, Pipeline, load_run
from tbmxai.cli import app
from tbmxai.models import NotFittedError


def _config(synthetic, tmp_path, model: str = "logistic_regression") -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "test_run",
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
            "view": {"modalities": ["tabular"], "metadata_columns": ["sex", "age"]},
            "split": {"name": "holdout", "params": {"test_size": 0.3}},
            "model": {"name": model},
            "evaluation": {"n_bootstrap": 50},
            "subgroups": {"columns": ["sex", "age"], "min_size": 5},
            "explain": {
                "methods": [{"name": "permutation_importance", "params": {"n_repeats": 2}}]
            },
        }
    )


@pytest.mark.parametrize("model", ["logistic_regression", "random_forest", "xgboost"])
def test_tabular_pipeline_end_to_end(synthetic, tmp_path, model) -> None:
    result = Pipeline(_config(synthetic, tmp_path, model)).run(tmp_path / "run")
    assert result.evaluations["test"]["auroc"].value > 0.6
    assert {"__overall__", "sex", "age"} <= set(result.subgroups["column"])
    assert result.explanations[0].attributions.index.name == "feature"
    run_dir = tmp_path / "run"
    for name in [
        "metrics.json",
        "report.md",
        "predictions_test.csv",
        "model.pkl",
        "splits.csv",
        "run_info.json",
        "manifest.json",
        "config.yaml",
        "tables/metrics.csv",
        "tables/subgroups.csv",
        "tables/cohort.csv",
        "tables/explanations.csv",
        "tables/schema.json",
    ]:
        assert (run_dir / name).exists(), name
    json.loads((run_dir / "metrics.json").read_text())
    run = load_run(run_dir)
    assert {"split", "metric", "value", "lower", "upper"} <= set(run["tables"]["metrics"].columns)
    splits = run["splits"]
    assert splits.groupby("patient_id")["split"].nunique().max() == 1  # patient-level


def test_unfitted_model_raises(synthetic, tmp_path) -> None:
    pipe = Pipeline(_config(synthetic, tmp_path))
    view = pipe.harmonize(pipe.ingest())
    with pytest.raises(NotFittedError):
        pipe.build_model().predict(view)


def test_config_yaml_roundtrip(synthetic, tmp_path) -> None:
    cfg = _config(synthetic, tmp_path)
    cfg.to_yaml(tmp_path / "c.yaml")
    assert ExperimentConfig.from_yaml(tmp_path / "c.yaml") == cfg


def test_cli_run_and_evaluate(synthetic, tmp_path) -> None:
    cfg = _config(synthetic, tmp_path)
    cfg.to_yaml(tmp_path / "c.yaml")
    runner = CliRunner()
    res = runner.invoke(app, ["run", str(tmp_path / "c.yaml"), "-o", str(tmp_path / "cli")])
    assert res.exit_code == 0, res.output
    preds = tmp_path / "cli" / "predictions_test.csv"
    assert "prob_TB" in pd.read_csv(preds).columns
    res = runner.invoke(
        app,
        [
            "evaluate",
            str(preds),
            "-m",
            "auroc",
            "-m",
            "ece",
            "--n-bootstrap",
            "50",
            "-g",
            "sex",
            "--min-size",
            "5",
        ],
    )
    assert res.exit_code == 0, res.output
    assert runner.invoke(app, ["components"]).exit_code == 0
