import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from tbmxai import ExperimentConfig, Pipeline
from tbmxai.cli import app
from tbmxai.tuning import set_by_path


def _cfg(
    synthetic, tuning: dict, model: dict, view=("tabular",), training=None
) -> ExperimentConfig:
    sources = [
        {"name": "tabular_file", "params": {"path": str(synthetic.ehr), "key_column": "pid"}}
    ]
    if "imaging" in view:
        sources.append(
            {
                "name": "cxr_manifest",
                "params": {
                    "path": str(synthetic.manifest),
                    "key_column": "pid",
                    "image_root": str(synthetic.images),
                },
            }
        )
    return ExperimentConfig.model_validate(
        {
            "name": "tune_test",
            "data": {
                "label_column": "tb_status",
                "class_names": ["non-TB", "TB"],
                "sources": sources,
            },
            "view": {"modalities": list(view), "metadata_columns": ["sex"]},
            "split": {"name": "holdout", "params": {"val_size": 0.2, "test_size": 0.2}},
            "model": model,
            "training": training or {},
            "evaluation": {"n_bootstrap": 20},
            "tuning": {"enabled": True, **tuning},
        }
    )


def test_set_by_path() -> None:
    tree: dict = {"model": {"params": {}}}
    set_by_path(tree, "model.params.encoders.imaging.params.backbone", "resnet18")
    assert tree["model"]["params"]["encoders"]["imaging"]["params"]["backbone"] == "resnet18"
    with pytest.raises(KeyError):
        set_by_path({"a": 1}, "a.b", 2)


def test_search_space_required() -> None:
    with pytest.raises(ValueError, match="search_space"):
        ExperimentConfig.model_validate(
            {
                "name": "x",
                "data": {"sources": [{"name": "tabular_file", "params": {"path": "x"}}]},
                "view": {"modalities": ["tabular"]},
                "model": {"name": "xgboost"},
                "tuning": {"enabled": True},
            }
        )


def test_tabular_tuning_refits_best(synthetic, tmp_path) -> None:
    cfg = _cfg(
        synthetic,
        {
            "n_trials": 4,
            "metric": "auroc",
            "sampler": "tpe",
            "search_space": {
                "model.params.max_depth": {"type": "int", "low": 2, "high": 5},
                "model.params.learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.3,
                    "log": True,
                },
            },
        },
        {"name": "xgboost", "params": {"n_estimators": 50}},
    )
    out = tmp_path / "run"
    result = Pipeline(cfg).run(out)
    best = json.loads((out / "tuning" / "best_params.json").read_text())
    assert best["direction"] == "maximize" and best["n_trials"] == 4
    assert result.config.model.params["max_depth"] == best["best_params"]["model.params.max_depth"]
    assert result.config.tuning.enabled is False  # refit config
    trials = pd.read_csv(out / "tables" / "tuning_trials.csv")
    assert len(trials) == 4 and "param:model.params.max_depth" in trials.columns
    assert (out / "config_best.yaml").exists() and (out / "tuning" / "optuna.db").exists()
    assert "Hyper-parameter tuning" in (out / "report.md").read_text()


def test_neural_tuning_with_pruning(synthetic, tmp_path) -> None:
    cfg = _cfg(
        synthetic,
        {
            "n_trials": 3,
            "metric": "auroc",
            "sampler": "random",
            "pruner": "median",
            "pruner_params": {"n_startup_trials": 1, "n_warmup_steps": 0},
            "search_space": {
                "training.optimizer.lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
                "model.params.fusion.name": {
                    "type": "categorical",
                    "choices": ["concat", "gated", "attention"],
                },
            },
        },
        {
            "name": "neural",
            "params": {
                "encoders": {
                    "imaging": {
                        "name": "timm_vision",
                        "params": {"backbone": "resnet18", "pretrained": False},
                    },
                    "tabular": {"name": "mlp_tabular", "params": {"hidden_dims": [16]}},
                },
                "fusion": {"name": "concat", "params": {"dim": 16}},
                "image": {"size": 32},
            },
        },
        view=("tabular", "imaging"),
        training={
            "epochs": 2,
            "batch_size": 32,
            "num_workers": 0,
            "device": "cpu",
            "early_stopping": {"monitor": "val_auroc"},
        },
    )
    result = Pipeline(cfg).run(tmp_path / "nn")
    assert result.tuning_best is not None
    assert result.config.model.params["fusion"]["name"] in {"concat", "gated", "attention"}
    assert not any(
        (tmp_path / "nn" / "tuning" / "trials").rglob("*.pt")
    )  # trial checkpoints cleaned
    assert (tmp_path / "nn" / "model.pkl").exists()


def test_cli_tune(synthetic, tmp_path) -> None:
    cfg = _cfg(
        synthetic,
        {
            "n_trials": 5,
            "search_space": {
                "model.params.C": {"type": "float", "low": 0.01, "high": 10, "log": True}
            },
        },
        {"name": "logistic_regression"},
    )
    cfg.to_yaml(tmp_path / "c.yaml")
    res = CliRunner().invoke(
        app, ["tune", str(tmp_path / "c.yaml"), "--n-trials", "2", "-o", str(tmp_path / "cli")]
    )
    assert res.exit_code == 0, res.output
    assert len(pd.read_csv(tmp_path / "cli" / "tuning" / "trials.csv")) == 2
