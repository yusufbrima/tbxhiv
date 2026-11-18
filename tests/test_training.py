import math

import pytest
import torch

from tbmxai import ExperimentConfig, Pipeline
from tbmxai.core.config import SchedulerConfig
from tbmxai.training import EarlyStopping, ModelCheckpoint, WarmupLRScheduler


def _opt(lr: float = 1.0) -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def test_warmup_then_cosine() -> None:
    opt = _opt()
    sched = WarmupLRScheduler(
        opt, SchedulerConfig(name="cosine", warmup_epochs=1, warmup_start_factor=0.1), 10, 3
    )
    assert sched.last_lr[0] == pytest.approx(0.1)
    for _ in range(10):
        sched.step_batch()
    assert sched.last_lr[0] == pytest.approx(1.0)  # warm-up done → peak
    for _ in range(20):
        sched.step_batch()
    assert sched.last_lr[0] == pytest.approx(0.0, abs=1e-9)  # cosine reaches floor


def test_step_and_plateau() -> None:
    opt = _opt()
    step = WarmupLRScheduler(opt, SchedulerConfig(name="step", step_size=2, gamma=0.5), 1, 10)
    for _ in range(4):
        step.step_epoch()
    assert step.last_lr[0] == pytest.approx(0.25)

    opt = _opt()
    plateau = WarmupLRScheduler(opt, SchedulerConfig(name="plateau", patience=1, gamma=0.1), 1, 10)
    for v in [1.0, 1.0, 1.0]:
        plateau.step_epoch(v)
    assert plateau.last_lr[0] == pytest.approx(0.1)


def test_early_stopping_modes() -> None:
    es = EarlyStopping("val_auroc", mode="auto", patience=2)
    assert es.mode == "max"
    assert not es.update({"val_auroc": 0.7})
    assert not es.update({"val_auroc": 0.6})
    assert es.update({"val_auroc": 0.65})
    assert EarlyStopping("val_loss").mode == "min"


def test_checkpoint_keeps_top_k(tmp_path) -> None:
    ckpt = ModelCheckpoint(tmp_path, "val_loss", save_top_k=2)
    for epoch, loss in enumerate([0.9, 0.5, 0.7, 0.3]):
        ckpt.update({"x": epoch}, {"val_loss": loss}, epoch)
    kept = sorted(p.name for p in tmp_path.glob("epoch*.pt"))
    assert len(kept) == 2 and ckpt.best_value == pytest.approx(0.3)
    assert (tmp_path / "last.pt").exists()


def test_neural_multimodal_pipeline(synthetic, tmp_path) -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "neural_smoke",
            "paths": {"data_root": str(synthetic.root)},
            "data": {
                "label_column": "tb_status",
                "class_names": ["non-TB", "TB"],
                "sources": [
                    {
                        "name": "tabular_file",
                        "params": {"path": "${data_root}/ehr.csv", "key_column": "pid"},
                    },
                    {
                        "name": "cxr_manifest",
                        "params": {
                            "path": "${data_root}/cxr_manifest.csv",
                            "key_column": "pid",
                            "image_root": "${data_root}/images",
                        },
                    },
                ],
            },
            "view": {"modalities": ["imaging", "tabular"], "metadata_columns": ["sex"]},
            "model": {
                "name": "neural",
                "params": {
                    "encoders": {
                        "imaging": {
                            "name": "timm_vision",
                            "params": {"backbone": "resnet18", "pretrained": False},
                        },
                        "tabular": {"name": "mlp_tabular", "params": {"hidden_dims": [32]}},
                    },
                    "fusion": {"name": "gated", "params": {"dim": 32}},
                    "image": {"size": 32},
                },
            },
            "training": {
                "epochs": 3,
                "batch_size": 32,
                "num_workers": 0,
                "device": "cpu",
                "scheduler": {"name": "cosine", "warmup_epochs": 0.5},
                "early_stopping": {"monitor": "val_auroc", "patience": 5},
                "optimizer": {"lr": 1e-3},
            },
            "evaluation": {"n_bootstrap": 20},
            "subgroups": {"columns": ["sex"], "min_size": 5},
        }
    )
    run_dir = tmp_path / "neural"
    result = Pipeline(cfg).run(run_dir)
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "tables" / "training_history.csv").exists()
    hist = result.training_history
    assert (
        hist is not None
        and len(hist) == 3
        and {"val_auroc", "lr", "train_loss"} <= set(hist.columns)
    )
    assert math.isfinite(result.evaluations["test"]["auroc"].value)
