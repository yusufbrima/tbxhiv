"""Supervised multi-label pretraining: manifest, training, export → timm_vision checkpoint."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from tbmxai.models.nn.encoders.vision import TimmVisionEncoder
from tbmxai.pretrain import PretrainConfig, Pretrainer, build_manifest
from tbmxai.pretrain.trainer import masked_bce

FINDINGS = ["Consolidation", "Pleural Effusion", "No Finding"]


def _fake_selection(tmp_path, n_patients=24):
    rng = np.random.default_rng(0)
    root, sel = tmp_path / "jpg", tmp_path / "selection"
    sel.mkdir()
    images, studies, chex = [], [], []
    for p in range(n_patients):
        subject = 10_000 + p
        split = "train" if p < 16 else ("validate" if p < 20 else "test")
        for s in range(2):
            study = subject * 10 + s
            path = f"files/p10/p{subject}/s{study}/img.jpg"
            arr = rng.integers(0, 80, (48, 40)).astype(np.uint8)
            cons = int(rng.random() < 0.5)
            if cons:
                arr[10:30, 10:30] += 150
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(root / path)
            images.append(
                {"subject_id": subject, "study_id": study, "official_split": split, "path": path}
            )
            studies.append(
                {"subject_id": subject, "study_id": study, "label_tb_episode": 1 if p == 0 else 0}
            )
            chex.append(
                {
                    "subject_id": subject,
                    "study_id": study,
                    "Consolidation": cons,
                    "Pleural Effusion": -1.0 if s else np.nan,
                    "No Finding": 1.0 - cons,
                }
            )
    pd.DataFrame(images).to_parquet(sel / "images.parquet")
    pd.DataFrame(
        {
            "subject_id": [10_000 + p for p in range(n_patients)],
            "dev_split": [
                "train" if p < 16 else "val" if p < 20 else "test" for p in range(n_patients)
            ],
        }
    ).to_csv(sel / "dev_split.csv", index=False)
    pd.DataFrame(studies).to_parquet(sel / "studies.parquet")
    pd.DataFrame(chex).to_csv(tmp_path / "chexpert.csv", index=False)
    return root, sel


def _config(tmp_path, root, sel, **data):
    return PretrainConfig.model_validate(
        {
            "name": "tiny",
            "output_dir": str(tmp_path / "out"),
            "data": {
                "selection_dir": str(sel),
                "image_root": str(root),
                "chexpert": str(tmp_path / "chexpert.csv"),
                "findings": FINDINGS,
                **data,
            },
            "model": {"backbone": "resnet18", "pretrained": False},
            "image": {"size": 32, "cache_dir": str(tmp_path / "cache"), "cache_size": 40},
            "training": {
                "epochs": 2,
                "batch_size": 8,
                "num_workers": 0,
                "precision": "32",
                "device": "cpu",
                "patience": None,
            },
        }
    )


def test_manifest_labels_splits_and_exclusion(tmp_path) -> None:
    root, sel = _fake_selection(tmp_path)
    m = build_manifest(_config(tmp_path, root, sel).data)
    assert 10_000 not in set(m["subject_id"])  # tb_episode patient excluded
    assert set(m.loc[m["split"] == "val", "subject_id"]) == set(range(10_016, 10_020))
    assert not set(m["subject_id"]) & set(range(10_020, 10_024))  # dev test never seen
    official = build_manifest(_config(tmp_path, root, sel, split="official").data)
    assert set(official.loc[official["split"] == "val", "subject_id"]) == set(range(10_016, 10_024))
    eff = m["Pleural Effusion"]
    assert eff.isna().sum() == (m["study_id"] % 10 == 1).sum()  # uncertain → masked
    assert (eff.dropna() == 0).all()  # blank → negative
    pos = build_manifest(_config(tmp_path, root, sel, uncertain="positive").data)
    assert pos["Pleural Effusion"].sum() == eff.isna().sum()
    with pytest.raises(KeyError):
        build_manifest(_config(tmp_path, root, sel, findings=["Nope"]).data)


def test_masked_bce_ignores_nan() -> None:
    logits = torch.tensor([[0.0, 5.0], [0.0, -5.0]])
    full = masked_bce(logits, torch.tensor([[1.0, float("nan")], [0.0, float("nan")]]))
    assert full == pytest.approx(float(torch.log(torch.tensor(2.0))), rel=1e-5)
    assert masked_bce(logits, torch.full((2, 2), float("nan"))) == 0.0


def test_pretrain_run_export_and_resume(tmp_path) -> None:
    root, sel = _fake_selection(tmp_path)
    cfg = _config(tmp_path, root, sel)
    summary = Pretrainer(cfg).run()
    out = tmp_path / "out" / "tiny"
    assert summary["epochs_run"] == 2 and 0 <= summary["best_epoch"] <= 1
    assert set(summary["best_val_auroc"]) == set(FINDINGS)
    assert len(pd.read_csv(out / "history.csv")) == 2
    assert json.loads((out / "metrics.json").read_text())["backbone"] == "resnet18"
    # backbone.pt loads into the experiment encoder with every key matched
    enc = TimmVisionEncoder("resnet18", pretrained=False, in_chans=1)
    missing, unexpected = enc.load_state_dict(torch.load(out / "backbone.pt"), strict=False)
    assert not missing and not unexpected
    TimmVisionEncoder("resnet18", pretrained=False, checkpoint=out / "backbone.pt")
    # resume continues after the last finished epoch
    more = cfg.model_copy(update={"training": cfg.training.model_copy(update={"epochs": 3})})
    assert Pretrainer(more).run(resume=True)["epochs_run"] == 3
