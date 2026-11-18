"""Pretraining manifest: one row per frontal MIMIC-CXR image with its study's CheXpert labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tbmxai.core.utils import get_logger
from tbmxai.pretrain.config import PretrainDataConfig

logger = get_logger(__name__)


def build_manifest(cfg: PretrainDataConfig) -> pd.DataFrame:
    """Images of a ``tbmxai mimic select`` output with multi-label targets.

    Returns columns ``image_path`` (absolute), ``subject_id``, ``study_id``, ``split``
    (``train`` / ``val``; dev-split ``test`` patients are dropped) and one float column
    per finding: 1.0, 0.0, or NaN (masked out of the loss).
    """
    sel = cfg.selection_dir.expanduser()
    images = pd.read_parquet(sel / "images.parquet")
    studies = pd.read_parquet(sel / "studies.parquet")

    excluded: set[int] = set()
    for name in cfg.exclude_labels:
        col = f"label_{name}"
        if col not in studies:
            raise KeyError(f"exclude_labels: '{name}' not in the selection ({col} missing)")
        excluded |= set(studies.loc[studies[col].fillna(0).eq(1), "subject_id"].astype(int))
    if excluded:
        logger.info("Excluding %d patients positive for %s", len(excluded), cfg.exclude_labels)
    images = images[~images["subject_id"].astype(int).isin(excluded)]

    chex = pd.read_csv(cfg.chexpert.expanduser())
    missing = [f for f in cfg.findings if f not in chex]
    if missing:
        raise KeyError(f"findings not in the CheXpert table: {missing}")
    chex = chex[["study_id", *cfg.findings]].drop_duplicates("study_id")
    df = images.merge(chex, on="study_id", how="left")

    for f in cfg.findings:
        raw = pd.to_numeric(df[f], errors="coerce")
        values = raw.copy()
        values[raw == -1] = {"ignore": np.nan, "positive": 1.0, "negative": 0.0}[cfg.uncertain]
        values[raw.isna()] = 0.0 if cfg.blank == "negative" else np.nan
        df[f] = values.astype(float)

    root = cfg.image_root.expanduser()
    df["image_path"] = df["path"].map(lambda p: str(root / p))
    if cfg.split == "dev":
        path = sel / "dev_split.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `tbmxai mimic dev-split CONFIG` first")
        dev = pd.read_csv(path).set_index("subject_id")["dev_split"]
        df["split"] = df["subject_id"].map(dev)
        if df["split"].isna().any():
            raise ValueError("dev_split.csv does not cover every selected patient")
        n_test = int(df["split"].eq("test").sum())
        df = df[df["split"] != "test"]  # never seen by pretraining
        logger.info("Dev split: %d test images held out of pretraining", n_test)
    else:
        df["split"] = np.where(df["official_split"].isin(cfg.val_splits), "val", "train")
    out = df[["image_path", "subject_id", "study_id", "split", *cfg.findings]].reset_index(
        drop=True
    )
    overlap = set(out.loc[out["split"] == "train", "subject_id"]) & set(
        out.loc[out["split"] == "val", "subject_id"]
    )
    if overlap:
        raise ValueError(f"{len(overlap)} patients are in both train and val splits")
    if cfg.max_train_images:
        train = out[out["split"] == "train"]
        keep = train.sample(min(len(train), cfg.max_train_images), random_state=0).index
        out = out[(out["split"] == "val") | out.index.isin(keep)].reset_index(drop=True)
    return out
