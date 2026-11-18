"""Configuration for supervised multi-label CXR pretraining (``tbmxai pretrain``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from tbmxai.core.config import InterpolatedConfig, _Strict

#: The 14 CheXpert findings labelled for every MIMIC-CXR study.
CHEXPERT_FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


class PretrainDataConfig(_Strict):
    """Which images and labels to pretrain on (MIMIC-CXR-JPG + CheXpert labels)."""

    selection_dir: Path = Field(description="`tbmxai mimic select` output (images.parquet).")
    image_root: Path = Field(description="Folder the selection's relative image paths start in.")
    chexpert: Path = Field(description="mimic-cxr-2.0.0-chexpert.csv(.gz).")
    findings: list[str] = Field(default_factory=lambda: list(CHEXPERT_FINDINGS), min_length=1)
    uncertain: Literal["ignore", "positive", "negative"] = Field(
        default="ignore", description="CheXpert -1: masked out of the loss, or U-ones / U-zeros."
    )
    blank: Literal["negative", "ignore"] = Field(
        default="negative", description="Finding not mentioned in the report."
    )
    exclude_labels: list[str] = Field(
        default_factory=lambda: ["tb_episode"],
        description="Drop every study of patients positive for these selection labels, so "
        "they stay unseen for later evaluation (e.g. MIMIC as an external TB site).",
    )
    split: Literal["dev", "official"] = Field(
        default="dev",
        description="dev: selection/dev_split.csv (`tbmxai mimic dev-split`) — train on "
        "train, early-stop on val, NEVER use test (kept unseen for downstream evaluation). "
        "official: MIMIC-CXR's split, `val_splits` for validation (tiny val/test).",
    )
    val_splits: list[str] = Field(
        default_factory=lambda: ["validate", "test"],
        description="split: official — MIMIC official splits used for validation.",
    )
    max_train_images: int | None = Field(default=None, ge=1, description="For quick runs.")


class PretrainModelConfig(_Strict):
    backbone: str = "densenet121"
    pretrained: bool = Field(default=True, description="Start from ImageNet weights.")
    drop_rate: float = Field(default=0.0, ge=0.0, lt=1.0)
    in_chans: int = 1


class PretrainImageConfig(_Strict):
    size: int = Field(default=320, ge=16)
    augment: bool = True
    mean: float = 0.5
    std: float = 0.25
    cache_dir: Path | None = Field(default=None, description="Decode-once image cache.")
    cache_size: int = Field(default=512, ge=16)


class PretrainTrainingConfig(_Strict):
    epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=4, ge=0)
    lr: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=1e-2, ge=0)
    warmup_epochs: float = Field(default=1.0, ge=0)
    min_lr_ratio: float = Field(default=0.01, ge=0, le=1)
    precision: Literal["32", "bf16-mixed", "16-mixed"] = "bf16-mixed"
    patience: int | None = Field(default=5, ge=1, description="Early stop on val mean AUROC.")
    device: str = "auto"


class PretrainConfig(InterpolatedConfig):
    """Top-level ``tbmxai pretrain`` config."""

    name: str
    seed: int = 42
    output_dir: Path
    data: PretrainDataConfig
    model: PretrainModelConfig = Field(default_factory=PretrainModelConfig)
    image: PretrainImageConfig = Field(default_factory=PretrainImageConfig)
    training: PretrainTrainingConfig = Field(default_factory=PretrainTrainingConfig)
