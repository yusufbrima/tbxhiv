"""Supervised multi-label pretraining of a CXR backbone.

A timm backbone (ImageNet-initialised by default) plus one linear output per finding
is trained with a masked binary cross-entropy (masked = uncertain or unmentioned,
depending on the config). The best epoch by validation mean AUROC is exported as
``backbone.pt``: a state dict whose keys match the ``timm_vision`` encoder, so any
experiment can start from it with::

    encoders:
      imaging:
        name: timm_vision
        params: {backbone: densenet121, pretrained: false, checkpoint: …/backbone.pt}

The image pipeline (resize, augmentation, normalisation) is the one experiments use,
so pretraining and fine-tuning see identically prepared images. ``last.pt`` allows
exact resumption.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from tbmxai.core.utils import get_logger, seed_everything
from tbmxai.data.image_cache import ImageCache
from tbmxai.data.torch_datasets import build_image_transform, load_image
from tbmxai.pretrain.config import PretrainConfig
from tbmxai.pretrain.mimic import build_manifest

logger = get_logger(__name__)


class MultiLabelImages(Dataset[tuple[Tensor, Tensor]]):
    """Images + multi-label targets (NaN = masked)."""

    def __init__(
        self,
        paths: list[str],
        targets: np.ndarray,
        transform: Any,
        cache: ImageCache | None = None,
    ) -> None:
        self.paths = paths
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.transform = transform
        self.cache = cache

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        path = self.paths[i]
        img = self.cache.load(path) if self.cache is not None else load_image(path)
        return self.transform(img), self.targets[i]


class MultiLabelNet(nn.Module):
    """timm backbone → dropout → linear head (one logit per finding)."""

    def __init__(self, backbone: str, n_out: int, pretrained: bool, in_chans: int, drop: float):
        super().__init__()
        import timm

        self.model = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=in_chans
        )
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(int(self.model.num_features), n_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.drop(self.model(x)))


def masked_bce(logits: Tensor, targets: Tensor) -> Tensor:
    """Binary cross-entropy averaged over observed (non-NaN) labels."""
    mask = ~torch.isnan(targets)
    if not mask.any():
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(
        logits[mask], targets[mask], reduction="mean"
    )


def per_label_auroc(targets: np.ndarray, probs: np.ndarray, names: list[str]) -> dict[str, float]:
    """AUROC per finding over observed labels (NaN when a finding has one class only)."""
    from sklearn.metrics import roc_auc_score

    out = {}
    for j, name in enumerate(names):
        ok = ~np.isnan(targets[:, j])
        y = targets[ok, j]
        out[name] = (
            float(roc_auc_score(y, probs[ok, j])) if len(np.unique(y)) == 2 else float("nan")
        )
    return out


class Pretrainer:
    """Runs one pretraining job described by a :class:`PretrainConfig`."""

    def __init__(self, config: PretrainConfig) -> None:
        self.cfg = config
        self.out = Path(config.output_dir).expanduser() / config.name
        self.findings = list(config.data.findings)

    # ------------------------------------------------------------------ setup

    def _device(self) -> torch.device:
        from tbmxai.training.trainer import resolve_device

        return resolve_device(self.cfg.training.device)

    def _loaders(self, manifest: pd.DataFrame) -> tuple[DataLoader, DataLoader]:
        cfg = self.cfg
        img = cfg.image
        cache = ImageCache(img.cache_dir, img.cache_size) if img.cache_dir else None
        common = {"size": img.size, "in_chans": cfg.model.in_chans, "mean": img.mean}
        tf_train = build_image_transform(train=True, augment=img.augment, std=img.std, **common)
        tf_eval = build_image_transform(train=False, std=img.std, **common)
        loaders = []
        for split, tf, shuffle in (("train", tf_train, True), ("val", tf_eval, False)):
            part = manifest[manifest["split"] == split]
            ds = MultiLabelImages(
                part["image_path"].tolist(), part[self.findings].to_numpy(float), tf, cache
            )
            loaders.append(
                DataLoader(
                    ds,
                    batch_size=cfg.training.batch_size,
                    shuffle=shuffle,
                    num_workers=cfg.training.num_workers,
                    pin_memory=torch.cuda.is_available(),
                    drop_last=shuffle and len(ds) > cfg.training.batch_size,
                    persistent_workers=cfg.training.num_workers > 0,
                )
            )
        return loaders[0], loaders[1]

    def _autocast(self, device: torch.device) -> Any:
        p = self.cfg.training.precision
        enabled = p != "32" and device.type in {"cuda", "cpu"}
        dtype = torch.bfloat16 if p == "bf16-mixed" else torch.float16
        if device.type == "cpu":
            dtype = torch.bfloat16
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)

    # ------------------------------------------------------------------- run

    def run(self, resume: bool = False) -> dict[str, Any]:
        cfg, tc = self.cfg, self.cfg.training
        seed_everything(cfg.seed)
        self.out.mkdir(parents=True, exist_ok=True)
        cfg.to_yaml(self.out / "config.yaml")

        manifest = build_manifest(cfg.data)
        manifest.to_csv(self.out / "manifest.csv", index=False)
        counts = manifest.groupby("split").agg(
            images=("image_path", "size"), patients=("subject_id", "nunique")
        )
        prevalence = manifest[manifest["split"] == "train"][self.findings].mean().round(4)
        logger.info("Pretraining manifest: %s", counts.to_dict(orient="index"))
        train_loader, val_loader = self._loaders(manifest)

        device = self._device()
        model = MultiLabelNet(
            cfg.model.backbone,
            len(self.findings),
            cfg.model.pretrained,
            cfg.model.in_chans,
            cfg.model.drop_rate,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        steps_per_epoch = max(1, len(train_loader))
        total, warm = tc.epochs * steps_per_epoch, int(tc.warmup_epochs * steps_per_epoch)

        def lr_at(step: int) -> float:
            if step < warm:
                return (step + 1) / max(1, warm)
            t = (step - warm) / max(1, total - warm)
            return tc.min_lr_ratio + (1 - tc.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        scaler = torch.amp.GradScaler(
            device.type, enabled=tc.precision == "16-mixed" and device.type == "cuda"
        )

        start, best, best_epoch, stale, history = 0, -1.0, -1, 0, []
        last = self.out / "last.pt"
        if resume and last.exists():
            state = torch.load(last, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            start, best, best_epoch, stale = (
                state["epoch"] + 1,
                state["best"],
                state["best_epoch"],
                state["stale"],
            )
            history = state["history"]
            logger.info("Resumed from %s at epoch %d", last, start)

        for epoch in range(start, tc.epochs):
            t0 = time.time()
            model.train()
            losses = []
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with self._autocast(device):
                    loss = masked_bce(model(x).float(), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(float(loss.detach()))
            val_loss, aurocs = self.evaluate(model, val_loader, device)
            mean_auc = float(np.nanmean(list(aurocs.values())))
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "val_loss": val_loss,
                "val_mean_auroc": mean_auc,
                "lr": opt.param_groups[0]["lr"],
                "epoch_time_s": round(time.time() - t0, 1),
                **{f"val_auroc_{k}": v for k, v in aurocs.items()},
            }
            history.append(row)
            pd.DataFrame(history).to_csv(self.out / "history.csv", index=False)
            logger.info(
                "epoch %d/%d train_loss=%.4f val_loss=%.4f val_mean_auroc=%.4f (%.0fs)",
                epoch + 1,
                tc.epochs,
                row["train_loss"],
                val_loss,
                mean_auc,
                row["epoch_time_s"],
            )
            if mean_auc > best:
                best, best_epoch, stale = mean_auc, epoch, 0
                self._export(model)
            else:
                stale += 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best": best,
                    "best_epoch": best_epoch,
                    "stale": stale,
                    "history": history,
                },
                last,
            )
            if tc.patience is not None and stale >= tc.patience:
                logger.info("Early stopping: no improvement for %d epochs", stale)
                break

        best_row = next(r for r in history if r["epoch"] == best_epoch)
        summary = {
            "name": cfg.name,
            "backbone": cfg.model.backbone,
            "findings": self.findings,
            "split_counts": counts.to_dict(orient="index"),
            "train_prevalence": prevalence.to_dict(),
            "best_epoch": best_epoch,
            "best_val_mean_auroc": best,
            "best_val_auroc": {k: best_row[f"val_auroc_{k}"] for k in self.findings},
            "epochs_run": len(history),
            "backbone_checkpoint": str(self.out / "backbone.pt"),
            "image": cfg.image.model_dump(mode="json"),
        }
        (self.out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    @torch.no_grad()
    def evaluate(
        self, model: nn.Module, loader: DataLoader, device: torch.device
    ) -> tuple[float, dict[str, float]]:
        model.eval()
        probs, targets, losses = [], [], []
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with self._autocast(device):
                logits = model(x).float()
            losses.append(float(masked_bce(logits, y)))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.cpu().numpy())
        p, t = np.concatenate(probs), np.concatenate(targets)
        return float(np.mean(losses)), per_label_auroc(t, p, self.findings)

    def _export(self, model: MultiLabelNet) -> None:
        """Backbone weights with ``timm_vision`` encoder keys (``model.*``) + the head."""
        backbone = {f"model.{k}": v.cpu() for k, v in model.model.state_dict().items()}
        torch.save(backbone, self.out / "backbone.pt")
        torch.save({"findings": self.findings, **model.head.state_dict()}, self.out / "head.pt")
