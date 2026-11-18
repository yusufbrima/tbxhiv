"""Configurable PyTorch training loop.

Features (all driven by :class:`~tbmxai.core.config.TrainingConfig`):
optimiser choice with encoder LR multiplier, linear warm-up + cosine / linear /
step / plateau / constant schedules, early stopping, top-k + last
checkpointing with exact resume, mixed precision (fp16 / bf16), gradient
accumulation, gradient clipping, class-weighted loss and label smoothing.

Batches are dicts ``{"inputs": {modality: tensor-or-dict}, "y": LongTensor}``.
Every epoch is logged (``history``) and written to ``history.csv`` so curves
can be plotted with any tool.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from tbmxai.core.config import TrainingConfig
from tbmxai.core.utils import get_logger
from tbmxai.evaluation.metrics import METRICS
from tbmxai.training.callbacks import EarlyStopping, ModelCheckpoint, resolve_mode
from tbmxai.training.schedule import WarmupLRScheduler, build_optimizer

logger = get_logger(__name__)


def resolve_device(spec: str) -> torch.device:
    """``auto`` → CUDA if available, else MPS, else CPU."""
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in nested dicts/lists to ``device``."""
    if isinstance(obj, Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, Mapping):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


@dataclass
class TrainingResult:
    """Outcome of :meth:`TorchTrainer.fit`."""

    history: pd.DataFrame
    best_epoch: int | None
    best_value: float | None
    best_checkpoint: Path | None
    stopped_early: bool
    extra: dict[str, Any] = field(default_factory=dict)


class TorchTrainer:
    """Train / validate / predict an ``nn.Module`` mapping ``inputs`` → logits.

    Args:
        config: Training configuration.
        output_dir: Where checkpoints and ``history.csv`` go (anywhere on disk).
    """

    def __init__(self, config: TrainingConfig, output_dir: str | Path) -> None:
        self.config = config
        self.output_dir = Path(output_dir).expanduser()
        self.device = resolve_device(config.device)
        if config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.amp_dtype = {"16-mixed": torch.float16, "bf16-mixed": torch.bfloat16}.get(
            config.precision
        )

    # ----------------------------------------------------------------- fit

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        class_weights: Tensor | None = None,
        epoch_callback: Callable[[int, dict[str, float]], None] | None = None,
    ) -> TrainingResult:
        """Train ``model`` in place (best weights restored if configured).

        Args:
            model: Network mapping ``inputs`` → logits.
            train_loader: Training batches.
            val_loader: Validation batches (enables val metrics / early stopping).
            class_weights: Optional per-class loss weights.
            epoch_callback: Called as ``callback(epoch, logs)`` after every epoch;
                may raise (e.g. ``optuna.TrialPruned``) to abort training.
        """
        cfg = self.config
        model.to(self.device)
        loss_fn = nn.CrossEntropyLoss(
            weight=None if class_weights is None else class_weights.to(self.device),
            label_smoothing=cfg.label_smoothing,
        )
        monitor = cfg.early_stopping.monitor if val_loader is not None else "train_loss"
        ckpt_monitor = (
            (cfg.checkpoint.monitor or monitor) if val_loader is not None else "train_loss"
        )

        optimizer = build_optimizer(model, cfg.optimizer)
        steps_per_epoch = max(1, len(train_loader) // cfg.accumulate_grad_batches)
        scheduler = WarmupLRScheduler(
            optimizer,
            cfg.scheduler,
            steps_per_epoch,
            cfg.epochs,
            monitor_mode=resolve_mode(monitor, cfg.early_stopping.mode),
        )
        stopper = (
            EarlyStopping(
                monitor,
                cfg.early_stopping.mode,
                cfg.early_stopping.patience,
                cfg.early_stopping.min_delta,
            )
            if cfg.early_stopping.enabled
            else None
        )
        checkpointer = (
            ModelCheckpoint(
                self.output_dir / "checkpoints",
                ckpt_monitor,
                cfg.checkpoint.mode,
                cfg.checkpoint.save_top_k,
                cfg.checkpoint.save_last,
            )
            if cfg.checkpoint.enabled
            else None
        )
        scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.amp_dtype == torch.float16 and self.device.type == "cuda"
        )

        history: list[dict[str, float]] = []
        start_epoch = 0
        if cfg.resume_from is not None:
            start_epoch, history = self._resume(
                cfg.resume_from, model, optimizer, scheduler, stopper
            )

        stopped_early = False
        for epoch in range(start_epoch, cfg.epochs):
            t0 = time.perf_counter()
            train_loss = self._train_epoch(
                model, train_loader, loss_fn, optimizer, scheduler, scaler
            )
            logs: dict[str, float] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": scheduler.last_lr[-1],
            }
            if val_loader is not None:
                logs.update(self._validate(model, val_loader, loss_fn))
            logs["epoch_time_s"] = time.perf_counter() - t0
            history.append(logs)
            logger.info(
                "epoch %d/%d %s",
                epoch + 1,
                cfg.epochs,
                " ".join(f"{k}={v:.4g}" for k, v in logs.items() if k != "epoch"),
            )
            if epoch_callback is not None:
                epoch_callback(epoch, logs)
            scheduler.step_epoch(logs.get(monitor))

            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "early_stopping": stopper.state_dict() if stopper else None,
                "history": history,
            }
            if checkpointer is not None:
                checkpointer.update(state, logs, epoch)
            if stopper is not None and stopper.update(logs):
                logger.info(
                    "Early stopping at epoch %d (best %s=%.4g)", epoch + 1, monitor, stopper.best
                )
                stopped_early = True
                break

        best_path = checkpointer.best_path if checkpointer else None
        if best_path is not None and cfg.checkpoint.restore_best:
            model.load_state_dict(
                torch.load(best_path, map_location=self.device, weights_only=False)["model"]
            )
            logger.info("Restored best weights from %s", best_path.name)

        frame = pd.DataFrame(history)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.output_dir / "history.csv", index=False)
        best_epoch = None
        if best_path is not None and not frame.empty and ckpt_monitor in frame:
            col = frame[ckpt_monitor]
            best_epoch = int(
                frame.loc[col.idxmin() if checkpointer.mode == "min" else col.idxmax(), "epoch"]
            )  # type: ignore[union-attr]
        return TrainingResult(
            history=frame,
            best_epoch=best_epoch,
            best_value=checkpointer.best_value if checkpointer else None,
            best_checkpoint=best_path,
            stopped_early=stopped_early,
        )

    # ------------------------------------------------------------ predict

    @torch.no_grad()
    def predict_logits(self, model: nn.Module, loader: DataLoader) -> np.ndarray:
        """Logits for every batch in ``loader`` (order preserved)."""
        model.to(self.device).eval()
        outputs = []
        for batch in loader:
            with self._autocast():
                logits = model(to_device(batch["inputs"], self.device))
            outputs.append(logits.float().cpu())
        return torch.cat(outputs).numpy()

    # ---------------------------------------------------------- internals

    def _autocast(self) -> torch.autocast:
        return torch.autocast(
            self.device.type,
            dtype=self.amp_dtype or torch.float32,
            enabled=self.amp_dtype is not None,
        )

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: WarmupLRScheduler,
        scaler: torch.amp.GradScaler,
    ) -> float:
        cfg = self.config
        model.train()
        total, count = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        n_batches = len(loader)
        for i, batch in enumerate(loader):
            y = batch["y"].to(self.device)
            with self._autocast():
                loss = loss_fn(model(to_device(batch["inputs"], self.device)), y)
            scaler.scale(loss / cfg.accumulate_grad_batches).backward()
            total += float(loss.detach()) * len(y)
            count += len(y)
            if (i + 1) % cfg.accumulate_grad_batches == 0 or i + 1 == n_batches:
                if cfg.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step_batch()
        return total / max(count, 1)

    @torch.no_grad()
    def _validate(
        self, model: nn.Module, loader: DataLoader, loss_fn: Callable[[Tensor, Tensor], Tensor]
    ) -> dict[str, float]:
        model.eval()
        logits, ys = [], []
        for batch in loader:
            with self._autocast():
                logits.append(model(to_device(batch["inputs"], self.device)).float())
            ys.append(batch["y"].to(self.device))
        all_logits, y = torch.cat(logits), torch.cat(ys)
        logs = {"val_loss": float(loss_fn(all_logits, y))}
        prob = torch.softmax(all_logits, -1).cpu().numpy()
        y_np = y.cpu().numpy()
        pred = prob.argmax(1)
        for name in ("auroc", "accuracy", "balanced_accuracy", "f1_macro", "ece"):
            logs[f"val_{name}"] = METRICS.build(name)(y_np, prob, pred)
        return logs

    def _resume(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: WarmupLRScheduler,
        stopper: EarlyStopping | None,
    ) -> tuple[int, list[dict[str, float]]]:
        state = torch.load(Path(path).expanduser(), map_location=self.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if stopper is not None and state.get("early_stopping"):
            stopper.load_state_dict(state["early_stopping"])
        logger.info("Resumed from %s (epoch %d)", path, state["epoch"] + 1)
        return int(state["epoch"]) + 1, list(state.get("history", []))
