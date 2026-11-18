"""Early stopping and checkpointing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tbmxai.core.utils import get_logger

logger = get_logger(__name__)


def resolve_mode(monitor: str, mode: str) -> str:
    """``auto`` → ``min`` for losses/calibration errors, ``max`` otherwise."""
    if mode != "auto":
        return mode
    lowered = monitor.lower()
    return "min" if any(k in lowered for k in ("loss", "brier", "ece", "error")) else "max"


def _improved(value: float, best: float | None, mode: str, min_delta: float = 0.0) -> bool:
    if not math.isfinite(value):
        return False
    if best is None:
        return True
    return value < best - min_delta if mode == "min" else value > best + min_delta


class EarlyStopping:
    """Stop training when ``monitor`` has not improved for ``patience`` epochs.

    Args:
        monitor: Metric key in the epoch log (e.g. ``val_loss``, ``val_auroc``).
        mode: ``min``, ``max`` or ``auto``.
        patience: Epochs without improvement before stopping.
        min_delta: Minimum change counted as improvement.
    """

    def __init__(
        self, monitor: str, mode: str = "auto", patience: int = 10, min_delta: float = 0.0
    ):
        self.monitor = monitor
        self.mode = resolve_mode(monitor, mode)
        self.patience = patience
        self.min_delta = min_delta
        self.best: float | None = None
        self.bad_epochs = 0

    def update(self, logs: dict[str, float]) -> bool:
        """Record an epoch; return ``True`` if training should stop."""
        if self.monitor not in logs:
            raise KeyError(f"Early-stopping monitor '{self.monitor}' not in logs {sorted(logs)}.")
        value = logs[self.monitor]
        if _improved(value, self.best, self.mode, self.min_delta):
            self.best, self.bad_epochs = value, 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best, self.bad_epochs = state["best"], state["bad_epochs"]


@dataclass(order=True)
class _Saved:
    sort_key: float
    path: Path = field(compare=False)
    value: float = field(compare=False)


class ModelCheckpoint:
    """Keep the top-k checkpoints by ``monitor`` and optionally ``last.pt``.

    Checkpoints contain model, optimiser, scheduler and callback state plus
    the epoch log, so training can be resumed exactly.

    Args:
        dirpath: Directory for ``.pt`` files (anywhere on disk).
        monitor: Metric key in the epoch log.
        mode: ``min``, ``max`` or ``auto``.
        save_top_k: Number of best checkpoints to keep (0 = none).
        save_last: Always write ``last.pt``.
    """

    def __init__(
        self,
        dirpath: str | Path,
        monitor: str,
        mode: str = "auto",
        save_top_k: int = 1,
        save_last: bool = True,
    ) -> None:
        self.dirpath = Path(dirpath).expanduser()
        self.monitor = monitor
        self.mode = resolve_mode(monitor, mode)
        self.save_top_k = save_top_k
        self.save_last = save_last
        self._saved: list[_Saved] = []

    @property
    def best_path(self) -> Path | None:
        """Path of the best checkpoint so far."""
        return min(self._saved).path if self._saved else None

    @property
    def best_value(self) -> float | None:
        """Monitored value of the best checkpoint."""
        return min(self._saved).value if self._saved else None

    def update(self, state: dict[str, Any], logs: dict[str, float], epoch: int) -> None:
        """Save ``state`` if it ranks in the top-k; always refresh ``last.pt``."""
        self.dirpath.mkdir(parents=True, exist_ok=True)
        if self.save_last:
            torch.save(state, self.dirpath / "last.pt")
        if self.save_top_k == 0:
            return
        value = logs.get(self.monitor, float("nan"))
        if not math.isfinite(value):
            return
        key = value if self.mode == "min" else -value
        if len(self._saved) < self.save_top_k or key < max(self._saved).sort_key:
            path = self.dirpath / f"epoch{epoch:03d}-{self.monitor}={value:.4f}.pt"
            torch.save(state, path)
            self._saved.append(_Saved(key, path, value))
            self._saved.sort()
            while len(self._saved) > self.save_top_k:
                worst = self._saved.pop()
                worst.path.unlink(missing_ok=True)
            logger.info("Checkpoint saved: %s", path.name)
