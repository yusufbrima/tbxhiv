"""Optimiser construction and learning-rate scheduling with warm-up."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from tbmxai.core.config import OptimizerConfig, SchedulerConfig


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    """Create the optimiser with two parameter groups.

    * Encoder parameters (``encoders.*``) use ``lr * encoder_lr_mult`` —
      discriminative fine-tuning for pretrained backbones.
    * Biases and normalisation parameters get no weight decay.
    """
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_encoder = name.startswith("encoders.")
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        groups.setdefault((is_encoder, no_decay), []).append(param)
    param_groups = [
        {
            "params": params,
            "lr": cfg.lr * (cfg.encoder_lr_mult if is_encoder else 1.0),
            "weight_decay": 0.0 if no_decay else cfg.weight_decay,
        }
        for (is_encoder, no_decay), params in groups.items()
    ]
    if cfg.name == "adamw":
        return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas)
    if cfg.name == "adam":
        return torch.optim.Adam(param_groups, lr=cfg.lr, betas=cfg.betas)
    return torch.optim.SGD(param_groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov)


class WarmupLRScheduler:
    """Linear warm-up followed by a configurable schedule.

    Updated per optimiser step via :meth:`step_batch` and per epoch via
    :meth:`step_epoch`. Each param group's LR is ``base_lr * factor``, where
    ``factor`` combines warm-up, the main schedule and plateau reductions.

    Args:
        optimizer: Optimiser whose group LRs at construction time are the base LRs.
        cfg: Scheduler configuration.
        steps_per_epoch: Optimiser steps per epoch (after gradient accumulation).
        epochs: Total epochs.
        monitor_mode: ``"min"`` or ``"max"`` for the plateau scheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        cfg: SchedulerConfig,
        steps_per_epoch: int,
        epochs: int,
        monitor_mode: str = "min",
    ) -> None:
        self.optimizer = optimizer
        self.cfg = cfg
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.total_steps = max(1, steps_per_epoch * epochs)
        self.warmup_steps = round(cfg.warmup_epochs * steps_per_epoch)
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.monitor_mode = monitor_mode
        self.step_count = 0
        self.epoch = 0
        self.plateau_factor = 1.0
        self._best: float | None = None
        self._bad_epochs = 0
        self._apply()

    # -------------------------------------------------------------- schedule

    def factor(self) -> float:
        """Current multiplicative LR factor."""
        cfg, step = self.cfg, self.step_count
        if step < self.warmup_steps:
            return (
                cfg.warmup_start_factor + (1 - cfg.warmup_start_factor) * step / self.warmup_steps
            )
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / decay_steps)
        floor = cfg.min_lr_ratio
        match cfg.name:
            case "cosine":
                return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
            case "linear":
                return floor + (1 - floor) * (1 - progress)
            case "step":
                return cfg.gamma ** (self.epoch // cfg.step_size)
            case "plateau":
                return self.plateau_factor
            case _:
                return 1.0

    def _apply(self) -> None:
        f = self.factor()
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * f

    def step_batch(self) -> None:
        """Advance one optimiser step."""
        self.step_count += 1
        self._apply()

    def step_epoch(self, monitored: float | None = None) -> None:
        """Advance one epoch; ``monitored`` drives the plateau schedule."""
        self.epoch += 1
        if self.cfg.name == "plateau" and monitored is not None and math.isfinite(monitored):
            better = self._best is None or (
                monitored < self._best if self.monitor_mode == "min" else monitored > self._best
            )
            if better:
                self._best, self._bad_epochs = monitored, 0
            else:
                self._bad_epochs += 1
                if self._bad_epochs > self.cfg.patience:
                    self.plateau_factor *= self.cfg.gamma
                    self._bad_epochs = 0
        self._apply()

    @property
    def last_lr(self) -> list[float]:
        """Current LR per param group."""
        return [g["lr"] for g in self.optimizer.param_groups]

    # ----------------------------------------------------------- persistence

    def state_dict(self) -> dict[str, Any]:
        """Serializable state (for checkpoint resume)."""
        return {
            "step_count": self.step_count,
            "epoch": self.epoch,
            "plateau_factor": self.plateau_factor,
            "best": self._best,
            "bad_epochs": self._bad_epochs,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`state_dict`."""
        self.step_count = state["step_count"]
        self.epoch = state["epoch"]
        self.plateau_factor = state["plateau_factor"]
        self._best = state["best"]
        self._bad_epochs = state["bad_epochs"]
        self._apply()
