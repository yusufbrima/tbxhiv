"""Small cross-cutting utilities: logging and seeding."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

_LOGGER_NAME = "tbmxai"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a package logger (``tbmxai`` or ``tbmxai.<name>``)."""
    return logging.getLogger(_LOGGER_NAME if not name else f"{_LOGGER_NAME}.{name}")


def configure_logging(level: int | str = logging.INFO) -> None:
    """Attach a rich console handler to the package logger (idempotent)."""
    from rich.logging import RichHandler

    logger = get_logger()
    logger.setLevel(level)
    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        logger.addHandler(RichHandler(rich_tracebacks=True, show_path=False))
    logger.propagate = False


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a declared dependency
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
