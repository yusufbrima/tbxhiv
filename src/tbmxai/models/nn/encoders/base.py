"""Neural encoder contract.

An encoder maps one modality to a pooled embedding ``(B, out_dim)`` and,
for token-level fusion (attention), to a token sequence ``(B, T, out_dim)``
plus an optional padding mask. Every encoder supports:

* **pretrained vs. scratch** — ``pretrained=True`` loads published weights
  (timm / Hugging Face); ``False`` initialises randomly;
* **custom weights** — ``checkpoint`` loads a ``state_dict`` (e.g. from
  domain-specific self-supervised pretraining), applied after construction;
* **freezing** — ``freeze=True`` disables gradients (linear probing).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import Tensor, nn

from tbmxai.core.registry import Registry
from tbmxai.core.types import Modality
from tbmxai.core.utils import get_logger

logger = get_logger(__name__)

#: ``(tokens (B, T, D), padding_mask (B, T) with True = padding, or None)``
TokenOutput = tuple[Tensor, Tensor | None]


class Encoder(nn.Module, ABC):
    """Base class for modality encoders.

    Subclasses must set :attr:`out_dim`, build their modules, then call
    :meth:`_post_init` at the end of ``__init__``.
    """

    modality: ClassVar[Modality]
    out_dim: int

    def _post_init(self, freeze: bool = False, checkpoint: str | Path | None = None) -> None:
        """Load optional custom weights, then optionally freeze."""
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state) if isinstance(state, dict) else state
            missing, unexpected = self.load_state_dict(state, strict=False)
            logger.info(
                "%s: loaded %s (missing=%d, unexpected=%d)",
                type(self).__name__,
                checkpoint,
                len(missing),
                len(unexpected),
            )
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    @abstractmethod
    def forward(self, x: Any) -> Tensor:
        """Pooled embedding of shape ``(B, out_dim)``."""

    def forward_tokens(self, x: Any) -> TokenOutput:
        """Token sequence for attention fusion; default: the pooled vector as one token."""
        return self.forward(x).unsqueeze(1), None


ENCODERS: Registry[Encoder] = Registry("encoder")
