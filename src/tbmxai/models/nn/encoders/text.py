"""Text encoders for radiology / clinical reports (Hugging Face transformers)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from tbmxai.core.types import Modality
from tbmxai.models.nn.encoders.base import ENCODERS, Encoder, TokenOutput

#: Suggested clinical / biomedical language models.
TEXT_BACKBONES: tuple[str, ...] = (
    "emilyalsentzer/Bio_ClinicalBERT",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "StanfordAIMI/RadBERT",
    "distilbert-base-uncased",
)


@ENCODERS.register("hf_text")
class HFTextEncoder(Encoder):
    """Transformer text encoder (pretrained or randomly initialised).

    Args:
        model_name: Hugging Face model id or local path (architecture + tokenizer).
        pretrained: Load pretrained weights; ``False`` builds the same
            architecture from its config with random weights.
        max_length: Tokenizer truncation length.
        pooling: ``"cls"`` (first token) or ``"mean"`` (mask-aware mean).
        config_overrides: Config changes for scratch models
            (e.g. ``{"num_hidden_layers": 4}``); ignored when pretrained.
        freeze / checkpoint: See :class:`Encoder`.

    Inputs are raw strings; call :meth:`tokenize` (done by the data pipeline)
    to obtain the ``input_ids`` / ``attention_mask`` batch.
    """

    modality = Modality.TEXT

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        pretrained: bool = True,
        max_length: int = 256,
        pooling: Literal["cls", "mean"] = "cls",
        config_overrides: dict[str, Any] | None = None,
        freeze: bool = False,
        checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name, **(config_overrides or {}))
            self.model = AutoModel.from_config(config)
        self.out_dim = int(self.model.config.hidden_size)
        self._post_init(freeze, checkpoint)

    def tokenize(self, texts: Sequence[str]) -> dict[str, Tensor]:
        """Tokenise a batch of strings into model inputs."""
        return dict(
            self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        )

    def forward_tokens(self, x: dict[str, Tensor]) -> TokenOutput:
        """Contextual tokens ``(B, T, D)`` and padding mask (True = pad)."""
        hidden = self.model(**x).last_hidden_state
        return hidden, x["attention_mask"] == 0

    def forward(self, x: dict[str, Tensor]) -> Tensor:
        """Pooled report embedding ``(B, D)``."""
        hidden, pad = self.forward_tokens(x)
        if self.pooling == "cls":
            return hidden[:, 0]
        keep = (~pad).unsqueeze(-1).to(hidden.dtype)
        return (hidden * keep).sum(1) / keep.sum(1).clamp_min(1.0)


def tokens_to_device(batch: dict[str, Tensor], device: torch.device | str) -> dict[str, Tensor]:
    """Move a tokenised batch to ``device``."""
    return {k: v.to(device) for k, v in batch.items()}
