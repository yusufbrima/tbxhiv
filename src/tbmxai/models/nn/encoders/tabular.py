"""Neural encoders for (preprocessed, numeric) tabular features.

Inputs are the dense matrix produced by
:func:`tbmxai.data.preprocessing.build_tabular_preprocessor`, so
``in_features`` is known only after the preprocessor is fit; the model builder
injects it.

"Pretrained" for tabular encoders means loading a ``checkpoint`` from your own
(e.g. self-supervised / masked-feature) pretraining run.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from tbmxai.core.types import Modality
from tbmxai.models.nn.encoders.base import ENCODERS, Encoder, TokenOutput


@ENCODERS.register("mlp_tabular")
class MLPTabularEncoder(Encoder):
    """Feed-forward encoder: ``[Linear → LayerNorm → GELU → Dropout] × L``.

    Args:
        in_features: Number of input features.
        hidden_dims: Hidden layer sizes; the last one is :attr:`out_dim`.
        dropout: Dropout probability.
        freeze / checkpoint: See :class:`Encoder`.
    """

    modality = Modality.TABULAR

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
        freeze: bool = False,
        checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        layers: list[nn.Module] = []
        prev = in_features
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout)]
            prev = dim
        self.net = nn.Sequential(*layers)
        self.out_dim = int(hidden_dims[-1])
        self._post_init(freeze, checkpoint)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, in_features)`` → ``(B, out_dim)``."""
        return self.net(x)


@ENCODERS.register("ft_transformer")
class FTTransformerEncoder(Encoder):
    """Feature-Tokenizer Transformer (Gorishniy et al., 2021), numeric variant.

    Each feature becomes a token ``x_j * W_j + b_j``; a ``[CLS]`` token is
    prepended and the sequence passes through a Transformer encoder. Exposes
    per-feature tokens for attention fusion.

    Args:
        in_features: Number of input features.
        d_token: Token dimension (= :attr:`out_dim`).
        n_layers: Transformer layers.
        n_heads: Attention heads (must divide ``d_token``).
        dropout: Attention/FFN dropout.
        freeze / checkpoint: See :class:`Encoder`.
    """

    modality = Modality.TABULAR

    def __init__(
        self,
        in_features: int,
        d_token: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        freeze: bool = False,
        checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, d_token))
        self.bias = nn.Parameter(torch.zeros(in_features, d_token))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_token,
            n_heads,
            dim_feedforward=4 * d_token,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_token)
        self.out_dim = d_token
        self._post_init(freeze, checkpoint)

    def _encode(self, x: Tensor) -> Tensor:
        tokens = x.unsqueeze(-1) * self.weight + self.bias  # (B, F, D)
        tokens = torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1)
        return self.norm(self.transformer(tokens))

    def forward(self, x: Tensor) -> Tensor:
        """``(B, in_features)`` → ``[CLS]`` embedding ``(B, d_token)``."""
        return self._encode(x)[:, 0]

    def forward_tokens(self, x: Tensor) -> TokenOutput:
        """Feature tokens ``(B, in_features, d_token)`` (without ``[CLS]``)."""
        return self._encode(x)[:, 1:], None
