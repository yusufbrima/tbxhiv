"""Fusion operators combining per-modality embeddings.

All operators share one interface so they are interchangeable from config::

    fusion = FUSIONS.build("attention", input_dims={"imaging": 1024, "tabular": 64}, dim=256)
    fused = fusion(features, masks)          # (B, fusion.out_dim)

* Pooled-feature operators (``requires_tokens = False``) receive ``(B, D_m)``.
* Token operators (``requires_tokens = True``) receive ``(B, T_m, D_m)`` plus
  optional padding masks; the network asks encoders for tokens accordingly.
* Logit-level operators (``produces_logits = True``, i.e. late fusion) return
  class logits directly; the network then skips its classification head.

Available: ``concat``, ``gated``, ``bilinear``, ``attention`` (joint
self-attention over all modality tokens), ``cross_attention`` (one modality
queries the others) and ``late`` (weighted logit averaging).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tbmxai.core.registry import Registry

Masks = Mapping[str, Tensor | None] | None
#: Modality → BoolTensor ``(B,)``, True where the modality is available.
Present = Mapping[str, Tensor] | None


class Fusion(nn.Module, ABC):
    """Base fusion operator.

    Args:
        input_dims: Modality name → embedding dimension. Order is canonicalised
            (sorted) so results do not depend on dict order.
    """

    requires_tokens: ClassVar[bool] = False
    produces_logits: ClassVar[bool] = False
    out_dim: int

    def __init__(self, input_dims: Mapping[str, int]) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("Fusion needs at least one modality.")
        self.input_dims = dict(sorted(input_dims.items()))
        self.modalities = list(self.input_dims)

    @abstractmethod
    def forward(
        self, features: Mapping[str, Tensor], masks: Masks = None, present: Present = None
    ) -> Tensor:
        """Fuse per-modality features into ``(B, out_dim)``.

        Args:
            features: Pooled ``(B, D_m)`` or token ``(B, T_m, D_m)`` features. Missing
                modalities have already been replaced by learned "missing" embeddings.
            masks: Token padding masks (True = ignore) for token operators.
            present: Per-sample availability; operators that weight modalities
                (gated, late) renormalise over the available ones.
        """


FUSIONS: Registry[Fusion] = Registry("fusion")


def _mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout)
    )


@FUSIONS.register("concat")
class ConcatFusion(Fusion):
    """Concatenate pooled embeddings, optionally followed by a projection.

    Args:
        input_dims: See :class:`Fusion`.
        dim: Projection size; ``None`` keeps the raw concatenation.
        dropout: Dropout after projection.
    """

    def __init__(self, input_dims: Mapping[str, int], dim: int | None = 256, dropout: float = 0.1):
        super().__init__(input_dims)
        total = sum(self.input_dims.values())
        self.proj = _mlp(total, dim, dropout) if dim else nn.Identity()
        self.out_dim = dim or total

    def forward(
        self, features: Mapping[str, Tensor], masks: Masks = None, present: Present = None
    ) -> Tensor:
        return self.proj(torch.cat([features[m] for m in self.modalities], dim=-1))


@FUSIONS.register("gated")
class GatedFusion(Fusion):
    """Gated multimodal unit: softmax gates weight projected modality embeddings.

    ``h_m = tanh(W_m x_m)``, ``g = softmax(W_g [x_1; …; x_M])``, ``z = Σ g_m h_m``.
    Gate values are exposed in :attr:`last_gates` for interpretability.
    """

    def __init__(self, input_dims: Mapping[str, int], dim: int = 256, dropout: float = 0.1):
        super().__init__(input_dims)
        self.proj = nn.ModuleDict({m: nn.Linear(d, dim) for m, d in self.input_dims.items()})
        self.gate = nn.Linear(sum(self.input_dims.values()), len(self.modalities))
        self.dropout = nn.Dropout(dropout)
        self.out_dim = dim
        self.last_gates: Tensor | None = None

    def forward(
        self, features: Mapping[str, Tensor], masks: Masks = None, present: Present = None
    ) -> Tensor:
        hidden = torch.stack([torch.tanh(self.proj[m](features[m])) for m in self.modalities], 1)
        logits = self.gate(torch.cat([features[m] for m in self.modalities], -1))
        if present is not None:
            avail = torch.stack([present[m] for m in self.modalities], 1)
            logits = logits.masked_fill(~avail, float("-inf"))
        gates = torch.softmax(logits, -1)
        self.last_gates = gates.detach()
        return self.dropout((gates.unsqueeze(-1) * hidden).sum(1))


@FUSIONS.register("bilinear")
class BilinearFusion(Fusion):
    """Bilinear pooling of exactly two modalities.

    ``rank=None`` uses a full ``nn.Bilinear``; an integer uses low-rank
    factorised bilinear pooling (MFB, Yu et al. 2017) with power and L2
    normalisation — far fewer parameters for large embeddings.
    """

    def __init__(
        self,
        input_dims: Mapping[str, int],
        dim: int = 256,
        rank: int | None = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(input_dims)
        if len(self.modalities) != 2:
            raise ValueError(f"Bilinear fusion needs exactly 2 modalities, got {self.modalities}.")
        da, db = (self.input_dims[m] for m in self.modalities)
        self.rank = rank
        if rank is None:
            self.bilinear = nn.Bilinear(da, db, dim)
        else:
            self.u = nn.Linear(da, dim * rank)
            self.v = nn.Linear(db, dim * rank)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = dim

    def forward(
        self, features: Mapping[str, Tensor], masks: Masks = None, present: Present = None
    ) -> Tensor:
        a, b = (features[m] for m in self.modalities)
        if self.rank is None:
            return self.dropout(self.bilinear(a, b))
        z = (self.u(a) * self.v(b)).view(a.shape[0], self.out_dim, self.rank).sum(-1)
        z = torch.sign(z) * torch.sqrt(z.abs() + 1e-8)
        return self.dropout(F.normalize(z, dim=-1))


@FUSIONS.register("attention")
class AttentionFusion(Fusion):
    """Joint self-attention over the tokens of all modalities.

    Each modality's tokens are projected to ``dim`` and tagged with a learned
    modality embedding; a ``[CLS]`` token is prepended and the sequence passes
    through a Transformer encoder. The ``[CLS]`` output is the fused vector.
    """

    requires_tokens = True

    def __init__(
        self,
        input_dims: Mapping[str, int],
        dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(input_dims)
        self.proj = nn.ModuleDict({m: nn.Linear(d, dim) for m, d in self.input_dims.items()})
        self.modality_embed = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(1, 1, dim)) for m in self.modalities}
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            dim, n_heads, 4 * dim, dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(
        self, features: Mapping[str, Tensor], masks: Masks = None, present: Present = None
    ) -> Tensor:
        batch = next(iter(features.values())).shape[0]
        tokens = [self.cls.expand(batch, -1, -1)]
        pads = [torch.zeros(batch, 1, dtype=torch.bool, device=self.cls.device)]
        for m in self.modalities:
            tok = self.proj[m](features[m]) + self.modality_embed[m]
            tokens.append(tok)
            mask = (masks or {}).get(m)
            pads.append(
                mask
                if mask is not None
                else torch.zeros(tok.shape[:2], dtype=torch.bool, device=tok.device)
            )
        out = self.encoder(torch.cat(tokens, 1), src_key_padding_mask=torch.cat(pads, 1))
        return self.norm(out[:, 0])


@FUSIONS.register("cross_attention")
class CrossAttentionFusion(Fusion):
    """One modality (``query``) attends to the tokens of the others.

    E.g. ``query="tabular"`` lets clinical context select relevant image
    regions; attention weights are kept in :attr:`last_attention` for
    visualisation. Output: ``[pooled query ; attended context]`` projected to ``dim``.
    """

    requires_tokens = True

    def __init__(
        self,
        input_dims: Mapping[str, int],
        query: str = "tabular",
        dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(input_dims)
        if query not in self.input_dims or len(self.modalities) < 2:
            raise ValueError(f"query '{query}' must be one of ≥2 modalities {self.modalities}.")
        self.query = query
        self.context = [m for m in self.modalities if m != query]
        self.proj = nn.ModuleDict({m: nn.Linear(d, dim) for m, d in self.input_dims.items()})
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.out = _mlp(2 * dim, dim, dropout)
        self.out_dim = dim
        self.last_attention: Tensor | None = None

    def forward(
        self, features: Mapping[str, Tensor], masks: Masks = None, present: Present = None
    ) -> Tensor:
        masks = masks or {}
        q = self.proj[self.query](features[self.query])  # (B, Tq, D)
        ctx = torch.cat([self.proj[m](features[m]) for m in self.context], 1)
        pad_parts = [
            masks.get(m)
            if masks.get(m) is not None
            else torch.zeros(features[m].shape[:2], dtype=torch.bool, device=q.device)
            for m in self.context
        ]
        attended, weights = self.attn(q, ctx, ctx, key_padding_mask=torch.cat(pad_parts, 1))  # type: ignore[arg-type]
        self.last_attention = weights.detach()
        return self.out(torch.cat([q.mean(1), attended.mean(1)], -1))


@FUSIONS.register("late")
class LateFusion(Fusion):
    """Per-modality classification heads, combined by (learnable) weighted logit averaging.

    Args:
        input_dims: See :class:`Fusion`.
        n_classes: Number of classes (injected by the network builder).
        weights: Initial per-modality weights (in :attr:`modalities` order);
            default uniform.
        learnable: Learn the weights (softmax-normalised).
    """

    produces_logits = True

    def __init__(
        self,
        input_dims: Mapping[str, int],
        n_classes: int,
        weights: Sequence[float] | None = None,
        learnable: bool = True,
    ) -> None:
        super().__init__(input_dims)
        self.heads = nn.ModuleDict({m: nn.Linear(d, n_classes) for m, d in self.input_dims.items()})
        init = (
            torch.tensor(weights, dtype=torch.float)
            if weights
            else torch.ones(len(self.modalities))
        )
        logw = torch.log(init / init.sum())
        self.log_weights = nn.Parameter(logw, requires_grad=learnable)
        self.out_dim = n_classes

    def forward(
        self, features: Mapping[str, Tensor], masks: Masks = None, present: Present = None
    ) -> Tensor:
        logits = torch.stack([self.heads[m](features[m]) for m in self.modalities], 0)
        logw = self.log_weights.view(-1, 1).expand(-1, logits.shape[1])  # (M, B)
        if present is not None:
            avail = torch.stack([present[m] for m in self.modalities], 0)
            logw = logw.masked_fill(~avail, float("-inf"))
        w = torch.softmax(logw, 0).unsqueeze(-1)  # weights over available modalities
        return (w * logits).sum(0)
