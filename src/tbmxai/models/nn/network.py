"""Assemble encoders + fusion + head into one network."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from tbmxai.core.config import ComponentConfig
from tbmxai.models.nn.encoders.base import ENCODERS, Encoder
from tbmxai.models.nn.fusion import FUSIONS, ConcatFusion, Fusion


class MultimodalNet(nn.Module):
    """Encoders per modality → (fusion) → classification head.

    * One encoder, no fusion → a unimodal network (e.g. CXR-only CNN).
    * Several encoders → fused by ``fusion`` (default: concatenation).
    * Logit-level fusion (``late``) → no extra head.

    The imaging branch stays addressable (``net.encoders["imaging"]``) so
    saliency methods can target it, with other inputs held fixed.

    Args:
        encoders: Modality name → encoder.
        n_classes: Number of output classes.
        fusion: Fusion operator (required semantics described above).
        head_hidden_dims: Hidden layers of the MLP head (empty = linear head).
        dropout: Head dropout.
    """

    def __init__(
        self,
        encoders: Mapping[str, Encoder],
        n_classes: int,
        fusion: Fusion | None = None,
        head_hidden_dims: Sequence[int] = (),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict(dict(encoders))
        if fusion is None and len(encoders) > 1:
            fusion = ConcatFusion({m: e.out_dim for m, e in encoders.items()})
        self.fusion = fusion
        self.n_classes = n_classes
        # Learned stand-ins for missing modalities (one vector per modality).
        self.missing_embeddings = nn.ParameterDict(
            {m: nn.Parameter(0.02 * torch.randn(e.out_dim)) for m, e in encoders.items()}
            if len(encoders) > 1
            else {}
        )

        if fusion is not None and fusion.produces_logits:
            self.head: nn.Module = nn.Identity()
            return
        in_dim = fusion.out_dim if fusion is not None else next(iter(encoders.values())).out_dim
        layers: list[nn.Module] = []
        for dim in head_hidden_dims:
            layers += [nn.Linear(in_dim, dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.head = nn.Sequential(*layers)

    def embed(self, inputs: Mapping[str, Any]) -> Tensor:
        """Fused representation before the head.

        ``inputs`` may contain ``"_present": {modality: BoolTensor(B)}``. For absent
        modalities the encoder output is replaced by a learned "missing" embedding;
        for token fusion all but one of that modality's tokens are masked out.
        """
        present: dict[str, Tensor] | None = inputs.get("_present")
        if self.fusion is None:
            ((name, encoder),) = self.encoders.items()
            return encoder(inputs[name])
        if self.fusion.requires_tokens:
            feats, masks = {}, {}
            for name, encoder in self.encoders.items():
                tok, mask = encoder.forward_tokens(inputs[name])
                if present is not None and name in present:
                    p = present[name].to(tok.device)
                    fill = self.missing_embeddings[name].to(tok.dtype).view(1, 1, -1).expand_as(tok)
                    tok = torch.where(p.view(-1, 1, 1), tok, fill)
                    hide = (~p).view(-1, 1) & (torch.arange(tok.shape[1], device=tok.device) > 0)
                    mask = hide if mask is None else (mask | hide)
                feats[name], masks[name] = tok, mask
            return self.fusion(feats, masks, present)
        feats = {}
        for name, encoder in self.encoders.items():
            f = encoder(inputs[name])
            if present is not None and name in present:
                p = present[name].to(f.device)
                fill = self.missing_embeddings[name].to(f.dtype).expand_as(f)
                f = torch.where(p.view(-1, 1), f, fill)
            feats[name] = f
        return self.fusion(feats, present=present)

    def forward(self, inputs: Mapping[str, Any]) -> Tensor:
        """Class logits ``(B, n_classes)``."""
        return self.head(self.embed(inputs))


def build_network(
    encoders: Mapping[str, ComponentConfig],
    n_classes: int,
    fusion: ComponentConfig | None = None,
    head: Mapping[str, Any] | None = None,
    encoder_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> MultimodalNet:
    """Build a :class:`MultimodalNet` from registry configs.

    Args:
        encoders: Modality → encoder config (``{name, params}``).
        n_classes: Number of classes.
        fusion: Fusion config; ignored for a single encoder.
        head: ``{hidden_dims: [...], dropout: float}``.
        encoder_overrides: Extra kwargs injected per modality at build time
            (e.g. ``{"tabular": {"in_features": 23}}`` once the preprocessor is fit).
    """
    overrides = encoder_overrides or {}
    built = {
        m: ENCODERS.build(spec.name, **{**spec.params, **overrides.get(m, {})})
        for m, spec in encoders.items()
    }
    fusion_module: Fusion | None = None
    if fusion is not None and len(built) > 1:
        cls = FUSIONS.get(fusion.name)
        kwargs = dict(fusion.params)
        if cls.produces_logits:
            kwargs["n_classes"] = n_classes
        fusion_module = cls(input_dims={m: e.out_dim for m, e in built.items()}, **kwargs)
    head = dict(head or {})
    return MultimodalNet(
        built,
        n_classes,
        fusion=fusion_module,
        head_hidden_dims=head.get("hidden_dims", ()),
        dropout=head.get("dropout", 0.1),
    )
