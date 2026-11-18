"""Vision encoders for chest X-rays (CNN, CNN-Transformer hybrids, Transformers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import Tensor

from tbmxai.core.types import Modality
from tbmxai.models.nn.encoders.base import ENCODERS, Encoder, TokenOutput

#: Curated, tested backbones by architecture family. Any other timm model name
#: also works; these are the ones we recommend and benchmark.
VISION_BACKBONES: dict[str, tuple[str, ...]] = {
    "cnn": (
        "resnet18",
        "resnet50",
        "densenet121",
        "efficientnet_b0",
        "convnext_tiny",
    ),
    "hybrid": (  # convolutional stem/stages + attention
        "coatnet_0_rw_224",
        "maxvit_tiny_tf_224",
        "levit_128",
    ),
    "transformer": (
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "deit_small_patch16_224",
        "swin_tiny_patch4_window7_224",
    ),
}


def backbone_family(name: str) -> str:
    """Family (``cnn``/``hybrid``/``transformer``) of a curated backbone, else ``custom``."""
    return next((fam for fam, names in VISION_BACKBONES.items() if name in names), "custom")


@ENCODERS.register("timm_vision")
class TimmVisionEncoder(Encoder):
    """Any `timm <https://github.com/huggingface/pytorch-image-models>`_ backbone.

    Args:
        backbone: timm model name (see :data:`VISION_BACKBONES`).
        pretrained: Load ImageNet weights (downloaded/cached by timm) or train from scratch.
        in_chans: Input channels (1 for grayscale CXR; timm adapts pretrained stems).
        image_size: Input size for architectures with fixed positional
            embeddings/windows (ViT, Swin, MaxViT…); ignored by CNNs.
        freeze: Freeze all backbone weights.
        checkpoint: Optional ``state_dict`` to load (e.g. CXR-domain pretraining).
            Keys are relative to this module (``model.*``).
        **timm_kwargs: Forwarded to ``timm.create_model`` (e.g. ``drop_path_rate``).
    """

    modality = Modality.IMAGING

    def __init__(
        self,
        backbone: str = "densenet121",
        pretrained: bool = True,
        in_chans: int = 1,
        image_size: int | None = None,
        freeze: bool = False,
        checkpoint: str | Path | None = None,
        **timm_kwargs: Any,
    ) -> None:
        super().__init__()
        import timm

        self.backbone = backbone
        self.family = backbone_family(backbone)
        kwargs: dict[str, Any] = {
            "pretrained": pretrained,
            "num_classes": 0,  # remove classifier → pooled features
            "in_chans": in_chans,
            **timm_kwargs,
        }
        if image_size is not None and self.family != "cnn":
            kwargs["img_size"] = image_size
        try:
            self.model = timm.create_model(backbone, **kwargs)
        except TypeError:
            # Architectures without an ``img_size`` argument (most CNNs).
            kwargs.pop("img_size", None)
            self.model = timm.create_model(backbone, **kwargs)
        self.out_dim = int(self.model.num_features)
        self._post_init(freeze, checkpoint)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, C, H, W)`` → ``(B, out_dim)``."""
        return self.model(x)

    def forward_tokens(self, x: Tensor) -> TokenOutput:
        """Spatial/patch tokens ``(B, T, out_dim)`` from ``forward_features``."""
        feats = self.model.forward_features(x)
        if feats.ndim == 4:
            # NCHW (CNNs) or NHWC (Swin/MaxViT in timm ≥ 0.9).
            if feats.shape[1] == self.out_dim:
                feats = feats.flatten(2).transpose(1, 2)
            else:
                feats = feats.flatten(1, 2)
        return feats, None
