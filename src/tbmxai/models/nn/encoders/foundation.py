"""Public chest X-ray foundation models as imaging encoders (reference baselines).

* ``xrv_densenet`` — TorchXRayVision DenseNet-121 (Cohen et al., 2022), supervised on several
  public CXR datasets. Weight ``densenet121-res224-all`` was trained on NIH, PadChest,
  CheXpert, MIMIC-CXR, Google, Open-i and RSNA (not Shenzhen, Montgomery or TBX11K).
* ``rad_dino`` — Microsoft RAD-DINO (Pérez-García et al., 2025), a DINOv2 ViT-B/14
  self-supervised on chest X-rays; the CLS embedding is used.

Both take the pipeline's standard image tensor — one channel normalised as
``(x - mean) / std`` with ``mean``/``std`` from the model's image config (0.5 / 0.25 by
default) — and convert it to the input each model expects, resizing inside the encoder.
Neither includes Shenzhen, Montgomery or TBX11K, but both saw MIMIC-CXR, so they are not
leakage-free references on MIMIC test patients.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from tbmxai.core.types import Modality
from tbmxai.models.nn.encoders.base import ENCODERS, Encoder


def _to_unit(x: Tensor, mean: float, std: float) -> Tensor:
    """Pipeline-normalised (B, C, H, W) → grayscale in [0, 1], shape (B, 1, H, W)."""
    raw = x * std + mean
    if raw.shape[1] > 1:
        raw = raw.mean(dim=1, keepdim=True)
    return raw.clamp(0.0, 1.0)


@ENCODERS.register("xrv_densenet")
class XRVDenseNetEncoder(Encoder):
    """TorchXRayVision DenseNet-121 features (1024-d), input resized to 224 x 224.

    Args:
        weights: TorchXRayVision weight name.
        freeze: Freeze the backbone (linear probing).
        checkpoint: Optional state dict loaded after construction.
        input_mean / input_std: The pipeline's image normalisation (to undo it).
    """

    modality = Modality.IMAGING

    def __init__(
        self,
        weights: str = "densenet121-res224-all",
        freeze: bool = False,
        checkpoint: str | Path | None = None,
        input_mean: float = 0.5,
        input_std: float = 0.25,
        **_: Any,
    ) -> None:
        super().__init__()
        import torchxrayvision as xrv

        self.model = xrv.models.DenseNet(weights=weights)
        self.model.op_threshs = None  # raw logits path; we only use features
        self.input_mean, self.input_std = input_mean, input_std
        self.out_dim = 1024
        self._post_init(freeze, checkpoint)

    def saliency_layer(self) -> tuple[str, int]:
        """CAM layer: the dense feature block (``features``, 1024 channels at 7 x 7)."""
        return "features", 1024

    def forward(self, x: Tensor) -> Tensor:
        unit = _to_unit(x, self.input_mean, self.input_std)
        unit = F.interpolate(unit, size=(224, 224), mode="bilinear", align_corners=False)
        xrv_in = unit * 2048.0 - 1024.0  # TorchXRayVision expects [-1024, 1024]
        feats = self.model.features(xrv_in)
        feats = F.relu(feats, inplace=False)
        return F.adaptive_avg_pool2d(feats, 1).flatten(1)


@ENCODERS.register("rad_dino")
class RadDinoEncoder(Encoder):
    """Microsoft RAD-DINO CLS embedding (768-d), input resized to ``size`` (default 518).

    Args:
        model_name: Hugging Face model id.
        size: Input side length (RAD-DINO was trained at 518).
        freeze: Freeze the backbone (linear probing).
        checkpoint: Optional state dict loaded after construction.
        input_mean / input_std: The pipeline's image normalisation (to undo it).
    """

    modality = Modality.IMAGING

    def __init__(
        self,
        model_name: str = "microsoft/rad-dino",
        size: int = 518,
        freeze: bool = False,
        checkpoint: str | Path | None = None,
        input_mean: float = 0.5,
        input_std: float = 0.25,
        **_: Any,
    ) -> None:
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.model = AutoModel.from_pretrained(model_name)
        proc = AutoImageProcessor.from_pretrained(model_name)
        self.register_buffer(
            "mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer("std", torch.tensor(proc.image_std).view(1, 3, 1, 1), persistent=False)
        self.size = size
        self.input_mean, self.input_std = input_mean, input_std
        self.out_dim = int(self.model.config.hidden_size)
        self._post_init(freeze, checkpoint)

    def saliency_layer(self) -> tuple[str, int]:
        """CAM layer: the last block's input norm (patch tokens still reach [CLS] there).

        Hooking the final output would give zero gradient on the patch tokens: only [CLS] feeds
        the pooled embedding, and the final LayerNorm acts per token.
        """
        last = self.model.config.num_hidden_layers - 1
        return f"encoder.layer.{last}.norm1", self.out_dim

    def forward(self, x: Tensor) -> Tensor:
        unit = _to_unit(x, self.input_mean, self.input_std)
        unit = F.interpolate(
            unit, size=(self.size, self.size), mode="bilinear", align_corners=False
        )
        rgb = (unit.repeat(1, 3, 1, 1) - self.mean) / self.std
        return self.model(pixel_values=rgb).pooler_output
