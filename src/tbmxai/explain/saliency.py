"""Saliency maps for CXR models and the imaging branch of multimodal models.

Both explainers work with :class:`~tbmxai.models.neural.NeuralClassifier` models
that have an imaging encoder, unimodal or fused. Saliency is computed through
the **full network**, with the other modalities of each sample held fixed at
their observed values, so maps answer: "which pixels drove *this* prediction,
given this patient's EHR?".

* ``grad_cam`` — Grad-CAM (Selvaraju et al., 2017) on the backbone's last feature
  stage, found automatically from timm ``feature_info``. Works for CNNs (NCHW),
  CNN-Transformer hybrids, Swin (NHWC) and ViT-style token outputs (prefix tokens
  removed, patches reshaped to a grid).
* ``integrated_gradients`` — Integrated Gradients (Sundararajan et al., 2017) on
  input pixels via captum, baseline = black image in normalised space.
* ``score_cam`` — Score-CAM (Wang et al., 2020): gradient-free; channel weights are
  the target-probability gain when only that channel's region is visible.

Which samples are explained is decided by the pipeline from ground truth only
(see :mod:`tbmxai.explain.selection` and :mod:`tbmxai.protocol`): every class is
represented and every model sees the same samples. Maps are computed for the
output of every class (``target_classes="all"``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tbmxai.core.types import Modality
from tbmxai.data.harmonize import MultimodalView
from tbmxai.explain.base import EXPLAINERS, Explainer, Explanation
from tbmxai.models.base import Classifier
from tbmxai.training.trainer import resolve_device, to_device

CATEGORIES = ("TP", "FN", "FP", "TN")


def _outcome(y: np.ndarray, p: np.ndarray, threshold: float) -> np.ndarray:
    pred = (p >= threshold).astype(int)
    return np.select(
        [(y == 1) & (pred == 1), (y == 1) & (pred == 0), (y == 0) & (pred == 1)],
        ["TP", "FN", "FP"],
        default="TN",
    )


class _ImagingSaliency(Explainer):
    """Shared batching and packaging for imaging saliency.

    The explainer explains **exactly the samples it is given** (the pipeline
    passes the shared, class-stratified explanation set), for **every target
    class** (``target_classes``), so maps are comparable across models.

    Args:
        model: Fitted ``NeuralClassifier`` with an imaging encoder.
        target_classes: ``"all"`` (default), ``"positive"`` (class 1) or a list of indices.
        threshold: Decision threshold used only to annotate outcome (TP/FN/FP/TN).
        batch_size: Samples per backward pass.
    """

    modality = Modality.IMAGING
    scope = "local"
    uses_xai_samples = True
    method: ClassVar[str]

    def __init__(
        self,
        model: Classifier,
        target_classes: str | list[int] = "all",
        threshold: float = 0.5,
        batch_size: int = 8,
    ) -> None:
        super().__init__(model)
        self.target_classes = target_classes
        self.threshold = threshold
        self.batch_size = batch_size
        self.target_class = 1  # set per pass in explain()

    @classmethod
    def supports(cls, model: Classifier) -> bool:
        from tbmxai.models.neural import NeuralClassifier

        return isinstance(model, NeuralClassifier) and Modality.IMAGING.value in model.encoder_specs

    def _targets(self, n_classes: int) -> list[int]:
        if self.target_classes == "all":
            return list(range(n_classes))
        if self.target_classes == "positive":
            return [1 if n_classes == 2 else n_classes - 1]
        return [int(k) for k in self.target_classes]  # type: ignore[union-attr]

    def explain(self, view: MultimodalView) -> Explanation:
        has_image = view.availability.get(Modality.IMAGING, np.ones(len(view), bool))
        if not has_image.all():
            raise ValueError(
                "Saliency samples must all have an image (use the explanation sample set)."
            )
        model: Any = self.model
        proba = model.predict_proba(view)
        targets = self._targets(view.n_classes)
        positive = 1 if view.n_classes == 2 else None
        category = (
            _outcome(view.y, proba[:, 1], self.threshold)
            if positive
            else np.where(proba.argmax(1) == view.y, "correct", "wrong")
        )
        net: nn.Module = model.network_
        device = resolve_device(model.training.device)
        net.to(device).eval()
        per_target, images = [], []
        for k in targets:
            self.target_class = k
            maps = []
            for batch in _eval_loader(model, view, self.batch_size):
                inputs = to_device(batch["inputs"], device)
                # Gradients must reach the image even when the backbone is frozen.
                image = inputs[Modality.IMAGING.value].detach().clone().requires_grad_(True)
                inputs[Modality.IMAGING.value] = image
                maps.append(self.saliency(net, inputs, image).detach().float().cpu().numpy())
                if k == targets[0]:
                    images.append(_display(image.detach().float().cpu().numpy()))
            per_target.append(np.concatenate(maps))
        attributions = np.stack(per_target, axis=1).astype(np.float32)  # (n, n_targets, H, W)
        return Explanation(
            method=self.method,
            modality=self.modality,
            scope=self.scope,
            attributions=attributions,
            sample_ids=view.sample_ids,
            extra={
                "images": np.concatenate(images),
                "patient_ids": view.patient_ids,
                "y_true": view.y,
                "class_names": view.class_names,
                "target_classes": targets,
                "proba": proba,
                "prob": proba[:, 1] if positive else proba.max(1),
                "category": category,
            },
        )

    def saliency(self, net: nn.Module, inputs: dict[str, Any], image: Tensor) -> Tensor:
        """Return normalised saliency ``(B, H, W)`` for one batch."""
        raise NotImplementedError

    def _forward_for(self, net: nn.Module, inputs: dict[str, Any]) -> Callable[[Tensor], Tensor]:
        """``f(image) -> target logit`` with the other inputs held fixed."""

        def f(image: Tensor) -> Tensor:
            return net({**inputs, Modality.IMAGING.value: image})[:, self.target_class]

        return f


@EXPLAINERS.register("grad_cam")
class GradCAMExplainer(_ImagingSaliency):
    """Grad-CAM on the vision backbone's last feature stage (any timm family).

    Encoders that are not timm models (TorchXRayVision, RAD-DINO) provide
    ``saliency_layer() -> (module name inside .model, channels)`` instead of ``feature_info``.

    Args:
        layer: Optional module name inside the timm backbone (default: last
            ``feature_info`` stage).
        **kwargs: See :class:`_ImagingSaliency`.
    """

    method = "grad_cam"

    def __init__(self, model: Classifier, layer: str | None = None, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.layer = layer

    def _target(self, net: nn.Module) -> tuple[nn.Module, int]:
        encoder = net.encoders[Modality.IMAGING.value]
        backbone = encoder.model
        if hasattr(encoder, "saliency_layer"):  # non-timm backbones name their own CAM layer
            name, channels = encoder.saliency_layer()
            name = self.layer or name
            return dict(backbone.named_modules())[name], int(channels)
        info = backbone.feature_info
        dicts = info if isinstance(info, list) else info.get_dicts()
        name = self.layer or dicts[-1]["module"]
        channels = next((d["num_chs"] for d in dicts if d["module"] == name), dicts[-1]["num_chs"])
        return dict(backbone.named_modules())[name], int(channels)

    def saliency(self, net: nn.Module, inputs: dict[str, Any], image: Tensor) -> Tensor:
        module, channels = self._target(net)
        store: dict[str, Tensor] = {}

        def hook(_m: nn.Module, _i: Any, out: Any) -> None:
            act = out[0] if isinstance(out, tuple) else out
            act.retain_grad()
            store["act"] = act

        handle = module.register_forward_hook(hook)
        try:
            with torch.enable_grad():
                net.zero_grad(set_to_none=True)
                score = self._forward_for(net, inputs)(image)
                score.sum().backward()
        finally:
            handle.remove()
        act, grad = _to_nchw(store["act"], channels), _to_nchw(store["act"].grad, channels)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        return _normalise(cam)


@EXPLAINERS.register("integrated_gradients")
class IntegratedGradientsExplainer(_ImagingSaliency):
    """Integrated Gradients on input pixels (captum).

    Args:
        n_steps: Riemann steps along the path.
        smooth_sigma: Gaussian smoothing (pixels) of |attributions| for display; 0 = none.
        **kwargs: See :class:`_ImagingSaliency`.
    """

    method = "integrated_gradients"

    def __init__(
        self, model: Classifier, n_steps: int = 32, smooth_sigma: float = 2.0, **kwargs: Any
    ) -> None:
        super().__init__(model, **kwargs)
        self.n_steps = n_steps
        self.smooth_sigma = smooth_sigma

    def saliency(self, net: nn.Module, inputs: dict[str, Any], image: Tensor) -> Tensor:
        from captum.attr import IntegratedGradients

        ig = IntegratedGradients(self._forward_for(net, inputs))
        with torch.enable_grad():
            baseline = torch.full_like(image, float(image.detach().min()))
            attr = ig.attribute(
                image, baselines=baseline, n_steps=self.n_steps, internal_batch_size=image.shape[0]
            )
        sal = attr.abs().sum(1, keepdim=True)
        if self.smooth_sigma > 0:
            sal = _gaussian_blur(sal, self.smooth_sigma)
        return _normalise(sal[:, 0])


@EXPLAINERS.register("score_cam")
class ScoreCAMExplainer(GradCAMExplainer):
    """Score-CAM (Wang et al., 2020): gradient-free class activation maps.

    Each channel of the target layer's activation is upsampled to the input
    size, min-max normalised to a mask, and the image is re-scored with only that
    region visible (``baseline + mask * (image - baseline)``, baseline = black).
    The channel weight is the increase of the target class **probability** over
    the baseline image; the map is ``ReLU(Σ_k w_k A_k)``. Other modalities of the
    sample are held fixed, so fusion models are explained in context.

    Costs one forward pass per channel per sample: ``max_channels`` keeps the
    channels with the highest mean activation (all if ``None``).

    Args:
        layer: Optional module name inside the timm backbone (default: last stage).
        max_channels: Channels used per sample (default 256).
        mask_batch_size: Masked images per forward pass.
        **kwargs: See :class:`_ImagingSaliency`.
    """

    method = "score_cam"

    def __init__(
        self,
        model: Classifier,
        layer: str | None = None,
        max_channels: int | None = 256,
        mask_batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, layer=layer, **kwargs)
        self.max_channels = max_channels
        self.mask_batch_size = mask_batch_size

    @torch.no_grad()
    def saliency(self, net: nn.Module, inputs: dict[str, Any], image: Tensor) -> Tensor:
        module, channels = self._target(net)
        store: dict[str, Tensor] = {}

        def hook(_m: nn.Module, _i: Any, out: Any) -> None:
            store["act"] = (out[0] if isinstance(out, tuple) else out).detach()

        handle = module.register_forward_hook(hook)
        try:
            net(inputs)
        finally:
            handle.remove()
        acts = _to_nchw(store["act"], channels).float()  # (B, K, h, w)
        image = image.detach()
        baseline = torch.full_like(image, float(image.min()))
        maps = []
        for i in range(image.shape[0]):
            act = acts[i]
            if self.max_channels is not None and act.shape[0] > self.max_channels:
                keep = act.mean((1, 2)).topk(self.max_channels).indices
                act = act[keep]
            masks = F.interpolate(
                act[:, None], size=image.shape[-2:], mode="bilinear", align_corners=False
            )
            flat = masks.flatten(1)
            lo, hi = flat.min(1, keepdim=True).values, flat.max(1, keepdim=True).values
            masks = ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(masks)  # (K, 1, H, W)
            base_score = self._probability(net, _take(inputs, i, 1), baseline[i : i + 1])[0]
            weights = []
            for start in range(0, masks.shape[0], self.mask_batch_size):
                m = masks[start : start + self.mask_batch_size]
                masked = baseline[i : i + 1] + m * (image[i : i + 1] - baseline[i : i + 1])
                weights.append(
                    self._probability(net, _take(inputs, i, len(m)), masked) - base_score
                )
            w = torch.cat(weights)  # (K,)
            cam = F.relu((w.view(-1, 1, 1) * act).sum(0, keepdim=True))[None]  # (1, 1, h, w)
            maps.append(
                F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[
                    0, 0
                ]
            )
        return _normalise(torch.stack(maps))

    def _probability(self, net: nn.Module, inputs: dict[str, Any], images: Tensor) -> Tensor:
        logits = net({**inputs, Modality.IMAGING.value: images})
        return torch.softmax(logits.float(), -1)[:, self.target_class]


# ------------------------------------------------------------------ helpers


def _eval_loader(model: Any, view: MultimodalView, batch_size: int) -> Any:
    """Deterministic eval loader over ``view`` with a small batch size (gradients need memory)."""
    original = model.training
    model.training = original.model_copy(update={"eval_batch_size": batch_size, "num_workers": 0})
    try:
        return model._loader(view, train=False)
    finally:
        model.training = original


def _to_nchw(t: Tensor, channels: int) -> Tensor:
    """Bring CNN (NCHW), Swin (NHWC) or token (B, T, C) activations to NCHW."""
    if t.ndim == 4:
        if t.shape[1] == channels:
            return t
        if t.shape[-1] == channels:
            return t.permute(0, 3, 1, 2)
        return t
    if t.ndim == 3:  # tokens: drop prefix ([CLS], registers), reshape patches to a square grid
        b, n, c = t.shape
        side = math.isqrt(n)
        tokens = t[:, n - side * side :, :]
        return tokens.transpose(1, 2).reshape(b, c, side, side)
    raise ValueError(f"Unsupported activation shape {tuple(t.shape)}")


def _normalise(sal: Tensor) -> Tensor:
    flat = sal.flatten(1)
    lo, hi = flat.min(1, keepdim=True).values, flat.max(1, keepdim=True).values
    return ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(sal)


def _gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kernel.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), kernel.view(1, 1, -1, 1))


def _display(images: np.ndarray) -> np.ndarray:
    """(B, C, H, W) normalised tensors → (B, H, W) in [0, 1] for overlays."""
    img = images.mean(1)
    lo = img.reshape(len(img), -1).min(1)[:, None, None]
    hi = img.reshape(len(img), -1).max(1)[:, None, None]
    return (img - lo) / np.maximum(hi - lo, 1e-8)


def _take(inputs: Any, i: int, repeat: int) -> Any:
    """Sample ``i`` of a (nested) batch, repeated ``repeat`` times along the batch axis."""
    if isinstance(inputs, Tensor):
        return inputs[i : i + 1].expand(repeat, *inputs.shape[1:]) if inputs.ndim else inputs
    if isinstance(inputs, dict):
        return {k: _take(v, i, repeat) for k, v in inputs.items()}
    return inputs
