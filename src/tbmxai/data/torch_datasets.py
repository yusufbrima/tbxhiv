"""PyTorch datasets, image loading/transforms and collation for :class:`MultimodalView`."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from tbmxai.core.types import Modality
from tbmxai.data.harmonize import MultimodalView
from tbmxai.data.image_cache import ImageCache


def load_image(path: str | Path) -> Image.Image:
    """Load a CXR as an 8-bit grayscale PIL image (PNG/JPEG/TIFF or DICOM).

    DICOMs are rescaled with their slope/intercept, min-max normalised and
    inverted when ``PhotometricInterpretation == MONOCHROME1``.
    """
    path = Path(path)
    if path.suffix.lower() == ".dcm":
        import pydicom

        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope", 1)) + float(
            getattr(ds, "RescaleIntercept", 0)
        )
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = 1.0 - arr
        return Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    with Image.open(path) as img:
        return img.convert("L")


def build_image_transform(
    size: int = 224,
    train: bool = False,
    augment: bool = True,
    in_chans: int = 1,
    mean: float = 0.5,
    std: float = 0.25,
) -> Callable[[Image.Image], torch.Tensor]:
    """Resize → (train-time augmentation) → tensor → normalise.

    Augmentation is deliberately mild (small affine + brightness/contrast) and
    excludes horizontal flips, which would swap cardiac/lung laterality.
    """
    from torchvision.transforms import v2

    steps: list[Any] = [v2.Resize((size, size), antialias=True)]
    if in_chans == 3:
        steps.append(v2.Grayscale(num_output_channels=3))
    if train and augment:
        steps += [
            v2.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    steps += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([mean] * in_chans, [std] * in_chans),
    ]
    return v2.Compose(steps)


PRESENT_KEY = "_present"


class MultimodalTorchDataset(Dataset[dict[str, Any]]):
    """Item: ``{"inputs": {modality: value, "_present": {modality: bool}}, "y": int}``.

    Missing modalities (from ``view.availability`` or modality dropout) get a
    placeholder input (zero image, empty text; tabular rows are imputed by the
    preprocessor) and ``_present[m] = False``; the network substitutes a learned
    "missing" embedding for them.

    Args:
        view: Harmonised view.
        modalities: Modalities to emit.
        tabular: Preprocessed tabular matrix ``(n, F)`` (required for tabular).
        image_transform: Callable applied to loaded images.
        image_shape: ``(C, H, W)`` of the zero placeholder for missing images.
        modality_dropout: Training-time probability of hiding each modality per
            sample (e.g. ``{"imaging": 0.3}``); at least one available modality is
            always kept. Use only for the training set.
        image_cache: Optional :class:`~tbmxai.data.image_cache.ImageCache` (decode once).
    """

    def __init__(
        self,
        view: MultimodalView,
        modalities: Sequence[Modality],
        tabular: np.ndarray | None = None,
        image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
        image_shape: tuple[int, int, int] = (1, 224, 224),
        modality_dropout: Mapping[str, float] | None = None,
        image_cache: ImageCache | None = None,
    ) -> None:
        self.view = view
        self.image_cache = image_cache
        self.modalities = list(modalities)
        self.tabular = None if tabular is None else torch.as_tensor(tabular, dtype=torch.float32)
        self.image_transform = image_transform or build_image_transform()
        self.image_shape = image_shape
        self.modality_dropout = {Modality(k): float(v) for k, v in (modality_dropout or {}).items()}
        if Modality.TABULAR in self.modalities and self.tabular is None:
            raise ValueError("Tabular modality requested but no preprocessed matrix given.")

    def __len__(self) -> int:
        return len(self.view)

    def _present(self, index: int) -> dict[Modality, bool]:
        present = {m: bool(self.view.availability[m][index]) for m in self.modalities}
        if self.modality_dropout:
            available = [m for m, ok in present.items() if ok]
            # torch RNG: differently seeded in every DataLoader worker.
            dropped = [
                m for m in available if float(torch.rand(())) < self.modality_dropout.get(m, 0.0)
            ]
            if len(dropped) == len(available) and available:
                dropped.pop(int(torch.randint(len(dropped), ())))  # keep one modality
            for m in dropped:
                present[m] = False
        return present

    def _load(self, path: str) -> Image.Image:
        return self.image_cache.load(path) if self.image_cache else load_image(path)

    def __getitem__(self, index: int) -> dict[str, Any]:
        present = self._present(index)
        inputs: dict[str, Any] = {}
        for m in self.modalities:
            if m is Modality.TABULAR:
                inputs[m.value] = self.tabular[index]  # type: ignore[index]
            elif m is Modality.IMAGING:
                inputs[m.value] = (
                    self.image_transform(self._load(self.view.image_paths[index]))  # type: ignore[index]
                    if present[m]
                    else torch.zeros(self.image_shape)
                )
            elif m is Modality.TEXT:
                inputs[m.value] = str(self.view.text[index]) if present[m] else ""  # type: ignore[index]
        inputs[PRESENT_KEY] = {m.value: ok for m, ok in present.items()}
        return {"inputs": inputs, "y": int(self.view.y[index])}


def make_collate(
    tokenize: Callable[[Sequence[str]], dict[str, torch.Tensor]] | None = None,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Collate: stack tensors, tokenise text, and stack ``_present`` flags to BoolTensors."""

    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for key in items[0]["inputs"]:
            values = [it["inputs"][key] for it in items]
            if key == PRESENT_KEY:
                inputs[key] = {
                    m: torch.tensor([v[m] for v in values], dtype=torch.bool) for m in values[0]
                }
            elif key == Modality.TEXT.value:
                if tokenize is None:
                    raise ValueError("Text inputs require a tokenizer.")
                inputs[key] = tokenize(values)
            else:
                inputs[key] = torch.stack(values)
        return {"inputs": inputs, "y": torch.tensor([it["y"] for it in items], dtype=torch.long)}

    return collate
