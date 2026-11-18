"""Neural classifier: any combination of encoders + fusion, trained by :class:`TorchTrainer`.

Configured entirely from YAML, e.g. CXR + EHR with cross-attention::

    model:
      name: neural
      params:
        encoders:
          imaging: {name: timm_vision, params: {backbone: convnext_tiny, pretrained: true}}
          tabular: {name: ft_transformer, params: {d_token: 64}}
        fusion: {name: cross_attention, params: {query: tabular, dim: 256}}
        head: {hidden_dims: [128], dropout: 0.2}
        image: {size: 224, augment: true}
    training: {...}   # top-level TrainingConfig section

A single encoder (and no fusion) gives a unimodal network.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tbmxai.core.config import ComponentConfig, TrainingConfig
from tbmxai.core.types import Modality
from tbmxai.core.utils import get_logger
from tbmxai.data.harmonize import MultimodalView
from tbmxai.data.image_cache import ImageCache
from tbmxai.data.preprocessing import build_tabular_preprocessor
from tbmxai.data.torch_datasets import MultimodalTorchDataset, build_image_transform, make_collate
from tbmxai.models.base import MODELS, Classifier

logger = get_logger(__name__)


def _as_component(spec: ComponentConfig | Mapping[str, Any]) -> ComponentConfig:
    return spec if isinstance(spec, ComponentConfig) else ComponentConfig(**spec)


@MODELS.register("neural")
class NeuralClassifier(Classifier):
    """Encoders (vision / tabular / text) → fusion → head, trained end-to-end.

    Handles missing modalities: absent inputs are replaced by learned "missing"
    embeddings and fusion operators renormalise over available modalities, so
    the same model serves patients with CXR+EHR, CXR only or EHR only.

    Args:
        encoders: Modality → encoder config. Keys define the required modalities.
        fusion: Fusion config (``concat``, ``gated``, ``bilinear``, ``attention``,
            ``cross_attention``, ``late``). Ignored for a single encoder.
        head: ``{hidden_dims: [...], dropout: float}``.
        image: Image pipeline: ``size``, ``augment``, ``in_chans``, ``mean``, ``std``,
            and optional ``cache_dir`` / ``cache_size`` (decode-once image cache;
            ``cache_size`` = shorter side, must be ≥ ``size``).
        modality_dropout: Training-time probability of hiding each modality per
            sample, e.g. ``{imaging: 0.3, tabular: 0.2}``. Teaches the network to
            predict when modalities are missing at inference (always ≥1 kept).
        training: Training config (the pipeline overrides it with the
            experiment's top-level ``training`` section via :meth:`setup`).
    """

    def __init__(
        self,
        encoders: Mapping[str, ComponentConfig | Mapping[str, Any]],
        fusion: ComponentConfig | Mapping[str, Any] | None = None,
        head: Mapping[str, Any] | None = None,
        image: Mapping[str, Any] | None = None,
        modality_dropout: Mapping[str, float] | None = None,
        training: TrainingConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.supports_missing_modalities = True
        if not encoders:
            raise ValueError("At least one encoder is required.")
        self.encoder_specs = {Modality(m).value: _as_component(s) for m, s in encoders.items()}
        self.required_modalities = frozenset(Modality(m) for m in self.encoder_specs)
        self.fusion_spec = None if fusion is None else _as_component(fusion)
        self.head = dict(head or {})
        self.image = {
            "size": 224,
            "augment": True,
            "in_chans": 1,
            "mean": 0.5,
            "std": 0.25,
            **(image or {}),
        }
        self.modality_dropout = {
            Modality(k).value: float(v) for k, v in (modality_dropout or {}).items()
        }
        unknown = set(self.modality_dropout) - set(self.encoder_specs)
        if unknown:
            raise ValueError(f"modality_dropout for modalities without encoder: {sorted(unknown)}")
        self.training = (
            training if isinstance(training, TrainingConfig) else TrainingConfig(**(training or {}))
        )
        self.work_dir: Path | None = None
        self.preprocessor_: Any = None
        self.network_: torch.nn.Module | None = None
        self.training_result_: Any = None
        #: Optional ``callback(epoch, logs)`` forwarded to the trainer (used for HPO pruning).
        self.epoch_callback: Any = None

    def setup(self, run_dir: Path, training: TrainingConfig | None = None) -> None:
        """Receive the run directory and experiment-level training config."""
        self.work_dir = Path(run_dir)
        if training is not None:
            self.training = training

    # ------------------------------------------------------------ helpers

    def _tabular_matrix(self, view: MultimodalView) -> np.ndarray | None:
        if Modality.TABULAR not in self.required_modalities:
            return None
        return np.asarray(self.preprocessor_.transform(view.tabular), dtype=np.float32)

    def _loader(self, view: MultimodalView, train: bool) -> DataLoader:
        img = self.image
        transform = build_image_transform(
            img["size"],
            train=train,
            augment=img["augment"],
            in_chans=img["in_chans"],
            mean=img["mean"],
            std=img["std"],
        )
        dataset = MultimodalTorchDataset(
            view,
            sorted(self.required_modalities),
            self._tabular_matrix(view),
            transform,
            image_shape=(img["in_chans"], img["size"], img["size"]),
            modality_dropout=self.modality_dropout if train else None,
            image_cache=self._image_cache(),
        )
        tokenize = None
        if Modality.TEXT in self.required_modalities:
            tokenize = self.network_.encoders[Modality.TEXT.value].tokenize  # type: ignore[union-attr,index]
        cfg = self.training
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size if train else (cfg.eval_batch_size or cfg.batch_size),
            shuffle=train,
            drop_last=train and len(dataset) > cfg.batch_size,
            num_workers=cfg.num_workers,
            collate_fn=make_collate(tokenize),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=cfg.num_workers > 0,
        )

    def _image_cache(self) -> ImageCache | None:
        img = self.image
        if not img.get("cache_dir"):
            return None
        size = int(img.get("cache_size") or max(512, img["size"]))
        if size < img["size"]:
            raise ValueError(f"image.cache_size ({size}) must be >= image.size ({img['size']})")
        return ImageCache(img["cache_dir"], size)

    def _class_weights(self, y: np.ndarray) -> torch.Tensor | None:
        spec = self.training.class_weights
        if spec is None:
            return None
        if spec == "balanced":
            counts = np.bincount(y, minlength=self.n_classes_ or 0).astype(float)
            weights = len(y) / (len(counts) * np.maximum(counts, 1.0))
            return torch.tensor(weights, dtype=torch.float32)
        return torch.tensor(spec, dtype=torch.float32)

    # --------------------------------------------------------- Classifier API

    def _fit(self, train: MultimodalView, val: MultimodalView | None) -> None:
        from tbmxai.models.nn.network import build_network
        from tbmxai.training.trainer import TorchTrainer

        overrides: dict[str, dict[str, Any]] = {}
        if Modality.TABULAR in self.required_modalities:
            self.preprocessor_ = build_tabular_preprocessor(train.tabular).fit(train.tabular)  # type: ignore[arg-type]
            overrides[Modality.TABULAR.value] = {
                "in_features": self._tabular_matrix(train).shape[1]
            }  # type: ignore[union-attr]
        if Modality.IMAGING in self.required_modalities:
            overrides[Modality.IMAGING.value] = {"in_chans": self.image["in_chans"]}

        self.network_ = build_network(
            self.encoder_specs, int(self.n_classes_ or 2), self.fusion_spec, self.head, overrides
        )
        if self.work_dir is None:
            self.work_dir = Path(tempfile.mkdtemp(prefix="tbmxai-"))
            logger.warning("No run directory set; training artefacts go to %s", self.work_dir)
        trainer = TorchTrainer(self.training, self.work_dir)
        self.training_result_ = trainer.fit(
            self.network_,
            self._loader(train, train=True),
            self._loader(val, train=False) if val is not None else None,
            class_weights=self._class_weights(train.y),
            epoch_callback=self.epoch_callback,
        )

    def _predict_proba(self, view: MultimodalView) -> np.ndarray:
        from tbmxai.training.trainer import TorchTrainer

        trainer = TorchTrainer(self.training, self.work_dir or Path(tempfile.gettempdir()))
        logits = trainer.predict_logits(self.network_, self._loader(view, train=False))  # type: ignore[arg-type]
        return torch.softmax(torch.from_numpy(logits), -1).numpy()

    def save(self, path: str | Path) -> Path:
        """Pickle with the network moved to CPU (loadable on CPU-only machines)."""
        if self.network_ is not None:
            self.network_.cpu()
        callback, self.epoch_callback = self.epoch_callback, None  # closures are not picklable
        try:
            return super().save(path)
        finally:
            self.epoch_callback = callback
