"""Deterministic, model-independent choice of samples to explain.

Explanations must be comparable across models and modalities (EHR-only,
CXR-only, multimodal), so the samples are chosen from **ground truth only**:

* ``n_per_class`` samples of **every** class (all target classes represented);
* one sample per patient (no patient dominates the gallery);
* only samples that have every required modality (so the same patients can be
  explained by EHR, CXR and multimodal models);
* patients ordered by a seeded hash of their id — the choice does not depend
  on row order, on the model, or on which other samples are in the view.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from tbmxai.core.types import Modality
from tbmxai.data.harmonize import MultimodalView


def _rank(patient: object, seed: int) -> str:
    return hashlib.sha1(f"{seed}|{patient}".encode()).hexdigest()


def select_explanation_samples(
    view: MultimodalView,
    n_per_class: int = 8,
    seed: int = 0,
    eligible: np.ndarray | None = None,
    require: Sequence[Modality] | None = None,
) -> np.ndarray:
    """Positional indices of the samples to explain (sorted).

    Args:
        view: Candidate samples (typically the test split).
        n_per_class: Samples per class (fewer if a class has fewer patients).
        seed: Seed of the patient ordering.
        eligible: Optional extra boolean mask (e.g. complete in every compared view).
        require: Modalities a sample must have (default: all selected modalities).
    """
    mods = list(require) if require is not None else list(view.availability)
    ok = np.ones(len(view), dtype=bool) if eligible is None else np.asarray(eligible, bool).copy()
    for m in mods:
        ok &= view.availability.get(m, np.ones(len(view), bool))
    frame = pd.DataFrame(
        {
            "pos": np.arange(len(view)),
            "patient": view.patient_ids.astype(str),
            "sample": view.sample_ids.astype(str),
            "y": view.y,
        }
    )[ok]
    chosen: list[int] = []
    for k in range(view.n_classes):
        cls = frame[frame["y"] == k].sort_values("sample").drop_duplicates("patient")
        cls = cls.assign(rank=cls["patient"].map(lambda p: _rank(p, seed))).sort_values("rank")
        chosen += cls["pos"].head(n_per_class).tolist()
    return np.sort(np.asarray(chosen, dtype=int))
