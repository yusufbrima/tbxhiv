"""Shared experimental protocol: identical samples, splits and explanation sets.

Comparing models (EHR-only vs CXR-only vs multimodal, or different fusion
methods) is only meaningful on the **same samples**. Each experiment's view can
differ — e.g. a CXR experiment that drops samples without an image — so a
protocol is built once from *all* configs to be compared:

1. every config is ingested and harmonised;
2. the **common samples** (present in every view, matched by stable sample id,
   with consistent labels) are kept;
3. the reference config's splitter assigns patients to train/val/test
   (patient-level, leakage-checked);
4. explanation samples are selected from the test split among samples that
   are complete in **every** view, stratified by class.

``samples.csv`` (``sample_id, patient_id, split, y, xai``) + ``protocol.json``
are written to the protocol directory. Experiments reference it via
``protocol.path``; they then train, test, run subgroup analyses and explain on
exactly these samples, and fail loudly if any protocol sample is missing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tbmxai.core.config import ExperimentConfig
from tbmxai.core.utils import get_logger
from tbmxai.data.harmonize import MultimodalView
from tbmxai.data.splits import DataSplits
from tbmxai.explain.selection import select_explanation_samples

logger = get_logger(__name__)

SAMPLES_FILE = "samples.csv"
INFO_FILE = "protocol.json"


class ProtocolError(ValueError):
    """Raised when an experiment is inconsistent with its protocol."""


@dataclass
class Protocol:
    """A fixed set of samples with split assignment and explanation flags."""

    samples: pd.DataFrame
    info: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- io

    @classmethod
    def load(cls, path: str | Path) -> Protocol:
        """Load from a protocol directory (or a ``samples.csv`` path)."""
        p = Path(path).expanduser()
        csv = p / SAMPLES_FILE if p.is_dir() else p
        if not csv.exists():
            raise FileNotFoundError(f"Protocol samples not found: {csv} (run `tbmxai protocol`).")
        samples = pd.read_csv(csv, dtype={"sample_id": str, "patient_id": str, "split": str})
        info_path = csv.parent / INFO_FILE
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        return cls(samples, info)

    def save(self, path: str | Path) -> Path:
        """Write ``samples.csv`` and ``protocol.json`` into directory ``path``."""
        d = Path(path).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        self.samples.to_csv(d / SAMPLES_FILE, index=False)
        (d / INFO_FILE).write_text(json.dumps(self.info, indent=2, default=str))
        return d

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the sample/split/xai assignment (recorded in run outputs)."""
        cols = ["sample_id", "split", "xai"]
        payload = self.samples[cols].sort_values("sample_id").to_csv(index=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    # ---------------------------------------------------------- application

    def restrict(self, view: MultimodalView) -> MultimodalView:
        """Subset ``view`` to the protocol samples (in protocol order).

        Raises:
            ProtocolError: If protocol samples are missing or labels disagree.
        """
        ids = view.sample_ids.astype(str)
        pos = pd.Series(np.arange(len(view)), index=ids)
        if pos.index.duplicated().any():
            raise ProtocolError("Duplicate sample ids in view.")
        wanted = self.samples["sample_id"].astype(str)
        missing = wanted[~wanted.isin(pos.index)]
        if len(missing):
            raise ProtocolError(
                f"{len(missing)} protocol samples are absent from this experiment's view "
                f"(e.g. {missing.head(3).tolist()}). The experiment must contain every protocol "
                "sample; check sources, view.modalities and view.missing_modalities."
            )
        sub = view.subset(pos[wanted].to_numpy())
        if "y" in self.samples and not np.array_equal(sub.y, self.samples["y"].to_numpy()):
            raise ProtocolError(
                "Labels differ from the protocol (different label column/definition?)."
            )
        return sub

    def splits(self, view: MultimodalView) -> DataSplits:
        """Protocol train/val/test assignment for a restricted view."""
        assignment = (
            self.samples.set_index("sample_id")["split"]
            .reindex(view.sample_ids.astype(str))
            .to_numpy()
        )
        parts = {s: view.subset(np.flatnonzero(assignment == s)) for s in ("train", "val", "test")}
        splits = DataSplits(
            train=parts["train"],
            val=parts["val"] if len(parts["val"]) else None,
            test=parts["test"],
        )
        splits.check_patient_disjoint()
        return splits

    def xai_ids(self) -> set[str]:
        """Sample ids flagged for explanation."""
        return set(self.samples.loc[self.samples["xai"].astype(bool), "sample_id"].astype(str))


def build_protocol(
    configs: Sequence[ExperimentConfig], reference: int = 0, n_per_class: int | None = None
) -> Protocol:
    """Build a protocol shared by ``configs`` (see module docstring).

    Args:
        configs: Experiment configs to be compared (≥ 1).
        reference: Index of the config whose splitter/seed define the split.
        n_per_class: Explanation samples per class (default: reference ``explain.samples``).
    """
    from tbmxai.pipeline import Pipeline

    if not configs:
        raise ValueError("At least one config is required.")
    views = []
    for cfg in configs:
        pipe = Pipeline(
            cfg.model_copy(update={"protocol": cfg.protocol.model_copy(update={"path": None})})
        )
        views.append(pipe.harmonize(pipe.ingest()))
        logger.info("Protocol: '%s' → %d samples", cfg.name, len(views[-1]))

    id_sets = [set(v.sample_ids.astype(str)) for v in views]
    common = set.intersection(*id_sets)
    if not common:
        raise ProtocolError("The configs share no samples (do they use the same sources/keys?).")
    ref_cfg, ref_view = configs[reference], views[reference]
    ref_ids = ref_view.sample_ids.astype(str)
    keep = np.flatnonzero(np.isin(ref_ids, list(common)))
    view = ref_view.subset(keep)

    labels = pd.Series(view.y, index=view.sample_ids.astype(str))
    for cfg, v in zip(configs, views, strict=True):
        other = pd.Series(v.y, index=v.sample_ids.astype(str)).reindex(labels.index)
        if not np.array_equal(other.to_numpy(), labels.to_numpy()):
            raise ProtocolError(f"Labels of '{cfg.name}' disagree with the reference config.")

    complete = np.ones(len(view), dtype=bool)
    for v in views:
        idx = pd.Series(np.arange(len(v)), index=v.sample_ids.astype(str)).reindex(
            view.sample_ids.astype(str)
        )
        full = (
            np.logical_and.reduce([m for m in v.availability.values()])
            if v.availability
            else np.ones(len(v), bool)
        )
        complete &= full[idx.to_numpy()]

    pipe = Pipeline(ref_cfg)
    splits = pipe.split(view)
    assignment = pd.Series("", index=view.sample_ids.astype(str))
    for name, part in splits.items():
        assignment[part.sample_ids.astype(str)] = name

    test = splits.test
    test_complete = pd.Series(complete, index=view.sample_ids.astype(str)).reindex(
        test.sample_ids.astype(str)
    )
    xcfg = ref_cfg.explain.samples
    chosen = select_explanation_samples(
        test,
        n_per_class=n_per_class or xcfg.n_per_class,
        seed=ref_cfg.seed if xcfg.seed is None else xcfg.seed,
        eligible=test_complete.to_numpy(),
        require=[],
    )
    xai = set(test.sample_ids[chosen].astype(str))

    samples = pd.DataFrame(
        {
            "sample_id": view.sample_ids.astype(str),
            "patient_id": view.patient_ids.astype(str),
            "split": assignment.to_numpy(),
            "y": view.y,
            "class_name": [view.class_names[k] for k in view.y],
            "xai": [s in xai for s in view.sample_ids.astype(str)],
            "complete_in_all": complete,
        }
    )
    info = {
        "created_utc": datetime.now(UTC).isoformat(),
        "configs": [c.name for c in configs],
        "reference": ref_cfg.name,
        "label_column": ref_cfg.data.label_column,
        "split": ref_cfg.split.model_dump(mode="json"),
        "seed": ref_cfg.seed,
        "n_samples": len(samples),
        "n_patients": int(samples["patient_id"].nunique()),
        "samples_per_config": {c.name: len(v) for c, v in zip(configs, views, strict=True)},
        "dropped_per_config": {
            c.name: len(v) - len(samples) for c, v in zip(configs, views, strict=True)
        },
        "split_sizes": samples["split"].value_counts().to_dict(),
        "xai_per_class": samples[samples["xai"]].groupby("class_name").size().to_dict(),
    }
    protocol = Protocol(samples, info)
    protocol.info["fingerprint"] = protocol.fingerprint
    return protocol
