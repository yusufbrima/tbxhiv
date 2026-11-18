"""Persist per-sample explanation artefacts in reusable formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tbmxai.explain.base import Explanation


def save_explanations(explanations: list[Explanation], out_dir: Path) -> list[Path]:
    """Write local explanations under ``out_dir/explanations/``.

    * imaging saliency → ``<method>.npz`` (``maps`` (n, target class, H, W), ``images``,
      ids, labels, probabilities, outcome category) + ``<method>_index.csv``;
    * tabular SHAP → ``shap_values.csv`` (sample × raw feature) and
      ``shap_feature_values.csv`` (the raw inputs, for beeswarm/dependence plots).
    """
    root = out_dir / "explanations"
    written: list[Path] = []
    for exp in explanations:
        if exp.scope != "local":
            continue
        root.mkdir(parents=True, exist_ok=True)
        if isinstance(exp.attributions, np.ndarray) and exp.attributions.ndim in (3, 4):
            extra = exp.extra
            npz = root / f"{exp.method}.npz"
            np.savez_compressed(
                npz,
                maps=exp.attributions.astype(np.float16),  # (n, n_target_classes, H, W)
                target_classes=np.asarray(extra.get("target_classes", [])),
                class_names=np.asarray(extra.get("class_names", [])).astype(str),
                images=np.asarray(extra.get("images"), dtype=np.float16),
                sample_ids=np.asarray(exp.sample_ids),
                patient_ids=np.asarray(extra.get("patient_ids")).astype(str),
                y_true=np.asarray(extra.get("y_true")),
                prob=np.asarray(extra.get("prob")),
                category=np.asarray(extra.get("category")).astype(str),
            )
            index = pd.DataFrame(
                {
                    "row": np.arange(len(exp.attributions)),
                    "sample_id": exp.sample_ids,
                    "patient_id": extra.get("patient_ids"),
                    "y_true": extra.get("y_true"),
                    "prob": extra.get("prob"),
                    "category": extra.get("category"),
                }
            )
            index.to_csv(root / f"{exp.method}_index.csv", index=False)
            written += [npz, root / f"{exp.method}_index.csv"]
        elif isinstance(exp.attributions, pd.DataFrame):
            values = root / f"{exp.method}_values.csv"
            exp.attributions.to_csv(values)
            written.append(values)
            if isinstance(exp.extra.get("feature_values"), pd.DataFrame):
                fv = root / f"{exp.method}_feature_values.csv"
                exp.extra["feature_values"].to_csv(fv)
                written.append(fv)
    return written
