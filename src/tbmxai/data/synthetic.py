"""Synthetic multimodal TB dataset for tests, demos and smoke runs.

Produces three linked sources keyed by ``pid`` (deliberately *not* the
canonical ``patient_id`` so the key-renaming path is exercised):

* ``ehr.csv`` — demographics, symptoms, labs and the ``tb_status`` label;
* ``cxr_manifest.csv`` + ``images/`` — 1–2 small PNG "radiographs" per patient,
  with a brighter upper-zone blob for TB-positive patients;
* ``reports.csv`` — templated radiology reports.

The signal is intentionally learnable but noisy. Nothing here is clinically
meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticPaths:
    """Locations of the generated files."""

    root: Path
    ehr: Path
    manifest: Path
    reports: Path
    images: Path


def make_synthetic_dataset(
    out_dir: str | Path,
    n_patients: int = 300,
    prevalence: float = 0.3,
    image_size: int = 64,
    seed: int = 0,
) -> SyntheticPaths:
    """Write a synthetic multimodal dataset to ``out_dir``.

    Args:
        out_dir: Output directory (created if needed).
        n_patients: Number of patients.
        prevalence: Fraction of TB-positive patients.
        image_size: Side length of the square grayscale PNGs.
        seed: RNG seed.

    Returns:
        Paths to the generated files.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    root = Path(out_dir)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    ids = [f"P{i:05d}" for i in range(n_patients)]
    tb = rng.random(n_patients) < prevalence
    hiv = rng.random(n_patients) < np.where(tb, 0.4, 0.15)
    ehr = pd.DataFrame(
        {
            "pid": ids,
            "age": rng.normal(38, 12, n_patients).clip(16, 85).round(),
            "sex": rng.choice(["F", "M"], n_patients),
            "hiv_status": np.where(hiv, "positive", "negative"),
            "cough_weeks": rng.poisson(np.where(tb, 4, 1.5)),
            "weight_loss": rng.random(n_patients) < np.where(tb, 0.6, 0.2),
            "crp": rng.lognormal(np.where(tb, 3.3, 2.4), 0.6).round(1),
            "cd4": np.where(hiv, rng.normal(350, 150, n_patients).clip(20).round(), np.nan),
            "tb_status": np.where(tb, "TB", "non-TB"),
        }
    )

    manifest_rows: list[dict[str, str]] = []
    reports: list[dict[str, str]] = []
    for pid, positive in zip(ids, tb, strict=True):
        for k in range(int(rng.integers(1, 3))):
            img = rng.normal(90, 20, (image_size, image_size))
            if positive:
                margin = max(1, image_size // 8)
                cy = rng.integers(image_size // 6, image_size // 2 + 1)
                cx = rng.integers(margin, image_size - margin)
                yy, xx = np.ogrid[:image_size, :image_size]
                img += 70 * np.exp(
                    -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (image_size / 10) ** 2)
                )
            name = f"{pid}_{k}.png"
            Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(images_dir / name)
            manifest_rows.append({"pid": pid, "path": name, "view": "PA"})
        finding = (
            rng.choice(["Upper lobe opacity with cavitation.", "Patchy consolidation, right apex."])
            if positive and rng.random() < 0.8
            else rng.choice(["Clear lung fields.", "No acute cardiopulmonary abnormality."])
        )
        reports.append({"pid": pid, "report": f"PA chest radiograph. {finding}"})

    paths = SyntheticPaths(
        root=root,
        ehr=root / "ehr.csv",
        manifest=root / "cxr_manifest.csv",
        reports=root / "reports.csv",
        images=images_dir,
    )
    ehr.to_csv(paths.ehr, index=False)
    pd.DataFrame(manifest_rows).to_csv(paths.manifest, index=False)
    pd.DataFrame(reports).to_csv(paths.reports, index=False)
    return paths
