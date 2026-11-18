"""Run provenance (for reproducibility) and loading of finished runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from tbmxai.core.config import ExperimentConfig

_TRACKED_PACKAGES = (
    "tbmxai",
    "torch",
    "torchvision",
    "timm",
    "transformers",
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "pydantic",
)


def _git_info() -> dict[str, Any]:
    """Git SHA / dirty flag of the current working directory, if it is a repo."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        )
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def file_fingerprint(path: str | Path, chunk: int = 1 << 20) -> str | None:
    """SHA-256 of a file (``None`` if missing). Directories are not hashed."""
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def collect_run_info(config: ExperimentConfig) -> dict[str, Any]:
    """Environment, versions, git state, config hash and input-file fingerprints."""
    packages = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            packages[pkg] = version(pkg)
        except PackageNotFoundError:
            packages[pkg] = None
    cuda = None
    try:
        import torch

        cuda = {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:  # pragma: no cover
        pass
    config_json = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    inputs = {
        f"{i}:{src.name}:{key}": {"path": str(val), "sha256": file_fingerprint(val)}
        for i, src in enumerate(config.data.sources)
        for key, val in src.params.items()
        if key == "path"
    }
    if config.data.dataset is not None:
        from tbmxai.data.dataset_format import DATASET_FILE, DatasetSpec

        try:
            spec = DatasetSpec.load(config.data.dataset)
            files = {"dataset.yaml": spec.root / DATASET_FILE, **spec.files()}
            for role, path in files.items():
                inputs[f"dataset:{role}"] = {"path": str(path), "sha256": file_fingerprint(path)}
        except (OSError, ValueError) as exc:
            inputs["dataset:error"] = {"path": str(config.data.dataset), "sha256": str(exc)}
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command": sys.argv,
        "git": _git_info(),
        "packages": packages,
        "cuda": cuda,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "seed": config.seed,
        "inputs": inputs,
    }


def write_manifest(out_dir: Path) -> Path:
    """List every file in the run directory (relative path, size) in ``manifest.json``."""
    files = [
        {"path": str(p.relative_to(out_dir)), "bytes": p.stat().st_size}
        for p in sorted(out_dir.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    ]
    path = out_dir / "manifest.json"
    path.write_text(json.dumps({"run_dir": str(out_dir), "files": files}, indent=2))
    return path


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load a finished run's machine-readable outputs.

    Returns:
        Dict with ``config`` (:class:`ExperimentConfig`), ``run_info`` (dict),
        ``tables`` (name → DataFrame from ``tables/``), ``predictions``
        (split → DataFrame) and ``splits`` (DataFrame).

    Example:
        >>> run = load_run("~/tbmxai_runs/my_exp/20261115-120000")
        >>> sns.pointplot(data=run["tables"]["subgroups"].query("metric == 'auroc'"),
        ...               x="value", y="group", hue="column")
    """
    root = Path(run_dir).expanduser()
    out: dict[str, Any] = {"config": ExperimentConfig.from_yaml(root / "config.yaml")}
    info = root / "run_info.json"
    out["run_info"] = json.loads(info.read_text()) if info.exists() else {}
    tables: dict[str, pd.DataFrame] = {}
    for file in sorted((root / "tables").glob("*")):
        if file.suffix == ".parquet":
            tables[file.stem] = pd.read_parquet(file)
        elif file.suffix == ".csv" and file.stem not in tables:
            tables[file.stem] = pd.read_csv(file)
    out["tables"] = tables
    out["predictions"] = {
        p.stem.removeprefix("predictions_"): pd.read_csv(p) for p in root.glob("predictions_*.csv")
    }
    splits = root / "splits.csv"
    out["splits"] = pd.read_csv(splits) if splits.exists() else None
    return out
