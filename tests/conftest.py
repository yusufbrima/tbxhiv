"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from tbmxai.data.synthetic import SyntheticPaths, make_synthetic_dataset


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticPaths:
    """A small synthetic multimodal dataset shared across tests."""
    root: Path = tmp_path_factory.mktemp("synthetic")
    return make_synthetic_dataset(root, n_patients=200, image_size=16, seed=1)
