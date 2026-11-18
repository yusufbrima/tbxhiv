"""Disk cache of resized grayscale images.

Full-resolution CXRs (≈2,500 × 3,000 px JPEG/DICOM) are expensive to decode;
on a machine with few CPU cores, decoding every image every epoch starves the
GPU. :class:`ImageCache` stores each image once, resized so its *shorter* side
equals ``size`` (aspect ratio preserved), as an uncompressed uint8 ``.npy`` file
that loads in microseconds. The training transform then resizes to the model
input size as before, so cached and uncached pipelines see the same pixels up
to resampling.

The cache is keyed by the source path, file size and mtime, so replacing an image
invalidates its entry. Entries are written atomically (safe with many DataLoader
workers). Build it ahead of training with ``tbmxai cache-images`` or let it fill
lazily during the first epoch.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from tbmxai.core.utils import get_logger

logger = get_logger(__name__)


class ImageCache:
    """Resized-image cache on local disk.

    Args:
        root: Cache directory (anywhere; one sub-directory per ``size``).
        size: Target length of the shorter image side in pixels.
    """

    def __init__(self, root: str | Path, size: int = 512) -> None:
        if size < 8:
            raise ValueError("size must be >= 8")
        self.size = int(size)
        self.root = Path(root).expanduser() / f"short{self.size}"

    def key(self, source: str | Path) -> Path:
        """Cache file for ``source`` (depends on path, size and mtime)."""
        src = Path(source)
        try:
            st = src.stat()
            stamp = f"{st.st_size}-{int(st.st_mtime)}"
        except OSError:
            stamp = "missing"
        digest = hashlib.sha1(f"{src.resolve()}|{stamp}".encode()).hexdigest()
        return self.root / digest[:2] / f"{digest}.npy"

    def resize(self, image: Image.Image) -> Image.Image:
        """Resize so that the shorter side equals :attr:`size` (never upsamples)."""
        w, h = image.size
        scale = self.size / min(w, h)
        if scale >= 1.0:
            return image
        return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)

    def load(self, source: str | Path, write: bool = True) -> Image.Image:
        """Return the cached image, creating the entry on a miss (if ``write``)."""
        from tbmxai.data.torch_datasets import load_image

        path = self.key(source)
        if path.exists():
            try:
                return Image.fromarray(np.load(path, allow_pickle=False), mode="L")
            except (OSError, ValueError):
                logger.warning("Corrupt cache entry %s; rebuilding", path)
        image = self.resize(load_image(source))
        if write:
            self._write(path, np.asarray(image, dtype=np.uint8))
        return image

    def _write(self, path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with tmp.open("wb") as fh:
            np.save(fh, array, allow_pickle=False)
        os.replace(tmp, path)  # atomic: concurrent workers never see partial files

    def build(self, sources: Iterable[str | Path], workers: int | None = None) -> dict[str, int]:
        """Populate the cache for ``sources`` in parallel; returns counts."""
        todo = [str(s) for s in sources if not self.key(s).exists()]
        stats = {"requested": 0, "built": 0, "failed": 0}
        stats["requested"] = len(todo)
        if not todo:
            return stats
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for ok in pool.map(
                _build_one, [(self.root.parent, self.size, s) for s in todo], chunksize=16
            ):
                stats["built" if ok else "failed"] += 1
        logger.info("Image cache %s: %s", self.root, stats)
        return stats


def _build_one(args: tuple[Path, int, str]) -> bool:
    root, size, source = args
    try:
        ImageCache(root, size).load(source)
        return True
    except Exception as exc:
        logger.warning("Could not cache %s: %s", source, exc)
        return False
