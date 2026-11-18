"""Shrink downloaded MIMIC-CXR JPGs in place (to fit large subsets on a small disk).

MIMIC-CXR-JPG images average ~1.9 MB at up to ~3000 px. Models here train at
256–512 px, so re-encoding to ``max_side`` (default 1024) on the long side keeps
generous headroom at ~10 % of the size. The step is idempotent: images already
within ``max_side`` are left untouched, so a resumed download only processes new
files. Writes are atomic (temporary file + rename), so an interruption never leaves
a truncated JPEG behind.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from tbmxai.core.utils import get_logger

logger = get_logger(__name__)

PHYSIONET_FILES = "https://physionet.org/files/"


def url_to_path(url: str, root: Path) -> Path:
    """Local path of a PhysioNet file URL under ``root`` (as written by wget --cut-dirs=1)."""
    if not url.startswith(PHYSIONET_FILES):
        raise ValueError(f"not a PhysioNet files URL: {url}")
    return root / url[len(PHYSIONET_FILES) :]


def shrink_jpeg(path: Path, max_side: int = 1024, quality: int = 95) -> str:
    """Re-encode ``path`` so its long side is at most ``max_side``.

    Returns ``"shrunk"``, ``"kept"`` (already small enough), ``"missing"`` or ``"error"``.
    """
    try:
        with Image.open(path) as im:
            w, h = im.size
            if max(w, h) <= max_side:
                return "kept"
            scale = max_side / max(w, h)
            small = im.convert("L" if im.mode in {"L", "I;16", "I"} else "RGB").resize(
                (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS
            )
        tmp = path.with_name(f".{path.name}.tmp")
        small.save(tmp, "JPEG", quality=quality, optimize=True)
        os.replace(tmp, path)
        return "shrunk"
    except FileNotFoundError:
        return "missing"
    except OSError as err:  # truncated / unreadable: leave it for the next download pass
        logger.warning("Cannot shrink %s: %s", path, err)
        return "error"


def _shrink_one(args: tuple[str, int, int]) -> str:
    path, max_side, quality = args
    return shrink_jpeg(Path(path), max_side, quality)


def shrink_many(
    paths: Iterable[Path], max_side: int = 1024, quality: int = 95, workers: int = 4
) -> dict[str, int]:
    """Shrink many JPGs in parallel; returns counts per outcome."""
    jobs = [(str(p), max_side, quality) for p in paths]
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for outcome in pool.map(_shrink_one, jobs, chunksize=32):
            counts[outcome] = counts.get(outcome, 0) + 1
    return counts


# ------------------------------------------------------- Google Cloud Storage copy


def gcs_token() -> str:
    """OAuth access token of the active ``gcloud`` account (valid ~1 hour)."""
    import subprocess

    return subprocess.run(
        ["gcloud", "auth", "print-access-token"], check=True, capture_output=True, text=True
    ).stdout.strip()


def fetch_from_gcs(
    urls: Iterable[str],
    root: Path,
    bucket: str,
    billing_project: str | None,
    workers: int = 32,
    prefix: str = "mimic-cxr-jpg/2.1.0/",
    retries: int = 3,
) -> dict[str, int]:
    """Copy PhysioNet files from their GCS mirror to the paths wget would use.

    ``https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/p10/…jpg`` →
    ``gs://<bucket>/files/p10/…jpg`` → ``root/mimic-cxr-jpg/2.1.0/files/p10/…jpg``.
    Requester-pays buckets need ``billing_project``. Existing complete JPGs are skipped;
    each file is written to a temporary name and renamed when complete.
    """
    import threading
    import time
    import urllib.error
    import urllib.parse
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    token = {"value": gcs_token(), "time": time.monotonic()}
    lock = threading.Lock()

    def auth() -> str:
        with lock:
            if time.monotonic() - token["time"] > 45 * 60:
                token["value"], token["time"] = gcs_token(), time.monotonic()
            return token["value"]

    def complete(path: Path) -> bool:
        try:
            with path.open("rb") as fh:
                fh.seek(-2, 2)
                return fh.read(2) == b"\xff\xd9"
        except OSError:
            return False

    def one(url: str) -> str:
        path = url_to_path(url, root)
        if complete(path):
            return "present"
        rel = url[len(PHYSIONET_FILES) :]
        if not rel.startswith(prefix):
            raise ValueError(f"{url} is not under {prefix}")
        obj = urllib.parse.quote(rel[len(prefix) :], safe="")
        api = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{obj}?alt=media"
        if billing_project:
            api += f"&userProject={urllib.parse.quote(billing_project)}"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.part")
        for attempt in range(retries):
            try:
                req = urllib.request.Request(api, headers={"Authorization": f"Bearer {auth()}"})
                with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
                os.replace(tmp, path)
                return "copied"
            except urllib.error.HTTPError as err:
                if err.code in (401, 403) and attempt == 0:
                    with lock:
                        token["value"], token["time"] = gcs_token(), time.monotonic()
                    continue
                if err.code == 404:
                    return "not_found"
                if attempt == retries - 1:
                    logger.warning("GCS %s: HTTP %s", rel, err.code)
            except OSError as err:
                if attempt == retries - 1:
                    logger.warning("GCS %s: %s", rel, err)
            time.sleep(2**attempt)
        tmp.unlink(missing_ok=True)
        return "failed"

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for outcome in pool.map(one, list(urls)):
            counts[outcome] = counts.get(outcome, 0) + 1
    return counts
