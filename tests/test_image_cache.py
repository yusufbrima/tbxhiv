"""Image cache: decoded, resized images are stored once and read back identically."""

import numpy as np

from tbmxai.data.image_cache import ImageCache


def test_image_cache_roundtrip(synthetic, tmp_path) -> None:
    cache = ImageCache(tmp_path / "c", size=16)
    path = next(synthetic.images.glob("*.png"))
    first = cache.load(path)
    assert cache.key(path).exists() and min(first.size) <= 16
    again = cache.load(path)
    assert np.array_equal(np.asarray(first), np.asarray(again))
    stats = cache.build(list(synthetic.images.glob("*.png"))[:10], workers=2)
    assert stats["failed"] == 0
