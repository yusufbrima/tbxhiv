"""One interface for turning raw data into a tbmxai standard dataset.

Every dataset — however it is shipped — becomes the same thing: a folder in the
:mod:`standard format <tbmxai.data.dataset_format>` (``dataset.yaml`` + one table
per modality, linked by ``patient_id``). There are two ways to describe how:

1. **A recipe** (``recipe.yaml``, no code): which raw files are samples, how ids,
   labels and features are read from file names, folders, side-car files and
   tables, and which columns go into which table. See :mod:`tbmxai.datasets.recipe`
   and ``configs/datasets/`` for examples. Most public datasets fit a recipe.
2. **A Python adapter**: a :class:`DatasetAdapter` subclass registered in
   :data:`ADAPTERS`, for logic a recipe should not express (e.g. MIMIC's
   time-windowed EHR features). Adapters can also be called by name.

Both run through ``tbmxai dataset build SOURCE RAW OUT``, where SOURCE is a recipe
file or an adapter name, and both end with the same validation and a
``build_info.json`` provenance record.

Small parsing functions that recipes call by name (``parse: nlm_reading``) live in
:data:`PARSERS`; register your own with :func:`register_parser`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tbmxai.core.registry import Registry

#: Parser name → function(str) → scalar or dict of fields.
PARSERS: dict[str, Callable[[str], Any]] = {}


def register_parser(name: str) -> Callable[[Callable[[str], Any]], Callable[[str], Any]]:
    """Register a text parser usable from recipes as ``parse: <name>``.

    A parser receives one string and returns either a single value or a dict
    whose keys become new fields (e.g. ``{"sex": "M", "age": 35.0}``).
    """

    def decorator(fn: Callable[[str], Any]) -> Callable[[str], Any]:
        key = name.lower()
        if key in PARSERS and PARSERS[key] is not fn:
            raise ValueError(f"parser '{key}' is already registered")
        PARSERS[key] = fn
        return fn

    return decorator


class DatasetAdapter(ABC):
    """Builds a standard dataset in ``out_dir`` from raw data in ``raw_dir``."""

    #: One-line description shown by ``tbmxai dataset adapters``.
    description: str = ""

    @abstractmethod
    def build(self, raw_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
        """Write the dataset and return a JSON-friendly build summary."""


ADAPTERS: Registry[DatasetAdapter] = Registry("dataset_adapter")


@ADAPTERS.register("recipe")
class RecipeAdapter(DatasetAdapter):
    """A declarative recipe file (see :mod:`tbmxai.datasets.recipe`).

    Args:
        recipe: Path to the recipe YAML.
    """

    description = "Declarative recipe (recipe.yaml): files/tables → standard dataset"

    def __init__(self, recipe: str | Path) -> None:
        from tbmxai.datasets.recipe import Recipe

        self.recipe = Recipe.load(recipe)

    def build(self, raw_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
        return self.recipe.build(raw_dir, out_dir)


@ADAPTERS.register("nlm_tb_legacy")
class NLMTBAdapter(DatasetAdapter):
    """The original Python converter for Shenzhen + Montgomery (reference implementation).

    ``configs/datasets/nlm_tb.yaml`` produces identical output; this adapter is kept
    so the equivalence stays tested.
    """

    description = "Shenzhen + Montgomery (NLM) TB sets — Python reference converter"

    def build(self, raw_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
        from tbmxai.datasets.nlm_tb import build_nlm_tb

        return build_nlm_tb(raw_dir, out_dir)


@ADAPTERS.register("mimic")
class MimicAdapter(DatasetAdapter):
    """MIMIC-IV + MIMIC-CXR builder (needs ``tbmxai mimic select`` and the images first).

    Raw locations and options come from the MIMIC build config, not ``raw_dir``.

    Args:
        config: MIMIC build config (e.g. ``configs/mimic/build.yaml``).
        memory_limit: DuckDB memory limit, e.g. ``"10GB"``.
    """

    description = "MIMIC-IV + MIMIC-CXR subset with leakage-safe EHR features (Python)"

    def __init__(self, config: str | Path, memory_limit: str | None = None) -> None:
        self.config = Path(config)
        self.memory_limit = memory_limit

    def build(self, raw_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
        from tbmxai.datasets.mimic import MimicBuildConfig, MimicBuilder

        cfg = MimicBuildConfig.from_yaml(self.config)
        want = cfg.output_dir.expanduser().resolve()
        if out_dir and Path(out_dir).expanduser().resolve() != want:
            raise ValueError(
                f"MIMIC writes to the config's output_dir ({want}), next to its selection; "
                "change output_dir in the config instead of passing another OUT."
            )
        return MimicBuilder(cfg, memory_limit=self.memory_limit).build()


# ------------------------------------------------------------- built-in parsers


@register_parser("nlm_reading")
def _nlm_reading(text: str) -> dict[str, Any]:
    """Sex, age and finding text of a Shenzhen / Montgomery clinical reading."""
    from tbmxai.datasets.nlm_tb import parse_reading

    return parse_reading(text)


def resolve_adapter(source: str, **params: Any) -> DatasetAdapter:
    """A recipe path (``*.yaml``/``*.yml`` file) or a registered adapter name → adapter."""
    path = Path(source).expanduser()
    if path.suffix.lower() in {".yaml", ".yml"}:
        if not path.exists():
            raise FileNotFoundError(f"Recipe {path} not found")
        return RecipeAdapter(path, **params)
    return ADAPTERS.build(source, **params)
