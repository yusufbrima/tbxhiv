"""Dataset adapters that turn public datasets into tbmxai-ready tables.

Adapters are imported lazily (``import tbmxai.datasets.mimic``) so their
dependencies are only loaded when used.
"""

from tbmxai.datasets.adapters import (
    ADAPTERS,
    PARSERS,
    DatasetAdapter,
    register_parser,
    resolve_adapter,
)

__all__ = ["ADAPTERS", "PARSERS", "DatasetAdapter", "register_parser", "resolve_adapter"]
