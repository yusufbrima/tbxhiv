"""Generic component registry.

Every swappable family of components (data sources, splitters, models, metrics,
explainers, report writers) owns a :class:`Registry`. Implementations register
themselves with a decorator and can then be instantiated by name from YAML
configs or the CLI::

    MODELS = Registry[Classifier]("model")

    @MODELS.register("logistic_regression")
    class LogisticRegressionClassifier(Classifier):
        ...

    model = MODELS.build("logistic_regression", C=0.5)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, ClassVar, Generic, TypeVar

T = TypeVar("T")


class RegistryError(LookupError):
    """Raised on unknown or duplicate registry keys."""


class Registry(Generic[T]):
    """Case-insensitive mapping from names to component classes.

    Args:
        kind: Human-readable name of the component family (e.g. ``"model"``).
            Used in error messages and as the key in :meth:`all`.
    """

    _index: ClassVar[dict[str, Registry[Any]]] = {}

    def __init__(self, kind: str) -> None:
        if kind in Registry._index:
            raise RegistryError(f"A registry for '{kind}' already exists.")
        self.kind = kind
        self._items: dict[str, type[T]] = {}
        Registry._index[kind] = self

    def register(
        self, name: str | None = None, *, override: bool = False
    ) -> Callable[[type[T]], type[T]]:
        """Return a class decorator that registers the class under ``name``.

        Args:
            name: Registry key. Defaults to the lower-cased class name.
            override: Allow replacing an existing entry (useful in notebooks).

        Returns:
            The decorator; it returns the class unchanged.
        """

        def decorator(cls: type[T]) -> type[T]:
            key = (name or cls.__name__).lower()
            if key in self._items and not override:
                existing = self._items[key].__qualname__
                raise RegistryError(f"{self.kind} '{key}' is already registered ({existing}).")
            self._items[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Look up a registered class by name.

        Raises:
            RegistryError: If ``name`` is not registered.
        """
        try:
            return self._items[name.lower()]
        except KeyError:
            available = ", ".join(self.names()) or "<none>"
            raise RegistryError(f"Unknown {self.kind} '{name}'. Available: {available}") from None

    def build(self, name: str, /, **kwargs: Any) -> T:
        """Instantiate the component registered under ``name`` with ``kwargs``."""
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        """Sorted list of registered names."""
        return sorted(self._items)

    @classmethod
    def all(cls) -> dict[str, Registry[Any]]:
        """All registries created so far, keyed by ``kind``."""
        return dict(cls._index)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry(kind={self.kind!r}, items={self.names()})"
