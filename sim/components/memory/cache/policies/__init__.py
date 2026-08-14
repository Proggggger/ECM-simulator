"""
Cache replacement policies, auto-discovered.

Mirrors ``sim/components/branch/predictors``: drop a module in this package
with a ``ReplacementPolicyBase`` subclass that has a ``name``, and it appears
in ``POLICY_CLASSES`` (and therefore in the API/UI) with no registration step.
"""
import importlib
import pkgutil

from .base import ReplacementPolicyBase


def _discover() -> dict:
    """Scan this package -> {name: class}."""
    registry = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name in ("base", "__init__"):
            continue
        mod = importlib.import_module(f"{__name__}.{info.name}")
        for attr in dir(mod):
            cls = getattr(mod, attr)
            if (isinstance(cls, type)
                    and issubclass(cls, ReplacementPolicyBase)
                    and cls is not ReplacementPolicyBase
                    and getattr(cls, "name", "")):
                registry[cls.name] = cls
    return registry


POLICY_CLASSES = _discover()
POLICIES = tuple(sorted(POLICY_CLASSES))
DEFAULT_POLICY = "lru"


def get_policy(name: str):
    """Look up a policy class by name, raising a helpful error if unknown."""
    if not isinstance(name, str) or name not in POLICY_CLASSES:
        raise ValueError(
            f"unknown replacement policy {name!r}; available: {list(POLICIES)}")
    return POLICY_CLASSES[name]


def policy_catalog() -> list:
    """Serialisable description of every policy (for GET /cache/policies)."""
    return [
        {
            "name": cls.name,
            "label": cls.label or cls.name.upper(),
            "description": cls.description,
        }
        for _n, cls in sorted(POLICY_CLASSES.items())
    ]


__all__ = ["ReplacementPolicyBase", "POLICY_CLASSES", "POLICIES",
           "DEFAULT_POLICY", "get_policy", "policy_catalog"]
