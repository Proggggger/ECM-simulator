"""
Replacement-policy contract for the cache model.

Mirrors ``sim/components/branch/predictors/base.py``: a small abstract base
that concrete policies subclass, discovered by scanning this package so the
API/UI can offer them without a hardcoded list.

A policy owns ONLY the recency/insertion bookkeeping for each set. It never
sees addresses, tags or data — the cache asks it two questions:

    touch(set_idx, way)          this way was just accessed (hit or fill)
    victim(set_idx) -> way       every way is valid; pick one to evict

Choosing an INVALID way on a cold miss is the cache's job, not the policy's,
so a policy is only ever asked for a victim once its set is full. That keeps
every policy small and makes them directly comparable.

Policies must be DETERMINISTIC: the same access sequence must always produce
the same eviction sequence, so cycle counts are reproducible and testable.
``RandomReplacement`` therefore uses a seeded PRNG rather than ``random``.
"""
from __future__ import annotations


class ReplacementPolicyBase:
    """Per-set replacement bookkeeping for an ``assoc``-way cache."""

    #: Short stable identifier used by the API/UI and by CacheConfig(policy=...)
    name: str = ""
    #: Human-readable label for the UI
    label: str = ""
    #: One-line explanation shown in the UI
    description: str = ""

    def __init__(self, num_sets: int, assoc: int):
        if num_sets < 1 or assoc < 1:
            raise ValueError("num_sets and assoc must both be >= 1")
        self.num_sets = num_sets
        self.assoc = assoc
        self.reset()

    # ── contract ──────────────────────────────────────────────────────
    def reset(self) -> None:
        """Clear all bookkeeping (called on construction and cache flush)."""
        raise NotImplementedError

    def touch(self, set_idx: int, way: int) -> None:
        """Record an access to ``way`` in ``set_idx`` (a hit or a fill)."""
        raise NotImplementedError

    def victim(self, set_idx: int) -> int:
        """Return the way to evict from a FULL set."""
        raise NotImplementedError

    # ── introspection (for the UI's per-set view) ─────────────────────
    def age_of(self, set_idx: int, way: int) -> int:
        """Rank of ``way`` in ``set_idx``: 0 = next victim, higher = safer.

        Purely informational — the UI draws it so a user can see *why* the
        next eviction picks what it picks. Policies without a meaningful total
        order may return 0 for every way.
        """
        return 0

    def __repr__(self) -> str:                      # pragma: no cover - debug aid
        return f"<{type(self).__name__} {self.num_sets}x{self.assoc}>"
