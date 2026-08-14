"""
Cache / memory-hierarchy modelling.

Public surface:

    CacheConfig            validated geometry + policy
    SetAssociativeCache    the metadata model (direct-mapped == assoc=1)
    CachedDataMemory       a DataMemory drop-in for any preset's `dmem=`
    POLICY_CLASSES         auto-discovered replacement policies
    policy_catalog()       serialisable policy list for the API

Usage:

    from sim.components.memory.cache import CachedDataMemory, CacheConfig
    from sim.isa.riscv.presets.pipeline import build

    dmem = CachedDataMemory(CacheConfig(size_kb=8, assoc=2, policy="lru"))
    cpu = build(program, dmem=dmem)
    ...
    dmem.cache.stats.hit_rate
"""
from .cached_dmem import CachedDataMemory
from .cached_multiport import CachedMultiPortDataMemory
from .model import (
    DEFAULT_ASSOC,
    DEFAULT_HIT_CYCLES,
    DEFAULT_LINE_BYTES,
    DEFAULT_MISS_PENALTY,
    DEFAULT_SIZE_KB,
    DEFAULT_WRITE_POLICY,
    WRITE_POLICIES,
    CacheConfig,
    CacheStats,
    SetAssociativeCache,
)
from .policies import (
    DEFAULT_POLICY,
    POLICIES,
    POLICY_CLASSES,
    ReplacementPolicyBase,
    get_policy,
    policy_catalog,
)

__all__ = [
    "CachedDataMemory",
    "CachedMultiPortDataMemory",
    "CacheConfig",
    "CacheStats",
    "SetAssociativeCache",
    "ReplacementPolicyBase",
    "POLICY_CLASSES",
    "POLICIES",
    "DEFAULT_POLICY",
    "get_policy",
    "policy_catalog",
    "WRITE_POLICIES",
    "DEFAULT_WRITE_POLICY",
    "DEFAULT_SIZE_KB",
    "DEFAULT_LINE_BYTES",
    "DEFAULT_ASSOC",
    "DEFAULT_HIT_CYCLES",
    "DEFAULT_MISS_PENALTY",
]


# ── Timing-mode support matrix ────────────────────────────────────────────
# Models whose presets wire the cache's stall/ready path AND are verified to
# produce exactly `misses x miss_penalty` extra cycles with an unchanged
# architectural result.
#
# Observe-only mode (the default) works on ALL 18 presets.
#
# Deliberately NOT included:
#   configurable  — shares single_cycle's datapath but routes commits through
#                   CycleBudgetController; the cache stall is not wired into
#                   that branch, and enabling timing there SILENTLY CORRUPTS
#                   results (verified: wrong values, zero cycle change).
#   pipeline      — needs a real MEM-stage stall: EX/MEM hold, MEM/WB bubble,
#                   ID/EX *hold* (not the hazard detector's bubble-inserting
#                   stall) and a gated branch redirect. Ports were added and
#                   wired; a program-level divergence remains. See
#                   cached_dmem.py.
#   multicycle / ooo / superscalar — not attempted.
TIMING_MODELS = ("single_cycle",)


def supports_timing(model: str) -> bool:
    """Whether `model` can be run with `timing_enabled=True` cache."""
    return model in TIMING_MODELS


def cached_dmem_for(model: str, config=None, timing_enabled: bool = False,
                    num_lanes: int = 2, **kw):
    """Build the right cache component for `model`, refusing unsafe combinations.

    Raises rather than silently returning a component whose stall output no
    preset consumes — that combination produces wrong ARCHITECTURAL results,
    not merely wrong cycle counts, so it must never be reachable by accident.
    """
    if timing_enabled and not supports_timing(model):
        raise ValueError(
            f"timing_enabled cache is not supported on the {model!r} model "
            f"(supported: {list(TIMING_MODELS)}). Its preset does not wire the "
            f"cache stall path, so enabling timing would corrupt results. Use "
            f"observe-only (timing_enabled=False), which works on all presets."
        )
    if model == "superscalar":
        return CachedMultiPortDataMemory(num_lanes=num_lanes, config=config, **kw)
    return CachedDataMemory(config, timing_enabled=timing_enabled, **kw)


__all__ += ["TIMING_MODELS", "supports_timing", "cached_dmem_for"]
