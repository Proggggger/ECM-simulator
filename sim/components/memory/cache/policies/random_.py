"""Deterministic pseudo-random replacement."""
from .base import ReplacementPolicyBase


class RandomReplacement(ReplacementPolicyBase):
    """Pick a victim with a seeded PRNG.

    Real hardware uses random replacement because it needs no per-line state
    and cannot be driven into LRU's cyclic-sweep worst case. It is included
    mainly so the UI can show that "dumber" is sometimes *better*.

    The PRNG is a seeded xorshift rather than ``random`` so a given access
    sequence always yields the same evictions — cycle counts must be
    reproducible for tests and for the /compare grid.
    """
    name = "random"
    label = "Random"
    description = "Evict a pseudo-random way (seeded, reproducible)"

    SEED = 0x2545F491

    def reset(self):
        self._state = self.SEED

    def touch(self, set_idx, way):
        pass                      # random replacement keeps no history

    def _next(self) -> int:
        # xorshift32 — cheap, deterministic, adequate for victim selection.
        x = self._state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self._state = x & 0xFFFFFFFF
        return self._state

    def victim(self, set_idx):
        return self._next() % self.assoc
