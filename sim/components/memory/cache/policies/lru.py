"""Least-Recently-Used replacement."""
from .base import ReplacementPolicyBase


class LRUReplacement(ReplacementPolicyBase):
    """Evict the way whose last access is furthest in the past.

    Each set keeps a list of ways in access order, most-recent LAST. A hit or
    fill moves the way to the back; the victim is the front.

    LRU is optimal for stack-like reuse and *pathological* for a cyclic sweep
    over ``assoc + 1`` lines that map to one set — every access misses, because
    the line about to be reused was just chosen as the victim. That failure
    mode is worth showing in the UI, so it is pinned by a test rather than
    hidden.
    """
    name = "lru"
    label = "LRU"
    description = "Evict the least recently used line"

    def reset(self):
        # order[s][0] is the next victim, order[s][-1] the most recent.
        self._order = [list(range(self.assoc)) for _ in range(self.num_sets)]

    def touch(self, set_idx, way):
        order = self._order[set_idx]
        order.remove(way)
        order.append(way)

    def victim(self, set_idx):
        return self._order[set_idx][0]

    def age_of(self, set_idx, way):
        # 0 = next victim, assoc-1 = most recently used.
        return self._order[set_idx].index(way)
