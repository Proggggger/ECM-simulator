"""First-In-First-Out (round-robin) replacement."""
from .base import ReplacementPolicyBase


class FIFOReplacement(ReplacementPolicyBase):
    """Evict the way that was FILLED longest ago, regardless of later hits.

    The distinction from LRU is the whole point: FIFO ignores hits. A line
    that is filled once and then hit repeatedly still ages out on schedule,
    so a workload that keeps re-touching one hot line performs worse under
    FIFO than under LRU. ``touch`` is deliberately a no-op for hits.
    """
    name = "fifo"
    label = "FIFO"
    description = "Evict the oldest fill, ignoring later hits"

    def reset(self):
        self._order = [list(range(self.assoc)) for _ in range(self.num_sets)]
        # A way is "filled" the first time it is touched while invalid; the
        # cache tells us via touch(), so track which ways have been filled to
        # distinguish the initial cold sequence from later hits.
        self._filled = [set() for _ in range(self.num_sets)]

    def touch(self, set_idx, way):
        # Only a FILL reorders a FIFO queue. The cache calls touch() for both
        # hits and fills, so treat the FIRST touch of a way as its fill and
        # ignore every subsequent hit.
        if way in self._filled[set_idx]:
            return
        self._filled[set_idx].add(way)
        order = self._order[set_idx]
        order.remove(way)
        order.append(way)

    def victim(self, set_idx):
        way = self._order[set_idx][0]
        # This way is about to be refilled, so its next touch is a fill.
        self._filled[set_idx].discard(way)
        return way

    def age_of(self, set_idx, way):
        return self._order[set_idx].index(way)
