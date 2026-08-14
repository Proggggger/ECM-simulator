"""
``CachedMultiPortDataMemory`` — the superscalar counterpart of
``CachedDataMemory``.

The superscalar presets wire a per-lane memory (``addr_0``, ``wdata_0``,
``mem_read_0``, ``mem_write_0``, ``rdata_0``, ...) rather than the single-port
bus every other model uses, so a single-port cache cannot be dropped in. This
subclass presents ``MultiPortDataMemory``'s exact port contract and feeds the
same cache model from all lanes.

Cache accesses are applied in LANE (program) order, which matters: two lanes
touching the same line in one cycle produce one miss and one hit, not two
misses — exactly as a real multi-ported L1 with a shared tag array would.

The data path is copied from ``MultiPortDataMemory`` verbatim, including
older-to-younger store forwarding within a fetch group and lane-order write
commit, so architectural results are identical to the uncached superscalar.
"""
from __future__ import annotations

from sim.component.base import ComponentBase, Port
from sim.components.memory.dmem import (
    DEFAULT_MEM_WORDS, STATE_WINDOW, memory_snapshot,
)

from .model import CacheConfig, SetAssociativeCache


class CachedMultiPortDataMemory(ComponentBase):
    """N-port data memory fronted by one shared modelled cache."""
    name = "cached_multiport_data_memory"
    ui_label = "Data Memory + Cache (multi-port)"
    ui_category = "memory"

    def __init__(self, num_lanes: int = 2, config: CacheConfig | None = None,
                 size: int = DEFAULT_MEM_WORDS,
                 timing_enabled: bool = False, **kw):
        self.num_lanes = num_lanes
        self.ports_spec = {}
        for i in range(num_lanes):
            self.ports_spec[f"addr_{i}"]      = Port(32, "in",  f"Lane {i} byte address")
            self.ports_spec[f"wdata_{i}"]     = Port(32, "in",  f"Lane {i} write data")
            self.ports_spec[f"mem_read_{i}"]  = Port(1,  "in",  f"Lane {i} is a load")
            self.ports_spec[f"mem_write_{i}"] = Port(1,  "in",  f"Lane {i} is a store")
            self.ports_spec[f"rdata_{i}"]     = Port(32, "out", f"Lane {i} read data")
        self.ports_spec["stall"] = Port(1, "out", "1 while a miss is being serviced")
        self.ports_spec["ready"] = Port(1, "out", "1 on the cycle the group completes")
        super().__init__(**kw)
        self._mem = [0] * size
        self._size = size
        self.cache = SetAssociativeCache(config or CacheConfig())
        self.timing_enabled = bool(timing_enabled)
        self._busy = 0
        self._last = None
        self._accessed_this_cycle = False

    def _word(self, byte_addr: int) -> int:
        return (byte_addr >> 2) % self._size

    # ── combinational (identical to MultiPortDataMemory) ──────────────
    def evaluate(self):
        for i in range(self.num_lanes):
            w = self._word(self[f"addr_{i}"])
            val = self._mem[w]
            # Store->load forwarding from older lanes (j<i) in this group.
            for j in range(i):
                if self[f"mem_write_{j}"] and self._word(self[f"addr_{j}"]) == w:
                    val = self[f"wdata_{j}"]
            self[f"rdata_{i}"] = val & 0xFFFF_FFFF

        if not self.timing_enabled:
            self["stall"] = 0
            self["ready"] = 1
        else:
            self["stall"] = 1 if self._busy > 0 else 0
            self["ready"] = 0 if self._busy > 0 else 1

    # ── clocked ───────────────────────────────────────────────────────
    def rising_edge(self):
        self._accessed_this_cycle = False
        if self.timing_enabled and self._busy > 0:
            self._busy -= 1
            self.cache.stats.stall_cycles += 1
            if self._busy > 0:
                return
            # fall through on the final cycle so the group commits

        # Feed the cache in lane (program) order so a same-line pair in one
        # group costs one miss + one hit, as a shared tag array would.
        results, worst = [], None
        for i in range(self.num_lanes):
            if not (self[f"mem_read_{i}"] or self[f"mem_write_{i}"]):
                continue
            r = self.cache.access(int(self[f"addr_{i}"]),
                                  is_write=bool(self[f"mem_write_{i}"]))
            r["lane"] = i
            results.append(r)
            if worst is None or r["cycles"] > worst["cycles"]:
                worst = r
        if results:
            self._last = results
            self._accessed_this_cycle = True

        if self.timing_enabled and worst is not None and not worst["hit"]:
            penalty = self.cache.config.miss_penalty
            if penalty > 0 and self._busy == 0:
                self._busy = penalty
                return

        # Commit stores in program (lane) order — younger same-word store wins.
        for i in range(self.num_lanes):
            if self[f"mem_write_{i}"]:
                self._mem[self._word(self[f"addr_{i}"])] = self[f"wdata_{i}"]

    # ── snapshot ──────────────────────────────────────────────────────
    def get_state(self):
        dense, hi = memory_snapshot(self._mem, self._size)
        active = next((i for i in range(self.num_lanes)
                       if self[f"mem_read_{i}"] or self[f"mem_write_{i}"]), -1)
        writes = [int(self[f"addr_{i}"]) for i in range(self.num_lanes)
                  if self[f"mem_write_{i}"]]
        focus = self.cache.config.set_of(
            int(self[f"addr_{active}"]) if active >= 0 else 0)
        state = {
            # ── MultiPortDataMemory-compatible keys ──
            "addr": f"0x{(self[f'addr_{active}'] if active >= 0 else 0):08x}",
            "rdata": self[f"rdata_{active}"] if active >= 0 else 0,
            "wdata": self[f"wdata_{active}"] if active >= 0 else 0,
            "wen": 1 if writes else 0,
            "write_addr": writes[-1] if writes else 0,
            "active_lanes": [i for i in range(self.num_lanes)
                             if self[f"mem_read_{i}"] or self[f"mem_write_{i}"]],
            "memory": dense,
            "memory_hi": hi,
            "size": self._size,
            "window": min(self._size, STATE_WINDOW),
            # ── cache additions ──
            "cache": self.cache.to_dict(focus_set=focus),
            "stall": self["stall"],
            "ready": self["ready"],
            "timing_enabled": self.timing_enabled,
        }
        has_access = self._accessed_this_cycle or self._busy > 0
        if self._last:
            state["last_access"] = [
                {"lane": r["lane"], "hit": r["hit"], "set": r["set"],
                 "way": r["way"], "evicted": r["evicted"],
                 "wrote_back": r["wrote_back"]}
                for r in self._last
            ]
        if has_access and self._last:
            state["cache_result"] = ("HIT" if all(r["hit"] for r in self._last)
                                     else "MISS")
        else:
            state["cache_result"] = "IDLE"
        return state
