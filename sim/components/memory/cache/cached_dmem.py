"""
``CachedDataMemory`` — a cache-modelling drop-in for ``DataMemory``.

Every one of the 18 presets already takes a ``dmem=`` argument, so this class
slots into any ISA x model combination without touching a preset:

    from sim.components.memory.cache import CachedDataMemory, CacheConfig
    cpu = build(program, dmem=CachedDataMemory(CacheConfig(assoc=2)))

It keeps ``DataMemory``'s exact port set and ``get_state()`` keys (so the UI,
``SimResult.memory`` and ``_arch_signature`` settle detection all keep working
unchanged) and ADDS:

    stall   1 while a miss is being serviced   (wire to fetch.stall)
    ready   1 on the cycle the access completes (gate the reg/mem write)
    cache_* statistics in get_state()

TIMING SUPPORT IS PER-MODEL (``timing_enabled=True``)
-----------------------------------------------------
**``single_cycle`` is the ONLY model that wires the stall path** (see
``TIMING_MODELS``). There it is verified exact: a miss occupies
``1 + miss_penalty`` cycles, so total cycles grow by exactly
``misses x penalty`` with the architectural result unchanged.

Every other model — ``configurable``, ``pipeline``, ``multicycle``, ``ooo``,
``superscalar`` — does NOT support timing mode. ``configurable`` routes commits
through ``CycleBudgetController`` and the stall is not wired into that branch;
the 5-stage pipeline needs a real MEM-stage stall (EX/MEM hold, MEM/WB bubble,
``ID_EX.hold`` rather than the hazard detector's bubble-inserting ``stall``,
and a gated branch redirect — those ports exist but wiring them was not
sufficient). Enabling timing there corrupts ARCHITECTURAL RESULTS, not just
cycle counts, so build the component through
``cached_dmem_for(model, ..., timing_enabled=...)``, which RAISES instead.

Observe-only mode (the default) works on all 18 presets.
"""
from __future__ import annotations

from sim.component.base import ComponentBase, Port
from sim.components.memory.dmem import (
    DEFAULT_MEM_WORDS, STATE_WINDOW, memory_snapshot,
)

from .model import CacheConfig, SetAssociativeCache


class CachedDataMemory(ComponentBase):
    """Word-addressed data memory fronted by a modelled cache.

    Data handling is byte-for-byte identical to ``DataMemory``: combinational
    read from ``addr``, synchronous write on ``wen`` using ``waddr`` when
    ``waddr_valid`` is set (the OoO store-commit path) and ``addr`` otherwise.
    The cache observes that address stream and maintains its metadata; it never
    intercepts the data. See ``model.py``'s docstring for why.
    """
    name = "cached_data_memory"
    ui_label = "Data Memory + Cache"
    ui_category = "memory"
    ports_spec = {
        # ── identical to DataMemory ──
        "addr":  Port(32, "in",  "Read address (byte)"),
        "waddr": Port(32, "in",  "Write address (byte)"),
        "waddr_valid": Port(1, "in", "1 = use waddr (else fall back to addr)"),
        "wdata": Port(32, "in",  "Write data"),
        "wen":   Port(1,  "in",  "Write enable"),
        "rdata": Port(32, "out", "Read data"),
        # ── added by the cache ──
        "mem_read": Port(1, "in", "1 = this cycle is a genuine load (optional)"),
        "stall": Port(1,  "out", "1 while a miss is being serviced"),
        "ready": Port(1,  "out", "1 on the cycle the access completes"),
    }

    def __init__(self, config: CacheConfig | None = None,
                 size: int = DEFAULT_MEM_WORDS,
                 timing_enabled: bool = False,
                 use_mem_read_port: bool = False, **kw):
        super().__init__(**kw)
        self._mem = [0] * size
        self._size = size
        self.cache = SetAssociativeCache(config or CacheConfig())
        self.timing_enabled = bool(timing_enabled)
        # See _is_reading(): without a read-enable the component falls back to
        # an address-change heuristic. A preset that wires decode.mem_read ->
        # dmem.mem_read can set this and get exact load accounting.
        self.use_mem_read_port = bool(use_mem_read_port)
        # Miss servicing state (only used when timing_enabled).
        self._wait = 0            # cycles still owed on the current miss
        self._pending = None      # (byte_addr, is_write) awaiting commit
        self._settling_addr = None  # suppress the one post-miss ready cycle
        # Per-cycle presentation state.
        self._last = None         # last access result, for get_state()
        self._accessed_this_cycle = False   # did an access land THIS cycle?

    # ── helpers ───────────────────────────────────────────────────────
    def _word(self, byte_addr: int) -> int:
        return (byte_addr >> 2) % self._size

    def _effective_addr(self) -> int:
        return int(self["waddr"] if self["waddr_valid"] else self["addr"])

    # ── combinational ─────────────────────────────────────────────────
    def evaluate(self):
        # Data path: unchanged from DataMemory. A cache miss delays *when* the
        # access commits, never *what* it returns.
        self["rdata"] = self._mem[self._word(self["addr"])]

        if not self.timing_enabled:
            self["stall"] = 0
            self["ready"] = 1
            return

        # A miss already in flight keeps the machine frozen.
        if self._wait > 0:
            self["stall"] = 1
            self["ready"] = 0
            return

        # Fresh look. The tag comparison is COMBINATIONAL in real hardware and
        # must be here too: `stall` has to be valid in the same cycle as the
        # access, and evaluate() runs before rising_edge(). probe() does not
        # mutate, so asking it here is free of side effects.
        if (self._settling_addr is None
                and self._accessing()
                and self.cache.probe(self._access_addr()) is None
                and self.cache.config.miss_penalty > 0):
            self["stall"] = 1
            self["ready"] = 0
        else:
            self["stall"] = 0
            self["ready"] = 1

    # ── clocked ───────────────────────────────────────────────────────
    def rising_edge(self):
        self._accessed_this_cycle = False
        if not self.timing_enabled:
            self._observe_only_edge()
            return

        # ── servicing a miss ──
        if self._wait > 0:
            self._wait -= 1
            self.cache.stats.stall_cycles += 1
            if self._wait == 0:
                # Fill complete: mark the address settled so the following
                # (ready) cycle does not re-count this access as a fresh hit.
                self._settling_addr = self._access_addr()
            return

        # ── the cycle right after a serviced miss ──
        # The instruction is still present (it commits at the end of THIS
        # cycle, since ready went high), so skip it: its hit/miss was already
        # counted and its write already applied. Only one cycle is suppressed.
        if self._settling_addr is not None:
            self._settling_addr = None
            return

        # ── a normal access ──
        if self._accessing():
            addr = self._access_addr()
            is_write = bool(self["wen"])
            self._last = self.cache.access(addr, is_write=is_write)
            self._accessed_this_cycle = True
            # Commit the data IMMEDIATELY, even on a miss.
            #
            # The tempting alternative — defer the store until the fill
            # completes — opens a stale-read window: in a pipelined datapath a
            # younger load can be reading `_mem` combinationally while an older
            # store is still pending, and would see the old value. Since this
            # model keeps ONE authoritative data array (see model.py), data
            # correctness must not depend on timing at all: the stall models
            # WHEN the instruction retires, never WHAT memory holds.
            if is_write:
                self._mem[self._word(addr)] = self["wdata"]
            if not self._last["hit"] and self.cache.config.miss_penalty > 0:
                # TIMING CONTRACT: a hit occupies 1 cycle; a miss occupies
                # 1 + penalty. `stall` is asserted for exactly `penalty`
                # cycles (this one plus penalty-1 more), so the instruction
                # commits on cycle `penalty`. Hence wait = penalty - 1, and
                # this cycle is itself counted as a stall cycle.
                self.cache.stats.stall_cycles += 1
                self._wait = self.cache.config.miss_penalty - 1
                if self._wait == 0:         # penalty == 1: settle immediately
                    self._settling_addr = addr


    def _observe_only_edge(self):
        """Step-1 behaviour: count traffic, never delay anything."""
        is_write = bool(self["wen"])
        addr = self._access_addr() if is_write else int(self["addr"])
        if is_write or self._is_reading():
            self._last = self.cache.access(addr, is_write=is_write)
            self._accessed_this_cycle = True
        if is_write:
            self._mem[self._word(addr)] = self["wdata"]

    def _accessing(self) -> bool:
        """Is this cycle a genuine memory access (load or store)?"""
        return bool(self["wen"]) or self._is_reading()

    def _access_addr(self) -> int:
        return self._effective_addr() if self["wen"] else int(self["addr"])

    def _is_reading(self) -> bool:
        """Whether this cycle presents a genuine load.

        EXACT PATH: if a preset wires ``decode.mem_read -> dmem.mem_read`` and
        constructs with ``use_mem_read_port=True``, the port is authoritative.

        FALLBACK (default): ``DataMemory`` has no read-enable — the datapath
        drives ``addr`` from the ALU every cycle, so for an ``ADD`` the address
        is just the arithmetic result. Counting every non-write cycle as a read
        would classify ordinary arithmetic as memory traffic and make the hit
        rate meaningless. Without a read-enable the honest approximation is to
        count a non-write cycle only when the address CHANGED, which filters
        the idle repetition of a held address.

        KNOWN LIMITATION of the fallback: it UNDERCOUNTS. Two consecutive loads
        from the SAME address register as one access, so the second load's hit
        is not recorded. Undercounting is the safer error — it never invents a
        hit that did not happen, and it never inflates the rate with non-memory
        instructions. Pinned by
        ``test_repeat_load_to_the_same_address_is_undercounted_without_the_port``.
        """
        if self.use_mem_read_port:
            return bool(self["mem_read"])
        addr = int(self["addr"])
        changed = addr != getattr(self, "_prev_read_addr", None)
        self._prev_read_addr = addr
        return changed

    # ── snapshot ──────────────────────────────────────────────────────
    def get_state(self):
        dense, hi = memory_snapshot(self._mem, self._size)
        focus = self.cache.config.set_of(int(self["addr"]))
        state = {
            # ── DataMemory-compatible keys (UI + _arch_signature) ──
            "addr": f"0x{self['addr']:08x}",
            "rdata": self["rdata"],
            "wdata": self["wdata"],
            "wen": self["wen"],
            "write_addr": self._effective_addr(),
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
        # `_last` is sticky by design (the UI wants to know where the most
        # recent access landed), but the HIT/MISS badge must describe THIS
        # cycle or it lies on every non-memory instruction — an ADD would show
        # the previous load's "HIT". Report IDLE unless an access was recorded
        # this cycle or a miss is still being serviced.
        active = self._accessed_this_cycle or self._wait > 0
        if self._last is not None:
            state["last_access"] = {
                "hit": self._last["hit"],
                "set": self._last["set"],
                "way": self._last["way"],
                "evicted": self._last["evicted"],
                "wrote_back": self._last["wrote_back"],
            }
        if active and self._last is not None:
            state["cache_result"] = "HIT" if self._last["hit"] else "MISS"
        else:
            state["cache_result"] = "IDLE"
        return state
