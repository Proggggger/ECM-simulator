"""
The cache model: address decomposition, tag/valid/dirty metadata, replacement,
and hit/miss/eviction/writeback accounting.

DESIGN DECISION — metadata model over a duplicated data array
-------------------------------------------------------------
This models a cache's METADATA (tags, valid/dirty bits, replacement state) and
its TIMING, but keeps exactly ONE authoritative copy of the data, in the
backing memory array that ``DataMemory`` already owns.

Why: the hit/miss sequence, the eviction sequence and the writeback count are
determined entirely by the stream of ADDRESSES and the metadata — not by where
the bytes physically sit. Storing a second copy of every cached line would add
a way for the simulator to return *wrong data* (a real risk, and a
teaching-hostile one) while adding nothing to what we report. With one array,
architectural state is correct by construction at every cycle, which is what
keeps the cache transparent to the single-cycle oracle and to
``_arch_signature`` settle detection.

The one thing this cannot show is write-back's *delayed* DRAM update as a
difference in memory CONTENTS mid-run. It is still fully visible where it
matters: in the writeback COUNT, in the eviction stream, and in the cycle cost
each policy incurs. ``write_policy`` therefore changes timing and statistics,
not observable data — documented, and pinned by tests.

Everything here is pure Python with no simulator dependency, so the whole model
is unit-testable without building a CPU.
"""
from __future__ import annotations

from .policies import DEFAULT_POLICY, get_policy

# ── defaults ──────────────────────────────────────────────────────────────
DEFAULT_LINE_BYTES = 64
DEFAULT_SIZE_KB = 8
DEFAULT_ASSOC = 4
DEFAULT_HIT_CYCLES = 1
DEFAULT_MISS_PENALTY = 10

WRITE_POLICIES = ("write_back", "write_through")
DEFAULT_WRITE_POLICY = "write_back"

# Clamp ranges — mirrors how api/app.py clamps cycle_costs to 1..64 so a
# hostile or fat-fingered request cannot allocate an enormous cache.
MIN_LINE_BYTES, MAX_LINE_BYTES = 4, 1024
MIN_SIZE_KB, MAX_SIZE_KB = 1, 1024
MIN_ASSOC, MAX_ASSOC = 1, 32
MAX_HIT_CYCLES = 64
MAX_MISS_PENALTY = 1000


def _is_pow2(n: int) -> bool:
    return n >= 1 and not (n & (n - 1))


class CacheConfig:
    """Validated cache geometry + policy.

    ``size_kb`` / ``line_bytes`` / ``assoc`` must all be powers of two (as in
    real hardware, so index/offset are bit-slices rather than divisions), and
    the geometry must yield at least one set.
    """

    def __init__(self, size_kb: int = DEFAULT_SIZE_KB,
                 line_bytes: int = DEFAULT_LINE_BYTES,
                 assoc: int = DEFAULT_ASSOC,
                 policy: str = DEFAULT_POLICY,
                 write_policy: str = DEFAULT_WRITE_POLICY,
                 write_allocate: bool = True,
                 hit_cycles: int = DEFAULT_HIT_CYCLES,
                 miss_penalty: int = DEFAULT_MISS_PENALTY):
        size_kb = int(size_kb)
        line_bytes = int(line_bytes)
        assoc = int(assoc)

        if not _is_pow2(size_kb):
            raise ValueError(f"size_kb must be a power of two, got {size_kb}")
        if not _is_pow2(line_bytes):
            raise ValueError(f"line_bytes must be a power of two, got {line_bytes}")
        if not _is_pow2(assoc):
            raise ValueError(f"assoc must be a power of two, got {assoc}")
        if not MIN_SIZE_KB <= size_kb <= MAX_SIZE_KB:
            raise ValueError(f"size_kb {size_kb} outside [{MIN_SIZE_KB}..{MAX_SIZE_KB}]")
        if not MIN_LINE_BYTES <= line_bytes <= MAX_LINE_BYTES:
            raise ValueError(
                f"line_bytes {line_bytes} outside [{MIN_LINE_BYTES}..{MAX_LINE_BYTES}]")
        if not MIN_ASSOC <= assoc <= MAX_ASSOC:
            raise ValueError(f"assoc {assoc} outside [{MIN_ASSOC}..{MAX_ASSOC}]")
        if write_policy not in WRITE_POLICIES:
            raise ValueError(
                f"write_policy must be one of {WRITE_POLICIES}, got {write_policy!r}")

        total_bytes = size_kb * 1024
        num_lines = total_bytes // line_bytes
        if num_lines < assoc:
            raise ValueError(
                f"{size_kb}KB / {line_bytes}B lines = {num_lines} lines, "
                f"which cannot be {assoc}-way associative")

        self.size_kb = size_kb
        self.line_bytes = line_bytes
        self.assoc = assoc
        self.policy = policy
        self.write_policy = write_policy
        self.write_allocate = bool(write_allocate)
        self.hit_cycles = max(1, min(int(hit_cycles), MAX_HIT_CYCLES))
        self.miss_penalty = max(0, min(int(miss_penalty), MAX_MISS_PENALTY))

        self.total_bytes = total_bytes
        self.num_lines = num_lines
        self.num_sets = num_lines // assoc
        # Bit-slice widths: [ tag | index | offset ]
        self.offset_bits = line_bytes.bit_length() - 1
        self.index_bits = self.num_sets.bit_length() - 1
        # Validate the policy name eagerly so a bad name fails at config time.
        get_policy(policy)

    # ── address decomposition ─────────────────────────────────────────
    def line_of(self, byte_addr: int) -> int:
        return byte_addr >> self.offset_bits

    def set_of(self, byte_addr: int) -> int:
        return self.line_of(byte_addr) & (self.num_sets - 1)

    def tag_of(self, byte_addr: int) -> int:
        return self.line_of(byte_addr) >> self.index_bits

    def to_dict(self) -> dict:
        return {
            "size_kb": self.size_kb,
            "line_bytes": self.line_bytes,
            "assoc": self.assoc,
            "policy": self.policy,
            "write_policy": self.write_policy,
            "write_allocate": self.write_allocate,
            "hit_cycles": self.hit_cycles,
            "miss_penalty": self.miss_penalty,
            "num_sets": self.num_sets,
            "num_lines": self.num_lines,
        }

    @classmethod
    def from_dict(cls, raw) -> "CacheConfig":
        """Build from an API payload, ignoring unknown keys."""
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("cache config must be an object")
        known = ("size_kb", "line_bytes", "assoc", "policy", "write_policy",
                 "write_allocate", "hit_cycles", "miss_penalty")
        return cls(**{k: raw[k] for k in known if k in raw})

    def __repr__(self):                             # pragma: no cover - debug aid
        return (f"<CacheConfig {self.size_kb}KB {self.assoc}-way "
                f"{self.line_bytes}B/line {self.policy} {self.write_policy}>")


class CacheStats:
    """Access counters. ``reads``/``writes`` split so the UI can show both."""

    __slots__ = ("read_hits", "read_misses", "write_hits", "write_misses",
                 "evictions", "writebacks", "stall_cycles")

    def __init__(self):
        self.reset()

    def reset(self):
        self.read_hits = 0
        self.read_misses = 0
        self.write_hits = 0
        self.write_misses = 0
        self.evictions = 0
        self.writebacks = 0
        self.stall_cycles = 0

    @property
    def hits(self) -> int:
        return self.read_hits + self.write_hits

    @property
    def misses(self) -> int:
        return self.read_misses + self.write_misses

    @property
    def accesses(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.accesses) if self.accesses else 0.0

    def to_dict(self) -> dict:
        return {
            "read_hits": self.read_hits,
            "read_misses": self.read_misses,
            "write_hits": self.write_hits,
            "write_misses": self.write_misses,
            "hits": self.hits,
            "misses": self.misses,
            "accesses": self.accesses,
            "hit_rate": round(self.hit_rate, 6),
            "evictions": self.evictions,
            "writebacks": self.writebacks,
            "stall_cycles": self.stall_cycles,
        }


class SetAssociativeCache:
    """Tag/valid/dirty array + replacement, with hit/miss accounting.

    Direct-mapped is simply ``assoc=1``; fully-associative is
    ``assoc == num_lines`` (one set). There is no separate class for either.

    ``access()`` is the whole interface: it reports whether an address hit,
    updates the metadata, and returns what happened so the caller can charge
    cycles. It performs NO data movement — see the module docstring.
    """

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self.stats = CacheStats()
        self._policy = get_policy(self.config.policy)(
            self.config.num_sets, self.config.assoc)
        self.flush()

    # ── state ─────────────────────────────────────────────────────────
    def flush(self):
        """Invalidate every line and clear replacement state (not stats)."""
        n = self.config.num_sets
        a = self.config.assoc
        self._valid = [[False] * a for _ in range(n)]
        self._dirty = [[False] * a for _ in range(n)]
        self._tags = [[0] * a for _ in range(n)]
        self._policy.reset()

    def reset(self):
        """Flush AND clear statistics."""
        self.flush()
        self.stats.reset()

    # ── the access path ───────────────────────────────────────────────
    def probe(self, byte_addr: int):
        """Non-mutating lookup -> way index, or None on a miss."""
        cfg = self.config
        s = cfg.set_of(byte_addr)
        tag = cfg.tag_of(byte_addr)
        for way in range(cfg.assoc):
            if self._valid[s][way] and self._tags[s][way] == tag:
                return way
        return None

    def access(self, byte_addr: int, is_write: bool = False) -> dict:
        """Look up ``byte_addr``, update metadata, and report what happened.

        Returns a dict with:
            hit          bool
            set, way     where it landed (way is None if a write miss was NOT
                         allocated, i.e. write-through + no-write-allocate)
            evicted      bool  — a valid line was displaced
            wrote_back   bool  — the displaced line was dirty
            cycles       int   — hit_cycles, plus miss_penalty on a miss
        """
        cfg = self.config
        s = cfg.set_of(byte_addr)
        tag = cfg.tag_of(byte_addr)
        way = self.probe(byte_addr)

        result = {"hit": way is not None, "set": s, "way": way,
                  "evicted": False, "wrote_back": False,
                  "cycles": cfg.hit_cycles}

        if way is not None:
            # ── HIT ──
            if is_write:
                self.stats.write_hits += 1
                if cfg.write_policy == "write_back":
                    self._dirty[s][way] = True
                else:
                    # write-through: DRAM is updated immediately, so the line
                    # never becomes dirty and never needs a writeback.
                    self.stats.writebacks += 1
            else:
                self.stats.read_hits += 1
            self._policy.touch(s, way)
            return result

        # ── MISS ──
        if is_write:
            self.stats.write_misses += 1
        else:
            self.stats.read_misses += 1
        result["cycles"] = cfg.hit_cycles + cfg.miss_penalty

        if is_write and cfg.write_policy == "write_through":
            # Write-through always pushes the store to DRAM. Whether it also
            # PULLS the line into the cache is the write-allocate choice.
            self.stats.writebacks += 1
            if not cfg.write_allocate:
                return result           # no-write-allocate: nothing cached
        elif is_write and not cfg.write_allocate:
            # write-back + no-write-allocate: store goes straight to DRAM.
            self.stats.writebacks += 1
            return result

        # Allocate a line: prefer an invalid way (cold), else ask the policy.
        victim = next((w for w in range(cfg.assoc) if not self._valid[s][w]), None)
        if victim is None:
            victim = self._policy.victim(s)
            result["evicted"] = True
            self.stats.evictions += 1
            if self._dirty[s][victim]:
                result["wrote_back"] = True
                self.stats.writebacks += 1

        self._valid[s][victim] = True
        self._tags[s][victim] = tag
        self._dirty[s][victim] = bool(is_write and cfg.write_policy == "write_back")
        self._policy.touch(s, victim)
        result["way"] = victim
        return result

    # ── introspection for the UI ──────────────────────────────────────
    def set_view(self, set_idx: int) -> list:
        """Per-way contents of one set, for the UI's set-array drawing."""
        return [
            {
                "way": w,
                "valid": self._valid[set_idx][w],
                "dirty": self._dirty[set_idx][w],
                "tag": self._tags[set_idx][w] if self._valid[set_idx][w] else None,
                "age": self._policy.age_of(set_idx, w),
            }
            for w in range(self.config.assoc)
        ]

    def occupancy(self) -> int:
        return sum(sum(1 for v in ways if v) for ways in self._valid)

    def dirty_lines(self) -> int:
        return sum(sum(1 for d in ways if d) for ways in self._dirty)

    def to_dict(self, focus_set: int | None = None) -> dict:
        """Serialisable snapshot. ``focus_set`` gets a dense per-way view.

        Only ONE set is expanded: a 1024-line cache would otherwise add 1024
        entries to every cycle's payload, the same bloat ``memory_snapshot``
        exists to avoid.
        """
        out = {
            "config": self.config.to_dict(),
            "stats": self.stats.to_dict(),
            "occupancy": self.occupancy(),
            "dirty_lines": self.dirty_lines(),
        }
        if focus_set is not None and 0 <= focus_set < self.config.num_sets:
            out["focus_set"] = focus_set
            out["ways"] = self.set_view(focus_set)
        return out
