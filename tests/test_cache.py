"""
Cache / memory-hierarchy model — unit + integration tests.

WHAT THIS FILE GUARDS
---------------------
The cache is a NEW subsystem, so it gets the treatment CACHE_DESIGN.md calls
for rather than a "hit rate > 0" smoke test:

  1. **Address decomposition** — tag/index/offset are bit-slices; every
     geometry rejects non-power-of-two and impossible configurations.
  2. **Hand-computed hit/miss counts** — a stride-1 walk over a known array
     with a known line size has an arithmetically predictable miss count (one
     per line). Those exact integers are asserted, not inequalities.
  3. **Policy differentiation** — access patterns where LRU and FIFO provably
     DISAGREE, and LRU's pathological cyclic sweep (assoc+1 lines -> 100%
     miss). If two policies never disagree anywhere, the policy plumbing is
     not actually wired up, and a test must catch that.
  4. **Monotonicity** — a bigger or more associative cache must never suffer
     MORE misses on the same trace. This is the same class of bug as the old
     "4 lanes slower than 2" superscalar regression.
  5. **Oracle equivalence on all 18 presets** — the cache must be
     architecturally TRANSPARENT: identical final registers and memory to an
     uncached run, on every ISA x model. This is the property the whole
     timing model will later be built on, so it is pinned first and hardest.

Run from repo root:  python3 -m pytest tests/test_cache.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.assembler import assemble
from sim.components.memory.cache import (
    POLICIES,
    POLICY_CLASSES,
    WRITE_POLICIES,
    CachedDataMemory,
    CachedMultiPortDataMemory,
    CacheConfig,
    ReplacementPolicyBase,
    SetAssociativeCache,
    get_policy,
    policy_catalog,
)
from sim.harness import PRESETS
from sim.runner_v2 import run_simulation

ISAS = ("riscv", "arm", "x86")
MODELS = ("single_cycle", "multicycle", "configurable", "pipeline",
          "ooo", "superscalar")


# ════════════════════════════════════════════════════════════════════════
# 1. CacheConfig — geometry + validation
# ════════════════════════════════════════════════════════════════════════
class TestCacheConfigGeometry:

    def test_default_geometry_is_self_consistent(self):
        c = CacheConfig()
        assert c.total_bytes == c.size_kb * 1024
        assert c.num_lines == c.total_bytes // c.line_bytes
        assert c.num_sets == c.num_lines // c.assoc
        assert 1 << c.offset_bits == c.line_bytes
        assert 1 << c.index_bits == c.num_sets

    @pytest.mark.parametrize("size_kb,line,assoc,sets", [
        (1, 64, 1, 16),      # direct-mapped: 1024/64 = 16 lines = 16 sets
        (1, 64, 2, 8),
        (1, 64, 4, 4),
        (1, 64, 16, 1),      # fully associative: one set
        (4, 32, 4, 32),      # 4096/32 = 128 lines / 4 = 32 sets
        (8, 64, 4, 32),      # the default
    ])
    def test_set_count_arithmetic(self, size_kb, line, assoc, sets):
        assert CacheConfig(size_kb=size_kb, line_bytes=line, assoc=assoc).num_sets == sets

    @pytest.mark.parametrize("kwargs", [
        {"size_kb": 3},              # not a power of two
        {"line_bytes": 48},
        {"assoc": 3},
        {"size_kb": 0},
        {"line_bytes": 0},
        {"assoc": 0},
        {"size_kb": 1 << 20},        # past MAX_SIZE_KB
        {"line_bytes": 4096},        # past MAX_LINE_BYTES
        {"assoc": 64},               # past MAX_ASSOC
        {"write_policy": "magic"},
        {"policy": "nonexistent"},
    ])
    def test_invalid_geometry_is_rejected(self, kwargs):
        with pytest.raises(ValueError):
            CacheConfig(**kwargs)

    def test_assoc_larger_than_line_count_is_rejected(self):
        # 1KB / 1024B lines = 1 line, cannot be 2-way.
        with pytest.raises(ValueError, match="associative"):
            CacheConfig(size_kb=1, line_bytes=1024, assoc=2)

    @pytest.mark.parametrize("policy", POLICIES)
    def test_every_discovered_policy_is_accepted(self, policy):
        # plru needs a power-of-two assoc, which every valid assoc already is.
        assert CacheConfig(policy=policy).policy == policy

    def test_hit_cycles_and_miss_penalty_are_clamped(self):
        c = CacheConfig(hit_cycles=0, miss_penalty=-5)
        assert c.hit_cycles == 1        # an access always costs >= 1
        assert c.miss_penalty == 0      # but a miss penalty may legitimately be 0
        c = CacheConfig(hit_cycles=10 ** 6, miss_penalty=10 ** 6)
        assert c.hit_cycles == 64
        assert c.miss_penalty == 1000

    def test_from_dict_ignores_unknown_keys(self):
        c = CacheConfig.from_dict({"assoc": 2, "bogus": 99})
        assert c.assoc == 2

    def test_from_dict_rejects_a_non_object(self):
        for bad in ("nope", [1], 5):
            with pytest.raises(ValueError):
                CacheConfig.from_dict(bad)

    def test_from_dict_of_none_is_the_default(self):
        assert CacheConfig.from_dict(None).to_dict() == CacheConfig().to_dict()

    def test_to_dict_roundtrips_through_from_dict(self):
        a = CacheConfig(size_kb=2, line_bytes=32, assoc=2, policy="fifo",
                        write_policy="write_through", write_allocate=False,
                        hit_cycles=2, miss_penalty=25)
        assert CacheConfig.from_dict(a.to_dict()).to_dict() == a.to_dict()


class TestAddressDecomposition:
    """[ tag | index | offset ] must be exact bit-slices."""

    @pytest.fixture()
    def cfg(self):
        # 1KB, 64B lines, 2-way -> 8 sets; offset=6 bits, index=3 bits
        return CacheConfig(size_kb=1, line_bytes=64, assoc=2)

    def test_addresses_within_one_line_share_set_and_tag(self, cfg):
        base = 0x400
        first = (cfg.set_of(base), cfg.tag_of(base))
        for off in range(cfg.line_bytes):
            assert (cfg.set_of(base + off), cfg.tag_of(base + off)) == first

    def test_next_line_moves_to_the_next_set(self, cfg):
        assert cfg.set_of(0) == 0
        assert cfg.set_of(64) == 1
        assert cfg.set_of(128) == 2

    def test_sets_wrap_after_num_sets_lines(self, cfg):
        stride = cfg.line_bytes * cfg.num_sets      # 64 * 8 = 512
        assert cfg.set_of(0) == cfg.set_of(stride) == 0
        # ...but the TAG differs, which is what makes them conflict, not alias.
        assert cfg.tag_of(0) != cfg.tag_of(stride)

    def test_line_index_is_monotonic_in_address(self, cfg):
        assert [cfg.line_of(a) for a in (0, 63, 64, 127, 128)] == [0, 0, 1, 1, 2]


# ════════════════════════════════════════════════════════════════════════
# 2. Hand-computed hit/miss counts
# ════════════════════════════════════════════════════════════════════════
class TestHitMissArithmetic:

    def test_cold_access_always_misses(self):
        c = SetAssociativeCache(CacheConfig())
        assert c.access(0)["hit"] is False
        assert c.stats.read_misses == 1

    def test_second_access_to_the_same_word_hits(self):
        c = SetAssociativeCache(CacheConfig())
        c.access(0)
        assert c.access(0)["hit"] is True
        assert c.stats.read_hits == 1

    @pytest.mark.parametrize("line_bytes,words,expected_misses", [
        (64, 64, 4),      # 64 words = 256B over 64B lines -> 4 lines
        (32, 64, 8),      # 256B over 32B lines -> 8 lines
        (16, 64, 16),
        (64, 16, 1),      # 16 words = 64B = exactly one line
        (64, 17, 2),      # one word past the line -> second line
    ])
    def test_sequential_walk_misses_once_per_line(self, line_bytes, words,
                                                  expected_misses):
        """The canonical arithmetic: a stride-1 sweep costs one miss per line."""
        c = SetAssociativeCache(CacheConfig(size_kb=4, line_bytes=line_bytes,
                                           assoc=4))
        for w in range(words):
            c.access(w * 4)
        assert c.stats.misses == expected_misses
        assert c.stats.hits == words - expected_misses
        assert c.stats.accesses == words

    def test_stride_equal_to_line_size_misses_every_time(self):
        """Touching one word per line defeats spatial locality entirely."""
        cfg = CacheConfig(size_kb=4, line_bytes=64, assoc=4)
        c = SetAssociativeCache(cfg)
        for i in range(20):
            c.access(i * cfg.line_bytes)
        assert c.stats.misses == 20
        assert c.stats.hits == 0

    def test_repeated_sweep_of_a_fitting_array_hits_after_the_first_pass(self):
        cfg = CacheConfig(size_kb=4, line_bytes=64, assoc=4)
        c = SetAssociativeCache(cfg)
        words = 256                       # 1KB, comfortably inside 4KB
        for _ in range(3):
            for w in range(words):
                c.access(w * 4)
        first_pass_misses = 1024 // cfg.line_bytes
        assert c.stats.misses == first_pass_misses      # only the cold pass
        assert c.stats.hits == words * 3 - first_pass_misses

    def test_hit_rate_and_accesses_are_consistent(self):
        c = SetAssociativeCache(CacheConfig())
        for w in range(50):
            c.access(w * 4)
        s = c.stats
        assert s.accesses == s.hits + s.misses == 50
        assert s.hit_rate == pytest.approx(s.hits / 50)

    def test_empty_cache_reports_zero_hit_rate_not_a_zero_division(self):
        assert SetAssociativeCache(CacheConfig()).stats.hit_rate == 0.0


class TestConflictMisses:

    def test_direct_mapped_thrashes_on_two_conflicting_lines(self):
        """Two addresses in the same set with different tags, assoc=1."""
        cfg = CacheConfig(size_kb=1, line_bytes=64, assoc=1)   # 16 sets
        c = SetAssociativeCache(cfg)
        a = 0
        b = cfg.line_bytes * cfg.num_sets      # same set, different tag
        assert cfg.set_of(a) == cfg.set_of(b)
        for _ in range(10):
            c.access(a)
            c.access(b)
        assert c.stats.misses == 20            # every single access misses
        assert c.stats.evictions == 19         # all but the very first fill

    def test_two_way_absorbs_the_same_pair_completely(self):
        """The identical trace on a 2-way cache: 2 cold misses, then all hits."""
        cfg = CacheConfig(size_kb=1, line_bytes=64, assoc=2)   # 8 sets
        c = SetAssociativeCache(cfg)
        a = 0
        b = cfg.line_bytes * cfg.num_sets
        assert cfg.set_of(a) == cfg.set_of(b)
        for _ in range(10):
            c.access(a)
            c.access(b)
        assert c.stats.misses == 2
        assert c.stats.evictions == 0

    def test_eviction_count_matches_fills_minus_capacity(self):
        cfg = CacheConfig(size_kb=1, line_bytes=64, assoc=1)   # 16 lines
        c = SetAssociativeCache(cfg)
        # Touch 32 distinct lines: 16 fills fit, the next 16 each evict.
        for i in range(32):
            c.access(i * cfg.line_bytes)
        assert c.stats.misses == 32
        assert c.stats.evictions == 32 - cfg.num_lines


# ════════════════════════════════════════════════════════════════════════
# 3. Replacement policies
# ════════════════════════════════════════════════════════════════════════
class TestPolicyRegistry:

    def test_discovery_found_the_expected_policies(self):
        assert set(POLICIES) >= {"lru", "fifo", "plru", "random"}

    @pytest.mark.parametrize("name", POLICIES)
    def test_every_policy_subclasses_the_base_and_has_a_unique_name(self, name):
        cls = POLICY_CLASSES[name]
        assert issubclass(cls, ReplacementPolicyBase)
        assert cls.name == name
    def test_names_are_unique(self):
        names = [c.name for c in POLICY_CLASSES.values()]
        assert len(names) == len(set(names))

    def test_get_policy_rejects_unknown_names(self):
        for bad in ("nope", "", None, 5, ["lru"]):
            with pytest.raises(ValueError):
                get_policy(bad)

    def test_catalog_is_serialisable_and_complete(self):
        cat = policy_catalog()
        assert {e["name"] for e in cat} == set(POLICIES)
        for e in cat:
            assert e["label"] and isinstance(e["description"], str)

    @pytest.mark.parametrize("name", POLICIES)
    def test_policy_is_deterministic_across_instances(self, name):
        """Same trace, two fresh caches -> identical statistics.

        Non-determinism would make cycle counts irreproducible and the
        /compare grid meaningless, so even `random` must be seeded.
        """
        trace = [(i * 64) % 4096 for i in range(200)]
        runs = []
        for _ in range(2):
            c = SetAssociativeCache(CacheConfig(size_kb=1, assoc=4, policy=name))
            for a in trace:
                c.access(a)
            runs.append(c.stats.to_dict())
        assert runs[0] == runs[1]


class TestPolicyBehaviourDiffers:
    """If no trace distinguishes two policies, the plumbing is not wired."""

    @staticmethod
    def _run(policy, trace, assoc=2):
        cfg = CacheConfig(size_kb=1, line_bytes=64, assoc=assoc, policy=policy)
        c = SetAssociativeCache(cfg)
        for a in trace:
            c.access(a)
        return c

    def test_lru_and_fifo_disagree_on_a_reuse_pattern(self):
        """A, B, A, C in one 2-way set.

        LRU evicts B (A was re-touched, so B is least recent) -> the later A
        still hits. FIFO ignores the re-touch of A and evicts A itself -> the
        later A misses. One trace, two different outcomes.
        """
        cfg = CacheConfig(size_kb=1, line_bytes=64, assoc=2)
        stride = cfg.line_bytes * cfg.num_sets
        A, B, C = 0, stride, 2 * stride
        trace = [A, B, A, C, A]
        lru = self._run("lru", trace)
        fifo = self._run("fifo", trace)
        assert lru.stats.misses != fifo.stats.misses, (
            "LRU and FIFO produced identical results — policy is not being used")
        assert lru.stats.misses == 3     # A,B,C cold; final A hits
        assert fifo.stats.misses == 4    # FIFO evicted A, so the final A misses

    def test_lru_is_pathological_on_a_cyclic_sweep_of_assoc_plus_one(self):
        """assoc+1 lines cycled through one set: LRU misses 100% of the time.

        The line about to be reused is always the one LRU just chose as the
        victim. Worth showing in the UI, so it is pinned here.
        """
        cfg = CacheConfig(size_kb=1, line_bytes=64, assoc=2, policy="lru")
        stride = cfg.line_bytes * cfg.num_sets
        lines = [i * stride for i in range(cfg.assoc + 1)]
        c = SetAssociativeCache(cfg)
        for _ in range(6):
            for a in lines:
                c.access(a)
        assert c.stats.hits == 0
        assert c.stats.misses == len(lines) * 6

    def test_random_survives_the_sweep_that_defeats_lru(self):
        """The point of random replacement: no adversarial worst case."""
        cfg = CacheConfig(size_kb=1, line_bytes=64, assoc=4, policy="random")
        stride = cfg.line_bytes * cfg.num_sets
        lines = [i * stride for i in range(cfg.assoc + 1)]
        c = SetAssociativeCache(cfg)
        for _ in range(20):
            for a in lines:
                c.access(a)
        assert c.stats.hits > 0, "random replacement should not be 100% miss"

    @pytest.mark.parametrize("policy", POLICIES)
    def test_every_policy_keeps_a_fitting_working_set_resident(self, policy):
        """Sanity floor: if the working set FITS, everything after the cold
        pass must hit, whatever the policy."""
        cfg = CacheConfig(size_kb=1, line_bytes=64, assoc=2, policy=policy)
        lines = [i * cfg.line_bytes for i in range(cfg.num_lines)]
        c = SetAssociativeCache(cfg)
        for _ in range(3):
            for a in lines:
                c.access(a)
        assert c.stats.misses == cfg.num_lines


class TestLRUMechanics:

    def test_age_ordering_reflects_access_recency(self):
        cfg = CacheConfig(size_kb=1, line_bytes=64, assoc=4, policy="lru")
        c = SetAssociativeCache(cfg)
        stride = cfg.line_bytes * cfg.num_sets
        for i in range(4):
            c.access(i * stride)          # fill set 0, ways 0..3 in order
        view = {w["way"]: w["age"] for w in c.set_view(0)}
        # way 0 was touched first -> age 0 (next victim); way 3 last -> age 3
        assert view[0] == 0 and view[3] == 3

    def test_touching_the_oldest_way_makes_it_safest(self):
        cfg = CacheConfig(size_kb=1, line_bytes=64, assoc=4, policy="lru")
        c = SetAssociativeCache(cfg)
        stride = cfg.line_bytes * cfg.num_sets
        for i in range(4):
            c.access(i * stride)
        c.access(0)                        # re-touch the oldest
        ages = {w["way"]: w["age"] for w in c.set_view(0)}
        assert ages[0] == 3                # now the most-recently-used


class TestPLRU:

    def test_plru_requires_a_power_of_two_associativity(self):
        cls = get_policy("plru")
        with pytest.raises(ValueError, match="power-of-two"):
            cls(num_sets=4, assoc=3)

    def test_plru_approximates_lru_on_a_fitting_working_set(self):
        cfg = CacheConfig(size_kb=1, line_bytes=64, assoc=4, policy="plru")
        c = SetAssociativeCache(cfg)
        lines = [i * cfg.line_bytes for i in range(cfg.num_lines)]
        for _ in range(3):
            for a in lines:
                c.access(a)
        assert c.stats.misses == cfg.num_lines

    def test_plru_victim_is_always_a_legal_way(self):
        for assoc in (1, 2, 4, 8, 16):
            pol = get_policy("plru")(num_sets=2, assoc=assoc)
            for step in range(50):
                pol.touch(0, step % assoc)
                assert 0 <= pol.victim(0) < assoc


# ════════════════════════════════════════════════════════════════════════
# 4. Monotonicity — bigger/more-associative must never be worse
# ════════════════════════════════════════════════════════════════════════
class TestMonotonicity:
    """Same class of bug as the old 'superscalar 4 lanes slower than 2'."""

    TRACE = [(i * 4) % 8192 for i in range(400)] + \
            [(i * 64) % 4096 for i in range(200)]

    @staticmethod
    def _misses(**kw):
        c = SetAssociativeCache(CacheConfig(**kw))
        for a in TestMonotonicity.TRACE:
            c.access(a)
        return c.stats.misses

    @pytest.mark.parametrize("policy", ["lru", "fifo", "plru"])
    def test_more_associativity_never_increases_misses(self, policy):
        prev = None
        for assoc in (1, 2, 4, 8):
            m = self._misses(size_kb=4, line_bytes=64, assoc=assoc, policy=policy)
            if prev is not None:
                assert m <= prev, f"assoc={assoc} ({policy}) got WORSE: {m} > {prev}"
            prev = m

    @pytest.mark.parametrize("policy", ["lru", "fifo", "plru"])
    def test_more_capacity_never_increases_misses(self, policy):
        prev = None
        for size_kb in (1, 2, 4, 8, 16):
            m = self._misses(size_kb=size_kb, line_bytes=64, assoc=4, policy=policy)
            if prev is not None:
                assert m <= prev, f"{size_kb}KB ({policy}) got WORSE: {m} > {prev}"
            prev = m

    def test_a_cache_holding_the_whole_trace_has_only_cold_misses(self):
        distinct_lines = len({a // 64 for a in self.TRACE})
        m = self._misses(size_kb=64, line_bytes=64, assoc=8)
        assert m == distinct_lines


# ════════════════════════════════════════════════════════════════════════
# 5. Write policies
# ════════════════════════════════════════════════════════════════════════
class TestWritePolicies:

    def test_write_back_marks_the_line_dirty(self):
        c = SetAssociativeCache(CacheConfig(write_policy="write_back"))
        c.access(0, is_write=True)
        assert c.dirty_lines() == 1

    def test_a_write_HIT_on_a_clean_line_dirties_it(self):
        """Distinct from the write-MISS path, which dirties on allocate.

        Read first (clean fill), then write the same line: the hit path must
        set the dirty bit. Found by mutation testing — disabling the hit-path
        assignment previously passed the whole suite.
        """
        c = SetAssociativeCache(CacheConfig(write_policy="write_back"))
        c.access(0)                          # read miss -> CLEAN fill
        assert c.dirty_lines() == 0
        c.access(0, is_write=True)           # write HIT -> must dirty
        assert c.stats.write_hits == 1
        assert c.dirty_lines() == 1

    def test_a_line_dirtied_by_a_write_hit_is_written_back_on_eviction(self):
        """The consequence that makes the dirty bit observable."""
        cfg = CacheConfig(size_kb=1, line_bytes=64, assoc=1,
                          write_policy="write_back")
        c = SetAssociativeCache(cfg)
        stride = cfg.line_bytes * cfg.num_sets
        c.access(0)                           # clean fill
        c.access(0, is_write=True)            # write hit -> dirty
        assert c.stats.writebacks == 0        # nothing flushed yet
        c.access(stride)                      # conflict evicts the dirty line
        assert c.stats.evictions == 1
        assert c.stats.writebacks == 1

    def test_write_through_hit_does_not_dirty_the_line(self):
        """The mirror case: write-through keeps DRAM current, so no dirty bit."""
        c = SetAssociativeCache(CacheConfig(write_policy="write_through"))
        c.access(0)                           # clean fill
        c.access(0, is_write=True)            # write hit
        assert c.stats.write_hits == 1
        assert c.dirty_lines() == 0

    def test_write_through_never_leaves_a_dirty_line(self):
        c = SetAssociativeCache(CacheConfig(write_policy="write_through"))
        for i in range(8):
            c.access(i * 4, is_write=True)
        assert c.dirty_lines() == 0

    def test_write_back_writes_back_exactly_once_per_dirty_eviction(self):
        cfg = CacheConfig(size_kb=1, line_bytes=64, assoc=1,
                          write_policy="write_back")
        c = SetAssociativeCache(cfg)
        stride = cfg.line_bytes * cfg.num_sets
        c.access(0, is_write=True)          # fill dirty
        c.access(stride)                    # conflict -> evict the dirty line
        assert c.stats.evictions == 1
        assert c.stats.writebacks == 1

    def test_write_back_does_not_write_back_a_clean_eviction(self):
        cfg = CacheConfig(size_kb=1, line_bytes=64, assoc=1,
                          write_policy="write_back")
        c = SetAssociativeCache(cfg)
        stride = cfg.line_bytes * cfg.num_sets
        c.access(0)                          # clean fill (a read)
        c.access(stride)                     # evicts it
        assert c.stats.evictions == 1
        assert c.stats.writebacks == 0

    def test_write_through_counts_a_writeback_per_store(self):
        c = SetAssociativeCache(CacheConfig(write_policy="write_through"))
        for i in range(5):
            c.access(0, is_write=True)       # same line, all hits
        assert c.stats.write_hits == 4 and c.stats.write_misses == 1
        assert c.stats.writebacks == 5       # every store reaches DRAM

    def test_no_write_allocate_does_not_pull_the_line_in(self):
        c = SetAssociativeCache(CacheConfig(write_allocate=False))
        c.access(0, is_write=True)
        assert c.occupancy() == 0            # nothing cached
        assert c.stats.write_misses == 1

    def test_write_allocate_does_pull_the_line_in(self):
        c = SetAssociativeCache(CacheConfig(write_allocate=True))
        c.access(0, is_write=True)
        assert c.occupancy() == 1
        assert c.access(0)["hit"] is True    # the following read now hits

    def test_read_and_write_counters_are_tracked_separately(self):
        c = SetAssociativeCache(CacheConfig())
        c.access(0)                 # read miss
        c.access(0)                 # read hit
        c.access(0, is_write=True)  # write hit
        c.access(4096, is_write=True)   # write miss
        s = c.stats
        assert (s.read_misses, s.read_hits, s.write_hits, s.write_misses) == (1, 1, 1, 1)


# ════════════════════════════════════════════════════════════════════════
# 6. Cache state / introspection
# ════════════════════════════════════════════════════════════════════════
class TestCacheStateAndFlush:

    def test_flush_invalidates_everything_but_keeps_stats(self):
        c = SetAssociativeCache(CacheConfig())
        for i in range(10):
            c.access(i * 64)
        before = c.stats.misses
        c.flush()
        assert c.occupancy() == 0
        assert c.stats.misses == before          # stats survive a flush
        assert c.access(0)["hit"] is False       # ...but the data does not

    def test_reset_clears_stats_too(self):
        c = SetAssociativeCache(CacheConfig())
        c.access(0)
        c.reset()
        assert c.stats.accesses == 0
        assert c.occupancy() == 0

    def test_probe_does_not_mutate_anything(self):
        c = SetAssociativeCache(CacheConfig())
        c.access(0)
        snapshot = c.stats.to_dict()
        assert c.probe(0) is not None
        assert c.probe(1 << 20) is None
        assert c.stats.to_dict() == snapshot

    def test_to_dict_expands_only_the_focus_set(self):
        c = SetAssociativeCache(CacheConfig(size_kb=8))     # 32 sets
        d = c.to_dict(focus_set=3)
        assert d["focus_set"] == 3
        assert len(d["ways"]) == c.config.assoc
        # No focus -> no per-way payload at all (keeps the snapshot small).
        assert "ways" not in c.to_dict()

    def test_to_dict_ignores_an_out_of_range_focus_set(self):
        c = SetAssociativeCache(CacheConfig())
        assert "ways" not in c.to_dict(focus_set=10 ** 6)
        assert "ways" not in c.to_dict(focus_set=-1)

    def test_set_view_reports_tag_valid_dirty(self):
        cfg = CacheConfig(size_kb=1, line_bytes=64, assoc=2)
        c = SetAssociativeCache(cfg)
        c.access(0, is_write=True)
        entry = next(w for w in c.set_view(0) if w["valid"])
        assert entry["dirty"] is True
        assert entry["tag"] == cfg.tag_of(0)

    def test_invalid_ways_report_no_tag(self):
        c = SetAssociativeCache(CacheConfig(size_kb=1, line_bytes=64, assoc=2))
        assert all(w["tag"] is None for w in c.set_view(0))


# ════════════════════════════════════════════════════════════════════════
# 7. CachedDataMemory — the DataMemory drop-in
# ════════════════════════════════════════════════════════════════════════
class TestCachedDataMemoryComponent:

    def test_presents_every_datamemory_port(self):
        from sim.components.memory.dmem import DataMemory
        cached = set(CachedDataMemory.ports_spec)
        assert set(DataMemory.ports_spec) <= cached
        assert {"stall", "ready"} <= cached          # plus the timing additions

    def test_get_state_carries_every_datamemory_key(self):
        from sim.components.memory.dmem import DataMemory
        plain, cached = DataMemory(), CachedDataMemory()
        plain.evaluate(); cached.evaluate()
        assert set(plain.get_state()) <= set(cached.get_state())

    def test_read_write_behaves_exactly_like_datamemory(self):
        from sim.components.memory.dmem import DataMemory
        plain, cached = DataMemory(), CachedDataMemory()
        for comp in (plain, cached):
            comp["addr"] = 0x40
            comp["wdata"] = 1234
            comp["wen"] = 1
            comp.evaluate()
            comp.rising_edge()
            comp["wen"] = 0
            comp.evaluate()
        assert cached["rdata"] == plain["rdata"] == 1234

    def test_waddr_valid_selects_the_dedicated_write_address(self):
        d = CachedDataMemory()
        d["addr"] = 0x80
        d["waddr"] = 0x00
        d["waddr_valid"] = 1
        d["wdata"] = 7
        d["wen"] = 1
        d.evaluate(); d.rising_edge()
        assert d._mem[0] == 7           # wrote to waddr, not addr

    def test_observe_only_mode_never_stalls(self):
        d = CachedDataMemory(timing_enabled=False)
        for i in range(20):
            d["addr"] = i * 64          # every access is a cold miss
            d.evaluate()
            assert d["stall"] == 0
            assert d["ready"] == 1
            d.rising_edge()
        assert d.cache.stats.misses > 0     # ...yet misses are still counted

    def test_statistics_accumulate_through_the_component(self):
        d = CachedDataMemory()
        for i in range(32):                 # 32 words = 128B = 2x64B lines
            d["addr"] = i * 4
            d.evaluate(); d.rising_edge()
        assert d.cache.stats.accesses == 32
        assert d.cache.stats.misses == 2

    def test_repeated_identical_address_is_not_counted_as_new_traffic(self):
        """The datapath drives `addr` every cycle even for an ADD, so a held
        address must not inflate the access count (see _is_reading)."""
        d = CachedDataMemory()
        d["addr"] = 0x40
        for _ in range(10):
            d.evaluate(); d.rising_edge()
        assert d.cache.stats.accesses == 1

    def test_state_reports_hit_miss_and_the_focused_set(self):
        d = CachedDataMemory()
        d["addr"] = 0x40
        d.evaluate(); d.rising_edge()
        st = d.get_state()
        assert st["cache_result"] == "MISS"
        assert st["cache"]["config"]["assoc"] == d.cache.config.assoc
        assert st["cache"]["focus_set"] == d.cache.config.set_of(0x40)
        # A DIFFERENT address in the same line hits (spatial locality).
        d["addr"] = 0x44
        d.evaluate(); d.rising_edge()
        assert d.get_state()["cache_result"] == "HIT"

    def test_repeat_load_to_the_same_address_is_undercounted_without_the_port(self):
        """Pins the documented fallback limitation (see _is_reading).

        With no read-enable wired, a held address is treated as idle, so two
        consecutive loads from the SAME address count as ONE access. This
        UNDERCOUNTS rather than inflating, which is the safer error — but it is
        a real approximation and must not be mistaken for exactness.
        """
        d = CachedDataMemory()
        d["addr"] = 0x40
        for _ in range(5):
            d.evaluate(); d.rising_edge()
        assert d.cache.stats.accesses == 1      # not 5

    def test_wiring_mem_read_gives_exact_load_accounting(self):
        """The opt-in exact path: with mem_read wired, repeats are counted."""
        d = CachedDataMemory(use_mem_read_port=True)
        d["addr"] = 0x40
        d["mem_read"] = 1
        for _ in range(5):
            d.evaluate(); d.rising_edge()
        assert d.cache.stats.accesses == 5
        assert d.cache.stats.read_misses == 1    # cold
        assert d.cache.stats.read_hits == 4      # the repeats now register

    def test_mem_read_low_means_no_traffic_even_on_a_new_address(self):
        d = CachedDataMemory(use_mem_read_port=True)
        for i in range(10):
            d["addr"] = i * 64          # ALU results, not loads
            d["mem_read"] = 0
            d.evaluate(); d.rising_edge()
        assert d.cache.stats.accesses == 0

    def test_idle_state_before_any_access(self):
        assert CachedDataMemory().get_state()["cache_result"] == "IDLE"

    def test_cache_result_reports_THIS_cycle_not_the_last_access(self):
        """REGRESSION PIN (fixed 2026-08-08).

        `_last` is sticky on purpose (the UI wants to know where the most
        recent access landed), but the HIT/MISS badge was derived from it
        directly, so it kept reporting the previous access forever. Scrubbing
        through cycles showed "HIT" on every non-memory instruction — an ADD
        displayed the previous load's result.
        """
        d = CachedDataMemory(use_mem_read_port=True)
        d["addr"] = 0x40
        d["mem_read"] = 1
        d.evaluate(); d.rising_edge()
        assert d.get_state()["cache_result"] == "MISS"

        # A non-memory cycle: same address on the bus, but mem_read is low.
        d["mem_read"] = 0
        d.evaluate(); d.rising_edge()
        assert d.get_state()["cache_result"] == "IDLE", (
            "badge is sticky — it is reporting the previous access")
        # ...while last_access is still available for the set-array view.
        # 0x40 with 64B lines is line 1 -> set 1 (not 0).
        assert d.get_state()["last_access"]["set"] == d.cache.config.set_of(0x40)

    def test_idle_cycles_dominate_a_realistic_program(self):
        """End-to-end version: most cycles are not memory accesses at all."""
        from collections import Counter
        asm = ("ADDI x1, x0, 0\n"
               "SW   x1, 0(x0)\n"
               "ADDI x2, x0, 1\n"
               "SW   x2, 4(x0)\n"
               "ADDI x4, x0, 9\n"
               "ADDI x5, x0, 8\n")
        dmem = CachedDataMemory(CacheConfig())
        states = run_simulation(
            PRESETS["riscv/single_cycle"]["build"](assemble("riscv", asm), dmem=dmem),
            num_cycles=60, include_reset=True, until_stable=True)
        counts = Counter(s.get("dmem", {}).get("cache_result") for s in states)
        # Two stores to the same 64B line: one cold miss, then a hit.
        assert counts["MISS"] == 1
        assert counts["HIT"] == 1
        assert counts["IDLE"] == len(states) - 2

    def test_addresses_in_one_line_share_set_and_tag_and_so_hit(self):
        """mem[0] and mem[4] are the SAME line, not a conflict.

        Same set + same tag means one line (spatial locality -> a hit). Same
        set + DIFFERENT tag is the conflict case. Worth pinning because the two
        look identical in the UI's tag column.
        """
        cfg = CacheConfig()                      # 64B lines
        assert cfg.set_of(0x0) == cfg.set_of(0x4)
        assert cfg.tag_of(0x0) == cfg.tag_of(0x4)
        c = SetAssociativeCache(cfg)
        assert c.access(0x0, is_write=True)["hit"] is False   # cold
        assert c.access(0x4, is_write=True)["hit"] is True    # same line

        # Contrast: same SET, different TAG -> a genuine conflict.
        stride = cfg.line_bytes * cfg.num_sets
        assert cfg.set_of(0x0) == cfg.set_of(stride)
        assert cfg.tag_of(0x0) != cfg.tag_of(stride)


# ════════════════════════════════════════════════════════════════════════
# 8. ORACLE EQUIVALENCE — the property everything else rests on
# ════════════════════════════════════════════════════════════════════════
PROGRAMS = {
    "riscv": ("ADDI x1,x0,16\nADDI x2,x0,77\nSW x2,0(x1)\nLW x3,0(x1)\n"
              "ADD x4,x3,x2\nSW x4,64(x1)\nLW x5,64(x1)\nADD x6,x5,x4\n"),
    "arm":   ("MOVZ X1,#16\nMOVZ X2,#77\nSTR X2,[X1]\nLDR X3,[X1]\n"
              "ADD X4,X3,X2\nSTR X4,[X1,#64]\nLDR X5,[X1,#64]\nADD X6,X5,X4\n"),
    "x86":   ("MOV ECX, 16\nMOV EBX, 77\nMOV [ECX], EBX\nMOV EAX, [ECX]\n"
              "ADD EAX, EBX\nMOV [ECX], EAX\n"),
}

CACHE_VARIANTS = [
    pytest.param({"size_kb": 1, "line_bytes": 64, "assoc": 1}, id="direct-mapped"),
    pytest.param({"size_kb": 1, "line_bytes": 64, "assoc": 2}, id="2-way"),
    pytest.param({"size_kb": 1, "line_bytes": 16, "assoc": 4}, id="small-lines"),
    pytest.param({"size_kb": 1, "line_bytes": 64, "assoc": 16}, id="fully-assoc"),
    pytest.param({"policy": "fifo"}, id="fifo"),
    pytest.param({"policy": "plru"}, id="plru"),
    pytest.param({"policy": "random"}, id="random"),
    pytest.param({"write_policy": "write_through"}, id="write-through"),
    pytest.param({"write_allocate": False}, id="no-write-allocate"),
]


def _dmem_for(model, cfg, num_lanes=2):
    if model == "superscalar":
        return CachedMultiPortDataMemory(num_lanes=num_lanes, config=cfg)
    return CachedDataMemory(cfg)


def _run(isa, model, program, dmem=None, cycles=140):
    build = PRESETS[f"{isa}/{model}"]["build"]
    kw = {"num_lanes": 2} if model == "superscalar" else {}
    if dmem is not None:
        kw["dmem"] = dmem
    states = run_simulation(build(program, **kw), num_cycles=cycles,
                            include_reset=True)
    last = states[-1]
    regs = tuple(last["regfile"]["registers"])
    mem = tuple(last["dmem"]["memory"][:128]) if "dmem" in last else ()
    return regs, mem


class TestOracleEquivalence:
    """The cache MUST be architecturally transparent.

    Identical final registers and memory to an uncached run, for every ISA x
    model x cache geometry. Only cycle COUNTS may differ (and in observe-only
    mode not even those). Everything the timing model will later do rests on
    this, so it is asserted across the full 18-preset matrix.
    """

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("isa", ISAS)
    def test_cached_run_matches_uncached_on_every_preset(self, isa, model):
        program = assemble(isa, PROGRAMS[isa])
        base = _run(isa, model, program)
        cached = _run(isa, model, program,
                      dmem=_dmem_for(model, CacheConfig(size_kb=1, assoc=2)))
        assert cached == base, f"{isa}/{model} diverged with a cache attached"

    @pytest.mark.parametrize("variant", CACHE_VARIANTS)
    @pytest.mark.parametrize("isa", ISAS)
    def test_every_cache_geometry_is_transparent(self, isa, variant):
        """Geometry and policy must never change the architectural result."""
        program = assemble(isa, PROGRAMS[isa])
        base = _run(isa, "single_cycle", program)
        cached = _run(isa, "single_cycle", program,
                      dmem=CachedDataMemory(CacheConfig(**variant)))
        assert cached == base, f"{isa} diverged with cache {variant}"

    @pytest.mark.parametrize("isa", ISAS)
    def test_observe_only_does_not_change_the_cycle_count(self, isa):
        """Step 1 is timing-NEUTRAL: same states, same length."""
        program = assemble(isa, PROGRAMS[isa])
        build = PRESETS[f"{isa}/pipeline"]["build"]
        plain = run_simulation(build(program), num_cycles=60, include_reset=True)
        cached = run_simulation(build(program, dmem=CachedDataMemory()),
                               num_cycles=60, include_reset=True)
        assert len(plain) == len(cached)

    @pytest.mark.parametrize("isa", ISAS)
    def test_the_cache_actually_saw_traffic(self, isa):
        """Guard against a vacuous pass: if the cache recorded no accesses,
        equivalence would be trivially true and prove nothing."""
        program = assemble(isa, PROGRAMS[isa])
        dmem = CachedDataMemory()
        run_simulation(PRESETS[f"{isa}/single_cycle"]["build"](program, dmem=dmem),
                       num_cycles=140, include_reset=True)
        assert dmem.cache.stats.accesses > 0
        assert dmem.cache.occupancy() > 0

    def test_memory_snapshot_keys_still_drive_settle_detection(self):
        """`_arch_signature` reads memory + memory_hi; both must be present or
        run_to_completion would settle immediately."""
        from sim.runner_v2 import _arch_signature
        program = assemble("riscv", PROGRAMS["riscv"])
        states = run_simulation(
            PRESETS["riscv/single_cycle"]["build"](program, dmem=CachedDataMemory()),
            num_cycles=20, include_reset=True)
        sig = _arch_signature(states[-1])
        assert any(entry[1] == "m" for entry in sig), "no memory in the signature"

    @pytest.mark.parametrize("isa", ISAS)
    def test_run_to_completion_still_terminates_with_a_cache(self, isa):
        program = assemble(isa, PROGRAMS[isa])
        states = run_simulation(
            PRESETS[f"{isa}/single_cycle"]["build"](program, dmem=CachedDataMemory()),
            num_cycles=4000, include_reset=True, until_stable=True)
        assert len(states) < 4000, "settle detection failed with a cache attached"


class TestMultiPortCachedMemory:

    def test_presents_every_multiport_port(self):
        from sim.components.memory.dmem import MultiPortDataMemory
        plain = MultiPortDataMemory(num_lanes=3)
        cached = CachedMultiPortDataMemory(num_lanes=3)
        assert set(plain.ports_spec) <= set(cached.ports_spec)

    @pytest.mark.parametrize("lanes", [1, 2, 4])
    def test_lane_count_is_honoured(self, lanes):
        c = CachedMultiPortDataMemory(num_lanes=lanes)
        for i in range(lanes):
            assert f"addr_{i}" in c.ports_spec
        assert f"addr_{lanes}" not in c.ports_spec

    def test_same_line_in_one_group_costs_one_miss_then_a_hit(self):
        """A shared tag array sees lane order, so two lanes on one line do not
        both miss."""
        c = CachedMultiPortDataMemory(num_lanes=2)
        c["addr_0"] = 0x00
        c["addr_1"] = 0x04          # same 64B line
        c["mem_read_0"] = 1
        c["mem_read_1"] = 1
        c.evaluate(); c.rising_edge()
        assert c.cache.stats.misses == 1
        assert c.cache.stats.hits == 1

    def test_older_lane_store_forwards_to_younger_lane_load(self):
        c = CachedMultiPortDataMemory(num_lanes=2)
        c["addr_0"] = 0x40
        c["wdata_0"] = 999
        c["mem_write_0"] = 1
        c["addr_1"] = 0x40
        c["mem_read_1"] = 1
        c.evaluate()
        assert c["rdata_1"] == 999      # sees the older same-group store

    def test_idle_lanes_generate_no_cache_traffic(self):
        c = CachedMultiPortDataMemory(num_lanes=2)
        c["addr_0"] = 0x40              # address present but no read/write flag
        c.evaluate(); c.rising_edge()
        assert c.cache.stats.accesses == 0


class TestExactAccessAccounting:
    """The hit rate must count REAL loads/stores, not bus noise.

    ``dmem.addr`` is driven from the ALU every cycle, so on an ``ADD`` the
    address bus carries an arithmetic result. Counting those as memory traffic
    made the headline statistic wrong by 60% on the loop below (2562 counted
    vs 1024 real), which would have shipped a plausible-looking but misleading
    hit rate.

    Fixed by having the presets wire their co-timed read-enable into the
    cache's optional ``mem_read`` port and flip it to exact accounting:
      single_cycle / configurable  ->  decode.mem_read
      pipeline                     ->  ex_mem.mem_read_out
      multicycle                   ->  mc_ctrl.mem_read_en
      superscalar                  ->  already exact (per-lane mem_read ports)
      ooo                          ->  still heuristic (see the test below)
    """

    # 512 iterations x (1 SW + 1 LW) = 1024 real accesses over a 2KB array.
    LOOP = ("ADDI x1, x0, 0\n"
            "ADDI x2, x0, 512\n"
            "ADDI x3, x0, 1024\n"
            "loop:\n"
            "SLLI x4, x1, 2\n"
            "ADD  x5, x3, x4\n"
            "SW   x1, 0(x5)\n"
            "LW   x6, 0(x5)\n"
            "ADDI x1, x1, 1\n"
            "BLT  x1, x2, loop\n")
    REAL_ACCESSES = 1024
    # 2KB touched / 64B lines = 32 lines -> exactly one cold miss each.
    COLD_MISSES = 32

    def _run(self, model, cfg=None):
        program = assemble("riscv", self.LOOP)
        dmem = CachedDataMemory(cfg or CacheConfig(size_kb=8, line_bytes=64, assoc=4))
        run_simulation(PRESETS[f"riscv/{model}"]["build"](program, dmem=dmem),
                       num_cycles=12000, include_reset=True, until_stable=True)
        return dmem

    @pytest.mark.parametrize("model", ["single_cycle", "configurable",
                                       "pipeline", "multicycle"])
    def test_access_count_matches_the_real_load_store_count(self, model):
        dmem = self._run(model)
        assert dmem.use_mem_read_port, f"{model} did not enable exact accounting"
        assert dmem.cache.stats.accesses == self.REAL_ACCESSES

    @pytest.mark.parametrize("model", ["single_cycle", "configurable",
                                       "pipeline", "multicycle"])
    def test_miss_count_is_one_per_line_of_the_array(self, model):
        """Arithmetic, not an inequality: 2KB / 64B = 32 cold misses."""
        assert self._run(model).cache.stats.misses == self.COLD_MISSES

    def test_every_exact_model_agrees_on_the_statistics(self):
        """Cache behaviour depends on the ADDRESS STREAM, which is the same
        program on every model — so the numbers must match exactly."""
        stats = {}
        for model in ("single_cycle", "configurable", "pipeline", "multicycle"):
            s = self._run(model).cache.stats
            stats[model] = (s.accesses, s.misses, s.hits)
        assert len(set(stats.values())) == 1, stats

    def test_ooo_still_uses_the_heuristic_and_overcounts(self):
        """Pins the remaining gap honestly rather than leaving it unstated.

        OoO's load path runs through the reservation station / ROB with no
        single co-timed read-enable at the dmem port, so it keeps the
        address-change heuristic and OVERCOUNTS (ALU results on the shared
        bus). If OoO ever gains an exact signal this test fails and should be
        folded into the parameterised cases above.
        """
        dmem = self._run("ooo")
        assert not dmem.use_mem_read_port
        assert dmem.cache.stats.accesses > self.REAL_ACCESSES

    def test_wiring_did_not_change_the_architectural_result(self):
        """The new wire must be observation-only."""
        program = assemble("riscv", self.LOOP)
        for model in ("single_cycle", "pipeline", "multicycle"):
            base = _run("riscv", model, program, cycles=12000)
            cached = _run("riscv", model, program,
                          dmem=CachedDataMemory(), cycles=12000)
            assert base == cached, model

    @pytest.mark.parametrize("line_bytes,expected_misses", [
        (16, 128),      # 2048B / 16B = 128 lines, one cold miss each
        (32, 64),
        (64, 32),
        (128, 16),
    ])
    def test_line_size_sets_the_miss_count_exactly(self, line_bytes,
                                                   expected_misses):
        """Bigger lines capture more spatial locality — exact arithmetic.

        The loop sweeps a 2KB array once, touching every word twice (SW then
        LW), so the miss count is purely 2048/line_bytes. Note this sweep has
        NO conflict misses (it never revisits a set with a new tag), which is
        why associativity is irrelevant here and line size is everything —
        conflict behaviour is covered by TestConflictMisses at the unit level.
        """
        cfg = CacheConfig(size_kb=8, line_bytes=line_bytes, assoc=4)
        stats = self._run("single_cycle", cfg).cache.stats
        assert stats.misses == expected_misses
        assert stats.accesses == self.REAL_ACCESSES
        assert stats.hits == self.REAL_ACCESSES - expected_misses

    def test_a_sequential_sweep_gains_nothing_from_associativity(self):
        """The flip side, asserted rather than assumed: no conflicts means
        assoc 1/2/4 are identical. A test claiming otherwise would be wrong."""
        rates = [self._run("single_cycle",
                           CacheConfig(size_kb=1, line_bytes=16, assoc=a))
                 .cache.stats.hit_rate
                 for a in (1, 2, 4)]
        assert len(set(rates)) == 1, rates


class TestPresetWiringIsOptional:
    """Plain DataMemory has no mem_read port; the wire must stay conditional."""

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("isa", ISAS)
    def test_uncached_presets_still_build(self, isa, model):
        program = assemble(isa, PROGRAMS[isa])
        kw = {"num_lanes": 2} if model == "superscalar" else {}
        PRESETS[f"{isa}/{model}"]["build"](program, **kw)   # must not raise

    @pytest.mark.parametrize("isa", ISAS)
    def test_plain_datamemory_has_no_mem_read_port(self, isa):
        from sim.components.memory.dmem import DataMemory
        assert "mem_read" not in DataMemory.ports_spec


class TestTimingModel:
    """Step 2: a miss STALLS the CPU, so cycle counts change.

    TIMING CONTRACT: a hit occupies 1 cycle, a miss occupies 1 + miss_penalty.
    So total cycles must grow by EXACTLY `misses x miss_penalty` while the
    architectural result stays identical to an uncached run.

    Only `single_cycle` wires the stall path today (see TIMING_MODELS). The
    guard tests below make sure the unsupported models cannot be used by
    accident — enabling timing where the preset ignores `stall` corrupts
    ARCHITECTURAL results, not just cycle counts, which is far worse than a
    missing feature.
    """

    LOOP = ("ADDI x1, x0, 0\n"
            "ADDI x2, x0, 128\n"
            "ADDI x3, x0, 1024\n"
            "loop:\n"
            "SLLI x4, x1, 2\n"
            "ADD  x5, x3, x4\n"
            "SW   x1, 0(x5)\n"
            "LW   x6, 0(x5)\n"
            "ADD  x7, x7, x6\n"
            "ADDI x1, x1, 1\n"
            "BLT  x1, x2, loop\n")
    SUM_0_TO_127 = 8128

    def _run(self, cfg=None, timing=False):
        program = assemble("riscv", self.LOOP)
        build = PRESETS["riscv/single_cycle"]["build"]
        dmem = (CachedDataMemory(cfg, timing_enabled=timing)
                if cfg is not None else None)
        kw = {"dmem": dmem} if dmem is not None else {}
        states = run_simulation(build(program, **kw), num_cycles=80000,
                                include_reset=True, until_stable=True)
        return states, dmem

    def test_baseline_program_is_correct(self):
        states, _ = self._run()
        assert states[-1]["regfile"]["registers"][7] == self.SUM_0_TO_127

    @pytest.mark.parametrize("penalty", [1, 5, 10, 25, 50])
    @pytest.mark.parametrize("line_bytes", [16, 64, 128])
    def test_extra_cycles_equal_misses_times_penalty(self, line_bytes, penalty):
        base, _ = self._run()
        cfg = CacheConfig(size_kb=8, line_bytes=line_bytes, assoc=4,
                          miss_penalty=penalty)
        timed, dmem = self._run(cfg, timing=True)
        added = len(timed) - len(base)
        assert added == dmem.cache.stats.misses * penalty, (
            f"{line_bytes}B/P={penalty}: +{added} cycles for "
            f"{dmem.cache.stats.misses} misses")

    @pytest.mark.parametrize("penalty", [1, 10, 50])
    @pytest.mark.parametrize("line_bytes", [16, 64, 128])
    def test_timing_never_changes_the_architectural_result(self, line_bytes,
                                                          penalty):
        """The whole point: cycles move, results do not."""
        base, _ = self._run()
        cfg = CacheConfig(size_kb=8, line_bytes=line_bytes, assoc=4,
                          miss_penalty=penalty)
        timed, _ = self._run(cfg, timing=True)
        assert (timed[-1]["regfile"]["registers"]
                == base[-1]["regfile"]["registers"])
        assert timed[-1]["dmem"]["memory"][:400] == base[-1]["dmem"]["memory"][:400]

    def test_zero_penalty_is_cycle_neutral(self):
        base, _ = self._run()
        timed, _ = self._run(CacheConfig(miss_penalty=0), timing=True)
        assert len(timed) == len(base)

    def test_stall_cycles_stat_matches_the_added_cycles(self):
        base, _ = self._run()
        timed, dmem = self._run(CacheConfig(size_kb=8, line_bytes=64, assoc=4,
                                            miss_penalty=10), timing=True)
        assert dmem.cache.stats.stall_cycles == len(timed) - len(base)

    def test_a_better_cache_is_actually_FASTER(self):
        """Cycle-level monotonicity — the payoff, and the anti-regression for
        the same class of bug as 'superscalar 4 lanes slower than 2'."""
        cycles = []
        for line_bytes in (16, 32, 64, 128):
            cfg = CacheConfig(size_kb=8, line_bytes=line_bytes, assoc=4,
                              miss_penalty=20)
            timed, _ = self._run(cfg, timing=True)
            cycles.append(len(timed))
        assert cycles == sorted(cycles, reverse=True), cycles

    def test_a_bigger_penalty_is_never_faster(self):
        cycles = [len(self._run(CacheConfig(size_kb=8, line_bytes=64, assoc=4,
                                            miss_penalty=p), timing=True)[0])
                  for p in (0, 5, 10, 20, 40)]
        assert cycles == sorted(cycles), cycles

    @pytest.mark.parametrize("penalty", [1, 3, 10])
    def test_a_single_miss_costs_exactly_one_plus_penalty_cycles(self, penalty):
        """Unit-level occupancy check, independent of any program."""
        d = CachedDataMemory(CacheConfig(miss_penalty=penalty),
                             timing_enabled=True, use_mem_read_port=True)
        d["addr"] = 0x40
        d["mem_read"] = 1
        occupancy = 0
        for _ in range(penalty + 5):
            d.evaluate()
            occupancy += 1
            if d["ready"]:
                break
            d.rising_edge()
        assert occupancy == 1 + penalty
        assert d.cache.stats.stall_cycles == penalty
        assert d.cache.stats.accesses == 1      # counted once, not per stall


class TestTimingSupportGuard:
    """`timing_enabled` on a model that ignores `stall` corrupts RESULTS.

    That is worse than an unimplemented feature, so it must be impossible to
    reach by accident rather than merely documented.
    """

    def test_only_single_cycle_is_advertised(self):
        from sim.components.memory.cache import TIMING_MODELS, supports_timing
        assert TIMING_MODELS == ("single_cycle",)
        assert supports_timing("single_cycle")
        for model in ("configurable", "pipeline", "multicycle", "ooo",
                      "superscalar"):
            assert not supports_timing(model)

    @pytest.mark.parametrize("model", ["configurable", "pipeline", "multicycle",
                                       "ooo", "superscalar"])
    def test_factory_refuses_timing_on_unsupported_models(self, model):
        from sim.components.memory.cache import cached_dmem_for
        with pytest.raises(ValueError, match="not supported"):
            cached_dmem_for(model, timing_enabled=True)

    @pytest.mark.parametrize("model", MODELS)
    def test_factory_allows_observe_only_everywhere(self, model):
        from sim.components.memory.cache import cached_dmem_for
        dmem = cached_dmem_for(model, CacheConfig())
        assert dmem is not None
        assert not getattr(dmem, "timing_enabled", False)

    def test_factory_returns_the_multiport_variant_for_superscalar(self):
        from sim.components.memory.cache import cached_dmem_for
        assert isinstance(cached_dmem_for("superscalar", num_lanes=3),
                          CachedMultiPortDataMemory)

    def test_configurable_corruption_is_why_it_is_excluded(self):
        """Pins the ACTUAL failure that got `configurable` excluded.

        Bypassing the factory and forcing timing on `configurable` diverges
        from the oracle. If this ever starts matching, the stall path has been
        wired into the configurable branch and it can join TIMING_MODELS.
        """
        program = assemble("riscv", TestTimingModel.LOOP)
        costs = {"alu": 1, "load": 1, "store": 1, "branch": 1}
        build = PRESETS["riscv/configurable"]["build"]
        base = run_simulation(build(program, cycle_costs=costs),
                              num_cycles=80000, include_reset=True,
                              until_stable=True)
        forced = run_simulation(
            build(program, cycle_costs=costs,
                  dmem=CachedDataMemory(CacheConfig(miss_penalty=10),
                                        timing_enabled=True)),
            num_cycles=80000, include_reset=True, until_stable=True)
        assert (forced[-1]["regfile"]["registers"]
                != base[-1]["regfile"]["registers"]), (
            "configurable + timing now agrees with the oracle — wire the stall "
            "path into the configurable branch and add it to TIMING_MODELS")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
