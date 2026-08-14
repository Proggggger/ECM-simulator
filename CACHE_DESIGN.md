# Cache / Memory Hierarchy — Design

Decision: **timing-accurate** (a miss stalls the CPU, so cycle counts change
and `/compare` becomes meaningful).

## Status

- **Step 1 — DONE** (observe-only cache, proven transparent on all 18 presets).
- **Step 2 — PARTIAL.** Timing works and is exact on **`single_cycle` only**.
  `pipeline` was attempted and reverted; `configurable`, `multicycle`, `ooo`,
  `superscalar` are not wired. Guarded by `cached_dmem_for()` so an unsupported
  combination raises instead of corrupting results.
- **Steps 4+5 (API + UI) — DONE.** `/cache/policies`, `/simulate`'s `cache`
  field + `cache_stats`, and the sidebar panel with a live set-array view.
  105 tests in `tests/test_cache_api.py`.
- Step 3 (L2 via `CacheHierarchy`) — pending. Extra policies landed in step 1.

252 tests in `tests/test_cache.py` + 105 in `tests/test_cache_api.py`.

### API

| route | purpose |
|---|---|
| `GET /cache/policies` | auto-discovered policies, legal pow2 geometry lists, defaults, `timing_models` |
| `POST /simulate` | accepts `cache` (object or `true`); returns `cache_stats`; per-cycle `dmem.cache` / `cache_result` / `last_access` |

Every documented rejection path returns JSON 400 — illegal geometry
(non-pow2, out of range, fewer lines than ways), unknown policy, wrong types,
the non-standard `Infinity` literal, and timing on an unsupported model.

### UI

Sidebar `#cache-panel`, reusing the configurable panel's `cfg-*` knob and
readout language so the two read as one system. The only new visual is the
**set array**: the accessed set drawn way-by-way with tag + valid/dirty + LRU
age, teal on access and rose on eviction, so a conflict miss is something you
watch happen rather than a number that changed.

Two details worth keeping:
- Geometry sliders are **log2** (the slider value is an exponent) because only
  powers of two are legal; the labels show the real values and the legal lists
  come from the server, so the limits have exactly one source of truth.
- The cache sliders reuse `.cfg-range` for styling, so the configurable
  handler's selector had to become `.cfg-range:not([data-cache])` — otherwise
  both handlers fire on one drag. Pinned by a test.

### Step 2: what works

On `single_cycle`, with `timing_enabled=True`:

**TIMING CONTRACT** — a hit occupies 1 cycle, a miss occupies `1 + miss_penalty`.
Verified across line sizes {16, 64, 128} x penalties {1, 5, 10, 25, 50}:

```
added_cycles == misses x miss_penalty        (exactly, every configuration)
architectural result == uncached run         (registers + memory)
stall_cycles stat    == added_cycles
```

Plus cycle-level monotonicity: a bigger line size is always *faster*, and a
bigger penalty is never faster. That is the anti-regression for the same class
of bug as the old "superscalar 4 lanes slower than 2".

### Step 2: what does not work, and why

**`pipeline` — attempted, reverted.** A MEM-stage stall needs far more than a
stall wire, and three of the four required pieces are now in place:

| requirement | status |
|---|---|
| `EX_MEM.stall` (hold the instruction in MEM) | port added |
| `MEM_WB.stall` (bubble — nothing completes) | port added |
| `ID_EX.hold` — distinct from `stall` | port added |
| gated branch redirect | wired |

The `ID_EX` distinction is the subtle one: `id_ex.stall` **inserts a bubble**
(that is how the load-use interlock works), so driving it from a cache stall
*destroys* the instruction in EX. A separate `hold` was needed. Even so, a
program-level divergence remains (off by one loop iteration's value on a
strided loop, and one extra counted miss), so the wiring was reverted rather
than shipped. All three new ports default to 0 when unwired, so they are inert
— confirmed by the full suite.

One real bug was found and fixed along the way: the store commit was being
**deferred** until the fill completed, which opens a stale-read window for a
younger load reading `_mem` combinationally. Data correctness must not depend
on timing in a single-array model — the stall models *when an instruction
retires*, never *what memory holds*. Fixing this also fixed `single_cycle`'s
largest-penalty case.

**`configurable` — excluded deliberately.** It shares `single_cycle`'s datapath
but routes commits through `CycleBudgetController`, and the cache stall is not
wired into that branch. Forcing timing there produced **wrong values and zero
cycle change** — pinned by
`test_configurable_corruption_is_why_it_is_excluded`, which fails if it ever
starts agreeing with the oracle.

Because that failure mode corrupts *architectural results* rather than merely
cycle counts, `cached_dmem_for(model, ..., timing_enabled=True)` raises for
every unsupported model. Observe-only mode remains valid on all 18.


### What step 1 delivered

```
sim/components/memory/cache/
  model.py            CacheConfig + CacheStats + SetAssociativeCache
  cached_dmem.py      CachedDataMemory        — the dmem= drop-in
  cached_multiport.py CachedMultiPortDataMemory — superscalar's per-lane variant
  policies/           base + lru, fifo, plru, random (auto-discovered)
```

Verified properties:
- **Architecturally transparent on all 18 presets** — identical final registers
  and memory to an uncached run, across 9 cache geometries/policies.
- Observe-only mode does not change the cycle count.
- Hit/miss counts match hand-computed arithmetic (one miss per line on a
  stride-1 sweep; exact miss counts for 16/32/64/128B lines).
- LRU vs FIFO provably disagree on a reuse trace; LRU's cyclic-sweep
  pathology (assoc+1 lines → 100% miss) is pinned; random survives it.
- Monotonicity: more capacity or associativity never increases misses.

### Two things found while building it

**1. Superscalar needs its own port contract.** The superscalar presets wire a
per-lane memory, so the single-port cache cannot drop in — hence
`CachedMultiPortDataMemory`. It feeds one shared cache in lane order, so two
lanes touching the same line in a cycle cost one miss + one hit (as a shared
tag array would), not two misses.

**2. The hit rate was 60% noise until the read-enable was wired.** `dmem.addr`
is driven from the ALU *every* cycle, so on an `ADD` the address bus carries an
arithmetic result. Counting those as memory traffic gave 2562 "accesses" on a
loop with 1024 real ones. Fixed by giving the cache an optional `mem_read`
port and having each preset wire its co-timed read-enable:

| model | signal | status |
|---|---|---|
| single_cycle / configurable | `decode.mem_read` | exact |
| pipeline | `ex_mem.mem_read_out` | exact |
| multicycle | `mc_ctrl.mem_read_en` | exact |
| superscalar | per-lane `mem_read_*` | exact by construction |
| ooo | — | **still heuristic, overcounts** |

The wire is conditional (`if "mem_read" in dmem.ports_spec`) so plain
`DataMemory` is unaffected. All four exact models now agree on identical
statistics for the same program, which is the correct invariant: cache
behaviour depends on the address stream, not the execution model.

OoO's load path runs through the RS/ROB with no single co-timed read-enable at
the dmem port, so it keeps the address-change heuristic and overcounts. Pinned
by `test_ooo_still_uses_the_heuristic_and_overcounts` rather than left unstated.

## Why this slots in cleanly

Two facts about the existing codebase make this a low-risk feature:

1. **Every one of the 18 presets already accepts `dmem=`.** Verified:
   `build(program, branch_predictor=None, dmem=None, alu=None, ...)` on all
   3 ISAs x 6 models. A cache that presents `DataMemory`'s port contract can be
   injected without editing a single preset file.
2. **`CycleBudgetController` already proves the stall-and-commit pattern.**
   It holds an instruction in flight (`stall=1`), commits exactly once on the
   final cycle (`commit=1` gating reg write / mem write / PC redirect), and
   *only the cycle count changes* — the architectural result stays identical to
   the single-cycle oracle. A cache miss penalty is the same shape: stall for
   N cycles, then let the access complete.

The branch predictors give the third precedent: a swappable, auto-discovered
(`/predictors` scans the package), UI-selectable **policy plugin** with a base
class defining the port contract. Cache replacement policies should follow it
exactly.

## Component design

New package `sim/components/memory/cache/`:

```
base.py            CacheBase — port contract + tag/index/offset math + stats
direct_mapped.py   1-way
set_associative.py N-way (parameterized; direct-mapped is the N=1 case)
fully_associative.py
policies/
  base.py          ReplacementPolicyBase  (mirrors BranchPredictorBase's shape)
  lru.py  fifo.py  random_.py  plru.py
hierarchy.py       CacheHierarchy — L1 -> (L2) -> DataMemory, the dmem= drop-in
```

### Port contract

`CacheHierarchy` must be a **superset-compatible drop-in** for `DataMemory`:
it keeps `addr / waddr / waddr_valid / wdata / wen / rdata`, and adds:

| Port | Dir | Meaning |
|---|---|---|
| `stall` | out | 1 while a miss is being serviced (hold the PC) |
| `ready` | out | 1 on the cycle the access completes |

`stall` wires to `fetch.stall` exactly as `cyc_ctrl.stall` does today, and the
reg/mem write enables get gated by `ready` through the existing `AndGate`
(`gate_mem_wen` / `gate_reg_wen`) — reusing the machinery already in
`single_cycle.py`'s `configurable` branch rather than inventing a new one.

### Configuration

```python
CacheConfig(
    levels=[
        CacheLevel(size_kb=8, line_bytes=64, assoc=4, policy="lru",
                   write_policy="write_back",   # | write_through
                   write_alloc=True,
                   hit_cycles=1, miss_penalty=10),
    ],
    dram_cycles=100,
)
```

Clamp everything server-side like `cycle_costs` does (`max(1, min(v, 64))`) —
sizes to powers of two, assoc to divisors of the set count, penalties to a sane
ceiling — and return `400 {error}` on garbage, per the API contract.

### Statistics (per level, per cycle)

`get_state()` returns hits, misses, hit rate, evictions, writebacks, plus a
**per-set view of tags + valid/dirty + LRU age** so the UI can draw the set
array and light up the accessed line. Keep the snapshot bounded the way
`memory_snapshot()` does — a 8 KB / 64 B cache is only 128 lines, but a large
config must not bloat every cycle's payload. Emit the touched set densely and
summarize the rest.

## Correctness strategy (this is the part that matters)

The cache must be **architecturally transparent**: same final registers, same
final memory, *different cycle count*. That gives a free, very strong test:

- **Oracle equivalence** — DONE, across all 18 presets x 9 geometries.
- **Cycle monotonicity** — DONE at the model level (more capacity/assoc never
  increases misses). Re-assert on CYCLES once step 2 lands.
- **Hand-computed hit/miss traces** — DONE, exact integers not inequalities.
- **Policy differentiation** — DONE (LRU vs FIFO disagree; LRU pathology pinned).
- **Write-back vs write-through** — DONE at the statistics level. Note the
  modelling limit below.

### Modelling limit: one data array

The cache models METADATA (tags, valid/dirty, replacement) and TIMING, but keeps
exactly ONE authoritative copy of the data — the backing array `DataMemory`
already owns. The hit/miss and eviction sequences depend only on the ADDRESS
stream and the metadata, so nothing is lost in what we report, and architectural
state stays correct by construction (which is what keeps the oracle equivalence
above trivially safe rather than luck).

The consequence: write-back's *delayed* DRAM update is not observable as a
difference in memory CONTENTS mid-run. It is still fully visible in the
writeback COUNT, the eviction stream, and (after step 2) the cycle cost. So
`write_policy` changes timing and statistics, not observable data.

### A note on how the write-hit path was found

Mutation testing caught a hole in the first version of the test suite:
disabling the dirty-bit assignment on a write **hit** passed all 166 tests,
because every write test happened to start from a write *miss* (which dirties
via the allocate path instead). Two tests were added for the
read-then-write-then-evict sequence. Worth repeating when step 2 adds the
stall path.


## UI

A "Memory" / "Cache" panel in the sidebar following `#configurable-panel`'s
established shape (knobs + live readout):

- Knobs: size, line size, associativity, policy, write policy, miss penalty,
  and a levels toggle (L1 / L1+L2).
- Live set-array visualization: sets x ways grid, each cell showing tag +
  valid/dirty, flashing on access, red on eviction.
- Readout: hits / misses / hit rate / writebacks, and **cycles with cache vs
  cycles with ideal memory** — the headline number, mirroring how the
  configurable panel shows cycles vs time.
- Reuse `layout_engine.js` + a new `layouts/cache.js` rather than bespoke SVG.

## API

- `/simulate` accepts `cache_config`; response gains `cache_stats` (per level)
  alongside the existing `time_model`.
- `/compare` gains cache columns so "no cache vs 2-way vs 4-way" is a grid.
- New `/cache/policies` auto-discovered by scanning `policies/`, exactly like
  `/predictors` scans the predictors package.

## Build order

1. `CacheBase` + direct-mapped + LRU, **observe-only** (no stall), injected via
   `dmem=`. Prove oracle equivalence across all 18 presets first.
2. Add the stall path. Re-prove oracle equivalence; now cycle counts move.
3. N-way + the remaining policies + write-back/write-through.
4. L2 via `CacheHierarchy`.
5. API + UI.

Step 1 is where the correctness risk lives; do not skip straight to step 2.
