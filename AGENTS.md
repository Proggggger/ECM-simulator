# AGENTS.md — Developer Guide

Orientation for coding agents working in this repository. Read this before
editing; it captures the invariants that are easy to break silently.

## What this project is

A pure-Python (stdlib-only core) CPU simulator plus a from-scratch C compiler:

- **3 ISAs** — RISC-V RV32I, ARM AArch64, x86 IA-32 — using real encodings.
- **6 execution models** — `single_cycle`, `multicycle` (FetDecExe),
  `configurable`, `pipeline`, `ooo` (Tomasulo), `superscalar` — as per-ISA
  **presets**, giving 18 combinations.
- A Flask web UI that renders the datapath, registers, and memory per cycle.
- Fork of `blakecragen/architecture_simulator`.

`single_cycle` is the **correctness oracle**: every model must commit identical
final registers and data memory. Cycle counts may differ; architectural state
may not.

## Commands

```bash
python3 appctl.py                  # server on http://127.0.0.1:5051
python3 appctl.py run --debug     # Flask debug + reloader
python3 appctl.py test            # full pytest suite (args forwarded)
python3 appctl.py test -k riscv -q
./run_tests.sh                    # pytest tests/ -v

# Backend only, no Flask — the fastest way to check behavior
python3 -m sim.harness --list-presets
python3 -m sim.harness --preset riscv/pipeline --asm prog.s --cycles 100 --json
```

On Windows without `python3`, use `python` or `venv/Scripts/python.exe`.
`appctl.py` is the only supported entry point (`app/flask_app.py` is retired).

## Repo map

| Path | Purpose |
|---|---|
| `sim/component/` | Framework: `Port`, `ComponentBase`, `Wire`, `CPU`, `CPUBuilder`, topology export |
| `sim/components/` | Hardware blocks by family: `alu/`, `branch/` (+ predictors), `fetch/`, `memory/` (+ `cache/`), `multicycle/`, `pipeline/`, `ooo/`, `regfile/`, `superscalar/`, `writeback/` |
| `sim/isa/<isa>/` | Per-ISA `config.py`, `constants.py`, `decoder_component.py`, and `presets/<model>.py` (each exports `build()`) |
| `sim/compiler/` | Core-C lexer → parser → AST → per-ISA codegen (`codegen/riscv.py`, `arm.py`, `x86.py`) |
| `sim/assembler/` | Two-pass per-ISA assemblers + `cheatsheet.py` |
| `sim/harness.py` | Canonical programmatic entry (`simulate`, `build_cpu`, `compile_and_simulate`) and CLI |
| `sim/runner_v2.py` | Tick loop, per-cycle snapshots, stable-state completion detection |
| `api/app.py` | Module-level Flask `app`; all HTTP routes; gunicorn target |
| `ui/templates/`, `ui/static/` | Pages + d3 layout engine; **no bundler** |
| `tests/` | pytest suite; no `conftest.py`, no pytest config |
| `programs/` | Hand-written asm per ISA (`programs/<isa>/…`) and Core-C samples (`programs/c/`) |
| `app/` | **Retired stub — never use** |
| `sim/execution/`, `sim/core/`, `sim/cpu.py` | Legacy Amaranth-era code; not on the live path |
| `sim/gpu/` | WIP: only `GPUSpec` so far (see `GRAPHICS_DESIGN.md`) |

## Data flow

```
Core-C source ──compile_c()──► asm text ──assemble()──► list[int] ──preset build()──► CPU ──run_simulation()──► cycles[]
programs/<isa>/**/*.asm ─────────────────────────────────┘
```

Two front doors over one backend:

1. `sim.harness.simulate(...)` → `SimResult` (registers, memory, snapshots).
2. `api/app.py` routes (thin Flask layer over harness + compiler + assembler).

## Non-negotiable invariants

1. **Oracle parity.** Any datapath/timing change must leave final registers and
   data memory identical to `single_cycle` for all applicable presets. Parity is
   enforced by `tests/test_cycle_state_contract.py`, `test_matrix.py`,
   `test_isa_model_predictor_matrix.py`, `test_program_results.py`,
   `test_example_results.py`.
2. **Topological eval order.** `CPU.tick()` runs three phases: (1) for each
   component in eval order, propagate incoming wires then `evaluate()`;
   (2) propagate all wires again (closes feedback/back-edges); (3) `rising_edge()`
   in order. A wrong order silently produces one-cycle-stale behavior.
3. **Snapshots are JSON.** Every component's `get_state()` must be serializable
   and bounded (dense memory window + sparse `memory_hi`; never dump huge arrays).
   Registers live under the `"registers"` key; the UI and contract tests depend
   on component names and port keys.
4. **Core stays dependency-free.** No numpy/third-party imports under `sim/`.
   Flask/flask-cors/gunicorn belong only to `api/`/deployment.
5. **Keep `api.app:app` module-level.** Gunicorn and tests import it directly.
6. **Prefix-safe URLs.** Templates must use `url_for` and JS must prepend
   `window.CPU_API_URL` (`{{ request.script_root }}`); never hardcode `/…` API
   paths.
7. **Validate at the boundary.** New/modified routes clamp inputs and return
   `400 {"error": ...}` — never a raw 500.
8. **Writes appear next cycle.** Register and memory commits happen on the clock
   edge; snapshots show them one cycle later. Do not "fix" this by peeking.
9. **Don't weaken tests.** If cycles legitimately change, update affected
   `Expected:` program headers/tables; if behavior must stay pinned (known
   limitations), keep the pinning test.

## Component framework contract

From `sim/component/base.py` and `wire.py`:

- Subclass `ComponentBase`; declare `ports_spec` (`Port` width/direction) and
  UI labels (`ui_label`, `ui_category`).
- Access ports via `self["name"]`; writes are masked to the port width.
- Implement `evaluate()` (combinational) and/or `rising_edge()` (stateful).
- Override `get_state()` for the UI; `reset()` zeroes **ports only** — reset
  internal state explicitly if needed.
- `CPUBuilder.add(name, comp).wire("a.port", "b.port").set_eval_order([...]).build()`
  validates ports and that eval order is unique and complete.
- Gotchas: widths truncate silently; unwired inputs keep their last value; when
  two wires drive one port, the later declaration wins; pipeline registers must
  evaluate before consumers; avoid combinational loops (gate writes/branches, not
  the memory write-enable itself).

## Recipes

### Add an instruction

1. Opcode/funct constants: `sim/isa/<isa>/constants.py`.
2. Decode it: `sim/isa/<isa>/decoder_component.py` (all ISAs expose a shared
   control-signal superset; add `AluOp`/`BranchCond` in `sim/core/signals.py`
   only if genuinely new).
3. Encode it: `sim/assembler/<isa>.py` (`_encode` + opcode tables). For x86 also
   update `_estimate_instruction_size` — it **must** agree with `_encode`.
4. Cheat sheet: `sim/assembler/cheatsheet.py`.
5. Tests: decoder suite + assembler round-trip (`tests/test_decoders_*.py`,
   `tests/test_assembler_roundtrip.py`). If Core-C should emit it, extend the
   compiler (below).

### Add an ISA

Clone the shape of `sim/isa/riscv/`: `config.py`, `constants.py`,
`decoder_component.py`, and six `presets/<model>.py` modules exporting
`build(program, ...) -> CPU`. Then register it:

- `sim/harness.py` — `ISA_CONFIGS` and `_ISA_DISPLAY` (presets auto-discover).
- `api/app.py` — `_VALID_ISAS` validation.
- UI — ISA dropdown in `ui/templates/cpu_simulator.html`, `program_lab.js` list,
  and an assembler/compiler backend if Core-C support is expected.
- The registry tests (`tests/test_harness.py`, `test_isa_model_predictor_matrix.py`)
  are the checklist; they will fail until every preset builds.

### Add / change an execution model or preset

- Add `sim/isa/<isa>/presets/<model>.py` with module-level `build()`.
- Add its label to `_MODEL_LABEL` in `sim/harness.py` (this defines `MODELS`).
- Route distinct kwargs (lanes, predictors, budgets, cache) through
  `harness.build_cpu` and the matching branch in `api/app.py`.
- `/topology` has a D3 auto-layout fallback, so a fixed layout is optional; add
  `ui/static/js/layouts/<model>.js` plus selectors in `cpu_simulator.js` /
  `program_lab.js` for a curated diagram.
- It must pass oracle parity and `tests/test_harness.py` preset-count checks.

### Add a Core-C feature

Lexer tables (`sim/compiler/lexer.py`) → AST node with `to_dict()`
(`sim/compiler/ast.py`) → parser production (`sim/compiler/parser.py`) →
dispatch in `sim/compiler/codegen/base.py` (`_gen_stmt` / `_gen_expr`, and
`_children` for traversal) → a new `emit_*` primitive in `base.py` **and all
three** backends (`riscv.py`, `arm.py`, `x86.py`). Keep `source_map` aligned
(x86's NOP peephole calls `_reindex_source_map`). Add tests in
`tests/test_compiler*.py` and `test_codegen_cross_isa.py`.

### Add an API route / field

Put the route in `api/app.py`, clamp inputs, return `400 {"error": ...}` on bad
input, and add coverage under `tests/test_api_*.py`. Changing an existing
response shape means updating `tests/test_api_response_contract.py` and the
matching UI consumer. There is **no `/step` endpoint**: stepping is client-side
scrubbing of the `/simulate` cycles array.

### Add a UI datapath block

1. Emit matching component/port state from the simulator's `get_state()`.
2. Add the component and wires to `ui/static/js/layouts/<model>.js` (keyed by
   state IDs; shapes `rect`/`mux`/`pipeline_bar`).
3. Add a `_summarize` case in `ui/static/js/layout_engine.js`.
4. Include any new script tag and selector in `cpu_simulator.js` /
   `program_lab.js`.

The UI uses ordered global `<script>` tags (no ES modules, no npm). D3 v7 comes
from a CDN. `tests/test_lab_page.py` asserts every `$('id')` lookup has a
matching template id and that referenced assets exist.

## Toolchain notes

### Compiler (Core-C)

Entry point: `sim.compiler.compile_c(source, isa)` → `CompileResult`
(`.asm`, `.tokens`, `.ast`, `.source_map`, `.symbols`).

Supported: `int`/`void`, `if`/`else`, `while`, `for`, `return`, `break`,
`continue`, `const`; functions with recursion (except x86), array/pointer
parameters, arrays, pointers, globals, and the `print()` builtin
(writes to `PRINT_ADDR = 2040`). Rejected: `char`, `float`, `struct`, `switch`,
`goto`, `malloc`, and pointer locals.

Model limits and conventions:

- One accumulator with spill slots: left operand in a temp slot, right in the
  accumulator, then "slot OP acc".
- Frames grow down from `STACK_TOP = 65528`; globals live in `[0, 2032)`;
  `STACK_LIMIT = 4096` guards collisions.
- Per-ISA conventions are documented in each backend's header: RISC-V uses `t0`
  accumulator / `sp` frame / `a0-a7` args; ARM uses `X9` accumulator / `X28`
  frame (SP aliases XZR) / 8-byte slots; x86 uses `EAX` accumulator / `EBP`
  fixed at 256 and does not support recursion (no CALL/RET in the ISA model).
- Errors: `CompilerError(message, line, col)`, `AssemblerError`; `/compile` and
  `/assemble` map them to HTTP 400.

### Assembler

`sim.assembler.assemble(isa, text)` → words (RISC-V/ARM) or bytes (x86).
Two passes: labels → PC, then encode. RISC-V/ARM are fixed-width; x86 is
variable-length with automatic rel8→rel32 branch relaxation (`_wide_jumps` is
sticky). Labels must match `[A-Za-z_]\w*`; generated labels are `L_<base>_<n>`.
Mnemonics are project-defined — check `sim/assembler/cheatsheet.py` and the
example programs before inventing syntax.

### Cache and GPU (in-progress areas)

- Cache (`sim/components/memory/cache/`): observe-only works on all 18 presets;
  **timing is implemented only for `single_cycle`**. `cached_dmem_for(...)` raises
  for other models with timing enabled. OoO cache statistics use an
  address-change heuristic and intentionally overcount (pinned by tests). See
  `CACHE_DESIGN.md`.
- GPU: only `sim/gpu/config.py` (`GPUSpec`) exists. `GRAPHICS_DESIGN.md` is the
  approved design; no GPU routes yet.

## Testing

- Run from the repo root: `python3 appctl.py test` (≈2,200 test functions in 47
  files). No `conftest.py`, no pytest/ruff/mypy config, no CI.
- Files add the repo root to `sys.path` themselves; helpers are sometimes shared
  across test modules by import.
- `run_simulation(until_stable=True)` stops after 64 architecturally-quiet
  cycles; PC is excluded from the signature (finished programs may spin).
- Intentional pins — do not "fix" without updating the pinning test and
  documenting why:
  - `x86/superscalar` is illustrative-only, excluded from correctness matrices.
  - `x86 POP` is a documented `xfail`.
  - Configurable + cache timing has a test asserting the *corruption* that
    justifies excluding it.
  - `/presets`' `models` label map omits `configurable` (pinned).
- After changing simulator behavior: run the full suite, then the parity suites
  above. Update `Expected:` headers in `programs/` only when behavior is
  intentionally corrected across **all** models.

## Deployment / environment

- `render.yaml` runs `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4
  --timeout 120 api.app:app` with `FLASK_ENV=production`.
- Env vars: `HOST` (default `127.0.0.1`), `PORT` (default `5051`),
  `FLASK_DEBUG` (truthy enables the reloader).
- Dependencies: `flask`, `flask-cors`, `gunicorn` only.

## Further design docs

- `README.md` — user-facing overview and quickstart.
- `CACHE_DESIGN.md` — cache status, timing contract, known limits.
- `CYCLE_EFFICIENCY.md` — pinned cross-model cycle tables and the
  "cycles ≠ time" rationale.
- `GRAPHICS_DESIGN.md` — approved, unimplemented GPU plan.
- `FUTURE_DIRECTIONS.md` — local-only notes (gitignored).
