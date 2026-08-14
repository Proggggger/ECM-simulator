"""
RESPONSE-SHAPE contract tests for the Flask API.

The JavaScript front-end (ui/static/js/cpu_simulator.js, layout_engine.js,
layouts/*.js, program_lab.js) reads response fields BY NAME. A silent key
rename or type change on the server would break the UI with no existing test
failing, because today's API tests mostly assert values, not shapes.

This file pins, for every route:
  * the exact top-level key set and each value's type
  * /simulate  — every cycle snapshot carries `_cycle` plus the component keys
                 the UI reads (regfile.registers, dmem.memory/memory_hi/
                 write_addr, fetch.pc, imem.*), across ALL 18 presets
  * /topology  — {nodes, edges, model}; node.id/label/category/ports and
                 edge.from/edge.to in the "component.port" form
                 layout_engine.js splits on '.'
  * /compare   — every grid cell carries EITHER a count or an `error`, never a
                 wrong number; Custom-column cells carry clock_period/total_time
  * /compile   — symbols entries are {name, kind, addr, size, stride, location}
  * /simulate on `configurable` — time_model = {costs, clock_period, cycles,
                 total_time} (+ `work`)

Kept fast with tiny programs and small cycle counts.

Run from repo root:  python3 -m pytest tests/test_api_response_contract.py -q
"""
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api.app as appmod
from api.app import app, PRESETS, ISA_CONFIGS, PREDICTOR_CLASSES

ISAS = ("riscv", "arm", "x86")
MODELS = ("single_cycle", "multicycle", "configurable", "pipeline", "ooo",
          "superscalar")
ALL_PRESETS = sorted(PRESETS)

# One tiny, ISA-correct program per ISA — enough to fetch/decode/execute a
# couple of instructions in every model without costing cycles.
TINY_ASM = {
    "riscv": "ADDI x1, x0, 5\nADDI x2, x1, 3\nNOP\n",
    "arm":   "MOVZ X1, #5\nADD X2, X1, XZR\nNOP\n",
    "x86":   "MOV EAX, 5\nMOV EBX, 3\nNOP\n",
}
TINY_C = "int main() { int a = 5; int b[2] = {1, 2}; return a + b[0]; }"


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    return app.test_client()


def get(client, path, status=200):
    resp = client.get(path)
    assert resp.status_code == status, f"GET {path}: {resp.data[:200]!r}"
    assert "application/json" in resp.content_type, resp.content_type
    return json.loads(resp.data)


def post(client, path, body, status=200):
    resp = client.post(path, json=body)
    assert resp.status_code == status, f"POST {path}: {resp.data[:200]!r}"
    assert "application/json" in resp.content_type, resp.content_type
    return json.loads(resp.data)


# ============================================================================
# 1. /presets
# ============================================================================
class TestPresetsShape:

    def test_top_level_keys_exact(self, client):
        assert set(get(client, "/presets")) == {"isas", "models", "presets"}

    def test_isas_entries(self, client):
        isas = get(client, "/presets")["isas"]
        assert set(isas) == set(ISAS)
        for name, entry in isas.items():
            assert set(entry) == {"display_name", "description", "program_format"}
            assert all(isinstance(v, str) for v in entry.values())
            assert entry["program_format"] in ("words", "bytes")

    def test_presets_entries(self, client):
        presets = get(client, "/presets")["presets"]
        assert set(presets) == set(ALL_PRESETS)
        assert len(presets) == 18
        for key, entry in presets.items():
            assert set(entry) == {"label", "isa", "model"}
            isa, model = key.split("/")
            assert entry["isa"] == isa and entry["model"] == model
            assert isinstance(entry["label"], str) and entry["label"]

    def test_models_map_is_str_to_str(self, client):
        models = get(client, "/presets")["models"]
        assert all(isinstance(k, str) and isinstance(v, str)
                   for k, v in models.items())

    def test_known_shortfall_configurable_missing_from_models_map(self, client):
        """`models` is a HARDCODED label map (api/app.py:186-192) that never
        gained the 6th model, so the UI supplies its own label for
        `configurable` (cpu_simulator.js:210). Pinned so the gap is visible and
        closing it is a deliberate change."""
        models = get(client, "/presets")["models"]
        registered = {p["model"] for p in PRESETS.values()}
        assert registered - set(models) == {"configurable"}


# ============================================================================
# 2. /isa/<name>
# ============================================================================
class TestIsaShape:

    ISA_KEYS = {"name", "display_name", "description", "num_regs", "reg_names",
                "program_format", "demo_program", "demo_program_text",
                "demo_program_asm"}

    @pytest.mark.parametrize("isa", ISAS)
    def test_keys_and_types(self, client, isa):
        data = get(client, f"/isa/{isa}")
        assert set(data) == self.ISA_KEYS
        assert data["name"] == isa
        assert isinstance(data["num_regs"], int) and data["num_regs"] > 0
        assert isinstance(data["reg_names"], list)
        assert len(data["reg_names"]) == data["num_regs"]
        assert all(isinstance(n, str) for n in data["reg_names"])
        assert data["program_format"] in ("words", "bytes")
        assert isinstance(data["demo_program"], list)
        assert all(isinstance(v, int) for v in data["demo_program"])
        assert isinstance(data["demo_program_text"], str)
        assert data["demo_program_text"].strip()
        assert isinstance(data["demo_program_asm"], str)

    @pytest.mark.parametrize("isa", ISAS)
    def test_demo_program_matches_the_registry(self, client, isa):
        data = get(client, f"/isa/{isa}")
        assert data["demo_program"] == list(ISA_CONFIGS[isa].demo_program())

    def test_x86_bytes_fit_in_a_byte(self, client):
        data = get(client, "/isa/x86")
        assert data["program_format"] == "bytes"
        assert all(0 <= b <= 0xFF for b in data["demo_program"])

    @pytest.mark.parametrize("isa", ("riscv", "arm"))
    def test_word_isas_fit_in_32_bits(self, client, isa):
        data = get(client, f"/isa/{isa}")
        assert data["program_format"] == "words"
        assert all(0 <= w <= 0xFFFFFFFF for w in data["demo_program"])


# ============================================================================
# 3. /predictors
# ============================================================================
class TestPredictorsShape:

    def test_shape(self, client):
        data = get(client, "/predictors")
        assert set(data) == {"predictors"}
        assert isinstance(data["predictors"], list) and data["predictors"]
        for entry in data["predictors"]:
            assert set(entry) == {"name", "label"}
            assert isinstance(entry["name"], str) and isinstance(entry["label"], str)

    def test_covers_the_registry_and_is_sorted(self, client):
        names = [p["name"] for p in get(client, "/predictors")["predictors"]]
        assert names == sorted(PREDICTOR_CLASSES)


# ============================================================================
# 4. /topology/<preset>  — what layout_engine.js consumes
# ============================================================================
class TestTopologyShape:

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_top_level_keys_and_model(self, client, preset):
        data = get(client, f"/topology/{preset}")
        assert set(data) == {"nodes", "edges", "model"}
        assert data["model"] == PRESETS[preset]["model"]
        assert isinstance(data["nodes"], list) and data["nodes"]
        assert isinstance(data["edges"], list) and data["edges"]

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_node_fields(self, client, preset):
        for node in get(client, f"/topology/{preset}")["nodes"]:
            assert set(node) == {"id", "label", "category", "ports"}
            assert isinstance(node["id"], str) and node["id"]
            assert isinstance(node["label"], str)
            assert isinstance(node["category"], str)
            assert isinstance(node["ports"], dict)
            for pname, port in node["ports"].items():
                assert set(port) == {"width", "direction", "desc"}
                assert port["direction"] in ("in", "out")
                assert isinstance(port["width"], int)

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_edge_endpoints_are_component_dot_port(self, client, preset):
        """layout_engine.js does `wire.from.split('.')[0]` / `[1]`
        (layout_engine.js:475-479, 865-866), so every endpoint must be exactly
        'component.port' and both halves must resolve against `nodes`."""
        data = get(client, f"/topology/{preset}")
        node_ports = {n["id"]: set(n["ports"]) for n in data["nodes"]}
        for edge in data["edges"]:
            assert set(edge) == {"from", "to", "label"}
            for side in ("from", "to"):
                ref = edge[side]
                assert isinstance(ref, str)
                assert ref.count(".") == 1, f"{preset} edge {side}={ref!r}"
                comp, port = ref.split(".")
                assert comp in node_ports, f"{preset}: unknown component {comp!r}"
                assert port in node_ports[comp], f"{preset}: unknown port {ref!r}"

    def test_node_ids_are_unique(self, client):
        for preset in ALL_PRESETS:
            ids = [n["id"] for n in get(client, f"/topology/{preset}")["nodes"]]
            assert len(ids) == len(set(ids)), preset

    def test_superscalar_lane_count_reflects_num_lanes(self, client):
        for lanes in (1, 2, 4):
            data = get(client, f"/topology/riscv/superscalar?num_lanes={lanes}")
            alus = [n["id"] for n in data["nodes"] if re.fullmatch(r"alu_\d+", n["id"])]
            assert len(alus) == lanes

    @pytest.mark.parametrize("preset", [p for p in ALL_PRESETS
                                        if not p.endswith("/superscalar")])
    def test_branch_predictor_threads_into_every_non_superscalar_topology(
            self, client, preset):
        """api/app.py:267-270 — a predictor is passed as a kwarg to any model
        that accepts one; the diagram must still render."""
        data = get(client, f"/topology/{preset}?branch_predictor=gshare"
                           "&prediction_stage=id")
        assert data["nodes"] and data["edges"]
        assert data["model"] == PRESETS[preset]["model"]


# ============================================================================
# 5. /simulate — the per-cycle contract, across all 18 presets
# ============================================================================
SIMULATE_KEYS = {"preset", "isa", "model", "reg_names", "program_format", "cycles"}


class TestSimulateShape:

    @staticmethod
    def _run(client, preset, cycles=6, **extra):
        isa = PRESETS[preset]["isa"]
        body = {"preset": preset, "cycles": cycles, "input_mode": "asm",
                "asm_text": TINY_ASM[isa]}
        body.update(extra)
        return post(client, "/simulate", body)

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_top_level_keys_exact(self, client, preset):
        data = self._run(client, preset)
        expected = set(SIMULATE_KEYS)
        if PRESETS[preset]["model"] == "configurable":
            expected.add("time_model")     # only this model gets a time model
        assert set(data) == expected, preset
        assert data["preset"] == preset
        assert data["isa"] == PRESETS[preset]["isa"]
        assert data["model"] == PRESETS[preset]["model"]
        assert data["program_format"] == ISA_CONFIGS[data["isa"]].program_format
        assert data["reg_names"] == ISA_CONFIGS[data["isa"]].register_names()

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_cycle_count_and_monotonic_cycle_tag(self, client, preset):
        data = self._run(client, preset, cycles=6)
        states = data["cycles"]
        assert len(states) == 7                       # 6 ticks + reset snapshot
        assert [s["_cycle"] for s in states] == list(range(7))

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_every_cycle_has_the_same_component_key_set(self, client, preset):
        """The UI indexes state by component name every cycle; a component that
        appears/disappears mid-run would break rendering."""
        states = self._run(client, preset)["cycles"]
        first = set(states[0])
        for s in states[1:]:
            assert set(s) == first, preset

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_ui_required_components_present_every_cycle(self, client, preset):
        """cpu_simulator.js reads state['fetch'], state['imem'], state['dmem']
        and the register file every cycle (cpu_simulator.js:829, 1059-1066,
        1163)."""
        for s in self._run(client, preset)["cycles"]:
            assert "fetch" in s and "pc" in s["fetch"]
            assert "imem" in s
            assert "dmem" in s
            regfiles = [v for v in s.values()
                        if isinstance(v, dict) and "registers" in v]
            assert len(regfiles) == 1, preset

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_dmem_state_fields(self, client, preset):
        """`memory` is the dense low window, `memory_hi` the sparse high map,
        `write_addr` is what the /lab print console watches (CLAUDE.md)."""
        dmem = self._run(client, preset)["cycles"][-1]["dmem"]
        for key in ("memory", "memory_hi", "write_addr", "wen", "size"):
            assert key in dmem, (preset, sorted(dmem))
        assert isinstance(dmem["memory"], list)
        assert len(dmem["memory"]) == 2048              # STATE_WINDOW
        assert all(isinstance(v, int) for v in dmem["memory"])
        assert isinstance(dmem["memory_hi"], dict)
        assert all(int(k) >= len(dmem["memory"]) for k in dmem["memory_hi"])

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_registers_are_ints_of_the_right_count(self, client, preset):
        data = self._run(client, preset)
        regs = data["cycles"][-1]["regfile"]["registers"]
        assert isinstance(regs, list)
        assert len(regs) == ISA_CONFIGS[data["isa"]].num_regs
        assert all(isinstance(v, int) and 0 <= v <= 0xFFFFFFFF for v in regs)

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_whole_response_is_json_round_trippable(self, client, preset):
        """No NaN/Infinity/bytes may sneak into a snapshot — JSON.parse() in the
        browser is stricter than Python's json module."""
        raw = json.dumps(self._run(client, preset, cycles=4))
        for bad in ("NaN", "Infinity", "-Infinity"):
            assert bad not in raw, preset
        json.loads(raw)

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_fetch_pc_is_a_hex_string(self, client, preset):
        for s in self._run(client, preset)["cycles"]:
            pc = s["fetch"]["pc"]
            assert isinstance(pc, str) and pc.startswith("0x"), (preset, pc)
            int(pc, 16)

    def test_run_to_completion_adds_exactly_two_keys(self, client):
        data = post(client, "/simulate", {
            "preset": "riscv/single_cycle", "run_to_completion": True,
            "input_mode": "asm", "asm_text": TINY_ASM["riscv"]})
        assert set(data) == SIMULATE_KEYS | {"run_to_completion", "completed"}
        assert data["run_to_completion"] is True
        assert isinstance(data["completed"], bool)

    def test_manual_mode_omits_the_completion_keys(self, client):
        data = self._run(client, "riscv/single_cycle", cycles=3)
        assert "completed" not in data and "run_to_completion" not in data


# ============================================================================
# 6. /simulate on `configurable` — the time_model contract
# ============================================================================
class TestTimeModelShape:

    @pytest.mark.parametrize("isa", ISAS)
    def test_time_model_keys_and_types(self, client, isa):
        data = post(client, "/simulate", {
            "preset": f"{isa}/configurable", "cycles": 8,
            "input_mode": "asm", "asm_text": TINY_ASM[isa],
            "cycle_costs": {"alu": 2, "load": 3, "store": 3, "branch": 2}})
        assert "time_model" in data
        tm = data["time_model"]
        assert set(tm) == {"costs", "clock_period", "work", "cycles", "total_time"}
        assert tm["costs"] == {"alu": 2, "load": 3, "store": 3, "branch": 2}
        assert set(tm["work"]) == set(appmod._CYCLE_CLASSES)
        assert isinstance(tm["cycles"], int) and tm["cycles"] >= 0
        assert isinstance(tm["clock_period"], (int, float)) and tm["clock_period"] > 0
        assert tm["total_time"] == pytest.approx(tm["cycles"] * tm["clock_period"])

    def test_clock_period_is_the_busiest_class(self, client):
        costs = {"alu": 1, "load": 4, "store": 2, "branch": 1}
        data = post(client, "/simulate", {
            "preset": "riscv/configurable", "cycles": 6,
            "input_mode": "asm", "asm_text": TINY_ASM["riscv"],
            "cycle_costs": costs})
        tm = data["time_model"]
        expected = max(appmod._CYCLE_WORK[c] / costs[c] for c in costs)
        assert tm["clock_period"] == pytest.approx(expected)

    @pytest.mark.parametrize("preset", [p for p in ALL_PRESETS
                                        if not p.endswith("/configurable")])
    def test_non_configurable_models_have_no_time_model(self, client, preset):
        isa = PRESETS[preset]["isa"]
        data = post(client, "/simulate", {
            "preset": preset, "cycles": 4, "input_mode": "asm",
            "asm_text": TINY_ASM[isa]})
        assert "time_model" not in data


# ============================================================================
# 7. /assemble
# ============================================================================
class TestAssembleShape:

    @pytest.mark.parametrize("isa", ISAS)
    def test_keys_and_program_range(self, client, isa):
        data = post(client, "/assemble", {"isa": isa, "text": TINY_ASM[isa]})
        assert set(data) == {"isa", "program"}
        assert data["isa"] == isa
        assert isinstance(data["program"], list) and data["program"]
        assert all(isinstance(v, int) for v in data["program"])
        hi = 0xFF if ISA_CONFIGS[isa].program_format == "bytes" else 0xFFFFFFFF
        assert all(0 <= v <= hi for v in data["program"])

    @pytest.mark.parametrize("isa", ISAS)
    def test_assembled_program_feeds_simulate_unchanged(self, client, isa):
        """Round-trip: /assemble's `program` is exactly what /simulate accepts
        in hex mode — the UI passes it straight through."""
        program = post(client, "/assemble",
                       {"isa": isa, "text": TINY_ASM[isa]})["program"]
        data = post(client, "/simulate", {
            "preset": f"{isa}/single_cycle", "cycles": 6,
            "input_mode": "hex", "program": program})
        regs = data["cycles"][-1]["regfile"]["registers"]
        # riscv/arm write x1/X1; x86's "MOV EAX, 5" writes index 0.
        assert regs[0 if isa == "x86" else 1] == 5


# ============================================================================
# 8. /compile — including the `symbols` debugger table
# ============================================================================
COMPILE_KEYS = {"isa", "backend", "asm", "stages", "source_map", "symbols",
                "program"}
SYMBOL_KEYS = {"name", "kind", "addr", "size", "stride", "location"}


class TestCompileShape:

    @pytest.mark.parametrize("isa", ISAS)
    def test_top_level_keys_exact(self, client, isa):
        data = post(client, "/compile", {"source": TINY_C, "isa": isa})
        assert set(data) == COMPILE_KEYS
        assert data["isa"] == isa
        assert data["backend"] == "python"
        assert isinstance(data["asm"], str) and data["asm"].strip()
        assert isinstance(data["program"], list) and data["program"]

    @pytest.mark.parametrize("isa", ISAS)
    def test_stages_keys_exact(self, client, isa):
        stages = post(client, "/compile", {"source": TINY_C, "isa": isa})["stages"]
        assert set(stages) == {"source", "tokens", "ast", "asm"}
        assert stages["source"] == TINY_C

    @pytest.mark.parametrize("isa", ISAS)
    def test_symbols_entry_shape(self, client, isa):
        """CLAUDE.md pins {name, kind, addr, size, stride, location} — the
        /lab Variables pane reads exactly these."""
        data = post(client, "/compile", {"source": TINY_C, "isa": isa})
        assert isinstance(data["symbols"], list) and data["symbols"]
        for sym in data["symbols"]:
            assert set(sym) == SYMBOL_KEYS, (isa, sym)
            assert isinstance(sym["name"], str) and sym["name"]
            assert sym["kind"] in ("var", "array")
            assert isinstance(sym["addr"], int) and sym["addr"] >= 0
            assert isinstance(sym["size"], int) and sym["size"] >= 1
            assert isinstance(sym["stride"], int) and sym["stride"] >= 1
            assert isinstance(sym["location"], str) and sym["location"]
        by_name = {s["name"]: s for s in data["symbols"]}
        assert by_name["a"]["kind"] == "var" and by_name["a"]["size"] == 1
        assert by_name["b"]["kind"] == "array" and by_name["b"]["size"] == 2

    @pytest.mark.parametrize("isa", ISAS)
    def test_source_map_entry_shape(self, client, isa):
        """program_lab.js joins the C and asm panes with {c_line, asm_line}
        pairs; both must be in-range 1-based line numbers."""
        data = post(client, "/compile", {"source": TINY_C, "isa": isa})
        smap = data["source_map"]
        assert isinstance(smap, list) and smap
        n_asm = len(data["asm"].splitlines())
        n_c = len(TINY_C.splitlines())
        for entry in smap:
            assert set(entry) == {"c_line", "asm_line"}
            assert 1 <= entry["asm_line"] <= n_asm, entry
            assert 1 <= entry["c_line"] <= n_c, entry

    @pytest.mark.parametrize("isa", ISAS)
    def test_program_matches_assembling_the_returned_asm(self, client, isa):
        data = post(client, "/compile", {"source": TINY_C, "isa": isa})
        again = post(client, "/assemble", {"isa": isa, "text": data["asm"]})
        assert data["program"] == again["program"]


# ============================================================================
# 9. /compare — the grid cell contract
# ============================================================================
COMPARE_KEYS = {"results", "models", "isas", "cap", "cycle_costs", "cost_classes"}


class TestCompareShape:

    def test_top_level_keys_exact(self, client):
        data = post(client, "/compare", {"asm_text": TINY_ASM["riscv"],
                                         "isa": "riscv"})
        assert set(data) == COMPARE_KEYS
        assert data["isas"] == ["riscv"]
        assert sorted(data["models"]) == sorted(MODELS)
        assert data["cap"] == appmod.MAX_AUTO_CYCLES
        assert set(data["cycle_costs"]) == set(appmod._CYCLE_CLASSES)
        assert data["cost_classes"] == list(appmod._CYCLE_CLASSES)

    @pytest.mark.parametrize("isa", ISAS)
    def test_every_cell_is_either_a_count_or_an_error(self, client, isa):
        """The whole point of the grid: a cell NEVER shows a number the model
        did not actually earn. Divergent / unavailable targets must carry
        `error` instead (api/app.py:530-556)."""
        data = post(client, "/compare", {"asm_text": TINY_ASM[isa], "isa": isa})
        cells = {r["model"]: r for r in data["results"]}
        assert set(cells) == set(data["models"])
        for model, cell in cells.items():
            assert cell["isa"] == isa
            assert cell["model"] == model
            assert cell["preset"] == f"{isa}/{model}"
            assert ("cycles" in cell) ^ ("error" in cell), cell
            if "error" in cell:
                assert isinstance(cell["error"], str) and cell["error"]
                assert "completed" not in cell
            else:
                assert isinstance(cell["cycles"], int) and cell["cycles"] >= 0
                assert isinstance(cell["completed"], bool)
                assert isinstance(cell["program_len"], int)

    def test_configurable_cells_carry_the_time_model_readout(self, client):
        data = post(client, "/compare", {
            "asm_text": TINY_ASM["riscv"], "isa": "riscv",
            "models": ["single_cycle", "configurable"],
            "cycle_costs": {"alu": 2, "load": 3, "store": 3, "branch": 2}})
        cells = {r["model"]: r for r in data["results"]}
        cfg = cells["configurable"]
        assert "clock_period" in cfg and "total_time" in cfg
        assert cfg["total_time"] == pytest.approx(cfg["cycles"] * cfg["clock_period"])
        assert "clock_period" not in cells["single_cycle"]
        assert "total_time" not in cells["single_cycle"]

    def test_x86_superscalar_is_always_an_error_cell(self, client):
        """The illustrative-only preset must never produce a number
        (api/app.py:538-542)."""
        data = post(client, "/compare", {"asm_text": TINY_ASM["x86"], "isa": "x86",
                                         "models": ["superscalar"]})
        cell = data["results"][0]
        assert "cycles" not in cell
        assert "illustrative-only" in cell["error"]

    def test_divergent_target_returns_error_not_a_wrong_number(self, client):
        """OoO never commits JAL's link-register write (CLAUDE.md /
        FUTURE_DIRECTIONS.md §OoO), so this program's OoO result diverges from
        the single_cycle oracle and the parity guard must blank the count."""
        data = post(client, "/compare", {
            "asm_text": "JAL x1, done\nNOP\ndone:\nNOP\n", "isa": "riscv",
            "models": ["single_cycle", "ooo"]})
        cells = {r["model"]: r for r in data["results"]}
        assert cells["single_cycle"]["completed"] is True
        assert "cycles" not in cells["ooo"]
        assert "diverges from single_cycle" in cells["ooo"]["error"]

    def test_unassemblable_program_errors_per_cell_not_per_request(self, client):
        data = post(client, "/compare", {"asm_text": "NOTAREALOP x1", "isa": "riscv",
                                         "models": ["single_cycle", "pipeline"]})
        for cell in data["results"]:
            assert "error" in cell and "cycles" not in cell

    def test_core_c_mode_spans_all_isas(self, client):
        data = post(client, "/compare", {"source": "int main(){return 3;}",
                                         "models": ["single_cycle"]})
        assert sorted(data["isas"]) == sorted(ISAS)
        assert len(data["results"]) == len(ISAS)
        for cell in data["results"]:
            assert ("cycles" in cell) ^ ("error" in cell), cell

    def test_cycle_counts_are_ordered_single_cycle_lowest(self, client):
        """A sanity invariant the UI's bar chart depends on: for the same
        program the single-cycle model always needs the fewest cycles."""
        data = post(client, "/compare", {"asm_text": TINY_ASM["riscv"],
                                         "isa": "riscv"})
        cells = {r["model"]: r for r in data["results"] if "cycles" in r}
        sc = cells["single_cycle"]["cycles"]
        for model, cell in cells.items():
            assert cell["cycles"] >= sc, model


# ============================================================================
# 10. /cheatsheet/<isa>
# ============================================================================
class TestCheatsheetShape:

    @pytest.mark.parametrize("isa", ISAS)
    def test_keys_and_entry_fields(self, client, isa):
        data = get(client, f"/cheatsheet/{isa}")
        assert set(data) == {"isa", "instructions"}
        assert data["isa"] == isa
        assert isinstance(data["instructions"], list) and data["instructions"]
        for item in data["instructions"]:
            assert {"category", "mnemonic", "syntax", "description",
                    "example"} <= set(item)
            assert all(isinstance(item[k], str) for k in
                       ("category", "mnemonic", "syntax", "description", "example"))


# ============================================================================
# 11. /examples and /examples/<path>
# ============================================================================
class TestExamplesShape:

    def test_catalog_nesting(self, client):
        catalog = get(client, "/examples")
        assert set(catalog) <= set(ISAS)
        for isa, categories in catalog.items():
            assert set(categories) <= set(appmod._CATEGORY_ORDER)
            for cat, payload in categories.items():
                assert set(payload) == {"label", "items"}
                assert isinstance(payload["label"], str) and payload["label"]
                assert isinstance(payload["items"], list) and payload["items"]
                for item in payload["items"]:
                    assert set(item) == {"name", "label", "file"}
                    assert item["file"] == f"{isa}/{cat}/{item['name']}.asm"

    def test_catalog_categories_are_a_subset_of_the_declared_order(self, client):
        """Flask serialises with sort_keys=True, so the JSON key ORDER is
        alphabetical, not _CATEGORY_ORDER — the UI must sort client-side. Pin
        the membership (and the fact that ordering is NOT conveyed)."""
        catalog = get(client, "/examples")
        for isa, categories in catalog.items():
            assert set(categories) <= set(appmod._CATEGORY_ORDER), isa
            assert list(categories) == sorted(categories), isa

    def test_single_file_shape(self, client):
        data = get(client, "/examples/riscv/algorithms/fibonacci.asm")
        assert set(data) == {"file", "content"}
        assert data["file"] == "riscv/algorithms/fibonacci.asm"
        assert isinstance(data["content"], str) and data["content"].strip()


# ============================================================================
# 12. /compiler/examples and /compiler/examples/<name>
# ============================================================================
class TestCompilerExamplesShape:

    def test_catalog_entry_shape(self, client):
        data = get(client, "/compiler/examples")
        assert set(data) == {"items"}
        items = data["items"]
        assert isinstance(items, list) and items
        for item in items:
            assert set(item) == {"name", "label", "file", "description", "targets"}
            assert item["file"] == item["name"] + ".c"
            assert re.fullmatch(r"[a-z0-9_]+", item["name"]), item["name"]
            assert isinstance(item["description"], str)
            assert isinstance(item["targets"], list)
            assert set(item["targets"]) <= set(ISAS)

    def test_catalog_is_sorted_by_name(self, client):
        names = [i["name"] for i in get(client, "/compiler/examples")["items"]]
        assert names == sorted(names)

    def test_single_sample_shape(self, client):
        data = get(client, "/compiler/examples/fib_iter")
        assert set(data) == {"name", "content"}
        assert data["name"] == "fib_iter"
        assert isinstance(data["content"], str) and data["content"].strip()

    def test_every_catalog_entry_is_fetchable(self, client):
        for item in get(client, "/compiler/examples")["items"]:
            data = get(client, f"/compiler/examples/{item['name']}")
            assert data["name"] == item["name"]
            assert data["content"].strip()


# ============================================================================
# 13. HTML routes  (the UI shell must keep serving)
# ============================================================================
class TestHtmlRoutes:

    @pytest.mark.parametrize("path", ["/", "/rtl_cpu", "/lab", "/program_lab"])
    def test_serves_html(self, client, path):
        resp = client.get(path)
        assert resp.status_code == 200
        assert "text/html" in resp.content_type
        body = resp.data.decode("utf-8", errors="replace")
        assert "<html" in body.lower()

    def test_lab_handoff_script_precedes_the_simulator_bundle(self, client):
        """CLAUDE.md: the inline `rtl_lab_handoff` consumer MUST stay BEFORE the
        cpu_simulator.js tag or the handoff from /lab silently breaks."""
        body = client.get("/").data.decode("utf-8", errors="replace")
        handoff = body.find("localStorage.getItem('rtl_lab_handoff')")
        bundle = body.find("<script src=")
        bundle = body.find("js/cpu_simulator.js", bundle)
        assert handoff != -1 and bundle != -1
        assert handoff < bundle


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
