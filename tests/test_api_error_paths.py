"""
Systematic ADVERSARIAL pass over every route in api/app.py.

CLAUDE.md documents the safety contract every route must satisfy:

    "All routes validate/clamp input and return JSON {error} with 400/404
     (never a raw 500 or a leaked file) on bad input."

The existing API tests (test_api_endpoints*.py, test_api_compile.py) cover the
happy path and a handful of negatives. This file attacks the contract itself:

  * missing / empty / non-JSON / wrong-Content-Type bodies on every POST route
  * wrong JSON type for every field (string-for-int, list-for-dict, null,
    nested object) on every route
  * unknown isa / model / preset / predictor / example name
  * PATH TRAVERSAL on the two file-serving routes (/examples/<isa>/<cat>/<f>.asm
    and /compiler/examples/<name>) — literal `../`, URL-encoded, double-encoded,
    backslash, null-byte, absolute-path and SYMLINK escapes
  * numeric clamping (cycles -> MAX_CYCLES, num_lanes -> MAX_LANES,
    cycle_costs -> 1..64, run_to_completion -> MAX_AUTO_CYCLES)
  * wrong-shape combinations (num_lanes on a non-superscalar model, x86 bytes
    given to RISC-V, both program+asm, neither)

Every request runs against a PRODUCTION-mode test client (TESTING=False,
PROPAGATE_EXCEPTIONS=False) so an unhandled exception surfaces as the HTML 500
a real browser would see, instead of being re-raised into the test.

This file FOUND three contract violations (unhashable JSON lookup keys, an
OverflowError from the non-standard `Infinity` literal, and an embedded NUL
reaching os.path.realpath). All three were fixed on 2026-08-08; the tests that
exposed them are kept as ordinary passing REGRESSION PINS in the sections at
the bottom of this file, so the guards cannot be removed without a failure.

Run from repo root:  python3 -m pytest tests/test_api_error_paths.py -q
"""
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api.app as appmod
from api.app import app

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Sentinels that must NEVER appear in a response body — if any shows up, a file
# outside the served directory leaked.
LEAK_SENTINELS = (
    b"from flask import",        # api/app.py
    b"Flask(__name__",           # api/app.py
    b"root:x:0:0",               # /etc/passwd
    b"BEGIN RSA PRIVATE KEY",
    b"# CLAUDE.md",              # repo instructions file
    b"def main(argv",            # appctl.py / sim/harness.py
)

POST_ROUTES = ("/simulate", "/assemble", "/compile", "/compare")
GET_ROUTES = (
    "/", "/lab", "/presets", "/predictors", "/examples", "/compiler/examples",
    "/isa/riscv", "/cheatsheet/riscv", "/topology/riscv/single_cycle",
    "/examples/riscv/algorithms/fibonacci.asm", "/compiler/examples/fib_iter",
)

TINY_ASM = {"riscv": "ADDI x1, x0, 5", "arm": "MOVZ X1, #5", "x86": "MOV EAX, 5"}


# ── fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture()
def client():
    """Production-mode Flask test client.

    TESTING/PROPAGATE_EXCEPTIONS are forced OFF so an unhandled route exception
    becomes the HTML 500 a browser would receive (which is exactly what the
    contract forbids) rather than being re-raised into the test.
    """
    old = (app.config.get("TESTING"), app.config.get("PROPAGATE_EXCEPTIONS"))
    app.config["TESTING"] = False
    app.config["PROPAGATE_EXCEPTIONS"] = False
    try:
        yield app.test_client()
    finally:
        app.config["TESTING"], app.config["PROPAGATE_EXCEPTIONS"] = old


# ── shared assertions ───────────────────────────────────────────────────────
def assert_no_leak(resp):
    """No response may ever contain the contents of a file outside the served
    directories (nor an HTML stack trace)."""
    for needle in LEAK_SENTINELS:
        assert needle not in resp.data, (
            f"response leaked {needle!r}: {resp.data[:200]!r}")
    assert b"Traceback (most recent call last)" not in resp.data


def assert_json_error(resp, allowed=(400, 404)):
    """The documented contract: JSON {error} with a 4xx status, never a 500."""
    assert_no_leak(resp)
    assert resp.status_code in allowed, (
        f"expected {allowed}, got {resp.status_code}: {resp.data[:200]!r}")
    assert "application/json" in resp.content_type, (
        f"error response was not JSON ({resp.content_type}): {resp.data[:200]!r}")
    body = json.loads(resp.data)       # must parse — no HTML error page
    assert isinstance(body, dict) and "error" in body, body
    return body


def assert_never_500(resp):
    """Weaker guard for inputs whose *acceptance* is a design choice: whatever
    the route decides, it must not be a server error and must stay JSON."""
    assert_no_leak(resp)
    assert resp.status_code < 500, (
        f"raw {resp.status_code}: {resp.data[:200]!r}")
    assert "application/json" in resp.content_type, resp.content_type
    json.loads(resp.data)
    return resp


# ============================================================================
# 1. MALFORMED / MISSING BODIES  (every POST route)
# ============================================================================
class TestMalformedBodies:
    """A POST route must reject anything that is not a JSON object with a
    JSON {error} 400 — never a 500, never an HTML page."""

    @pytest.mark.parametrize("route", POST_ROUTES)
    def test_no_body_at_all(self, client, route):
        assert_json_error(client.post(route), allowed=(400,))

    @pytest.mark.parametrize("route", POST_ROUTES)
    def test_empty_body_json_content_type(self, client, route):
        assert_json_error(
            client.post(route, data="", content_type="application/json"),
            allowed=(400,))

    @pytest.mark.parametrize("route", POST_ROUTES)
    def test_unparseable_json(self, client, route):
        assert_json_error(
            client.post(route, data="{not valid json", content_type="application/json"),
            allowed=(400,))

    @pytest.mark.parametrize("route", POST_ROUTES)
    def test_truncated_json(self, client, route):
        assert_json_error(
            client.post(route, data='{"isa": "ris', content_type="application/json"),
            allowed=(400,))

    @pytest.mark.parametrize("route", POST_ROUTES)
    @pytest.mark.parametrize("payload", [[1, 2, 3], "hello", None, 42, True, 3.5])
    def test_json_body_that_is_not_an_object(self, client, route, payload):
        assert_json_error(client.post(route, json=payload), allowed=(400,))

    @pytest.mark.parametrize("route", POST_ROUTES)
    def test_form_encoded_body_rejected(self, client, route):
        assert_json_error(client.post(route, data={"isa": "riscv"}), allowed=(400,))

    @pytest.mark.parametrize("route", POST_ROUTES)
    def test_wrong_content_type_with_valid_json_object(self, client, route):
        """get_json(force=True) means text/plain is tolerated; whatever the route
        decides, it must not 500 and must answer in JSON."""
        assert_never_500(client.post(route, data='{"isa": "riscv"}',
                                     content_type="text/plain"))

    @pytest.mark.parametrize("route", POST_ROUTES)
    def test_binary_garbage_body(self, client, route):
        assert_json_error(
            client.post(route, data=b"\x00\x01\x02\xff\xfe",
                        content_type="application/json"),
            allowed=(400,))

    @pytest.mark.parametrize("route", POST_ROUTES)
    def test_empty_json_object_never_500s(self, client, route):
        """{} is legal JSON; /simulate defaults to the demo program, the others
        complain about their missing required field. Neither may 500."""
        assert_never_500(client.post(route, json={}))

    @pytest.mark.parametrize("route", POST_ROUTES)
    def test_get_on_post_route_is_405(self, client, route):
        resp = client.get(route)
        assert resp.status_code == 405
        assert_no_leak(resp)


# ============================================================================
# 2. UNKNOWN IDENTIFIERS  (isa / model / preset / predictor / example)
# ============================================================================
class TestUnknownIdentifiers:

    @pytest.mark.parametrize("name", ["mips", "sparc", "ppc", "RISCV", "riscv ",
                                      "", "0", "__init__", "riscv/riscv"])
    def test_unknown_isa_on_isa_route(self, client, name):
        resp = client.get(f"/isa/{name}")
        assert_no_leak(resp)
        assert resp.status_code in (404, 308), resp.status_code

    @pytest.mark.parametrize("name", ["mips", "sparc", "x87", "riscv32"])
    def test_unknown_isa_on_cheatsheet(self, client, name):
        assert_json_error(client.get(f"/cheatsheet/{name}"), allowed=(404,))

    @pytest.mark.parametrize("name", ["RISCV", "RiscV", "  riscv  "])
    def test_cheatsheet_isa_lookup_is_case_and_space_insensitive(self, client, name):
        """`get_cheatsheet` lowercases+strips (sim/assembler/cheatsheet.py:10),
        so these are the SAME ISA, not unknown ones. Pinned so a future
        tightening is a deliberate change, not a silent one."""
        resp = client.get(f"/cheatsheet/{name}")
        assert resp.status_code == 200, resp.data[:200]
        assert json.loads(resp.data)["instructions"]

    @pytest.mark.parametrize("preset", [
        "riscv/nonexistent", "sparc/single_cycle", "riscv", "single_cycle",
        "riscv/single_cycle/extra", "RISCV/SINGLE_CYCLE", "../api/app",
    ])
    def test_unknown_preset_on_topology_is_404_json(self, client, preset):
        resp = client.get(f"/topology/{preset}")
        assert_no_leak(resp)
        # Werkzeug may normalise some paths away before routing -> plain 404.
        assert resp.status_code == 404, resp.status_code
        if "application/json" in resp.content_type:
            assert "error" in json.loads(resp.data)

    @pytest.mark.parametrize("preset", [
        "riscv/nonexistent", "sparc/single_cycle", "riscv", "", "single_cycle",
    ])
    def test_unknown_preset_on_simulate_is_400_json(self, client, preset):
        assert_json_error(
            client.post("/simulate", json={"preset": preset, "cycles": 1}),
            allowed=(400,))

    @pytest.mark.parametrize("predictor", ["nope", "GSHARE", "gshare2", "base",
                                           "__init__", "0"])
    def test_unknown_predictor_on_simulate(self, client, predictor):
        assert_json_error(
            client.post("/simulate", json={"preset": "riscv/pipeline", "cycles": 1,
                                           "branch_predictor": predictor}),
            allowed=(400,))

    @pytest.mark.parametrize("predictor", ["nope", "GSHARE", "base"])
    def test_unknown_predictor_on_topology(self, client, predictor):
        assert_json_error(
            client.get(f"/topology/riscv/pipeline?branch_predictor={predictor}"),
            allowed=(400,))

    @pytest.mark.parametrize("isa", ["mips", "sparc", "x87", ""])
    def test_unknown_isa_on_assemble(self, client, isa):
        assert_json_error(client.post("/assemble", json={"isa": isa, "text": "NOP"}),
                          allowed=(400,))

    def test_assemble_isa_is_case_insensitive_but_echoed_verbatim(self, client):
        """`assemble()` lowercases the ISA key (sim/assembler/__init__.py:27) yet
        the route echoes the RAW string back — the UI must not treat the echoed
        `isa` as canonical."""
        resp = client.post("/assemble", json={"isa": "RISCV", "text": "NOP"})
        assert resp.status_code == 200, resp.data[:200]
        assert json.loads(resp.data)["isa"] == "RISCV"

    @pytest.mark.parametrize("isa", ["mips", "sparc", "RISCV", "", "0"])
    def test_unknown_isa_on_compile(self, client, isa):
        assert_json_error(
            client.post("/compile", json={"source": "int main(){return 1;}", "isa": isa}),
            allowed=(400,))

    @pytest.mark.parametrize("isa", ["mips", "sparc", "RISCV", ""])
    def test_unknown_isa_on_compare(self, client, isa):
        assert_json_error(
            client.post("/compare", json={"asm_text": "NOP", "isa": isa,
                                          "models": ["single_cycle"]}),
            allowed=(400,))

    @pytest.mark.parametrize("models", [["warp"], ["single_cycle", "warp"],
                                        ["SINGLE_CYCLE"], [""], ["ooo", "nope"]])
    def test_unknown_model_on_compare(self, client, models):
        assert_json_error(
            client.post("/compare", json={"asm_text": "NOP", "isa": "riscv",
                                          "models": models}),
            allowed=(400,))

    @pytest.mark.parametrize("name", ["no_such_example", "zzz", "main"])
    def test_unknown_c_example_name_is_404(self, client, name):
        assert_json_error(client.get(f"/compiler/examples/{name}"), allowed=(404,))

    @pytest.mark.parametrize("name", ["Bad-Name", "collatz.c", "UPPER", "a b",
                                      "foo$bar", "foo-bar", "föö"])
    def test_malformed_c_example_name_is_400(self, client, name):
        assert_json_error(client.get(f"/compiler/examples/{name}"), allowed=(400,))

    def test_unknown_example_file_is_404(self, client):
        assert_json_error(
            client.get("/examples/riscv/algorithms/no_such_program.asm"),
            allowed=(404,))

    def test_unknown_example_category_is_404(self, client):
        assert_json_error(
            client.get("/examples/riscv/no_such_category/fibonacci.asm"),
            allowed=(404,))


# ============================================================================
# 3. WRONG JSON TYPE FOR EVERY FIELD
# ============================================================================
WRONG_TYPES = [
    pytest.param("string", id="string"),
    pytest.param(123, id="int"),
    pytest.param(1.5, id="float"),
    pytest.param(True, id="bool"),
    pytest.param(None, id="null"),
    pytest.param({"nested": {"deep": 1}}, id="object"),
    pytest.param([1, 2, 3], id="list"),
]


class TestWrongFieldTypes:
    """Every field of every POST route, given every wrong JSON type, must not
    produce a server error."""

    @pytest.mark.parametrize("value", WRONG_TYPES)
    @pytest.mark.parametrize("field", ["program", "input_mode", "asm_text",
                                       "cycles", "run_to_completion",
                                       "num_lanes", "prediction_stage",
                                       "cycle_costs"])
    def test_simulate_field_wrong_type(self, client, field, value):
        body = {"preset": "riscv/single_cycle", "cycles": 2, field: value}
        assert_never_500(client.post("/simulate", json=body))

    @pytest.mark.parametrize("value", WRONG_TYPES)
    @pytest.mark.parametrize("field", ["isa", "text"])
    def test_assemble_field_wrong_type(self, client, field, value):
        body = {"isa": "riscv", "text": "NOP", field: value}
        assert_never_500(client.post("/assemble", json=body))

    @pytest.mark.parametrize("value", WRONG_TYPES)
    def test_compile_source_wrong_type(self, client, value):
        assert_never_500(client.post("/compile", json={"source": value, "isa": "riscv"}))

    @pytest.mark.parametrize("value", WRONG_TYPES)
    @pytest.mark.parametrize("field", ["source", "asm_text", "cycle_costs"])
    def test_compare_field_wrong_type(self, client, field, value):
        body = {"asm_text": "NOP", "isa": "riscv", "models": ["single_cycle"],
                field: value}
        assert_never_500(client.post("/compare", json=body))

    @pytest.mark.parametrize("value", [["single_cycle"], "single_cycle", 5,
                                       None, {"m": 1}])
    def test_compare_models_wrong_type(self, client, value):
        assert_never_500(client.post("/compare", json={
            "asm_text": "NOP", "isa": "riscv", "models": value}))

    def test_simulate_cycle_costs_non_object_on_configurable(self, client):
        assert_json_error(client.post("/simulate", json={
            "preset": "riscv/configurable", "cycles": 2,
            "cycle_costs": [1, 2, 3]}), allowed=(400,))

    @pytest.mark.parametrize("bad", ["x", [1], {"a": 1}, "1.5", None])
    def test_simulate_cycle_costs_non_int_value(self, client, bad):
        """A non-integer per-class budget is a 400 (None means 'use default')."""
        resp = client.post("/simulate", json={
            "preset": "riscv/configurable", "cycles": 2, "cycle_costs": {"alu": bad}})
        assert_never_500(resp)
        if bad is not None:
            assert resp.status_code == 400, resp.data[:200]

    def test_simulate_program_entries_wrong_type(self, client):
        for bad in ([None], [[1]], [{"a": 1}], ["zz"], [True, "0x1"]):
            resp = client.post("/simulate", json={
                "preset": "riscv/single_cycle", "cycles": 2, "program": bad})
            assert_never_500(resp)

    def test_simulate_asm_text_non_string_is_400(self, client):
        assert_json_error(client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": 2,
            "input_mode": "asm", "asm_text": ["ADDI x1,x0,1"]}), allowed=(400,))


# ============================================================================
# 4. PATH TRAVERSAL  (the highest-value tests in this file)
# ============================================================================
EXAMPLES_TRAVERSAL = [
    "/examples/riscv/../../api/app.py",
    "/examples/riscv/algorithms/../../../api/app.py",
    "/examples/riscv/algorithms/..%2f..%2f..%2fapi%2fapp.py",
    "/examples/riscv/%2e%2e/%2e%2e/api/app.py",
    "/examples/riscv/algorithms/%2e%2e%2f%2e%2e%2f%2e%2e%2fapi%2fapp.py",
    "/examples/riscv/..%252f..%252fapi%252fapp.py",          # double-encoded
    "/examples/riscv/algorithms/....//....//api/app.py",     # filter-bypass form
    "/examples/riscv/..%5c..%5capi%5capp.py",                # backslash
    "/examples/riscv/\\..\\..\\api\\app.py",
    "/examples//etc/passwd",                                 # absolute-ish
    "/examples/riscv/algorithms/%2fetc%2fpasswd",
    "/examples/riscv/algorithms/fibonacci.asm%00.py",        # null byte
    "/examples/riscv/algorithms/..%00/fibonacci.asm",
    "/examples/riscv/algorithms/CLAUDE.md",                  # wrong extension
    "/examples/riscv/algorithms/app.py",
    "/examples/riscv/algorithms/.env",
    "/examples/riscv/../c/fib_iter.c",
    "/examples/riscv/algorithms/fibonacci.ASM",              # case-tricked ext
    "/examples/riscv/./algorithms/fibonacci.asm",            # dot segment
    "/examples/riscv/algorithms/./fibonacci.asm",
    "/examples/riscv/algorithms/subdir/../fibonacci.asm",
    "/examples/mips/algorithms/fibonacci.asm",               # unknown isa
    "/examples/riscv/onlytwo.asm",                           # wrong part count
    "/examples/riscv/a/b/c.asm",
    "/examples/~/secrets.asm",
    "/examples/riscv/algorithms/",
]

C_EXAMPLE_TRAVERSAL = [
    "/compiler/examples/../../api/app",
    "/compiler/examples/..%2f..%2fapi%2fapp",
    "/compiler/examples/%2e%2e%2f%2e%2e%2fapi%2fapp",
    "/compiler/examples/..%252f..%252fapi%252fapp",
    "/compiler/examples/....//....//api/app",
    "/compiler/examples/fib_iter/../../../api/app",
    "/compiler/examples/..%5c..%5capi%5capp",
    "/compiler/examples/%2e%2e",
    "/compiler/examples/fib_iter%00",
    "/compiler/examples/fib_iter%00.py",
    "/compiler/examples/fib_iter.c",
    "/compiler/examples/%2Fetc%2Fpasswd",
    "/compiler/examples/FIB_ITER",
    "/compiler/examples/",
    "/compiler/examples/.",
    "/compiler/examples/..",
]


class TestPathTraversal:
    """Neither file-serving route may ever return a byte from outside its
    directory, under any encoding, and never a 500."""

    @pytest.mark.parametrize("path", EXAMPLES_TRAVERSAL)
    def test_examples_route_rejects_traversal(self, client, path):
        resp = client.get(path, follow_redirects=True)
        assert_no_leak(resp)
        assert resp.status_code in (400, 404), (
            f"{path} -> {resp.status_code}: {resp.data[:200]!r}")
        if "application/json" in resp.content_type:
            assert "error" in json.loads(resp.data)

    @pytest.mark.parametrize("path", C_EXAMPLE_TRAVERSAL)
    def test_compiler_examples_route_rejects_traversal(self, client, path):
        resp = client.get(path, follow_redirects=True)
        assert_no_leak(resp)
        assert resp.status_code in (400, 404), (
            f"{path} -> {resp.status_code}: {resp.data[:200]!r}")
        assert b"int main" not in resp.data       # never leak sample source either
        if "application/json" in resp.content_type:
            assert "error" in json.loads(resp.data)

    @pytest.mark.parametrize("filename", ["..foo.asm", "x..y.asm", "..%2ffoo.asm",
                                          "a%2Fb.asm", "%2E%2E.asm"])
    def test_examples_rejects_any_dotdot_in_the_filename(self, client, filename):
        """The `'..' in filepath or filename != basename` gate
        (api/app.py:807-808) rejects a dot-dot ANYWHERE in the path, even inside
        an otherwise-harmless filename — a deliberately blunt instrument."""
        assert_json_error(
            client.get(f"/examples/riscv/algorithms/{filename}"), allowed=(400,))

    def test_examples_symlink_escape_is_blocked(self, client):
        """A symlink planted INSIDE programs/ that points at api/app.py must not
        be followed — the realpath containment guard (api/app.py:816-818) is the
        only thing standing between a writable examples dir and source disclosure."""
        link = os.path.join(appmod.PROGRAMS_DIR, "riscv", "algorithms",
                            "_pytest_symlink_escape.asm")
        target = os.path.join(os.path.realpath(ROOT), "api", "app.py")
        if os.path.lexists(link):
            os.unlink(link)
        os.symlink(target, link)
        try:
            resp = client.get("/examples/riscv/algorithms/_pytest_symlink_escape.asm")
            assert_json_error(resp, allowed=(400, 404))
        finally:
            os.unlink(link)

    def test_compiler_examples_symlink_escape_is_blocked(self, client):
        link = os.path.join(appmod.C_EXAMPLES_DIR, "_pytest_symlink_escape.c")
        target = os.path.join(os.path.realpath(ROOT), "api", "app.py")
        if os.path.lexists(link):
            os.unlink(link)
        os.symlink(target, link)
        try:
            resp = client.get("/compiler/examples/_pytest_symlink_escape")
            assert_json_error(resp, allowed=(400, 404))
        finally:
            os.unlink(link)

    def test_legitimate_example_still_serves(self, client):
        """The guards must not break the happy path."""
        resp = client.get("/examples/riscv/algorithms/fibonacci.asm")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["file"] == "riscv/algorithms/fibonacci.asm"
        assert data["content"].strip()

    def test_legitimate_c_example_still_serves(self, client):
        resp = client.get("/compiler/examples/fib_iter")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["name"] == "fib_iter"
        assert data["content"].strip()

    def test_every_catalog_path_is_inside_programs_dir(self, client):
        """Cross-check: nothing the catalog advertises resolves outside
        programs/ (a poisoned catalog entry would be just as bad as a
        traversal)."""
        real_root = os.path.realpath(appmod.PROGRAMS_DIR) + os.sep
        catalog = json.loads(client.get("/examples").data)
        seen = 0
        for categories in catalog.values():
            for payload in categories.values():
                for item in payload["items"]:
                    real = os.path.realpath(
                        os.path.join(appmod.PROGRAMS_DIR, item["file"]))
                    assert real.startswith(real_root), item["file"]
                    seen += 1
        assert seen > 0


# ============================================================================
# 5. NUMERIC CLAMPING
# ============================================================================
class TestCycleClamping:
    """`cycles` is clamped to 0..MAX_CYCLES; the response always carries
    cycles+1 snapshots (the +1 is the reset cycle)."""

    @pytest.mark.parametrize("requested,expected", [
        (0, 0), (1, 1), (-1, 0), (-100000, 0),
        (appmod.MAX_CYCLES, appmod.MAX_CYCLES),
        (appmod.MAX_CYCLES + 1, appmod.MAX_CYCLES),
        (10 ** 9, appmod.MAX_CYCLES),
    ])
    def test_cycles_clamped(self, client, requested, expected):
        resp = client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": requested,
            "input_mode": "asm", "asm_text": TINY_ASM["riscv"]})
        assert resp.status_code == 200, resp.data[:200]
        assert len(json.loads(resp.data)["cycles"]) == expected + 1

    def test_cycles_numeric_string_is_accepted(self, client):
        """`_coerce_int` deliberately accepts "50" (HTML form values are strings)."""
        resp = client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": "50",
            "input_mode": "asm", "asm_text": TINY_ASM["riscv"]})
        assert resp.status_code == 200
        assert len(json.loads(resp.data)["cycles"]) == 51

    def test_cycles_float_is_truncated_not_rejected(self, client):
        resp = client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": 5.9,
            "input_mode": "asm", "asm_text": TINY_ASM["riscv"]})
        assert resp.status_code == 200
        assert len(json.loads(resp.data)["cycles"]) == 6   # int(5.9) == 5

    @pytest.mark.parametrize("bad", ["fifty", "", [10], {"n": 10}, "0x10"])
    def test_non_numeric_cycles_is_400(self, client, bad):
        assert_json_error(client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": bad}), allowed=(400,))

    def test_cycles_null_falls_back_to_default(self, client):
        resp = client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": None,
            "input_mode": "asm", "asm_text": TINY_ASM["riscv"]})
        assert resp.status_code == 200
        assert len(json.loads(resp.data)["cycles"]) == 21   # default 20 + reset

    def test_program_length_cap(self, client):
        long_prog = [0x00000013] * (appmod.MAX_PROGRAM_WORDS + 1)
        body = assert_json_error(client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": 1, "program": long_prog}),
            allowed=(400,))
        assert "too long" in body["error"]

    def test_program_exactly_at_cap_is_accepted(self, client):
        prog = [0x00000013] * appmod.MAX_PROGRAM_WORDS
        resp = client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": 1, "program": prog})
        assert resp.status_code == 200, resp.data[:200]


class TestLaneClamping:
    """`num_lanes` is clamped to 1..MAX_LANES for the superscalar model."""

    @pytest.mark.parametrize("lanes,expected", [
        (-1, 1), (0, 1), (1, 1), (2, 2), (6, 6), (7, 7),
        (appmod.MAX_LANES, appmod.MAX_LANES),
        (appmod.MAX_LANES + 1, appmod.MAX_LANES),
        (1000, appmod.MAX_LANES), ("3", 3),
    ])
    def test_num_lanes_clamped_on_simulate(self, client, lanes, expected):
        resp = client.post("/simulate", json={
            "preset": "riscv/superscalar", "cycles": 3, "num_lanes": lanes,
            "input_mode": "asm", "asm_text": TINY_ASM["riscv"]})
        assert resp.status_code == 200, resp.data[:200]
        state = json.loads(resp.data)["cycles"][-1]
        n_alus = sum(1 for k in state if re.fullmatch(r"alu_\d+", k))
        assert n_alus == expected

    @pytest.mark.parametrize("lanes", [-1, 0, 1, 7, 9, 1000])
    def test_num_lanes_clamped_on_topology(self, client, lanes):
        resp = client.get(f"/topology/riscv/superscalar?num_lanes={lanes}")
        assert resp.status_code == 200, resp.data[:200]
        assert "nodes" in json.loads(resp.data)

    @pytest.mark.parametrize("bad", ["abc", "", "1.5", "1e9"])
    def test_non_numeric_num_lanes_is_400(self, client, bad):
        assert_json_error(
            client.get(f"/topology/riscv/superscalar?num_lanes={bad}"), allowed=(400,))
        assert_json_error(client.post("/simulate", json={
            "preset": "riscv/superscalar", "cycles": 2, "num_lanes": bad}),
            allowed=(400,))

    def test_num_lanes_on_non_superscalar_model_is_ignored(self, client):
        """A stray num_lanes must not change (or break) an in-order model."""
        base = client.post("/simulate", json={
            "preset": "riscv/pipeline", "cycles": 6,
            "input_mode": "asm", "asm_text": TINY_ASM["riscv"]})
        with_lanes = client.post("/simulate", json={
            "preset": "riscv/pipeline", "cycles": 6, "num_lanes": 4,
            "input_mode": "asm", "asm_text": TINY_ASM["riscv"]})
        assert base.status_code == with_lanes.status_code == 200
        assert json.loads(base.data)["cycles"] == json.loads(with_lanes.data)["cycles"]


class TestCycleCostClamping:
    """`cycle_costs` per-class budgets are clamped to 1..64 on the
    `configurable` model, and ignored everywhere else."""

    @pytest.mark.parametrize("value,expected", [
        (0, 1), (-1, 1), (-10 ** 9, 1), (1, 1), (64, 64), (65, 64),
        (10 ** 9, 64), ("7", 7),
    ])
    def test_alu_cost_clamped(self, client, value, expected):
        resp = client.post("/simulate", json={
            "preset": "riscv/configurable", "cycles": 2,
            "cycle_costs": {"alu": value}})
        assert resp.status_code == 200, resp.data[:200]
        assert json.loads(resp.data)["time_model"]["costs"]["alu"] == expected

    def test_missing_keys_default(self, client):
        resp = client.post("/simulate", json={
            "preset": "riscv/configurable", "cycles": 2, "cycle_costs": {}})
        assert resp.status_code == 200
        costs = json.loads(resp.data)["time_model"]["costs"]
        assert set(costs) == set(appmod._CYCLE_CLASSES)
        assert costs == dict(appmod._DEFAULT_COSTS)

    def test_extra_unknown_keys_are_ignored(self, client):
        resp = client.post("/simulate", json={
            "preset": "riscv/configurable", "cycles": 2,
            "cycle_costs": {"alu": 2, "teleport": 99, "": 3}})
        assert resp.status_code == 200
        costs = json.loads(resp.data)["time_model"]["costs"]
        assert set(costs) == set(appmod._CYCLE_CLASSES)
        assert costs["alu"] == 2

    def test_cycle_costs_on_non_configurable_model_is_ignored(self, client):
        """Even an INVALID cycle_costs must be ignored by a non-configurable
        model (the route only validates it when it will be used)."""
        resp = client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": 4,
            "cycle_costs": {"alu": "not-an-int"},
            "input_mode": "asm", "asm_text": TINY_ASM["riscv"]})
        assert resp.status_code == 200, resp.data[:200]
        assert "time_model" not in json.loads(resp.data)

    def test_compare_clamps_cycle_costs_the_same_way(self, client):
        resp = client.post("/compare", json={
            "asm_text": TINY_ASM["riscv"], "isa": "riscv",
            "models": ["configurable"],
            "cycle_costs": {"alu": -5, "load": 0, "store": 10 ** 9, "branch": 1}})
        assert resp.status_code == 200
        echoed = json.loads(resp.data)["cycle_costs"]
        assert echoed == {"alu": 1, "load": 1, "store": 64, "branch": 1}

    @pytest.mark.parametrize("bad", ["nope", 5, [1, 2], True])
    def test_compare_non_object_cycle_costs_is_400(self, client, bad):
        resp = client.post("/compare", json={
            "asm_text": "NOP", "isa": "riscv", "models": ["single_cycle"],
            "cycle_costs": bad})
        assert_never_500(resp)
        if bad not in (0, "", None, False):    # falsy -> `or {}` default
            assert resp.status_code == 400, resp.data[:200]

    @pytest.mark.parametrize("bad", ["abc", "", "1.5", [4]])
    def test_compare_non_integer_num_lanes_is_400(self, client, bad):
        """api/app.py:453-455 — /compare validates num_lanes exactly like
        /simulate does."""
        assert_json_error(client.post("/compare", json={
            "asm_text": "NOP", "isa": "riscv", "models": ["single_cycle"],
            "num_lanes": bad}), allowed=(400,))

    @pytest.mark.parametrize("cls", ["alu", "load", "store", "branch"])
    def test_compare_non_integer_cycle_cost_class_is_400(self, client, cls):
        """api/app.py:466-469 — each per-class budget is validated by name."""
        body = assert_json_error(client.post("/compare", json={
            "asm_text": "NOP", "isa": "riscv", "models": ["configurable"],
            "cycle_costs": {cls: "not-an-int"}}), allowed=(400,))
        assert cls in body["error"]


class TestRunToCompletion:
    """`run_to_completion` ignores `cycles`, caps at MAX_AUTO_CYCLES and always
    answers with a boolean `completed`."""

    def test_completed_true_for_a_settling_program(self, client):
        resp = client.post("/simulate", json={
            "preset": "riscv/single_cycle", "run_to_completion": True,
            "input_mode": "asm", "asm_text": "ADDI x1, x0, 5\nNOP\n"})
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["run_to_completion"] is True
        assert isinstance(data["completed"], bool) and data["completed"] is True
        assert len(data["cycles"]) < appmod.MAX_AUTO_CYCLES

    def test_completed_false_and_capped_for_a_never_settling_program(self, client, monkeypatch):
        monkeypatch.setattr(appmod, "MAX_AUTO_CYCLES", 200)
        resp = client.post("/simulate", json={
            "preset": "riscv/single_cycle", "run_to_completion": True,
            "input_mode": "asm", "asm_text": "loop:\nADDI x1, x1, 1\nJAL x0, loop\n"})
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["completed"] is False
        assert len(data["cycles"]) == 200 + 1

    @pytest.mark.parametrize("truthy", [True, 1, "yes", "false", [0], {"a": 1}])
    def test_truthy_values_enable_auto_mode(self, client, truthy):
        """bool() coercion: any truthy JSON value turns auto mode on. (Note the
        string "false" is truthy in Python — documented, not a crash.)"""
        resp = client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": 2,
            "run_to_completion": truthy,
            "input_mode": "asm", "asm_text": TINY_ASM["riscv"]})
        assert resp.status_code == 200
        assert json.loads(resp.data)["completed"] is True

    @pytest.mark.parametrize("falsy", [False, 0, "", None, [], {}])
    def test_falsy_values_keep_manual_mode(self, client, falsy):
        resp = client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": 3,
            "run_to_completion": falsy,
            "input_mode": "asm", "asm_text": TINY_ASM["riscv"]})
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "completed" not in data
        assert len(data["cycles"]) == 4


# ============================================================================
# 6. WRONG-SHAPE COMBINATIONS
# ============================================================================
class TestWrongShapeCombinations:

    def test_x86_byte_program_on_riscv_preset_does_not_500(self, client):
        """Bytes where 32-bit words are expected: garbage in, garbage out —
        but never a crash."""
        resp = client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": 5,
            "program": [0xB9, 0x0A, 0x00, 0x00, 0x00]})
        assert_never_500(resp)
        assert resp.status_code == 200

    def test_riscv_word_program_on_x86_preset_does_not_500(self, client):
        resp = client.post("/simulate", json={
            "preset": "x86/single_cycle", "cycles": 5, "program": [0x02A00093]})
        assert_never_500(resp)
        assert resp.status_code == 200

    def test_arm_asm_on_riscv_preset_is_400(self, client):
        assert_json_error(client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": 3,
            "input_mode": "asm", "asm_text": "MOVZ X1, #5"}), allowed=(400,))

    def test_both_program_and_asm_supplied_asm_wins(self, client):
        """input_mode=asm + a non-empty asm_text overrides `program` — pin the
        precedence so it can't silently flip."""
        resp = client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": 4,
            "program": [0x02A00093],                 # ADDI x1, x0, 42
            "input_mode": "asm", "asm_text": "ADDI x1, x0, 7"})
        assert resp.status_code == 200
        assert json.loads(resp.data)["cycles"][-1]["regfile"]["registers"][1] == 7

    def test_program_used_when_input_mode_is_hex(self, client):
        resp = client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": 4, "input_mode": "hex",
            "program": [0x02A00093], "asm_text": "ADDI x1, x0, 7"})
        assert resp.status_code == 200
        assert json.loads(resp.data)["cycles"][-1]["regfile"]["registers"][1] == 42

    def test_neither_program_nor_asm_uses_the_demo(self, client):
        resp = client.post("/simulate", json={"preset": "riscv/single_cycle",
                                              "cycles": 4})
        assert resp.status_code == 200
        assert json.loads(resp.data)["cycles"][-1]["regfile"]["registers"][1] != 0

    def test_empty_program_list_falls_back_to_demo(self, client):
        resp = client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": 4, "program": []})
        assert resp.status_code == 200

    def test_unknown_input_mode_falls_back_to_program(self, client):
        resp = client.post("/simulate", json={
            "preset": "riscv/single_cycle", "cycles": 4, "input_mode": "telepathy",
            "program": [0x02A00093], "asm_text": "ADDI x1, x0, 7"})
        assert resp.status_code == 200
        assert json.loads(resp.data)["cycles"][-1]["regfile"]["registers"][1] == 42

    def test_compare_requires_exactly_one_of_source_or_asm(self, client):
        for body in ({}, {"source": "", "asm_text": ""},
                     {"source": "int main(){return 1;}", "asm_text": "NOP"},
                     {"source": "   ", "asm_text": "   "}):
            assert_json_error(client.post("/compare", json=body), allowed=(400,))

    def test_compare_isa_ignored_in_source_mode(self, client):
        """Core-C mode always spans all ISAs; a stray `isa` must not narrow or
        break it (and an unknown one must not 400 either)."""
        resp = client.post("/compare", json={
            "source": "int main(){return 1;}", "isa": "mips",
            "models": ["single_cycle"]})
        assert resp.status_code == 200, resp.data[:200]
        assert sorted(json.loads(resp.data)["isas"]) == ["arm", "riscv", "x86"]

    def test_compare_empty_models_list_yields_empty_grid(self, client):
        resp = client.post("/compare", json={
            "asm_text": "NOP", "isa": "riscv", "models": []})
        assert resp.status_code == 200
        assert json.loads(resp.data)["results"] == []

    def test_prediction_stage_garbage_does_not_500(self, client):
        for stage in ("zzz", "", "IF", "0"):
            resp = client.post("/simulate", json={
                "preset": "riscv/pipeline", "cycles": 4,
                "branch_predictor": "gshare", "prediction_stage": stage,
                "input_mode": "asm", "asm_text": TINY_ASM["riscv"]})
            assert_never_500(resp)


# ============================================================================
# 7. INTERNAL-FAILURE PATHS  (a broken preset must still answer in JSON)
# ============================================================================
class TestInternalFailuresStayJson:
    """api/app.py wraps CPU construction / simulation in try/except and answers
    with a JSON {error} 500 rather than an HTML crash page. Pin that."""

    @staticmethod
    def _boom(*a, **k):
        raise RuntimeError("synthetic preset failure")

    def test_topology_build_failure_is_json(self, client, monkeypatch):
        monkeypatch.setitem(appmod.PRESETS["riscv/single_cycle"], "build", self._boom)
        resp = client.get("/topology/riscv/single_cycle")
        assert_no_leak(resp)
        assert "application/json" in resp.content_type
        assert "error" in json.loads(resp.data)

    def test_simulate_build_failure_is_json(self, client, monkeypatch):
        monkeypatch.setitem(appmod.PRESETS["riscv/single_cycle"], "build", self._boom)
        resp = client.post("/simulate", json={"preset": "riscv/single_cycle",
                                              "cycles": 2})
        assert_no_leak(resp)
        assert "application/json" in resp.content_type
        assert "error" in json.loads(resp.data)

    def test_compare_target_failure_becomes_an_error_cell(self, client, monkeypatch):
        monkeypatch.setitem(appmod.PRESETS["riscv/pipeline"], "build", self._boom)
        resp = client.post("/compare", json={
            "asm_text": TINY_ASM["riscv"], "isa": "riscv",
            "models": ["single_cycle", "pipeline"]})
        assert resp.status_code == 200, resp.data[:200]
        cells = {r["model"]: r for r in json.loads(resp.data)["results"]}
        assert "error" in cells["pipeline"]
        assert "cycles" not in cells["pipeline"]
        assert cells["single_cycle"].get("completed") is True

    def test_compare_reference_failure_still_returns_a_grid(self, client, monkeypatch):
        """If the single_cycle oracle itself cannot run, the parity check is
        skipped rather than exploding."""
        monkeypatch.setitem(appmod.PRESETS["riscv/single_cycle"], "build", self._boom)
        resp = client.post("/compare", json={
            "asm_text": TINY_ASM["riscv"], "isa": "riscv", "models": ["pipeline"]})
        assert resp.status_code == 200, resp.data[:200]
        assert len(json.loads(resp.data)["results"]) == 1

    def test_compare_unavailable_preset_becomes_an_error_cell(self, client, monkeypatch):
        """A preset that dropped out of the registry (see the defensive
        try/except imports in the CLAUDE.md notes) yields an error cell."""
        trimmed = {k: v for k, v in appmod.PRESETS.items() if k != "arm/ooo"}
        monkeypatch.setattr(appmod, "PRESETS", trimmed)
        resp = client.post("/compare", json={
            "asm_text": TINY_ASM["arm"], "isa": "arm", "models": ["ooo"]})
        assert resp.status_code == 200, resp.data[:200]
        cell = json.loads(resp.data)["results"][0]
        assert cell["error"] == "preset not available"

    def test_compile_non_compiler_exception_is_400(self, client, monkeypatch):
        """Untrusted source must never raw-500 even if the compiler raises
        something that is not a CompilerError (api/app.py:625-626)."""
        def boom(source, isa):
            raise RuntimeError("synthetic codegen crash")
        monkeypatch.setattr(appmod, "compile_c", boom)
        assert_json_error(client.post("/compile", json={
            "source": "int main(){return 1;}", "isa": "riscv"}), allowed=(400,))

    def test_compile_assembly_failure_is_400(self, client, monkeypatch):
        """Bad codegen must surface as a 400 at /compile, not downstream
        (api/app.py:632-633)."""
        def boom(isa, text):
            raise RuntimeError("synthetic assembler crash")
        monkeypatch.setattr(appmod, "assemble", boom)
        body = assert_json_error(client.post("/compile", json={
            "source": "int main(){return 1;}", "isa": "riscv"}), allowed=(400,))
        assert "Assembly error" in body["error"]

    def test_compare_frontend_failure_becomes_an_error_cell(self, client):
        """A program that will not assemble yields per-cell errors, not a 400
        for the whole grid (api/app.py:509-510, 534-537)."""
        resp = client.post("/compare", json={
            "asm_text": "TOTALLY_NOT_AN_INSTRUCTION x1", "isa": "riscv",
            "models": ["single_cycle", "pipeline"]})
        assert resp.status_code == 200, resp.data[:200]
        for cell in json.loads(resp.data)["results"]:
            assert "error" in cell and "cycles" not in cell


# ============================================================================
# 8. CATALOG ROUTES WITH A HOSTILE / MISSING FILESYSTEM
# ============================================================================
class TestCatalogRoutesTolerateOddDirectories:

    def test_examples_catalog_with_missing_programs_dir(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(appmod, "PROGRAMS_DIR", str(tmp_path / "gone"))
        resp = client.get("/examples")
        assert resp.status_code == 200
        assert json.loads(resp.data) == {}

    def test_examples_catalog_skips_non_asm_and_unknown_isa_dirs(self, client,
                                                                 monkeypatch, tmp_path):
        (tmp_path / "riscv" / "algorithms").mkdir(parents=True)
        (tmp_path / "riscv" / "algorithms" / "good.asm").write_text("; === Good ===\nNOP\n")
        (tmp_path / "riscv" / "algorithms" / "notes.txt").write_text("ignore me")
        (tmp_path / "riscv" / "algorithms" / "secret.py").write_text("import os")
        (tmp_path / "riscv" / "unlisted_category").mkdir()
        (tmp_path / "mips").mkdir()
        (tmp_path / "mips" / "algorithms").mkdir()
        (tmp_path / "mips" / "algorithms" / "evil.asm").write_text("NOP")
        monkeypatch.setattr(appmod, "PROGRAMS_DIR", str(tmp_path))
        data = json.loads(client.get("/examples").data)
        assert set(data) == {"riscv"}
        assert set(data["riscv"]) == {"algorithms"}
        names = [i["name"] for i in data["riscv"]["algorithms"]["items"]]
        assert names == ["good"]

    def test_examples_catalog_untitled_file_falls_back_to_filename(self, client,
                                                                   monkeypatch, tmp_path):
        (tmp_path / "riscv" / "hazards").mkdir(parents=True)
        (tmp_path / "riscv" / "hazards" / "load_use_stall.asm").write_text("NOP\n")
        monkeypatch.setattr(appmod, "PROGRAMS_DIR", str(tmp_path))
        data = json.loads(client.get("/examples").data)
        assert data["riscv"]["hazards"]["items"][0]["label"] == "Load Use Stall"

    def test_c_catalog_with_missing_dir(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(appmod, "C_EXAMPLES_DIR", str(tmp_path / "gone"))
        resp = client.get("/compiler/examples")
        assert resp.status_code == 200
        assert json.loads(resp.data) == {"items": []}

    def test_c_catalog_skips_non_c_files_and_handles_headerless(self, client,
                                                                monkeypatch, tmp_path):
        (tmp_path / "plain.c").write_text("int main(){return 0;}\n")
        (tmp_path / "readme.md").write_text("# nope")
        (tmp_path / "script.py").write_text("import os")
        monkeypatch.setattr(appmod, "C_EXAMPLES_DIR", str(tmp_path))
        items = json.loads(client.get("/compiler/examples").data)["items"]
        assert [i["name"] for i in items] == ["plain"]
        assert items[0] == {"name": "plain", "label": "Plain", "file": "plain.c",
                            "description": "", "targets": []}


# ============================================================================
# 9. HELPER-LEVEL EDGE CASES  (branches unreachable through a normal request)
# ============================================================================
class TestHelperEdgeCases:

    def test_coerce_int_none_returns_default(self):
        assert appmod._coerce_int(None, 7) == 7

    @pytest.mark.parametrize("bad", ["abc", [1], {"a": 1}, object()])
    def test_coerce_int_bad_value_returns_none(self, bad):
        assert appmod._coerce_int(bad, 7) is None

    def test_coerce_int_accepts_numeric_strings_and_floats(self):
        assert appmod._coerce_int("42", 0) == 42
        assert appmod._coerce_int(4.9, 0) == 4

    def test_demo_comments_unknown_isa_is_empty(self):
        assert appmod._get_demo_comments("mips") == []
        assert appmod._get_demo_comments("x86") == []

    def test_format_demo_words_without_comments(self):
        """A program longer than the annotation list emits bare hex lines."""
        class _Stub:
            name = "mips"
            @staticmethod
            def demo_program():
                return [0x11223344, 0x55667788]
        text = appmod._format_demo_text_words(_Stub())
        assert text.splitlines() == ["0x11223344", "0x55667788"]

    def test_format_demo_bytes_without_annotations(self, monkeypatch):
        monkeypatch.setattr(appmod, "_get_x86_demo_annotations", lambda: [])
        class _Stub:
            @staticmethod
            def demo_program():
                return [0xB9, 0x0A]
        assert appmod._format_demo_text_bytes(_Stub()).splitlines() == ["B9", "0A"]

    def test_c_example_meta_on_unreadable_path(self):
        assert appmod._c_example_meta("/no/such/dir/file.c") == ("", [])

    def test_extract_title_on_unreadable_path_falls_back_to_filename(self):
        assert appmod._extract_title("/no/such/dir/bubble_sort.asm") == "Bubble Sort"


# ============================================================================
# 10. GLOBAL INVARIANTS
# ============================================================================
class TestGlobalInvariants:

    @pytest.mark.parametrize("path", GET_ROUTES)
    def test_get_routes_never_500(self, client, path):
        resp = client.get(path)
        assert resp.status_code < 500, f"{path}: {resp.data[:200]!r}"
        assert_no_leak(resp)

    @pytest.mark.parametrize("path", GET_ROUTES)
    def test_get_routes_ignore_junk_query_params(self, client, path):
        resp = client.get(path + "?num_lanes=&branch_predictor=&foo=%00&x[]=1")
        assert resp.status_code < 500, f"{path}: {resp.data[:200]!r}"
        assert_no_leak(resp)

    def test_unrouted_path_is_404(self, client):
        resp = client.get("/definitely/not/a/route")
        assert resp.status_code == 404
        assert_no_leak(resp)

    def test_deeply_nested_json_does_not_500(self, client):
        deep = {"source": "x"}
        node = deep
        for _ in range(50):
            node["nested"] = {}
            node = node["nested"]
        assert_never_500(client.post("/compile", json=deep))

    def test_unicode_and_control_chars_in_source_are_400_not_500(self, client):
        for src in ("int main(){ 中文 }", "int main(){\x00}", "\x00\x1b[31m"):
            assert_never_500(client.post("/compile", json={"source": src,
                                                           "isa": "riscv"}))

    def test_unicode_in_assembly_is_400_not_500(self, client):
        for text in ("ADDI x1, x0, 中", "\x00", "ADDI\tx1,\tx0,\t 1"):
            assert_never_500(client.post("/assemble", json={"isa": "riscv",
                                                            "text": text}))


# ============================================================================
# 11. REGRESSION PINS for FIXED contract violations (2026-08-08)
# ============================================================================
#
# Each test below sends input a browser/JS client could realistically produce.
# All of them USED TO return a RAW HTML 500, violating the documented
# never-500 contract. Two root causes, both now fixed:
#
#   (A) UNHASHABLE LOOKUP KEYS — api/app.py did `value not in SOME_DICT`
#       before checking the type, so a JSON list/object raised
#       TypeError: unhashable type.
#         * /simulate  preset            (api/app.py:311)
#         * /simulate  branch_predictor  (api/app.py:306)
#         * /compare   isa               (api/app.py:449)
#         * /compare   models            (api/app.py:442, `set(models)`)
#         * /compile   isa               (api/app.py:618)
#       FIXED: an `isinstance(..., str)` guard precedes each lookup, and
#       /compare's `models` checks every element is a str before set().
#
#   (B) OverflowError FROM int(float('inf')) — `_coerce_int` (api/app.py:59-70)
#       caught only (TypeError, ValueError), but Python's JSON parser accepts
#       the non-standard literals Infinity / -Infinity / NaN, and
#       int(float("inf")) raises OverflowError.
#         * /simulate  cycles / num_lanes / cycle_costs[*]
#         * /compare   num_lanes / cycle_costs[*]
#       FIXED: OverflowError added to the except clause.
#
# These are kept as ordinary passing tests so the guards cannot be removed
# without a failure.
# ============================================================================
class TestContractViolationsUnhashableKeys:

    @pytest.mark.parametrize("preset", [["riscv/single_cycle"], {"a": 1}])
    def test_simulate_non_scalar_preset_should_be_400(self, client, preset):
        assert_json_error(client.post("/simulate", json={"preset": preset,
                                                         "cycles": 1}))

    @pytest.mark.parametrize("bp", [["gshare"], {"name": "gshare"}])
    def test_simulate_non_scalar_branch_predictor_should_be_400(self, client, bp):
        assert_json_error(client.post("/simulate", json={
            "preset": "riscv/pipeline", "cycles": 1, "branch_predictor": bp}))

    def test_topology_branch_predictor_is_correctly_guarded(self, client):
        """Contrast case: /topology reads branch_predictor from the QUERY
        STRING, which is always a str, so its identical `not in
        PREDICTOR_CLASSES` lookup (api/app.py:257) can never hit the
        unhashable-key bug. Only the JSON routes are exposed."""
        assert_json_error(
            client.get("/topology/riscv/pipeline?branch_predictor=nope"),
            allowed=(400,))

    def test_compare_non_scalar_isa_should_be_400(self, client):
        assert_json_error(client.post("/compare", json={
            "asm_text": "NOP", "isa": ["riscv"], "models": ["single_cycle"]}))

    def test_compare_models_containing_objects_should_be_400(self, client):
        assert_json_error(client.post("/compare", json={
            "asm_text": "NOP", "isa": "riscv", "models": [{"m": "single_cycle"}]}))

    def test_compile_non_scalar_isa_should_be_400(self, client):
        assert_json_error(client.post("/compile", json={
            "source": "int main(){return 1;}", "isa": ["riscv"]}))


class TestContractViolationsInfinityOverflow:
    """Python's json module ACCEPTS the non-standard literals Infinity /
    -Infinity / NaN, and `_coerce_int` does not catch the OverflowError that
    int(float('inf')) raises. NaN happens to be caught (int(nan) is a
    ValueError), so only the infinities blow up."""

    @staticmethod
    def _raw(client, route, body):
        return client.post(route, data=body, content_type="application/json")

    def test_nan_cycles_is_correctly_rejected(self, client):
        """Control case: NaN -> ValueError -> handled 400. Proves the gap below
        is specifically OverflowError, not 'floats' in general."""
        assert_json_error(self._raw(client, "/simulate", '{"cycles": NaN}'),
                          allowed=(400,))

    @pytest.mark.parametrize("literal", ["Infinity", "-Infinity"])
    def test_infinite_cycles_should_be_400(self, client, literal):
        assert_json_error(self._raw(client, "/simulate",
                                    '{"cycles": %s}' % literal), allowed=(400,))

    def test_infinite_num_lanes_should_be_400(self, client):
        assert_json_error(self._raw(
            client, "/simulate",
            '{"preset":"riscv/superscalar","cycles":2,"num_lanes":Infinity}'),
            allowed=(400,))

    def test_infinite_cycle_cost_should_be_400(self, client):
        assert_json_error(self._raw(
            client, "/simulate",
            '{"preset":"riscv/configurable","cycles":2,'
            '"cycle_costs":{"alu":Infinity}}'), allowed=(400,))

    def test_compare_infinite_num_lanes_should_be_400(self, client):
        assert_json_error(self._raw(
            client, "/compare",
            '{"asm_text":"NOP","isa":"riscv","models":["single_cycle"],'
            '"num_lanes":Infinity}'), allowed=(400,))

    def test_topology_infinity_query_param_is_correctly_rejected(self, client):
        """Contrast: query params arrive as the STRING "Infinity", int() raises
        ValueError, and the route answers 400 as documented."""
        assert_json_error(
            client.get("/topology/riscv/superscalar?num_lanes=Infinity"),
            allowed=(400,))


# ============================================================================
# REGRESSION PIN — embedded NUL in an otherwise-valid example path (FIXED)
#
# The traversal guards reject anything containing ".." or not ending in
# ".asm" BEFORE touching the filesystem, so every ".."/wrong-extension NUL
# vector was already correctly refused. But a path that was traversal-clean
# and DID end in ".asm" reached `os.path.realpath(fpath)` (api/app.py:816),
# and realpath raises ValueError("embedded null character in path") -> raw 500.
#
# No file ever leaked (it failed before any read), so this was a robustness /
# contract bug rather than a disclosure one.
#
# FIXED 2026-08-08: `if "\x00" in filepath` is now checked alongside the ".."
# guard, before any filesystem call.
# ============================================================================
class TestContractViolationNullByteInPath:

    @pytest.mark.parametrize("path", [
        "/examples/riscv/basic/test%00.asm",
        "/examples/riscv/basic/fibonacci%00x.asm",
    ])
    def test_nul_in_examples_filename_should_be_400_not_500(self, client, path):
        resp = client.get(path)
        assert_no_leak(resp)
        assert_json_error(resp)

    def test_nul_vectors_that_are_caught_earlier_are_fine(self, client):
        """Contrast: these NUL paths also contain ".." or a bad extension, so
        the string-level guards reject them before realpath is reached."""
        for path in ("/examples/riscv/basic/x%00.py",
                     "/examples/riscv/basic/..%00/x.asm"):
            resp = client.get(path)
            assert_no_leak(resp)
            assert resp.status_code in (400, 404), path


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
