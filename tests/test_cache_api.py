"""
Cache API + UI contract tests.

WHY THIS FILE EXISTS
--------------------
The cache panel is driven entirely by two endpoints, and the JavaScript reads
specific keys out of them with no null-guarding on the nested shapes. A silent
rename in `api/app.py` or `CacheStats.to_dict()` would leave the panel showing
em-dashes with nothing failing, so the KEY NAMES are pinned here — the same
reason `test_lab_page.py` pins the `/lab` id contract.

Covered:
  * GET  /cache/policies — the catalog + legal value lists + timing matrix
  * POST /simulate       — the `cache` request field and the `cache_stats`
                           response, plus the per-cycle set view
  * every documented rejection path (bad geometry, bad policy, wrong types,
    Infinity, and timing on an unsupported model) returning JSON 400
  * the HTML/JS id contract for the panel

Run from repo root:  python3 -m pytest tests/test_cache_api.py -q
"""
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.app import app
from sim.assembler import assemble

ROOT = os.path.join(os.path.dirname(__file__), "..")

# A program with a store, a load that hits the same line, and a use.
ASM = ("ADDI x1, x0, 16\n"
       "ADDI x2, x0, 77\n"
       "SW   x2, 0(x1)\n"
       "LW   x3, 0(x1)\n"
       "ADD  x4, x3, x2\n")


@pytest.fixture()
def client():
    old = (app.config.get("TESTING"), app.config.get("PROPAGATE_EXCEPTIONS"))
    app.config["TESTING"] = False
    app.config["PROPAGATE_EXCEPTIONS"] = False
    try:
        yield app.test_client()
    finally:
        app.config["TESTING"], app.config["PROPAGATE_EXCEPTIONS"] = old


@pytest.fixture()
def program():
    return assemble("riscv", ASM)


def _simulate(client, program, cache=None, preset="riscv/single_cycle"):
    body = {"preset": preset, "program": program, "run_to_completion": True}
    if cache is not None:
        body["cache"] = cache
    return client.post("/simulate", json=body)


# ════════════════════════════════════════════════════════════════════════
# GET /cache/policies
# ════════════════════════════════════════════════════════════════════════
class TestCachePoliciesEndpoint:

    def test_returns_json_200(self, client):
        res = client.get("/cache/policies")
        assert res.status_code == 200
        assert "application/json" in res.content_type

    def test_lists_every_discovered_policy(self, client):
        from sim.components.memory.cache import POLICIES
        body = client.get("/cache/policies").get_json()
        assert {p["name"] for p in body["policies"]} == set(POLICIES)

    def test_each_policy_entry_has_the_fields_the_ui_renders(self, client):
        """The <option> needs a label and a title tooltip."""
        for entry in client.get("/cache/policies").get_json()["policies"]:
            assert entry["name"] and entry["label"]
            assert isinstance(entry["description"], str)

    def test_write_policies_match_the_model(self, client):
        from sim.components.memory.cache import WRITE_POLICIES
        body = client.get("/cache/policies").get_json()
        assert body["write_policies"] == list(WRITE_POLICIES)

    @pytest.mark.parametrize("key", ["size_kb", "line_bytes", "assoc"])
    def test_choices_are_powers_of_two_in_ascending_order(self, client, key):
        """The geometry sliders index INTO these lists, so order and
        pow2-ness are load-bearing, not decorative."""
        choices = client.get("/cache/policies").get_json()["choices"][key]
        assert choices == sorted(choices)
        assert all(v > 0 and not (v & (v - 1)) for v in choices)

    @pytest.mark.parametrize("key", ["size_kb", "line_bytes", "assoc",
                                     "policy", "write_policy", "miss_penalty",
                                     "write_allocate"])
    def test_defaults_cover_every_knob(self, client, key):
        assert key in client.get("/cache/policies").get_json()["defaults"]

    def test_defaults_are_themselves_legal_choices(self, client):
        body = client.get("/cache/policies").get_json()
        for key in ("size_kb", "line_bytes", "assoc"):
            assert body["defaults"][key] in body["choices"][key]
        assert body["defaults"]["policy"] in {p["name"] for p in body["policies"]}
        assert body["defaults"]["write_policy"] in body["write_policies"]

    def test_timing_models_matches_the_model_layer(self, client):
        """The UI greys out the stall checkbox from this list, so it must not
        drift from the component's own support matrix."""
        from sim.components.memory.cache import TIMING_MODELS
        body = client.get("/cache/policies").get_json()
        assert body["timing_models"] == list(TIMING_MODELS)

    def test_a_default_config_actually_builds(self, client):
        """A default the server advertises but cannot construct would be a
        first-click failure."""
        from sim.components.memory.cache import CacheConfig
        d = client.get("/cache/policies").get_json()["defaults"]
        CacheConfig(**{k: d[k] for k in
                       ("size_kb", "line_bytes", "assoc", "policy",
                        "write_policy", "write_allocate", "miss_penalty")})


# ════════════════════════════════════════════════════════════════════════
# POST /simulate — the cache request/response contract
# ════════════════════════════════════════════════════════════════════════
class TestSimulateCacheContract:

    def test_no_cache_field_means_no_cache_stats(self, client, program):
        """The plain DataMemory path must be untouched."""
        body = _simulate(client, program).get_json()
        assert "cache_stats" not in body

    def test_cache_true_uses_the_defaults(self, client, program):
        from sim.components.memory.cache import DEFAULT_ASSOC
        body = _simulate(client, program, cache=True).get_json()
        assert body["cache_stats"]["config"]["assoc"] == DEFAULT_ASSOC

    def test_cache_object_is_honoured(self, client, program):
        cfg = {"size_kb": 2, "line_bytes": 32, "assoc": 2, "policy": "fifo",
               "write_policy": "write_through"}
        got = _simulate(client, program, cache=cfg).get_json()["cache_stats"]["config"]
        for key, val in cfg.items():
            assert got[key] == val

    @pytest.mark.parametrize("key", ["accesses", "hits", "misses", "hit_rate",
                                     "evictions", "writebacks", "stall_cycles"])
    def test_stats_expose_every_field_the_readout_shows(self, client, program, key):
        stats = _simulate(client, program, cache=True).get_json()["cache_stats"]["stats"]
        assert key in stats

    def test_timing_enabled_is_reported_back(self, client, program):
        body = _simulate(client, program, cache={"timing_enabled": False})
        assert body.get_json()["cache_stats"]["timing_enabled"] is False

    def test_the_program_actually_generated_traffic(self, client, program):
        """Guard against a vacuous pass: 1 store + 1 load to the same line."""
        stats = _simulate(client, program, cache=True).get_json()["cache_stats"]["stats"]
        assert stats["accesses"] == 2
        assert stats["misses"] == 1        # the store misses cold
        assert stats["hits"] == 1          # the load hits the filled line

    def test_timing_adds_exactly_misses_times_penalty(self, client, program):
        base = len(_simulate(client, program).get_json()["cycles"])
        body = _simulate(client, program,
                         cache={"miss_penalty": 10, "timing_enabled": True}).get_json()
        stats = body["cache_stats"]["stats"]
        assert len(body["cycles"]) - base == stats["misses"] * 10
        assert stats["stall_cycles"] == stats["misses"] * 10

    def test_observe_only_does_not_change_the_cycle_count(self, client, program):
        base = len(_simulate(client, program).get_json()["cycles"])
        cached = len(_simulate(client, program, cache=True).get_json()["cycles"])
        assert cached == base

    def test_results_are_unchanged_by_the_cache(self, client, program):
        """Transparency, asserted through the API the UI actually calls."""
        plain = _simulate(client, program).get_json()["cycles"][-1]
        cached = _simulate(client, program,
                           cache={"miss_penalty": 10,
                                  "timing_enabled": True}).get_json()["cycles"][-1]
        assert cached["regfile"]["registers"] == plain["regfile"]["registers"]

    @pytest.mark.parametrize("model", ["single_cycle", "multicycle",
                                       "configurable", "pipeline", "ooo",
                                       "superscalar"])
    def test_observe_only_works_on_every_model(self, client, program, model):
        res = _simulate(client, program, cache=True, preset=f"riscv/{model}")
        assert res.status_code == 200
        assert "cache_stats" in res.get_json()


class TestPerCycleSetView:
    """The set-array renderer reads these nested keys with no guards."""

    @pytest.fixture()
    def access_cycle(self, client, program):
        body = _simulate(client, program,
                         cache={"size_kb": 1, "line_bytes": 64, "assoc": 2}).get_json()
        for cyc in body["cycles"]:
            if cyc.get("dmem", {}).get("cache_result") in ("HIT", "MISS"):
                return cyc["dmem"]
        pytest.fail("no cycle reported a cache access")

    def test_cache_result_is_a_known_label(self, access_cycle):
        assert access_cycle["cache_result"] in ("HIT", "MISS", "IDLE")

    def test_focus_set_and_num_sets_are_present(self, access_cycle):
        cache = access_cycle["cache"]
        assert isinstance(cache["focus_set"], int)
        assert cache["config"]["num_sets"] >= 1
        assert 0 <= cache["focus_set"] < cache["config"]["num_sets"]

    def test_ways_has_one_entry_per_way_with_render_fields(self, access_cycle):
        cache = access_cycle["cache"]
        ways = cache["ways"]
        assert len(ways) == cache["config"]["assoc"]
        for w in ways:
            assert set(w) >= {"way", "valid", "dirty", "tag", "age"}

    def test_last_access_locates_the_way_that_was_touched(self, access_cycle):
        la = access_cycle["last_access"]
        assert set(la) >= {"hit", "set", "way", "evicted", "wrote_back"}

    def test_datamemory_keys_still_present_for_the_memory_viewer(self, access_cycle):
        """The cache panel must not cost us the existing memory view."""
        for key in ("addr", "rdata", "wdata", "wen", "write_addr", "memory",
                    "memory_hi", "size", "window"):
            assert key in access_cycle

    def test_only_one_set_is_expanded_per_cycle(self, client, program):
        """Payload discipline: a large cache must not add every set to every
        cycle (the same bloat memory_snapshot's window exists to avoid)."""
        body = _simulate(client, program,
                         cache={"size_kb": 512, "line_bytes": 16, "assoc": 2}).get_json()
        cache = body["cycles"][-1]["dmem"]["cache"]
        assert cache["config"]["num_sets"] > 1000
        assert len(cache["ways"]) == 2


# ════════════════════════════════════════════════════════════════════════
# Rejection paths — the documented never-500 contract
# ════════════════════════════════════════════════════════════════════════
class TestCacheRequestValidation:

    def _err(self, client, program, cache):
        res = _simulate(client, program, cache=cache)
        assert res.status_code == 400, f"{cache} -> {res.status_code}"
        assert "application/json" in res.content_type
        assert "error" in json.loads(res.data)
        assert b"Traceback" not in res.data
        return res

    @pytest.mark.parametrize("cache", [
        {"size_kb": 3}, {"line_bytes": 48}, {"assoc": 3},
        {"size_kb": 0}, {"line_bytes": 0}, {"assoc": 0},
        {"size_kb": 10 ** 9}, {"line_bytes": 4096}, {"assoc": 64},
        {"size_kb": 1, "line_bytes": 1024, "assoc": 2},   # 1 line, 2-way
    ])
    def test_illegal_geometry_is_rejected(self, client, program, cache):
        self._err(client, program, cache)

    @pytest.mark.parametrize("cache", [
        {"policy": "nonexistent"}, {"policy": 123}, {"policy": ["lru"]},
        {"write_policy": "magic"}, {"write_policy": 1},
    ])
    def test_illegal_policy_is_rejected(self, client, program, cache):
        self._err(client, program, cache)

    @pytest.mark.parametrize("cache", [
        {"write_allocate": "yes"}, {"timing_enabled": "yes"},
        {"timing_enabled": 1}, {"write_allocate": None},
    ])
    def test_non_boolean_flags_are_rejected(self, client, program, cache):
        self._err(client, program, cache)

    @pytest.mark.parametrize("cache", [
        {"size_kb": "eight"}, {"size_kb": [8]}, {"size_kb": {"v": 8}},
        {"miss_penalty": "ten"}, {"assoc": None},
    ])
    def test_non_integer_numerics_are_rejected(self, client, program, cache):
        self._err(client, program, cache)

    @pytest.mark.parametrize("cache", ["on", [1, 2], 5, 1.5])
    def test_cache_field_of_the_wrong_type_is_rejected(self, client, program, cache):
        self._err(client, program, cache)

    @pytest.mark.parametrize("cache", [False, None])
    def test_falsey_cache_means_disabled_not_an_error(self, client, program, cache):
        res = _simulate(client, program, cache=cache)
        assert res.status_code == 200
        assert "cache_stats" not in res.get_json()

    def test_infinity_in_a_cache_field_is_rejected(self, client, program):
        """json accepts the non-standard Infinity literal; int(inf) overflows."""
        res = client.post("/simulate",
                          data=json.dumps({"preset": "riscv/single_cycle",
                                           "program": program}).rstrip("}")
                               + ', "cache": {"size_kb": Infinity}}',
                          content_type="application/json")
        assert res.status_code == 400
        assert "application/json" in res.content_type

    @pytest.mark.parametrize("model", ["configurable", "multicycle", "pipeline",
                                       "ooo", "superscalar"])
    def test_timing_on_an_unsupported_model_is_refused(self, client, program, model):
        """Silently ignoring this would corrupt ARCHITECTURAL results, so it
        must be a hard 400 naming the supported models."""
        res = _simulate(client, program, cache={"timing_enabled": True},
                        preset=f"riscv/{model}")
        assert res.status_code == 400
        assert "single_cycle" in json.loads(res.data)["error"]

    def test_timing_on_single_cycle_is_allowed(self, client, program):
        res = _simulate(client, program, cache={"timing_enabled": True})
        assert res.status_code == 200
        assert res.get_json()["cache_stats"]["timing_enabled"] is True


# ════════════════════════════════════════════════════════════════════════
# UI contract — the panel's ids and the assets that drive it
# ════════════════════════════════════════════════════════════════════════
class TestCachePanelMarkup:

    @pytest.fixture()
    def page(self, client):
        return client.get("/").get_data(as_text=True)

    @pytest.mark.parametrize("element_id", [
        "cache-panel", "cache-enable", "cache-body", "cache-policy",
        "cache-write-policy", "cache-timing", "cache-timing-wrap",
        "cache-timing-note", "cache-ways", "cache-set-label",
        "cache-set-result", "cache-accesses", "cache-hitmiss", "cache-evict",
        "cache-stall", "cache-hitrate",
    ])
    def test_panel_element_exists(self, page, element_id):
        assert f'id="{element_id}"' in page

    @pytest.mark.parametrize("knob", ["size_kb", "line_bytes", "assoc",
                                      "miss_penalty"])
    def test_every_knob_and_its_readout_exist(self, page, knob):
        assert f'data-cache="{knob}"' in page
        assert f'data-cache-val="{knob}"' in page

    def test_every_js_cache_id_lookup_resolves_in_the_html(self, page):
        """The JS is not null-guarded for these — a missing id is a TypeError."""
        with open(os.path.join(ROOT, "ui", "static", "js",
                               "cpu_simulator.js")) as fh:
            js = fh.read()
        ids = {i for i in re.findall(r"getElementById\('([\w-]+)'\)", js)
               if i.startswith("cache")}
        assert ids, "expected cache getElementById lookups"
        missing = sorted(i for i in ids if f'id="{i}"' not in page)
        assert missing == [], f"HTML missing ids used by the JS: {missing}"

    def test_js_cache_preset_buttons_all_exist_in_the_html(self, page):
        with open(os.path.join(ROOT, "ui", "static", "js",
                               "cpu_simulator.js")) as fh:
            js = fh.read()
        presets = set(re.findall(r"^\s{4}(tiny|typical|big):", js, re.M))
        for name in presets:
            assert f'data-cache-preset="{name}"' in page

    def test_cache_css_classes_used_by_the_js_are_defined(self):
        """A class the JS toggles but the stylesheet never defines is a
        silently invisible state (e.g. an eviction that does not show)."""
        with open(os.path.join(ROOT, "ui", "static", "js",
                               "cpu_simulator.js")) as fh:
            js = fh.read()
        with open(os.path.join(ROOT, "ui", "static", "css",
                               "cpu_simulator.css")) as fh:
            css = fh.read()
        used = set(re.findall(r"cache-(?:way|badge|timing)--[\w-]+", js))
        assert used, "expected state classes in the JS"
        missing = sorted(c for c in used if f".{c}" not in css)
        assert missing == [], f"CSS missing classes the JS toggles: {missing}"

    def test_the_generic_cfg_range_selector_excludes_cache_knobs(self):
        """The configurable model's cfg-range handler must not also fire for
        the cache sliders — they share the visual class but not the payload."""
        with open(os.path.join(ROOT, "ui", "static", "js",
                               "cpu_simulator.js")) as fh:
            js = fh.read()
        assert ".cfg-range:not([data-cache])" in js


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
