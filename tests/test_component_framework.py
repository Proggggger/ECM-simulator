"""Unit tests for the component FRAMEWORK: Port / ComponentBase / Wire /
CPU / CPUBuilder / generate_topology / run_simulation.

These modules are the substrate every preset is built on, but until now they
were only exercised *indirectly* through full-CPU preset runs — so the error
paths (bad wire endpoints, malformed eval orders), the port width-masking
contract, and tick()'s three-phase ordering had no direct coverage.

Everything here is hand-built from tiny toy components so a failure points at
the framework itself rather than at an ISA decoder.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.component.base import (
    ComponentBase, Port, mask32, sign_extend, to_signed32,
)
from sim.component.topology import generate_topology
from sim.component.wire import CPU, CPUBuilder, Wire
from sim.runner_v2 import STABLE_WINDOW, _arch_signature, run_simulation


# ── toy components used by the framework tests ──────────────────────────────

class Counter(ComponentBase):
    """Free-running counter. Sequential-only: value appears on `out`."""
    name = "counter"
    ui_label = "Counter"
    ui_category = "fetch"
    ports_spec = {"out": Port(8, "out", "counter value")}

    def __init__(self, start: int = 0, **kw):
        super().__init__(**kw)
        self._n = start

    def evaluate(self):
        self["out"] = self._n

    def rising_edge(self):
        self._n = (self._n + 1) & 0xFF


class Doubler(ComponentBase):
    """Purely combinational: out = in * 2 (mod 256)."""
    name = "doubler"
    ui_category = "execute"
    ports_spec = {
        "inp": Port(8, "in", "operand"),
        "out": Port(8, "out", "doubled"),
    }

    def evaluate(self):
        self["out"] = (self["inp"] * 2) & 0xFF


class Sink(ComponentBase):
    """Sequential sink: samples `inp` at the rising edge into `_seen`."""
    name = "sink"
    ui_category = "writeback"
    ports_spec = {"inp": Port(8, "in", "captured value")}

    def __init__(self, **kw):
        super().__init__(**kw)
        self._seen = []

    def rising_edge(self):
        self._seen.append(self["inp"])

    def get_state(self):
        return {"inp": self["inp"], "seen": list(self._seen)}


class Accumulator(ComponentBase):
    """Registered accumulator with a combinational feedback (back-edge) input."""
    ports_spec = {
        "fb":  Port(8, "in", "feedback"),
        "out": Port(8, "out", "accumulator value"),
    }

    def __init__(self, start: int = 1, **kw):
        super().__init__(**kw)
        self._v = start

    def evaluate(self):
        self["out"] = self._v

    def rising_edge(self):
        self._v = self["fb"]

    def get_state(self):
        return {"v": self._v, "out": self["out"], "fb": self["fb"]}


class ArchLike(ComponentBase):
    """Component whose get_state() looks 'architectural' to _arch_signature."""
    ports_spec = {"o": Port(32, "out", "")}

    def __init__(self, stop: int = 3, **kw):
        super().__init__(**kw)
        self._n = 0
        self._stop = stop

    def evaluate(self):
        self["o"] = self._n

    def rising_edge(self):
        if self._n < self._stop:
            self._n += 1

    def get_state(self):
        return {"registers": [self._n], "o": self["o"]}


# ── helpers ─────────────────────────────────────────────────────────────────

class TestMaskAndSignHelpers(unittest.TestCase):

    def test_mask32_boundaries(self):
        self.assertEqual(mask32(0), 0)
        self.assertEqual(mask32(1), 1)
        self.assertEqual(mask32(0x7FFF_FFFF), 0x7FFF_FFFF)
        self.assertEqual(mask32(0x8000_0000), 0x8000_0000)
        self.assertEqual(mask32(0xFFFF_FFFF), 0xFFFF_FFFF)

    def test_mask32_wraps_at_2_32(self):
        self.assertEqual(mask32(1 << 32), 0)
        self.assertEqual(mask32((1 << 32) + 5), 5)

    def test_mask32_of_negative_is_twos_complement(self):
        self.assertEqual(mask32(-1), 0xFFFF_FFFF)
        self.assertEqual(mask32(-2), 0xFFFF_FFFE)
        self.assertEqual(mask32(-(1 << 31)), 0x8000_0000)

    def test_sign_extend_12_bit(self):
        self.assertEqual(sign_extend(0x000, 12), 0)
        self.assertEqual(sign_extend(0x001, 12), 1)
        self.assertEqual(sign_extend(0x7FF, 12), 2047)     # largest positive
        self.assertEqual(sign_extend(0x800, 12), -2048)    # smallest negative
        self.assertEqual(sign_extend(0xFFF, 12), -1)

    def test_sign_extend_widths_1_and_32(self):
        self.assertEqual(sign_extend(0, 1), 0)
        self.assertEqual(sign_extend(1, 1), -1)
        self.assertEqual(sign_extend(0x7FFF_FFFF, 32), 0x7FFF_FFFF)
        self.assertEqual(sign_extend(0x8000_0000, 32), -(1 << 31))
        self.assertEqual(sign_extend(0xFFFF_FFFF, 32), -1)

    def test_sign_extend_zero_width_is_unsupported(self):
        """bits=0 is a degenerate call; it raises rather than silently lying."""
        with self.assertRaises(ValueError):
            sign_extend(0, 0)

    def test_to_signed32(self):
        self.assertEqual(to_signed32(0), 0)
        self.assertEqual(to_signed32(1), 1)
        self.assertEqual(to_signed32(0x7FFF_FFFF), 2147483647)
        self.assertEqual(to_signed32(0x8000_0000), -2147483648)
        self.assertEqual(to_signed32(0xFFFF_FFFF), -1)

    def test_to_signed32_masks_before_interpreting(self):
        # Inputs wider than 32 bits (or negative) are masked first.
        self.assertEqual(to_signed32(-1), -1)
        self.assertEqual(to_signed32((1 << 32) + 7), 7)
        self.assertEqual(to_signed32((1 << 33) | 0x8000_0000), -2147483648)

    def test_sign_extend_and_to_signed32_agree_at_32_bits(self):
        for v in (0, 1, 0x7FFF_FFFF, 0x8000_0000, 0xDEAD_BEEF, 0xFFFF_FFFF):
            self.assertEqual(sign_extend(v, 32), to_signed32(v), hex(v))


# ── Port ────────────────────────────────────────────────────────────────────

class TestPort(unittest.TestCase):

    def test_defaults(self):
        p = Port()
        self.assertEqual(p.width, 32)
        self.assertEqual(p.direction, "in")
        self.assertEqual(p.desc, "")

    def test_explicit_fields(self):
        p = Port(5, "out", "write address")
        self.assertEqual((p.width, p.direction, p.desc), (5, "out", "write address"))

    def test_repr_shows_width_and_direction(self):
        self.assertEqual(repr(Port(5, "out", "ignored")), "Port(w=5, out)")
        self.assertEqual(repr(Port()), "Port(w=32, in)")

    def test_slots_reject_extra_attributes(self):
        p = Port()
        with self.assertRaises(AttributeError):
            p.bogus = 1


# ── ComponentBase ───────────────────────────────────────────────────────────

class TestComponentBase(unittest.TestCase):

    def test_bare_component_has_no_ports_and_noop_hooks(self):
        c = ComponentBase()
        self.assertEqual(c.get_state(), {})
        self.assertIsNone(c.evaluate())
        self.assertIsNone(c.rising_edge())
        self.assertEqual(c.name, "")
        self.assertEqual(c.ui_label, "")
        self.assertEqual(c.ui_category, "")

    def test_ports_initialise_to_zero(self):
        d = Doubler()
        self.assertEqual(d["inp"], 0)
        self.assertEqual(d["out"], 0)

    def test_each_instance_gets_its_own_port_storage(self):
        a, b = Doubler(), Doubler()
        a["inp"] = 9
        self.assertEqual(b["inp"], 0)
        # ...but the SPEC is shared class state (declarative, read-only).
        self.assertIs(a.ports_spec, b.ports_spec)
        self.assertIs(a.ports_spec, Doubler.ports_spec)

    def test_unknown_port_raises_keyerror_on_read_and_write(self):
        d = Doubler()
        with self.assertRaises(KeyError):
            d["nope"]
        with self.assertRaises(KeyError):
            d["nope"] = 1

    def test_writes_are_masked_to_declared_width(self):
        class Widths(ComponentBase):
            ports_spec = {
                "w1":  Port(1,  "in", ""),
                "w4":  Port(4,  "in", ""),
                "w5":  Port(5,  "in", ""),
                "w32": Port(32, "in", ""),
                "w64": Port(64, "in", ""),
            }
        c = Widths()
        c["w1"] = 1
        self.assertEqual(c["w1"], 1)
        c["w1"] = 2
        self.assertEqual(c["w1"], 0, "bit 1 must be dropped by a 1-bit port")
        c["w1"] = -1
        self.assertEqual(c["w1"], 1)

        c["w4"] = 15
        self.assertEqual(c["w4"], 15)
        c["w4"] = 16
        self.assertEqual(c["w4"], 0)

        c["w5"] = 31
        self.assertEqual(c["w5"], 31)
        c["w5"] = 32
        self.assertEqual(c["w5"], 0)
        c["w5"] = 33
        self.assertEqual(c["w5"], 1)

        c["w32"] = 0xFFFF_FFFF
        self.assertEqual(c["w32"], 0xFFFF_FFFF)
        c["w32"] = 1 << 32
        self.assertEqual(c["w32"], 0)
        c["w32"] = -1
        self.assertEqual(c["w32"], 0xFFFF_FFFF)
        c["w32"] = 0x8000_0000
        self.assertEqual(c["w32"], 0x8000_0000)

        c["w64"] = (1 << 64) - 1
        self.assertEqual(c["w64"], 0xFFFF_FFFF_FFFF_FFFF)
        c["w64"] = 1 << 64
        self.assertEqual(c["w64"], 0)

    def test_get_state_defaults_to_a_copy_of_the_ports(self):
        d = Doubler()
        d["inp"] = 3
        d.evaluate()
        st = d.get_state()
        self.assertEqual(st, {"inp": 3, "out": 6})
        st["inp"] = 999            # mutating the snapshot must not corrupt the component
        self.assertEqual(d["inp"], 3)

    def test_reset_zeroes_every_port(self):
        d = Doubler()
        d["inp"] = 7
        d.evaluate()
        self.assertNotEqual(d["out"], 0)
        d.reset()
        self.assertEqual(d["inp"], 0)
        self.assertEqual(d["out"], 0)
        self.assertEqual(set(d.get_state()), {"inp", "out"})

    def test_reset_does_not_touch_sequential_state(self):
        """reset() is documented as zeroing PORTS only, not internal registers."""
        c = Counter(start=5)
        c.evaluate()
        c.reset()
        self.assertEqual(c["out"], 0)
        c.evaluate()
        self.assertEqual(c["out"], 5, "internal counter survives a port reset")

    def test_ui_label_constructor_override(self):
        self.assertEqual(Counter().ui_label, "Counter")
        self.assertEqual(Counter(ui_label="Renamed").ui_label, "Renamed")
        # A falsy override is ignored (documented `if ui_label:` guard).
        self.assertEqual(Counter(ui_label="").ui_label, "Counter")

    def test_ui_label_override_is_instance_local(self):
        Counter(ui_label="Renamed")
        self.assertEqual(Counter().ui_label, "Counter")


# ── Wire ────────────────────────────────────────────────────────────────────

class TestWire(unittest.TestCase):

    def test_propagate_copies_source_port_to_destination_port(self):
        src, dst = Counter(), Doubler()
        src["out"] = 21
        comps = {"src": src, "dst": dst}
        Wire("src", "out", "dst", "inp").propagate(comps)
        self.assertEqual(dst["inp"], 21)

    def test_propagate_applies_the_destination_width_mask(self):
        """A wire is not a bus check: a wide source narrows silently at the sink."""
        class Wide(ComponentBase):
            ports_spec = {"o": Port(32, "out", "")}

        class Narrow(ComponentBase):
            ports_spec = {"i": Port(8, "in", "")}

        w, n = Wide(), Narrow()
        w["o"] = 0x1FF
        Wire("w", "o", "n", "i").propagate({"w": w, "n": n})
        self.assertEqual(n["i"], 0xFF)

    def test_repr(self):
        self.assertEqual(repr(Wire("a", "o", "b", "i")), "a.o → b.i")

    def test_slots_reject_extra_attributes(self):
        with self.assertRaises(AttributeError):
            Wire("a", "o", "b", "i").extra = 1


# ── CPUBuilder ──────────────────────────────────────────────────────────────

class TestCPUBuilder(unittest.TestCase):

    def _two(self):
        b = CPUBuilder()
        b.add("src", Counter()).add("dst", Doubler())
        return b

    def test_add_and_wire_are_fluent(self):
        b = CPUBuilder()
        self.assertIs(b.add("src", Counter()), b)
        self.assertIs(b.add("dst", Doubler()), b)
        self.assertIs(b.wire("src.out", "dst.inp"), b)
        self.assertIs(b.set_eval_order(["src", "dst"]), b)

    def test_build_returns_a_cpu_with_the_registered_parts(self):
        b = self._two().wire("src.out", "dst.inp")
        cpu = b.build()
        self.assertIsInstance(cpu, CPU)
        self.assertEqual(set(cpu.components), {"src", "dst"})
        self.assertEqual(len(cpu.wires), 1)

    def test_add_with_a_duplicate_name_replaces_the_component(self):
        first, second = Counter(), Counter()
        cpu = CPUBuilder().add("c", first).add("c", second).build()
        self.assertIs(cpu.components["c"], second)
        self.assertEqual(len(cpu.components), 1)

    def test_wire_rejects_unknown_source_component(self):
        with self.assertRaises(ValueError) as cm:
            self._two().wire("nope.out", "dst.inp")
        self.assertIn("Unknown source component 'nope'", str(cm.exception))

    def test_wire_rejects_unknown_destination_component(self):
        with self.assertRaises(ValueError) as cm:
            self._two().wire("src.out", "nope.inp")
        self.assertIn("Unknown destination component 'nope'", str(cm.exception))

    def test_wire_rejects_unknown_source_port(self):
        with self.assertRaises(ValueError) as cm:
            self._two().wire("src.bogus", "dst.inp")
        self.assertIn("No port 'bogus' on 'src'", str(cm.exception))

    def test_wire_rejects_unknown_destination_port(self):
        with self.assertRaises(ValueError) as cm:
            self._two().wire("src.out", "dst.bogus")
        self.assertIn("No port 'bogus' on 'dst'", str(cm.exception))

    def test_wire_rejects_endpoint_without_a_dot(self):
        with self.assertRaises(ValueError):
            self._two().wire("srcout", "dst.inp")
        with self.assertRaises(ValueError):
            self._two().wire("src.out", "dstinp")

    def test_wire_splits_only_on_the_first_dot(self):
        """Port names may contain dots; the component name may not."""
        class Dotted(ComponentBase):
            ports_spec = {"a.b": Port(8, "in", "")}

        b = CPUBuilder().add("src", Counter()).add("d", Dotted())
        b.wire("src.out", "d.a.b")
        w = b.build().wires[0]
        self.assertEqual((w.dst_comp, w.dst_port), ("d", "a.b"))

    def test_a_rejected_wire_is_not_recorded(self):
        b = self._two()
        with self.assertRaises(ValueError):
            b.wire("src.bogus", "dst.inp")
        self.assertEqual(b.build().wires, [])

    def test_default_eval_order_is_insertion_order(self):
        cpu = self._two().build()
        self.assertEqual(cpu.eval_order, ["src", "dst"])

    def test_explicit_eval_order_is_preserved(self):
        cpu = self._two().set_eval_order(["dst", "src"]).build()
        self.assertEqual(cpu.eval_order, ["dst", "src"])

    def test_eval_order_rejects_duplicates(self):
        with self.assertRaises(ValueError) as cm:
            self._two().set_eval_order(["src", "src", "dst"]).build()
        self.assertIn("Duplicate components in eval order", str(cm.exception))
        self.assertIn("src", str(cm.exception))

    def test_eval_order_rejects_missing_components(self):
        """A component left out of the order is never evaluated OR clocked."""
        with self.assertRaises(ValueError) as cm:
            self._two().set_eval_order(["src"]).build()
        msg = str(cm.exception)
        self.assertIn("missing: ['dst']", msg)
        self.assertIn("unknown: []", msg)

    def test_eval_order_rejects_unknown_components(self):
        with self.assertRaises(ValueError) as cm:
            self._two().set_eval_order(["src", "dst", "ghost"]).build()
        msg = str(cm.exception)
        self.assertIn("missing: []", msg)
        self.assertIn("unknown: ['ghost']", msg)

    def test_eval_order_reports_missing_and_unknown_together(self):
        with self.assertRaises(ValueError) as cm:
            self._two().set_eval_order(["src", "ghost"]).build()
        msg = str(cm.exception)
        self.assertIn("missing: ['dst']", msg)
        self.assertIn("unknown: ['ghost']", msg)

    def test_empty_builder_builds_an_empty_cpu(self):
        cpu = CPUBuilder().build()
        self.assertEqual(cpu.components, {})
        self.assertEqual(cpu.eval_order, [])
        cpu.tick()  # must not explode
        self.assertEqual(cpu.get_cycle_state(), {})


# ── CPU.tick() three-phase semantics ────────────────────────────────────────

class TestCPUTick(unittest.TestCase):
    """A hand-built 3-component toy CPU: counter -> doubler -> sink."""

    def _chain(self):
        b = CPUBuilder()
        b.add("counter", Counter()).add("doubler", Doubler()).add("sink", Sink())
        b.wire("counter.out", "doubler.inp")
        b.wire("doubler.out", "sink.inp")
        b.set_eval_order(["counter", "doubler", "sink"])
        return b.build()

    def test_phase1_propagates_then_evaluates_in_order(self):
        cpu = self._chain()
        cpu.tick()
        st = cpu.get_cycle_state()
        # counter starts at 0 -> doubler sees 0 -> out 0
        self.assertEqual(st["counter"]["out"], 0)
        self.assertEqual(st["doubler"], {"inp": 0, "out": 0})
        cpu.tick()
        st = cpu.get_cycle_state()
        self.assertEqual(st["counter"]["out"], 1)
        self.assertEqual(st["doubler"], {"inp": 1, "out": 2})

    def test_combinational_chain_settles_within_one_tick(self):
        cpu = self._chain()
        for _ in range(5):
            cpu.tick()
        st = cpu.get_cycle_state()
        self.assertEqual(st["doubler"]["out"], st["counter"]["out"] * 2)

    def test_phase3_rising_edge_sees_phase2_propagated_values(self):
        """The sink is clocked; it must capture THIS cycle's doubler output."""
        cpu = self._chain()
        for _ in range(4):
            cpu.tick()
        self.assertEqual(cpu.components["sink"]._seen, [0, 2, 4, 6])

    def test_rising_edge_runs_for_every_component_in_eval_order(self):
        order = []

        class Recorder(ComponentBase):
            ports_spec = {}

            def __init__(self, tag, **kw):
                super().__init__(**kw)
                self.tag = tag

            def evaluate(self):
                order.append(("eval", self.tag))

            def rising_edge(self):
                order.append(("edge", self.tag))

        b = CPUBuilder().add("a", Recorder("a")).add("b", Recorder("b"))
        b.set_eval_order(["b", "a"])
        b.build().tick()
        self.assertEqual(order, [("eval", "b"), ("eval", "a"),
                                 ("edge", "b"), ("edge", "a")])

    def test_phase2_propagation_closes_a_back_edge(self):
        """A feedback loop resolves via phase 2 + rising_edge, not phase 1."""
        b = CPUBuilder().add("acc", Accumulator(start=1)).add("dbl", Doubler())
        b.wire("acc.out", "dbl.inp").wire("dbl.out", "acc.fb")
        b.set_eval_order(["acc", "dbl"])
        cpu = b.build()
        seen = []
        for _ in range(4):
            cpu.tick()
            seen.append(cpu.get_cycle_state()["acc"]["v"])
        self.assertEqual(seen, [2, 4, 8, 16])

    def test_wrong_eval_order_produces_a_stale_datapath(self):
        """Eval order must be a topological order; a bad one is observably wrong.

        Evaluating the consumer BEFORE the producer costs a cycle of latency,
        which is exactly the failure mode set_eval_order() exists to avoid.
        """
        b = CPUBuilder().add("acc", Accumulator(start=1)).add("dbl", Doubler())
        b.wire("acc.out", "dbl.inp").wire("dbl.out", "acc.fb")
        b.set_eval_order(["dbl", "acc"])   # consumer first: WRONG
        cpu = b.build()
        seen = []
        for _ in range(4):
            cpu.tick()
            seen.append(cpu.get_cycle_state()["acc"]["v"])
        self.assertNotEqual(seen, [2, 4, 8, 16])
        self.assertEqual(seen, [0, 2, 0, 4])

    def test_unwired_input_ports_stay_at_their_last_written_value(self):
        d = Doubler()
        cpu = CPUBuilder().add("dbl", d).build()
        d["inp"] = 5
        cpu.tick()
        self.assertEqual(d["out"], 10)
        cpu.tick()
        self.assertEqual(d["out"], 10, "no wire drives inp, so it is sticky")

    def test_a_wire_can_fan_out_to_several_sinks(self):
        b = CPUBuilder()
        b.add("counter", Counter()).add("d1", Doubler()).add("d2", Doubler())
        b.wire("counter.out", "d1.inp").wire("counter.out", "d2.inp")
        b.set_eval_order(["counter", "d1", "d2"])
        cpu = b.build()
        cpu.tick()
        cpu.tick()
        st = cpu.get_cycle_state()
        self.assertEqual(st["d1"]["out"], 2)
        self.assertEqual(st["d2"]["out"], 2)

    def test_last_wire_wins_when_two_sources_drive_one_input(self):
        """Wires are applied in declaration order; there is no bus arbitration."""
        b = CPUBuilder()
        b.add("a", Counter(start=1)).add("bb", Counter(start=9)).add("d", Doubler())
        b.wire("a.out", "d.inp").wire("bb.out", "d.inp")
        b.set_eval_order(["a", "bb", "d"])
        cpu = b.build()
        cpu.tick()
        self.assertEqual(cpu.components["d"]["inp"], 9)

    def test_get_cycle_state_covers_every_component(self):
        cpu = self._chain()
        cpu.tick()
        st = cpu.get_cycle_state()
        self.assertEqual(set(st), set(cpu.components))
        json.dumps(st)   # must be JSON-serialisable for the API

    def test_get_cycle_state_uses_each_components_get_state_override(self):
        cpu = self._chain()
        cpu.tick()
        self.assertIn("seen", cpu.get_cycle_state()["sink"])

    def test_get_cycle_state_before_any_tick_is_the_reset_snapshot(self):
        cpu = self._chain()
        st = cpu.get_cycle_state()
        self.assertEqual(st["counter"]["out"], 0)
        self.assertEqual(st["sink"]["seen"], [])


# ── topology ────────────────────────────────────────────────────────────────

class TestGenerateTopology(unittest.TestCase):

    def _chain(self):
        b = CPUBuilder()
        b.add("counter", Counter()).add("doubler", Doubler())
        b.wire("counter.out", "doubler.inp")
        return b.build()

    def test_shape(self):
        topo = generate_topology(self._chain())
        self.assertEqual(set(topo), {"nodes", "edges"})
        self.assertEqual(len(topo["nodes"]), 2)
        self.assertEqual(len(topo["edges"]), 1)

    def test_node_fields(self):
        node = generate_topology(self._chain())["nodes"][0]
        self.assertEqual(set(node), {"id", "label", "category", "ports"})
        self.assertEqual(node["id"], "counter")
        self.assertEqual(node["label"], "Counter")
        self.assertEqual(node["category"], "fetch")
        self.assertEqual(node["ports"], {
            "out": {"width": 8, "direction": "out", "desc": "counter value"},
        })

    def test_label_falls_back_ui_label_then_name_then_key(self):
        class Anon(ComponentBase):
            ports_spec = {"o": Port(1, "out", "")}

        class Named(ComponentBase):
            name = "the_name"
            ports_spec = {"o": Port(1, "out", "")}

        def label(key, comp):
            return generate_topology(CPUBuilder().add(key, comp).build())["nodes"][0]["label"]

        self.assertEqual(label("anon", Anon()), "anon")            # neither set
        self.assertEqual(label("x", Named()), "the_name")          # name only
        self.assertEqual(label("x", Named(ui_label="Pretty")), "Pretty")
        self.assertEqual(label("x", Counter()), "Counter")         # class ui_label

    def test_category_defaults_to_empty_string(self):
        class Anon(ComponentBase):
            ports_spec = {"o": Port(1, "out", "")}
        topo = generate_topology(CPUBuilder().add("anon", Anon()).build())
        self.assertEqual(topo["nodes"][0]["category"], "")

    def test_edge_fields_and_label(self):
        edge = generate_topology(self._chain())["edges"][0]
        self.assertEqual(edge, {
            "from": "counter.out",
            "to": "doubler.inp",
            "label": "counter value",     # the SOURCE port's desc
        })

    def test_edge_label_falls_back_to_the_source_port_name(self):
        class NoDesc(ComponentBase):
            ports_spec = {"o": Port(1, "out", ""), "i": Port(1, "in", "")}
        b = CPUBuilder().add("a", NoDesc()).add("b", NoDesc())
        b.wire("a.o", "b.i")
        self.assertEqual(generate_topology(b.build())["edges"][0]["label"], "o")

    def test_empty_cpu_yields_empty_graph(self):
        self.assertEqual(generate_topology(CPUBuilder().build()),
                         {"nodes": [], "edges": []})

    def test_topology_is_json_serialisable(self):
        json.dumps(generate_topology(self._chain()))

    def test_every_edge_endpoint_resolves_to_a_declared_node_port(self):
        """Guards against the diagram referencing ports the UI can't place."""
        from sim.isa.riscv.presets.single_cycle import build
        topo = generate_topology(build([0x00500093, 0x00300113]))
        ports = {n["id"]: set(n["ports"]) for n in topo["nodes"]}
        self.assertTrue(topo["edges"])
        for e in topo["edges"]:
            for endpoint in (e["from"], e["to"]):
                comp, _, port = endpoint.partition(".")
                self.assertIn(comp, ports, endpoint)
                self.assertIn(port, ports[comp], endpoint)


# ── runner_v2 ───────────────────────────────────────────────────────────────

class TestRunSimulation(unittest.TestCase):

    def _cpu(self, stop=3):
        return CPUBuilder().add("c", ArchLike(stop=stop)).build()

    def test_returns_one_state_per_cycle(self):
        states = run_simulation(self._cpu(), num_cycles=5)
        self.assertEqual(len(states), 5)
        self.assertEqual([s["_cycle"] for s in states], [0, 1, 2, 3, 4])

    def test_include_reset_prepends_the_pre_tick_snapshot(self):
        states = run_simulation(self._cpu(), num_cycles=5, include_reset=True)
        self.assertEqual(len(states), 6)
        self.assertEqual([s["_cycle"] for s in states], [0, 1, 2, 3, 4, 5])
        self.assertEqual(states[0]["c"]["registers"], [0], "cycle 0 = reset state")
        self.assertEqual(states[1]["c"]["registers"], [1])

    def test_zero_cycles(self):
        self.assertEqual(run_simulation(self._cpu(), num_cycles=0), [])
        only = run_simulation(self._cpu(), num_cycles=0, include_reset=True)
        self.assertEqual(len(only), 1)
        self.assertEqual(only[0]["_cycle"], 0)

    def test_states_are_independent_snapshots(self):
        states = run_simulation(self._cpu(), num_cycles=4)
        self.assertEqual([s["c"]["registers"] for s in states],
                         [[1], [2], [3], [3]])

    def test_until_stable_stops_once_state_settles(self):
        states = run_simulation(self._cpu(stop=3), num_cycles=100,
                                until_stable=True, stable_window=4)
        # 3 changing cycles, then `stable_window` further identical ones.
        self.assertEqual(len(states), 3 + 4)
        self.assertEqual(states[-1]["c"]["registers"], [3])

    def test_until_stable_still_honours_num_cycles_as_a_cap(self):
        states = run_simulation(self._cpu(stop=10_000), num_cycles=7,
                                until_stable=True, stable_window=3)
        self.assertEqual(len(states), 7)

    def test_until_stable_off_by_default_runs_the_full_count(self):
        states = run_simulation(self._cpu(stop=1), num_cycles=20)
        self.assertEqual(len(states), 20)

    def test_stable_window_default_is_the_documented_64(self):
        self.assertEqual(STABLE_WINDOW, 64)

    def test_arch_signature_hashes_registers_and_data_memory(self):
        sig = _arch_signature({
            "regfile": {"registers": [1, 2]},
            "dmem": {"memory": [7], "memory_hi": {"5": 9}},
        })
        self.assertEqual(sig, (("dmem", "m", ((0, 7), (5, 9))),
                               ("regfile", "r", (1, 2))))
        hash(sig)  # must be hashable / comparable

    def test_arch_signature_excludes_instruction_memories(self):
        for name in ("imem", "IMEM", "byte_imem"):
            self.assertEqual(_arch_signature({name: {"memory": [1, 2]}}), (),
                             f"{name} must not count as architectural state")

    def test_arch_signature_ignores_non_dict_entries(self):
        self.assertEqual(_arch_signature({"_cycle": 3}), ())

    def test_arch_signature_includes_the_sparse_high_memory_map(self):
        """Hashing only the dense window would miss software-stack churn."""
        a = _arch_signature({"dmem": {"memory": [0], "memory_hi": {}}})
        b = _arch_signature({"dmem": {"memory": [0], "memory_hi": {16382: 1}}})
        self.assertNotEqual(a, b)

    def test_arch_signature_is_stable_across_key_ordering(self):
        one = _arch_signature({"a": {"registers": [1]}, "b": {"registers": [2]}})
        two = _arch_signature({"b": {"registers": [2]}, "a": {"registers": [1]}})
        self.assertEqual(one, two)

    def test_until_stable_notices_high_memory_only_changes(self):
        """A CPU whose ONLY activity is above the dense window must not settle."""
        from sim.components.memory.dmem import STATE_WINDOW

        class HighWriter(ComponentBase):
            ports_spec = {}

            def __init__(self, **kw):
                super().__init__(**kw)
                self._hi = {}
                self._n = 0

            def rising_edge(self):
                self._n += 1
                self._hi[STATE_WINDOW + self._n] = self._n

            def get_state(self):
                return {"memory": [0], "memory_hi": dict(self._hi)}

        cpu = CPUBuilder().add("dmem", HighWriter()).build()
        states = run_simulation(cpu, num_cycles=12, until_stable=True,
                                stable_window=3)
        self.assertEqual(len(states), 12, "never settles: high memory keeps moving")


if __name__ == "__main__":
    unittest.main()
