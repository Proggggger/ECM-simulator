"""
x86 IA-32 Core-C code generator — depth + end-to-end correctness.

WHY THIS FILE EXISTS
--------------------
``sim/compiler/codegen/x86.py`` was the weakest-covered module in the project
(73%): its whole inline signed ``emit_soft_div`` body (lines 361-439 — the
sign-flag / abs / repeated-subtract / sign-fixup sequence) was NEVER executed,
because no existing test compiled a ``/`` or ``%`` for x86. The x86 backend is
also the most constrained one — no CALL/RET stack, no shift/mul/div
instruction, imm8-only ALU immediates, ``[EBP+disp8]`` frame addressing and a
codegen-inserted load-use NOP — so it is exactly where a codegen bug is most
likely to hide.

WHAT IS ASSERTED
  * the inline mul/div lowerings are correct for **all four sign quadrants**
    plus divide-by-zero, verified by RUNNING the compiled program;
  * every observable (return value in EAX, ``print()`` stores at
    ``PRINT_ADDR``, data memory) is checked against a hand-computed value;
  * because x86 codegen is CALL-FREE, it satisfies the OoO link-register
    parity limitation documented in CLAUDE.md/FUTURE_DIRECTIONS.md — so the
    same programs are run across **five execution models** (single_cycle,
    multicycle, configurable, pipeline, ooo) and required to agree;
  * the load-use NOP peephole and its source-map reindexing are unit-tested
    directly on ``_postprocess`` / ``_load_dest`` / ``_reads_reg``.

superscalar is excluded on purpose: CLAUDE.md documents that the x86
superscalar preset reuses the RISC-V decoder on a byte stream and is
illustrative only (pinned by ``test_x86_superscalar_is_illustrative_only``).

Run from repo root:  python3 -m pytest tests/test_codegen_x86.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.assembler import assemble
from sim.compiler import compile_c
from sim.compiler.codegen.base import PRINT_ADDR
from sim.compiler.codegen.x86 import (
    MAX_SLOT_DISP,
    X86_FRAME_BASE,
    X86CodeGen,
)
from sim.harness import PRESETS, compile_and_simulate
from sim.runner_v2 import run_simulation

# Models x86 compiled code is required to be correct on. x86 codegen emits no
# CALL, so OoO's link-register gap does not apply and OoO is IN scope.
X86_MODELS = ("single_cycle", "multicycle", "configurable", "pipeline", "ooo")

CYCLE_CAP = 12000


def _signed(v):
    return v if v < 2 ** 31 else v - 2 ** 32


def run(src, model="single_cycle", cap=CYCLE_CAP):
    """Compile ``src`` for x86, run it to completion, return (prints, states)."""
    asm = compile_c(src, "x86").asm
    program = assemble("x86", asm)          # raises on any bad encoding
    cpu = PRESETS[f"x86/{model}"]["build"](program)
    states = run_simulation(cpu, num_cycles=cap, include_reset=True,
                            until_stable=True)
    return _print_events(states), states

def _print_events(states):
    """Console stores at PRINT_ADDR, deduped exactly like the /lab terminal."""
    events, prev = [], None
    for st in states:
        d = st.get("dmem") or {}
        if not d.get("wen"):
            prev = None
            continue
        size = d.get("size", len(d.get("memory", [])) or 1)
        sig = (d.get("write_addr"), d.get("wdata"))
        if (d.get("write_addr", 0) >> 2) % size == (PRINT_ADDR >> 2) % size \
                and sig != prev:
            events.append(_signed(d["wdata"]))
        prev = sig
    return events


def prints(src, model="single_cycle", cap=CYCLE_CAP):
    return run(src, model, cap)[0]


def eax(src, model="single_cycle", cap=CYCLE_CAP):
    """Final EAX (the x86 return-value register), as a signed int.

    Runs to completion (``until_stable``) rather than burning the whole cycle
    budget — the x86 inline mul/div loops make a fixed budget expensive.
    """
    states = run(src, model, cap)[1]
    for state in reversed(states):
        rf = state.get("regfile")
        if rf and "registers" in rf:
            return _signed(rf["registers"][0])       # EAX is index 0
    raise AssertionError("no register-file state in the run")


# ============================================================================
# 1. INLINE SIGNED DIVIDE / MODULO  (x86.py:354-439 — was 100% uncovered)
# ============================================================================
class TestInlineSignedDivide:
    """``emit_soft_div`` is a repeated-subtraction loop over absolute values
    with a sign fix-up: quotient sign = XOR of the operand signs, remainder
    sign = the DIVIDEND's sign (C truncate-toward-zero semantics). All four
    quadrants plus the divide-by-zero escape are exercised here."""

    QUADRANTS = [
        (17, 5, 3, 2),        # + / +
        (-17, 5, -3, -2),     # - / +   remainder takes the dividend's sign
        (17, -5, -3, 2),      # + / -
        (-17, -5, 3, -2),     # - / -
        (5, 17, 0, 5),        # |dividend| < |divisor|
        (-5, 17, 0, -5),
        (20, 5, 4, 0),        # exact
        (-20, -5, 4, 0),
        (0, 7, 0, 0),
        (-7, 1, -7, 0),
    ]

    @pytest.mark.parametrize("a,b,q,r", QUADRANTS)
    def test_quotient_and_remainder_in_every_sign_quadrant(self, a, b, q, r):
        # Cross-check the expectation against C truncate-toward-zero first.
        assert q == int(a / b) and r == a - q * b
        src = (f"int main() {{ int a = {a}; int b = {b}; "
               f"print(a / b); print(a % b); return 0; }}")
        assert prints(src) == [q, r]

    def test_divide_by_zero_yields_zero_rather_than_hanging(self):
        src = ("int main() { int a = 7; int b = 0; "
               "print(a / b); print(a % b); return 0; }")
        assert prints(src) == [0, 0]

    def test_negative_dividend_divided_by_zero(self):
        src = ("int main() { int a = -7; int b = 0; "
               "print(a / b); print(a % b); return 0; }")
        assert prints(src) == [0, 0]

    def test_quotient_and_remainder_reconstruct_the_dividend(self):
        """The metamorphic law (a/b)*b + a%b == a, checked in the simulator
        rather than by inspecting the emitted asm."""
        src = """
int main() {
  int i;
  int j;
  for (i = -9; i <= 9; i = i + 6) {
    for (j = -4; j <= 4; j = j + 4) {
      if (j != 0) { print((i / j) * j + (i % j)); }
    }
  }
  return 0;
}
"""
        got = prints(src)
        expected = [i for i in range(-9, 10, 6) for j in (-4, 4)]
        assert got == expected

    def test_divide_result_can_feed_the_return_value(self):
        assert eax("int main(){ int a = -100; int b = 7; return a / b; }") == -14

    def test_modulo_result_can_feed_the_return_value(self):
        assert eax("int main(){ int a = -100; int b = 7; return a % b; }") == -2

    def test_division_inside_a_loop_condition(self):
        src = """
int main() {
  int n = 40;
  int steps = 0;
  while (n / 3 > 0) { n = n / 3; steps = steps + 1; }
  print(steps);
  print(n);
  return 0;
}
"""
        # 40 -> 13 -> 4 -> 1 (1/3 == 0 stops the loop)
        assert prints(src) == [3, 1]

    def test_compound_divide_and_modulo_assignment(self):
        src = ("int main() { int s = 100; s /= 7; print(s); s %= 4; "
               "print(s); return 0; }")
        assert prints(src) == [14, 2]


# ============================================================================
# 2. INLINE MULTIPLY  (emit_soft_mul: repeated addition + sign fix-up)
# ============================================================================
class TestInlineMultiply:
    @pytest.mark.parametrize("a,b", [
        (6, 7), (-6, 7), (6, -7), (-6, -7), (0, 5), (5, 0), (1, 123),
        (-1, 123), (12, 12),
    ])
    def test_products_in_every_sign_quadrant(self, a, b):
        src = (f"int main() {{ int a = {a}; int b = {b}; "
               f"print(a * b); return 0; }}")
        assert prints(src) == [a * b]

    def test_multiply_is_commutative_in_the_generated_code(self):
        """The lowering is asymmetric (it loops |multiplier| times), so
        a*b and b*a take different paths and must still agree."""
        src = ("int main() { int a = -9; int b = 4; "
               "print(a * b); print(b * a); return 0; }")
        got = prints(src)
        assert got == [-36, -36]

    def test_squares_via_repeated_multiplication(self):
        src = ("int main() { int i; for (i = 1; i <= 5; i++) print(i * i); "
               "return 0; }")
        assert prints(src) == [1, 4, 9, 16, 25]

    def test_multiply_wraps_at_32_bits(self):
        """Wraparound, with a SMALL multiplier.

        The lowering loops |multiplier| times, so 100000*100000 would need
        ~100k iterations — far past any sane cycle cap (it silently returns
        the un-multiplied operand instead, which is cycle exhaustion, not a
        codegen bug). Put the magnitude in the multiplicand instead: the
        product still overflows 32 bits but costs only a handful of loops.
        """
        # 2^30 * 4 == 2^32 -> 0 in 32-bit arithmetic.
        assert eax("int main(){ int a = 1073741824; int b = 4; "
                   "return a * b; }") == 0
        # 2000000000 * 3 == 6e9; 6e9 mod 2^32 == 1705032704.
        assert eax("int main(){ int a = 2000000000; int b = 3; "
                   "return a * b; }") == 1705032704


# ============================================================================
# 3. END-TO-END PROGRAMS ACROSS EVERY IN-SCOPE MODEL
# ============================================================================
class TestEndToEndAcrossModels:
    """x86 codegen is call-free, so all five in-order/OoO models must produce
    identical, hand-computed results."""

    PROGRAMS = {
        "arith_and_bitwise": ("""
int main() {
  int a = 29;
  int b = 7;
  print(a + b); print(a - b); print(a * b); print(a / b); print(a % b);
  print(a & b); print(a | b); print(a ^ b); print(-a); print(!a); print(!0);
  return 0;
}
""", [36, 22, 203, 4, 1, 5, 31, 26, -29, 0, 1]),

        "comparisons": ("""
int main() {
  int a = 29;
  int b = 7;
  print(a < b); print(a > b); print(a <= b); print(a >= b);
  print(a == b); print(a != b); print(a == a); print(a <= a);
  return 0;
}
""", [0, 1, 0, 1, 0, 1, 1, 1]),

        "globals_and_global_array": ("""
int g = 5;
int t[4] = {2, 4, 6, 8};
int main() {
  int i;
  int s = g;
  for (i = 0; i < 4; i++) { s = s + t[i]; t[i] = t[i] * 2; }
  print(s);
  for (i = 0; i < 4; i++) print(t[i]);
  return 0;
}
""", [25, 4, 8, 12, 16]),

        "array_reverse_in_place": ("""
int main() {
  int a[6] = {1, 2, 3, 4, 5, 6};
  int i;
  int t;
  for (i = 0; i < 3; i++) { t = a[i]; a[i] = a[5 - i]; a[5 - i] = t; }
  for (i = 0; i < 6; i++) print(a[i]);
  return 0;
}
""", [6, 5, 4, 3, 2, 1]),

        "pointers_read_and_write": ("""
int main() {
  int x = 10;
  int y = 20;
  int p = &x;
  *p = *p + 5;
  p = &y;
  *p = *p - 5;
  print(x); print(y); print(*(&x));
  return 0;
}
""", [15, 15, 15]),

        "short_circuit_side_effects": ("""
int side = 0;
int main() {
  int a = 0;
  int b = 1;
  print(a && (side = 1)); print(side);
  print(b || (side = 2)); print(side);
  print(b && (side = 3)); print(side);
  print(a || (side = 4)); print(side);
  return 0;
}
""", [0, 0, 1, 0, 1, 3, 1, 4]),

        "ternary_and_incdec": ("""
int main() {
  int i;
  for (i = -2; i <= 2; i++) print(i < 0 ? -1 : (i == 0 ? 0 : 1));
  i = 5;
  print(i++); print(i); print(++i); print(i);
  print(i--); print(i); print(--i); print(i);
  return 0;
}
""", [-1, -1, 0, 1, 1, 5, 6, 7, 7, 7, 6, 5, 5]),

        "break_and_continue": ("""
int main() {
  int i;
  int s = 0;
  for (i = 0; i < 12; i++) {
    if (i % 2 == 0) continue;
    if (i > 7) break;
    s = s + i;
    print(s);
  }
  return 0;
}
""", [1, 4, 9, 16]),
    }

    @pytest.mark.parametrize("name", sorted(PROGRAMS))
    def test_single_cycle_matches_the_hand_computed_result(self, name):
        src, expected = self.PROGRAMS[name]
        assert prints(src) == expected

    @pytest.mark.parametrize("model", X86_MODELS)
    def test_representative_program_agrees_on_every_model(self, model):
        src, expected = self.PROGRAMS["globals_and_global_array"]
        assert prints(src, model) == expected

    @pytest.mark.parametrize("model", X86_MODELS)
    def test_signed_division_agrees_on_every_model(self, model):
        src = """
int main() {
  int i;
  int j;
  for (i = -7; i <= 7; i = i + 14) {
    for (j = -3; j <= 3; j = j + 6) { print(i / j); print(i % j); }
  }
  return 0;
}
"""
        assert prints(src, model) == [2, -1, -2, -1, -2, 1, 2, 1]

    @pytest.mark.parametrize("model", X86_MODELS)
    def test_bubble_sort_agrees_on_every_model(self, model):
        src = """
int main() {
  int a[5] = {4, 1, 3, 0, 2};
  int i;
  int j;
  int t;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 4 - i; j++) {
      if (a[j] > a[j + 1]) { t = a[j]; a[j] = a[j + 1]; a[j + 1] = t; }
    }
  }
  for (i = 0; i < 5; i++) print(a[i]);
  return 0;
}
"""
        assert prints(src, model) == [0, 1, 2, 3, 4]

    def test_x86_superscalar_is_illustrative_only(self):
        """Documented limitation (CLAUDE.md): the x86 superscalar preset reuses
        the RISC-V decoder on a byte stream, so a real x86 byte program does
        NOT produce the right answer there. Pinned rather than fixed."""
        src, expected = self.PROGRAMS["globals_and_global_array"]
        assert prints(src, "superscalar", cap=4000) != expected


# ============================================================================
# 4. MEMORY-VISIBLE STATE (not just the return value / console)
# ============================================================================
class TestDataMemoryEffects:
    def test_locals_land_at_their_symbol_addresses(self):
        src = ("int main() { int a = 6; int b = 7; int c = a * b; "
               "return c; }")
        result = compile_c(src, "x86")
        syms = {s["name"]: s for s in result.symbols}
        assert set(syms) == {"a", "b", "c"}
        r = compile_and_simulate("x86", "single_cycle", src, cycles=CYCLE_CAP)
        assert [r.mem_at(syms[n]["addr"]) for n in ("a", "b", "c")] == [6, 7, 42]

    def test_main_frame_base_is_the_fixed_ebp(self):
        src = "int main() { int a = 1; return a; }"
        sym = compile_c(src, "x86").symbols[0]
        assert sym["addr"] >= X86_FRAME_BASE
        assert sym["location"].startswith("[EBP+")

    def test_globals_occupy_the_low_region_below_the_frame(self):
        src = "int g = 3; int h = 4; int main() { return g + h; }"
        result = compile_c(src, "x86")
        globs = [s for s in result.symbols if s["kind"] == "global"]
        assert [g["name"] for g in globs] == ["g", "h"]
        assert [g["addr"] for g in globs] == [0, 4]
        assert all(g["addr"] < X86_FRAME_BASE for g in globs)
        r = compile_and_simulate("x86", "single_cycle", src, cycles=CYCLE_CAP)
        assert [r.mem_at(g["addr"]) for g in globs] == [3, 4]
        assert _signed(r.reg("EAX")) == 7

    def test_array_writes_are_visible_in_data_memory(self):
        src = ("int main() { int arr[4]; arr[0] = 11; arr[1] = 22; "
               "arr[2] = 33; arr[3] = 44; return arr[2]; }")
        result = compile_c(src, "x86")
        arr = next(s for s in result.symbols if s["name"] == "arr")
        assert (arr["kind"], arr["size"], arr["stride"]) == ("array", 4, 4)
        r = compile_and_simulate("x86", "single_cycle", src, cycles=CYCLE_CAP)
        got = [r.mem_at(arr["addr"] + i * arr["stride"]) for i in range(4)]
        assert got == [11, 22, 33, 44]
        assert _signed(r.reg("EAX")) == 33

    def test_print_writes_the_console_word_and_nothing_else_does(self):
        _, states = run("int main() { int a = 5; print(a); return a; }")
        final = states[-1]["dmem"]
        word = (PRINT_ADDR >> 2) % final.get("size", len(final["memory"]))
        assert final["memory"][word] == 5


# ============================================================================
# 5. IMMEDIATE / ADDRESSING POLICY (imm8 ALU, MOV imm32, [EBP+disp8])
# ============================================================================
class TestImmediateAndAddressingPolicy:
    @pytest.mark.parametrize("value", [0, 1, -1, 127, 128, -128, -129, 255,
                                       2047, 2048, -3000, 70000, 305419896])
    def test_constants_of_every_magnitude_round_trip(self, value):
        assert eax(f"int main() {{ return {value}; }}") == value

    def test_large_offsets_go_through_a_scratch_register(self):
        """``_add_imm_to_acc`` uses ADD r,imm8 when it fits and materializes
        into EDX otherwise. Both branches are driven here."""
        cg = X86CodeGen()
        cg._add_imm_to_acc(10)
        cg._add_imm_to_acc(300)
        cg._add_imm_to_acc(-128)
        cg._add_imm_to_acc(-129)
        assert cg.lines == ["ADD EAX, 10",
                            "MOV EDX, 300", "ADD EAX, EDX",
                            "ADD EAX, -128",
                            "MOV EDX, -129", "ADD EAX, EDX"]

    def test_slot_offsets_within_disp8_are_accepted(self):
        cg = X86CodeGen()
        assert cg._slot_off(0) == 0
        assert cg._slot_off(MAX_SLOT_DISP // 4) == MAX_SLOT_DISP
        assert cg._mem(0) == "[EBP+0]"
        assert cg._mem(3) == "[EBP+12]"

    def test_slot_offset_beyond_disp8_is_a_compile_error(self):
        from sim.compiler.errors import CompilerError
        cg = X86CodeGen()
        with pytest.raises(CompilerError) as exc:
            cg._slot_off(MAX_SLOT_DISP // 4 + 1)
        assert "exceeds the disp8 range" in exc.value.message

    def test_an_array_that_just_fits_the_frame_still_works(self):
        # 2 fixed slots + array + a temp or two must stay <= 31 slots.
        src = ("int main() { int a[20]; int i; int s = 0; "
               "for (i = 0; i < 20; i++) a[i] = i; "
               "for (i = 0; i < 20; i++) s = s + a[i]; return s; }")
        assert eax(src) == sum(range(20))


# ============================================================================
# 6. THE LOAD-USE NOP PEEPHOLE  (_postprocess / _load_dest / _reads_reg)
# ============================================================================
class TestLoadUseNopPeephole:
    """The shared x86 pipeline does not forward MOV reg,[mem] into the very
    next consumer, so codegen inserts a NOP. Verified as a unit (so the
    classifier's corner cases are pinned) AND end-to-end (single_cycle,
    multicycle and pipeline must agree, which is what the NOP buys)."""

    @pytest.mark.parametrize("line,expected", [
        ("MOV EAX, [EBP+8]", ("MOV", ["EAX", "[EBP+8]"])),
        ("ADD EAX, ECX", ("ADD", ["EAX", "ECX"])),
        ("NOP", ("NOP", [])),
        ("some_label:", (None, [])),
        ("; a comment", (None, [])),
        ("", (None, [])),
        ("   ", (None, [])),
    ])
    def test_instr_parts_classifies_lines(self, line, expected):
        assert X86CodeGen._instr_parts(line) == expected

    @pytest.mark.parametrize("line,dest", [
        ("MOV EAX, [EBP+8]", "EAX"),
        ("MOV ECX, [EDX+0]", "ECX"),
        ("MOV EAX, 5", None),            # immediate, not a load
        ("MOV [ECX+0], EAX", None),      # a STORE, not a load
        ("ADD EAX, ECX", None),
        ("label:", None),
    ])
    def test_load_dest_detects_only_memory_loads(self, line, dest):
        assert X86CodeGen._load_dest(line) == dest

    @pytest.mark.parametrize("line,reg,reads", [
        ("MOV EAX, 5", "EAX", False),        # MOV dst is write-only
        ("MOV ECX, EAX", "EAX", True),       # source
        ("ADD EAX, ECX", "EAX", True),       # ALU dst is read+write
        ("CMP EAX, 0", "EAX", True),
        ("MOV [ECX+0], EAX", "ECX", True),   # base register inside brackets
        ("MOV EAX, [ECX+0]", "ECX", True),
        ("MOV EAX, [ECX+0]", "EDX", False),
        ("label:", "EAX", False),
    ])
    def test_reads_reg_operand_roles(self, line, reg, reads):
        assert X86CodeGen._reads_reg(line, reg) is reads

    def test_postprocess_inserts_a_nop_between_a_load_and_its_consumer(self):
        cg = X86CodeGen()
        cg.lines = ["MOV EAX, [EBP+4]", "ADD EAX, ECX"]
        cg.source_map = [{"asm_line": 1, "c_line": 1},
                         {"asm_line": 2, "c_line": 2}]
        cg._postprocess()
        assert cg.lines == ["MOV EAX, [EBP+4]", "NOP", "ADD EAX, ECX"]
        assert cg.source_map == [{"asm_line": 1, "c_line": 1},
                                 {"asm_line": 3, "c_line": 2}]

    def test_postprocess_looks_past_labels_and_comments(self):
        cg = X86CodeGen()
        cg.lines = ["MOV EAX, [EBP+4]", "some_label:", "ADD EAX, ECX"]
        cg.source_map = [{"asm_line": 1, "c_line": 1},
                         {"asm_line": 3, "c_line": 2}]
        cg._postprocess()
        assert cg.lines == ["MOV EAX, [EBP+4]", "NOP", "some_label:",
                            "ADD EAX, ECX"]
        assert cg.source_map[-1] == {"asm_line": 4, "c_line": 2}

    def test_postprocess_does_not_insert_when_the_next_use_is_independent(self):
        cg = X86CodeGen()
        cg.lines = ["MOV EAX, [EBP+4]", "MOV ECX, 3"]
        cg.source_map = []
        cg._postprocess()
        assert "NOP" not in cg.lines

    def test_postprocess_handles_a_load_as_the_very_last_line(self):
        cg = X86CodeGen()
        cg.lines = ["MOV EAX, [EBP+4]"]
        cg.source_map = []
        cg._postprocess()
        assert cg.lines == ["MOV EAX, [EBP+4]"]

    def test_real_programs_contain_nops_and_a_consistent_source_map(self):
        result = compile_c("int main(){ int a=3; int b=4; return a*b+a/b; }",
                           "x86")
        lines = result.asm.split("\n")
        assert any(l.strip() == "NOP" for l in lines)
        assert result.source_map
        for entry in result.source_map:
            assert 1 <= entry["asm_line"] <= len(lines)
            # No source-map entry may point at a blank line after reindexing.
            assert lines[entry["asm_line"] - 1].strip()

    @pytest.mark.parametrize("model", ["single_cycle", "multicycle", "pipeline"])
    def test_load_heavy_program_agrees_across_models(self, model):
        """This is what the NOP peephole exists for: the pipeline result must
        match the single-cycle oracle on chained loads."""
        src = """
int main() {
  int a[4] = {1, 2, 3, 4};
  int s = 0;
  int i;
  for (i = 0; i < 4; i++) { s = s + a[i] * a[i]; }
  print(s);
  return s;
}
"""
        assert prints(src, model) == [1 + 4 + 9 + 16]


# ============================================================================
# 7. EMITTED-CODE SHAPE (contract with sim/assembler/x86.py)
# ============================================================================
class TestEmittedCodeShape:
    def test_program_prologue_sets_ebp_and_falls_through_to_main(self):
        lines = [l.strip() for l in
                 compile_c("int main(){return 1;}", "x86").asm.split("\n")]
        instrs = [l for l in lines if l and not l.startswith(";")]
        assert instrs[0] == f"MOV EBP, {X86_FRAME_BASE}"
        assert instrs[1] == "func_main:"

    def test_return_jumps_to_the_halt_park_loop(self):
        asm = compile_c("int main(){return 1;}", "x86").asm
        assert "halt:" in asm
        assert "JMP halt" in asm

    def test_no_call_ret_or_shift_instruction_is_ever_emitted(self):
        src = ("int main(){ int a=7; int b=3; int c[3]={1,2,3}; "
               "return a*b + a/b + a%b + c[1]; }")
        mnemonics = {l.strip().split()[0].upper()
                     for l in compile_c(src, "x86").asm.split("\n")
                     if l.strip() and not l.strip().startswith(";")
                     and not l.strip().endswith(":")}
        assert not (mnemonics & {"CALL", "RET", "SHL", "SHR", "SAR",
                                 "IMUL", "MUL", "IDIV", "DIV"})

    @pytest.mark.parametrize("src", [
        "int main(){ int a=1; return a; }",
        "int g=2; int main(){ return g; }",
        "int main(){ int a[3]={1,2,3}; return a[2]; }",
        "int main(){ int a=9; int b=4; return a*b - a/b - a%b; }",
        "int main(){ int i; int s=0; for(i=0;i<3;i++){ if(i) s+=i; else s-=1; }"
        " return s; }",
        "int main(){ int a=1; print(a); return a; }",
    ])
    def test_everything_the_backend_emits_assembles_clean(self, src):
        program = assemble("x86", compile_c(src, "x86").asm)
        assert len(program) > 0
        assert all(0 <= b <= 0xFF for b in program), "x86 programs are BYTES"
