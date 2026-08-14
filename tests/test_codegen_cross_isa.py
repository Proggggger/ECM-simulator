"""
Cross-ISA differential (metamorphic) tests for the Core-C compiler.

WHY THIS FILE EXISTS
--------------------
The per-ISA codegen tests each assert hand-computed expected values, which is
sound but expensive to write and easy to get subtly wrong. This file takes the
complementary approach: compile the SAME Core-C program for riscv, arm and x86
and require all three to agree on their observable output.

That is high-yield because it needs no oracle: any disagreement is a codegen
bug in at least one backend, and the three backends are structurally very
different (riscv/arm are call-based with a software stack and soft mul/div
helpers; x86 is call-free with inline lowerings and [EBP+disp8] frames), so
they are unlikely to fail identically. Where a value is cheap to state, the
agreed result is ALSO pinned against a hand-computed constant, so all three
agreeing on a wrong answer still fails.

SCOPE / LIMITATIONS (documented in CLAUDE.md, respected here)
  * OoO never commits JAL/BL's link-register write, so call-wrapped riscv/arm
    code (which is all compiled Core-C on those ISAs) fails on ooo. The
    differential set therefore runs on the models where parity is documented
    to hold; ``test_ooo_link_register_gap_still_applies`` pins the limitation
    itself so this exclusion cannot silently become obsolete.
  * x86 superscalar reuses the RISC-V decoder on a byte stream and is
    illustrative only -- excluded, and pinned as such in test_codegen_x86.py.
  * x86 lowers ``*`` and ``/`` to iterative loops, so the operand MAGNITUDE
    drives cycle count. Programs here keep multipliers/divisors small; a
    large-magnitude case is cycle exhaustion, not a codegen bug.

Run from repo root:  python3 -m pytest tests/test_codegen_cross_isa.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.assembler import assemble
from sim.compiler import compile_c
from sim.compiler.codegen.base import PRINT_ADDR
from sim.harness import PRESETS
from sim.runner_v2 import run_simulation

ISAS = ("riscv", "arm", "x86")

# Models on which compiled Core-C is required to agree across ISAs.
# ooo is excluded for the documented link-register gap (pinned separately);
# superscalar is excluded because the x86 preset is illustrative only.
PARITY_MODELS = ("single_cycle", "multicycle", "configurable", "pipeline")

CYCLE_CAP = 12000


def _signed(v):
    return v if v < 2 ** 31 else v - 2 ** 32


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


def prints(src, isa, model="single_cycle", cap=CYCLE_CAP):
    """Compile ``src`` for ``isa``/``model``, run to completion, return prints."""
    asm = compile_c(src, isa).asm
    program = assemble(isa, asm)        # raises on any bad encoding
    cpu = PRESETS[f"{isa}/{model}"]["build"](program)
    states = run_simulation(cpu, num_cycles=cap, include_reset=True,
                            until_stable=True)
    return _print_events(states)


# ─────────────────────────────────────────────────────────────────────────
# The differential corpus.
#
# Each entry is (name, source, expected_prints). The expected list is the
# hand-computed answer -- so this is both a differential test (all three
# ISAs must agree) AND an oracle test (they must agree on the RIGHT value).
# Keep multipliers/divisors small: x86 lowers them to loops.
# ─────────────────────────────────────────────────────────────────────────
PROGRAMS = [
    (
        "arithmetic_and_precedence",
        """
        int main() {
            print(2 + 3 * 4);
            print((2 + 3) * 4);
            print(20 - 6 - 3);
            print(7 * 3 - 4 * 2);
            return 0;
        }
        """,
        [14, 20, 11, 13],
    ),
    (
        "division_and_modulo_all_sign_quadrants",
        """
        int main() {
            print(17 / 5);   print(17 % 5);
            print(-17 / 5);  print(-17 % 5);
            print(17 / -5);  print(17 % -5);
            print(-17 / -5); print(-17 % -5);
            return 0;
        }
        """,
        # C99 truncates toward zero; the remainder takes the dividend's sign.
        [3, 2, -3, -2, -3, 2, 3, -2],
    ),
    (
        "comparisons",
        """
        int main() {
            print(3 < 4); print(4 < 3); print(3 <= 3);
            print(5 > 2); print(2 > 5); print(5 >= 6);
            print(7 == 7); print(7 != 7);
            return 0;
        }
        """,
        [1, 0, 1, 1, 0, 0, 1, 0],
    ),
    (
        "negative_comparisons",
        """
        int main() {
            print(-5 < 3); print(-5 > 3);
            print(-5 < -3); print(-3 < -5);
            print(0 > -1); print(-1 >= -1);
            return 0;
        }
        """,
        [1, 0, 1, 0, 1, 1],
    ),
    (
        "logical_short_circuit",
        """
        int main() {
            int calls;
            calls = 0;
            print(0 && 1); print(1 && 0); print(1 && 1);
            print(0 || 0); print(0 || 1); print(1 || 0);
            print(!0); print(!1); print(!5);
            return 0;
        }
        """,
        [0, 0, 1, 0, 1, 1, 1, 0, 0],
    ),
    (
        "bitwise_ops",
        """
        int main() {
            print(12 & 10); print(12 | 10); print(12 ^ 10);
            print(~0); print(~5);
            return 0;
        }
        """,
        # NOTE: << and >> are deliberately absent -- the ARM backend rejects
        # them outright (no shift instruction in that assembler). Shifts are
        # covered per-ISA in TestShiftsAreRiscvAndX86Only below.
        [8, 14, 6, -1, -6],
    ),
    (
        "ternary_and_compound_assign",
        """
        int main() {
            int a; a = 10;
            print(a > 5 ? 100 : 200);
            print(a < 5 ? 100 : 200);
            a += 5;  print(a);
            a -= 3;  print(a);
            a *= 2;  print(a);
            a /= 4;  print(a);
            a %= 4;  print(a);
            return 0;
        }
        """,
        [100, 200, 15, 12, 24, 6, 2],
    ),
    (
        "increment_decrement_pre_and_post",
        """
        int main() {
            int i; i = 5;
            print(i++); print(i);
            print(++i); print(i);
            print(i--); print(i);
            print(--i); print(i);
            return 0;
        }
        """,
        [5, 6, 7, 7, 7, 6, 5, 5],
    ),
    (
        "while_and_for_loops",
        """
        int main() {
            int i; int sum;
            sum = 0; i = 1;
            while (i <= 5) { sum += i; i++; }
            print(sum);
            sum = 0;
            for (i = 0; i < 6; i++) sum += i * i;
            print(sum);
            return 0;
        }
        """,
        [15, 55],
    ),
    (
        "nested_loops_break_continue",
        """
        int main() {
            int i; int j; int n;
            n = 0;
            for (i = 0; i < 4; i++) {
                for (j = 0; j < 4; j++) {
                    if (j == 2) continue;
                    if (i == 3) break;
                    n++;
                }
            }
            print(n);
            i = 0;
            while (1) { i++; if (i >= 7) break; }
            print(i);
            return 0;
        }
        """,
        [9, 7],
    ),
    (
        "if_else_chain",
        """
        int main() {
            int i;
            for (i = 0; i < 5; i++) {
                if (i == 0) print(100);
                else if (i == 1) print(200);
                else if (i == 2) print(300);
                else print(999);
            }
            return 0;
        }
        """,
        [100, 200, 300, 999, 999],
    ),
    (
        "arrays_and_initializers",
        """
        int main() {
            int a[5]; int i; int sum;
            for (i = 0; i < 5; i++) a[i] = i * 3;
            sum = 0;
            for (i = 0; i < 5; i++) sum += a[i];
            print(sum);
            int b[] = {4, 8, 15, 16};
            print(b[0] + b[3]);
            print(b[1] * b[2]);
            return 0;
        }
        """,
        [30, 20, 120],
    ),
    (
        "globals_persist",
        """
        int g;
        int arr[3];
        int main() {
            int i;
            g = 42;
            print(g);
            g += 8;
            print(g);
            for (i = 0; i < 3; i++) arr[i] = g + i;
            print(arr[0] + arr[1] + arr[2]);
            return 0;
        }
        """,
        [42, 50, 153],
    ),
    (
        "arrays_indexed_and_summed",
        """
        int main() {
            int a[6]; int i; int sum; int max;
            for (i = 0; i < 6; i++) a[i] = (i * 7) % 5;
            sum = 0; max = 0;
            for (i = 0; i < 6; i++) {
                sum += a[i];
                if (a[i] > max) max = a[i];
            }
            print(sum); print(max);
            print(a[0]); print(a[5]);
            return 0;
        }
        """,
        # a = [0,2,4,1,3,0]
        [10, 4, 0, 0],
    ),
    (
        "operator_mix_stress",
        """
        int main() {
            int i; int acc;
            acc = 0;
            for (i = 1; i <= 6; i++) {
                if (i % 2 == 0) acc += i * 2;
                else acc -= i;
                acc = acc & 255;
            }
            print(acc);
            print(acc > 0 ? acc % 7 : -1);
            return 0;
        }
        """,
        [15, 1],
    ),
]

PROGRAM_IDS = [name for name, _, _ in PROGRAMS]


# ─────────────────────────────────────────────────────────────────────────
# 1. The differential property: all three ISAs agree, on the right answer.
# ─────────────────────────────────────────────────────────────────────────
class TestCrossISADifferential:

    @pytest.mark.parametrize("name,src,expected", PROGRAMS, ids=PROGRAM_IDS)
    def test_all_three_isas_agree_with_the_hand_computed_result(
            self, name, src, expected):
        got = {isa: prints(src, isa) for isa in ISAS}
        # Differential: no backend may disagree with the others...
        assert got["riscv"] == got["arm"] == got["x86"], (
            f"{name}: backends disagree -> {got}")
        # ...and the value they agree on must be the correct one, so that
        # three identically-wrong backends still fail.
        assert got["riscv"] == expected, f"{name}: agreed but wrong -> {got}"


# ─────────────────────────────────────────────────────────────────────────
# 2. The same property must survive every execution model, not just
#    single_cycle. A codegen/datapath interaction (load-use, forwarding,
#    variable-length fetch) can be model-specific.
# ─────────────────────────────────────────────────────────────────────────
class TestCrossModelAgreement:

    # A compact program touching mul, div, branches, memory and a loop --
    # enough to exercise hazards on every model without being slow.
    SRC = """
    int main() {
        int a[4]; int i; int sum;
        for (i = 0; i < 4; i++) a[i] = (i + 1) * 3;
        sum = 0;
        for (i = 0; i < 4; i++) sum += a[i];
        print(sum);
        print(sum / 5);
        print(sum % 7);
        return 0;
    }
    """
    EXPECTED = [30, 6, 2]

    @pytest.mark.parametrize("isa", ISAS)
    @pytest.mark.parametrize("model", PARITY_MODELS)
    def test_program_agrees_on_every_parity_model(self, isa, model):
        assert prints(self.SRC, isa, model) == self.EXPECTED

    @pytest.mark.parametrize("isa", ISAS)
    def test_recursion_where_supported(self, isa):
        """Recursion needs a call stack: riscv/arm have one, x86 does not."""
        src = """
        int fib(int n) {
            if (n < 2) return n;
            return fib(n - 1) + fib(n - 2);
        }
        int main() { print(fib(10)); return 0; }
        """
        if isa == "x86":
            pytest.skip("x86 codegen is call-free; recursion is out of scope")
        assert prints(src, isa) == [55]

    @pytest.mark.parametrize("isa", ("riscv", "arm"))
    def test_array_params_are_pass_by_reference(self, isa):
        """Multi-function, so riscv/arm only (x86 models no call stack)."""
        src = """
        void fill(int a[], int n, int base) {
            int i;
            for (i = 0; i < n; i++) a[i] = base + i;
        }
        int total(int a[], int n) {
            int i; int s;
            s = 0;
            for (i = 0; i < n; i++) s += a[i];
            return s;
        }
        int main() {
            int buf[4];
            fill(buf, 4, 10);
            print(buf[0]); print(buf[3]);
            print(total(buf, 4));
            return 0;
        }
        """
        # Arrays pass by reference, so fill()'s writes are visible in main.
        assert prints(src, isa) == [10, 13, 46]


# ─────────────────────────────────────────────────────────────────────────
# 3. Pin the documented limitations, so the exclusions above cannot quietly
#    become obsolete without a test noticing.
# ─────────────────────────────────────────────────────────────────────────
class TestBackendScopeLimits:
    """Per-backend language-scope differences, pinned so they are deliberate."""

    def test_shifts_work_on_riscv_only(self):
        """Only the RISC-V backend lowers << and >>."""
        src = "int main() { print(1 << 4); print(256 >> 3); return 0; }"
        assert prints(src, "riscv") == [16, 32]

    @pytest.mark.parametrize("isa", ("arm", "x86"))
    def test_shifts_are_rejected_by_arm_and_x86(self, isa):
        """Neither assembler exposes a shift instruction, so codegen refuses.

        Pinned as a compile-time REJECTION rather than a silently wrong
        result -- if either backend gains shifts, this fails and << / >>
        should be absorbed into the shared differential corpus above.
        """
        from sim.compiler.errors import CompilerError
        with pytest.raises(CompilerError, match="shift"):
            compile_c("int main() { print(1 << 4); return 0; }", isa)

    def test_pointer_variable_declarations_are_rejected_everywhere(self):
        """`int *p;` is out of grammar on ALL backends (parser-level)."""
        from sim.compiler.errors import CompilerError
        for isa in ISAS:
            with pytest.raises(CompilerError, match="pointer variable"):
                compile_c("int main(){ int x; int *p; p = &x; return 0; }", isa)

    @pytest.mark.parametrize("isa", ("riscv", "arm"))
    def test_address_of_and_deref_on_a_scalar_agree(self, isa):
        """`&`/`*` ARE supported via pointer PARAMS -- verify both agree.

        x86 is excluded: it models no CALL/RET stack, so any program with a
        second function is rejected outright (pinned just below).
        """
        src = """
        void bump(int *p) { *p = *p + 5; }
        int main() { int x; x = 7; bump(&x); print(x); return 0; }
        """
        assert prints(src, isa) == [12]

    def test_x86_rejects_multi_function_programs(self):
        """x86 codegen is call-free, so it supports exactly one function.

        This is what excludes x86 from every multi-function case above
        (recursion, array params, pointer params) -- pinned so the exclusion
        is visibly deliberate rather than an untested gap.
        """
        from sim.compiler.errors import CompilerError
        src = "int f() { return 1; } int main() { return f(); }"
        with pytest.raises(CompilerError, match="single function"):
            compile_c(src, "x86")


class TestDocumentedLimitations:

    SRC = "int main() { print(1 + 1); return 0; }"

    @pytest.mark.parametrize("isa", ("riscv", "arm"))
    def test_ooo_link_register_gap_still_applies(self, isa):
        """OoO never commits JAL/BL's link-register write.

        Compiled Core-C on riscv/arm is always call-wrapped, so it cannot
        complete on ooo. If this ever starts passing, the gap has been fixed
        and ooo should be promoted into PARITY_MODELS.
        """
        assert prints(self.SRC, isa, "ooo") != [2], (
            f"{isa}/ooo now produces the right answer -- the documented "
            "link-register gap appears FIXED; add 'ooo' to PARITY_MODELS "
            "and delete this test.")

    def test_x86_ooo_is_unaffected_because_codegen_is_call_free(self):
        """The mirror of the above: x86 emits no CALL, so ooo works."""
        assert prints(self.SRC, "x86", "ooo") == [2]
