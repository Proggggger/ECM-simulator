"""
Tests for the per-ISA configuration objects and the ISABase abstract surface.

`sim/isa/<arch>/config.py` defines the `ISABase` subclass (RISCV / ARM / X86)
that every preset and every API route reads to learn how an ISA behaves:
register count and names, the zero register, the PC reset value, the NOP
encoding, the program format ("words" vs "bytes"), and the demo program the UI
loads on an ISA switch.

The three big properties pinned here:

  1. `program_format` really is "words" / "words" / "bytes" — the whole
     byte-addressed x86 fetch path branches on this string.
  2. `demo_program()` and `demo_program_asm()` are the SAME program: the asm
     text must assemble, through the project's own assembler, to exactly the
     hex list.  Nothing else in the suite checks that these two hand-maintained
     copies stay in sync, so they can silently drift.
  3. Each demo program actually RUNS and computes the value its comments claim
     (fib(10) = 55 / 89), on the single-cycle oracle.

`sim/isa/base.py` is also covered here: its four default-returning methods are
the fallback contract a future ISA inherits.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.assembler import assemble
from sim.harness import simulate
from sim.isa.arm.config import ARM
from sim.isa.arm.constants import REGISTER_NAMES as ARM_REGISTER_NAMES
from sim.isa.base import ISABase
from sim.isa.riscv.config import RISCV
from sim.isa.riscv.constants import REGISTER_NAMES as RISCV_REGISTER_NAMES
from sim.isa.x86.config import X86
from sim.isa.x86.constants import REGISTER_NAMES as X86_REGISTER_NAMES


ISA_CLASSES = [
    pytest.param(RISCV, id="riscv"),
    pytest.param(ARM, id="arm"),
    pytest.param(X86, id="x86"),
]


# ════════════════════════════════════════════════════════════════════════════
#  ISABase — the abstract surface a new ISA inherits
# ════════════════════════════════════════════════════════════════════════════

def test_isabase_class_level_defaults():
    assert ISABase.name == "unknown"
    assert ISABase.display_name == "Unknown"
    assert ISABase.description == ""
    assert ISABase.num_regs == 32
    assert ISABase.zero_reg is True
    assert ISABase.zero_reg_index == 0
    assert ISABase.pc_reset == 0
    assert ISABase.program_format == "words"


def test_isabase_register_names_defaults_to_x0_style_numbering():
    """The generic fallback: x0 .. x{num_regs-1}, derived from num_regs."""
    base = ISABase()
    names = base.register_names()
    assert len(names) == 32
    assert names[0] == "x0"
    assert names[31] == "x31"
    assert names == [f"x{i}" for i in range(32)]


def test_isabase_register_names_tracks_a_subclass_num_regs():
    class Tiny(ISABase):
        num_regs = 4

    assert Tiny().register_names() == ["x0", "x1", "x2", "x3"]


def test_isabase_default_nop_is_zero():
    assert ISABase().default_nop() == 0


def test_isabase_demo_program_is_empty():
    assert ISABase().demo_program() == []


def test_isabase_demo_program_asm_is_empty():
    assert ISABase().demo_program_asm() == ""


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_every_isa_config_subclasses_isabase(isa_cls):
    assert issubclass(isa_cls, ISABase)
    assert isinstance(isa_cls(), ISABase)


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_every_isa_overrides_all_four_abstract_methods(isa_cls):
    """A real ISA must not fall back to ISABase's empty defaults."""
    isa = isa_cls()
    assert isa.register_names() != ISABase().register_names() or \
        isa_cls is not X86      # x86 has 8 regs, so it always differs
    assert isa.default_nop() != 0
    assert isa.demo_program() != []
    assert isa.demo_program_asm().strip() != ""


# ════════════════════════════════════════════════════════════════════════════
#  Identity / metadata
# ════════════════════════════════════════════════════════════════════════════

EXPECTED_IDENTITY = {
    RISCV: ("riscv", "RISC-V (RV32I)"),
    ARM:   ("arm",   "ARM (AArch64)"),
    X86:   ("x86",   "x86 (IA-32)"),
}


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_isa_identity_strings(isa_cls):
    """`name` keys the preset registry and every API route — it must be exact."""
    isa = isa_cls()
    expected_name, expected_display = EXPECTED_IDENTITY[isa_cls]
    assert isa.name == expected_name
    assert isa.display_name == expected_display
    assert isa.description, "the UI shows this in the ISA picker"


def test_isa_names_are_unique_and_lowercase():
    names = [cls().name for cls in (RISCV, ARM, X86)]
    assert names == ["riscv", "arm", "x86"]
    assert len(set(names)) == 3
    assert all(n == n.lower() for n in names)


# ════════════════════════════════════════════════════════════════════════════
#  program_format
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("isa_cls,expected", [
    (RISCV, "words"),
    (ARM,   "words"),
    (X86,   "bytes"),
])
def test_program_format(isa_cls, expected):
    """The whole byte-addressed x86 fetch path branches on this exact string."""
    assert isa_cls().program_format == expected


def test_program_format_is_one_of_the_two_supported_values():
    for cls in (RISCV, ARM, X86):
        assert cls().program_format in ("words", "bytes")


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_demo_program_elements_fit_the_declared_program_format(isa_cls):
    """"bytes" means every element is 0..255; "words" means 0..2^32-1."""
    isa = isa_cls()
    program = isa.demo_program()
    limit = 0xFF if isa.program_format == "bytes" else 0xFFFF_FFFF
    for i, value in enumerate(program):
        assert isinstance(value, int)
        assert 0 <= value <= limit, (
            f"{isa.name} demo_program()[{i}] = {value:#x} exceeds the "
            f"{isa.program_format} range")


# ════════════════════════════════════════════════════════════════════════════
#  register_names()
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("isa_cls,num_regs", [(RISCV, 32), (ARM, 32), (X86, 8)])
def test_register_names_length_matches_num_regs(isa_cls, num_regs):
    isa = isa_cls()
    assert isa.num_regs == num_regs
    assert len(isa.register_names()) == num_regs


@pytest.mark.parametrize("isa_cls,constants_table", [
    (RISCV, RISCV_REGISTER_NAMES),
    (ARM,   ARM_REGISTER_NAMES),
    (X86,   X86_REGISTER_NAMES),
])
def test_register_names_mirrors_the_constants_table(isa_cls, constants_table):
    assert isa_cls().register_names() == list(constants_table)


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_register_names_returns_a_defensive_copy(isa_cls):
    """Every config does `list(REGISTER_NAMES)` — mutating the result must not
    corrupt the shared module-level table for the rest of the process."""
    isa = isa_cls()
    first = isa.register_names()
    first[0] = "CLOBBERED"
    assert isa.register_names()[0] != "CLOBBERED"


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_register_names_are_unique_non_empty_strings(isa_cls):
    names = isa_cls().register_names()
    assert all(isinstance(n, str) and n for n in names)
    assert len(set(names)) == len(names), "duplicate register name"


def test_riscv_register_names_are_the_abi_names():
    names = RISCV().register_names()
    assert names[0] == "zero"
    assert names[1] == "ra"
    assert names[2] == "sp"
    assert names[10] == "a0"
    assert names[31] == "t6"


def test_arm_register_names_end_with_xzr_at_index_31():
    names = ARM().register_names()
    assert names[0] == "X0"
    assert names[30] == "X30"
    assert names[31] == "XZR"


def test_x86_register_names_are_the_eight_ia32_gprs_in_encoding_order():
    assert X86().register_names() == ["EAX", "ECX", "EDX", "EBX",
                                      "ESP", "EBP", "ESI", "EDI"]


# ════════════════════════════════════════════════════════════════════════════
#  zero register / PC reset
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("isa_cls,zero_reg,zero_index", [
    (RISCV, True,  0),    # x0 is hardwired zero
    (ARM,   True,  31),   # XZR is index 31, NOT 0 — X0 is a normal GPR
    (X86,   False, 0),    # every IA-32 GPR is writable
])
def test_zero_register_configuration(isa_cls, zero_reg, zero_index):
    """RAT / ForwardingUnit / hazard detectors are parameterised on this.

    Getting ARM's index wrong (0 instead of 31) would make X0 unwritable and
    let XZR hold state — the exact bug the zero-reg parameterisation fixed.
    """
    isa = isa_cls()
    assert isa.zero_reg is zero_reg
    assert isa.zero_reg_index == zero_index


def test_arm_zero_reg_index_is_explicitly_overridden_not_inherited():
    """ARM must not silently inherit ISABase's zero_reg_index = 0."""
    assert "zero_reg_index" in vars(ARM)
    assert ARM.zero_reg_index == 31 != ISABase.zero_reg_index


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_pc_reset_is_zero_for_every_modelled_isa(isa_cls):
    assert isa_cls().pc_reset == 0


# ════════════════════════════════════════════════════════════════════════════
#  default_nop()
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("isa_cls,nop,note", [
    (RISCV, 0x00000013, "ADDI x0, x0, 0"),
    (ARM,   0xD503201F, "A64 NOP (hint #0)"),
    (X86,   0x90,       "single-byte 0x90"),
])
def test_default_nop_encoding(isa_cls, nop, note):
    assert isa_cls().default_nop() == nop, note


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_default_nop_decodes_to_an_inert_instruction(isa_cls):
    """The padding word must not touch memory or redirect the PC.

    RISC-V is the interesting case: its NOP is `ADDI x0, x0, 0`, which really
    DOES assert reg_write — the write is neutralised by x0 being hardwired to
    zero in the register file, not by the decoder.  ARM's NOP and x86's 0x90
    are recognised as no-ops by their decoders and assert nothing at all.
    """
    from sim.isa.arm.decoder_component import ARMDecoder
    from sim.isa.riscv.decoder_component import RISCVDecoder
    from sim.isa.x86.decoder_component import X86Decoder

    decoder_for = {RISCV: RISCVDecoder, ARM: ARMDecoder, X86: X86Decoder}
    isa = isa_cls()
    decoder = decoder_for[isa_cls]()
    decoder["instr_in"] = isa.default_nop()
    decoder.evaluate()

    for port in ("mem_read", "mem_write", "branch", "jal", "jalr"):
        assert decoder[port] == 0, f"{isa_cls.__name__} NOP asserted {port}"

    if isa_cls is RISCV:
        # ADDI x0, x0, 0 — a write that the zero register discards.
        assert decoder["reg_write"] == 1
        assert decoder["rd"] == isa.zero_reg_index == 0
        assert isa.zero_reg is True, "the NOP relies on x0 being hardwired zero"
    else:
        assert decoder["reg_write"] == 0


@pytest.mark.parametrize("isa_cls,text", [
    (RISCV, "NOP"),
    (ARM,   "NOP"),
    (X86,   "NOP"),
])
def test_default_nop_matches_what_the_assembler_emits_for_nop(isa_cls, text):
    isa = isa_cls()
    assert assemble(isa.name, text) == [isa.default_nop()]


def test_x86_default_nop_fits_in_a_single_byte():
    """x86 programs are byte lists, so the NOP must be byte-sized."""
    assert 0 <= X86().default_nop() <= 0xFF


# ════════════════════════════════════════════════════════════════════════════
#  demo_program()  <->  demo_program_asm()  round trip
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_demo_program_asm_assembles_to_exactly_demo_program(isa_cls):
    """THE key invariant of this file.

    `demo_program()` (hand-written hex) and `demo_program_asm()` (hand-written
    text) are two independent copies of the same program.  The UI shows the
    text and the simulator runs the hex, so a drift between them is invisible
    at runtime but produces a UI that lies about what is executing.
    """
    isa = isa_cls()
    expected = isa.demo_program()
    got = assemble(isa.name, isa.demo_program_asm())
    assert len(got) == len(expected), (
        f"{isa.name}: asm assembles to {len(got)} unit(s) but demo_program() "
        f"has {len(expected)}")
    for i, (g, e) in enumerate(zip(got, expected)):
        assert g == e, (
            f"{isa.name} unit {i}: assembled {g:#0{2 if isa.program_format == 'bytes' else 10}x} "
            f"!= demo_program() {e:#0{2 if isa.program_format == 'bytes' else 10}x}")


@pytest.mark.parametrize("isa_cls,expected_len", [
    (RISCV, 8),     # 8 instruction words
    (ARM,   9),     # 9 instruction words (ARM needs an explicit CMP)
    (X86,   31),    # 31 BYTES (variable-length), not 31 instructions
])
def test_demo_program_length(isa_cls, expected_len):
    assert len(isa_cls().demo_program()) == expected_len


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_demo_program_asm_is_non_empty_and_has_a_loop_label(isa_cls):
    """All three demos are the same iterative Fibonacci with a `loop:` label."""
    text = isa_cls().demo_program_asm()
    assert text.strip()
    assert "loop:" in text
    assert not text.endswith("\n"), "demo asm should not have a trailing newline"


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_demo_program_asm_comment_style_is_semicolon(isa_cls):
    """`;` is the one comment marker all three assemblers accept.

    ARM in particular cannot use `#` (it is the immediate prefix), so a demo
    written with `#` comments would fail to assemble for ARM only.
    """
    text = isa_cls().demo_program_asm()
    assert ";" in text
    for line in text.splitlines():
        code = line.split(";")[0]
        assert "#" not in code or isa_cls is ARM, (
            "'#' outside a comment is only valid as an ARM immediate prefix")


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_demo_program_asm_reassembles_deterministically(isa_cls):
    isa = isa_cls()
    first = assemble(isa.name, isa.demo_program_asm())
    second = assemble(isa.name, isa.demo_program_asm())
    assert first == second


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_demo_program_returns_a_fresh_list_each_call(isa_cls):
    """Callers (api/app.py, the harness) mutate/slice this; it must not be shared."""
    isa = isa_cls()
    first = isa.demo_program()
    first.append(0)
    assert len(isa.demo_program()) == len(first) - 1


# ════════════════════════════════════════════════════════════════════════════
#  demo programs actually RUN and compute what their comments claim
# ════════════════════════════════════════════════════════════════════════════
# All three demos are iterative Fibonacci with N = 10:
#   prev = 0; curr = 1
#   repeat 10x: temp = prev + curr; prev = curr; curr = temp
# After 10 iterations prev = fib(10) = 55 and curr = fib(11) = 89.

FIB_PREV = 55
FIB_CURR = 89

# Where each ISA's demo leaves prev / curr / temp.
DEMO_RESULT_REGS = {
    RISCV: {"x2": FIB_PREV, "x3": FIB_CURR, "x4": FIB_CURR},
    ARM:   {"X2": FIB_PREV, "X3": FIB_CURR, "X4": FIB_CURR},
    X86:   {"EAX": FIB_PREV, "EBX": FIB_CURR, "EDX": FIB_CURR},
}


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_demo_program_runs_and_computes_fib_10(isa_cls):
    """The demo is what a first-time user sees; it must produce a real answer."""
    isa = isa_cls()
    result = simulate(isa.name, "single_cycle",
                      program=isa.demo_program(), cycles=200)
    for reg, expected in DEMO_RESULT_REGS[isa_cls].items():
        assert result.reg(reg) == expected, (
            f"{isa.name} demo: {reg} = {result.reg(reg)}, expected {expected}")


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_demo_program_and_its_asm_simulate_identically(isa_cls):
    """Running the hex list and running the assembled text must agree.

    This is the round-trip check made end-to-end: same registers, same memory.
    """
    isa = isa_cls()
    from_hex = simulate(isa.name, "single_cycle",
                        program=isa.demo_program(), cycles=200)
    from_asm = simulate(isa.name, "single_cycle",
                        asm=isa.demo_program_asm(), cycles=200)
    assert from_hex.registers == from_asm.registers
    assert from_hex.memory == from_asm.memory


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_demo_program_terminates_within_a_reasonable_cycle_budget(isa_cls):
    """The loop must actually exit — a demo that never settles hangs the UI's
    run_to_completion path."""
    isa = isa_cls()
    settled = simulate(isa.name, "single_cycle",
                       program=isa.demo_program(), cycles=100)
    later = simulate(isa.name, "single_cycle",
                     program=isa.demo_program(), cycles=400)
    assert settled.registers == later.registers, "the demo has not settled by cycle 100"


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
@pytest.mark.parametrize("model", ["single_cycle", "multicycle", "pipeline"])
def test_demo_program_agrees_across_the_in_order_models(isa_cls, model):
    """The demo must give the same architectural answer on every in-order model.

    (OoO / superscalar are excluded: they are documented as illustrative for
    loop-heavy code — see CLAUDE.md "Known limitations".)
    """
    isa = isa_cls()
    result = simulate(isa.name, model, program=isa.demo_program(), cycles=600)
    for reg, expected in DEMO_RESULT_REGS[isa_cls].items():
        assert result.reg(reg) == expected, f"{isa.name}/{model}: {reg}"


# ════════════════════════════════════════════════════════════════════════════
#  Cross-ISA consistency
# ════════════════════════════════════════════════════════════════════════════

def test_all_three_isas_are_registered_in_the_harness():
    from sim.harness import ISA_CONFIGS, ISAS

    assert set(ISAS) == {"riscv", "arm", "x86"}
    for name, cfg in ISA_CONFIGS.items():
        assert cfg.name == name, "registry key must match the config's own name"
        assert isinstance(cfg, ISABase)


@pytest.mark.parametrize("isa_cls", ISA_CLASSES)
def test_harness_registry_holds_the_same_config_class(isa_cls):
    from sim.harness import ISA_CONFIGS

    assert isinstance(ISA_CONFIGS[isa_cls().name], isa_cls)


def test_only_x86_uses_the_byte_program_format():
    from sim.harness import ISA_CONFIGS

    byte_isas = {n for n, c in ISA_CONFIGS.items() if c.program_format == "bytes"}
    assert byte_isas == {"x86"}


def test_register_counts_across_isas():
    from sim.harness import ISA_CONFIGS

    assert {n: c.num_regs for n, c in ISA_CONFIGS.items()} == \
        {"riscv": 32, "arm": 32, "x86": 8}
