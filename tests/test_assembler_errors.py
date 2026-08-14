"""
ERROR-PATH tests for the four assembler modules (`sim/assembler/`).

WHY THIS FILE EXISTS
--------------------
`tests/test_assemblers.py` pins the HAPPY path exhaustively (exact encodings
for every supported mnemonic in all three ISAs) but only four of the dozens of
``raise AssemblerError`` sites are ever reached.  The assembler is the front
door for untrusted user text (`POST /assemble`, `POST /compile`, the UI's
editor), so every rejection path is a real product surface: a silently
truncated immediate or a raw `ValueError` leaking out of the encoder is a bug
users see.

This file walks each module's error conditions systematically:

  * `base.py`     — AssemblerError formatting, `_parse_immediate`, the three
                    `_require_*` range/alignment guards, the abstract
                    `_encode`, duplicate labels, pass-2 exception wrapping.
  * `riscv.py`    — register parsing, operand arity, per-format immediate
                    ranges (I/S/B/J/shamt), branch alignment, `offset(base)`.
  * `arm.py`      — the WEAKEST-covered module: register parsing, arity,
                    imm12 shift rules, MOVZ hw rules, LDR/STR scaling, branch
                    alignment, and the unsupported-condition rejection.
                    Also pins that `#` is an IMMEDIATE PREFIX, not a comment.
  * `x86.py`      — arity, register parsing, the sign-extended 8-bit field
                    guard (disp8/imm8/rel8) and branch relaxation boundaries.

ASSERTION STYLE
---------------
Every test asserts the EXCEPTION TYPE *and* matches the message, so a
refactor that keeps "it raises something" but loses the diagnostic still
fails.  Two error-quality bugs this file found (ARM's unclosed-bracket raw
`ValueError`, and encoder errors never naming the source line) were fixed on
2026-08-08; `TestErrorLineContext` and `TestARMUnclosedBracket` are now
ordinary passing REGRESSION PINS.

Everything here is pure string -> int; no CPU is ever built.

Run from repo root:  python3 -m pytest tests/test_assembler_errors.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.assembler import assemble
from sim.assembler.arm import ARMAssembler
from sim.assembler.arm import _parse_reg as arm_parse_reg
from sim.assembler.base import AssemblerBase, AssemblerError
from sim.assembler.riscv import RISCVAssembler
from sim.assembler.riscv import _parse_reg as riscv_parse_reg
from sim.assembler.x86 import X86Assembler
from sim.assembler.x86 import _imm8_byte, _is_mem, _is_reg, _modrm
from sim.assembler.x86 import _parse_reg as x86_parse_reg


def err(fn, *args, **kw):
    """Run *fn*, require an AssemblerError, return it for message assertions."""
    with pytest.raises(AssemblerError) as excinfo:
        fn(*args, **kw)
    return excinfo.value


# ════════════════════════════════════════════════════════════════════════════
# base.py — AssemblerError formatting + shared helpers
# ════════════════════════════════════════════════════════════════════════════

class TestAssemblerErrorFormatting:
    """The exception carries structured line info AND renders it in str()."""

    def test_message_only_has_no_line_prefix(self):
        e = AssemblerError("boom")
        assert str(e) == "boom"
        assert e.line_number is None
        assert e.line_text is None

    def test_line_number_is_prefixed(self):
        e = AssemblerError("boom", 7)
        assert str(e) == "line 7: boom"
        assert e.line_number == 7

    def test_line_text_is_quoted_and_stripped(self):
        e = AssemblerError("boom", 7, "   ADD x1  ")
        assert str(e) == "line 7: 'ADD x1' -- boom"
        assert e.line_text == "   ADD x1  "     # raw text preserved on the attr

    def test_line_text_without_number_still_renders(self):
        # base.py:18 — the `if line_text` branch independent of line_number.
        e = AssemblerError("boom", None, "  ADD x1  ")
        assert str(e) == "'ADD x1' -- boom"

    def test_is_an_exception_subclass(self):
        assert issubclass(AssemblerError, Exception)


class TestBaseParseImmediate:
    """`_parse_immediate` accepts dec / hex / bin / '#'-prefixed / negative."""

    @pytest.fixture()
    def base(self):
        return AssemblerBase()

    @pytest.mark.parametrize("token,expected", [
        ("5", 5), ("0", 0), ("-7", -7),
        ("0x1F", 31), ("0X1F", 31), ("-0x10", -16),
        ("0b1010", 10), ("0B1010", 10), ("-0b101", -5),
        ("#5", 5), ("#-5", -5), ("-#5", -5),      # ARM immediate prefix
        ("  12  ", 12),
    ])
    def test_accepted_literals(self, base, token, expected):
        assert base._parse_immediate(token) == expected

    @pytest.mark.parametrize("token", ["0xZZ", "0b12", "abc", "", "1.5", "1_2_"])
    def test_malformed_literals_raise_valueerror(self, base, token):
        # _parse_immediate itself raises the raw ValueError; _pass2 is what
        # wraps it into an AssemblerError with line context.
        with pytest.raises(ValueError):
            base._parse_immediate(token)

    @pytest.mark.parametrize("token", ["+0x10", "+0b101"])
    def test_explicit_plus_before_a_radix_prefix_is_rejected(self, base, token):
        # QUIRK: base.py:151 only peels a leading '-', so '-0x10' works but
        # '+0x10' falls through to int(token) and dies. Pinned as observed.
        with pytest.raises(ValueError, match="base 10"):
            base._parse_immediate(token)
        assert base._parse_immediate(token.replace("+", "")) > 0
        assert base._parse_immediate(token.replace("+", "-")) < 0

    def test_resolve_label_wins_over_number(self, base):
        assert base._resolve_label_or_imm("lbl", 4, {"lbl": 12}) == 8

    def test_resolve_falls_through_to_immediate(self, base):
        assert base._resolve_label_or_imm("+8", 4, {}) == 8

    def test_unknown_label_is_parsed_as_a_number_and_fails(self, base):
        # There is no separate "undefined label" check: an unresolved name
        # falls through to _parse_immediate and dies as a bad literal.
        with pytest.raises(ValueError, match="invalid literal"):
            base._resolve_label_or_imm("nowhere", 0, {})


class TestBaseRangeGuards:
    """`_require_signed` / `_require_unsigned` / `_require_aligned`."""

    @pytest.fixture()
    def base(self):
        return AssemblerBase()

    def test_require_signed_accepts_boundaries(self, base):
        assert base._require_signed(-2048, 12) == -2048
        assert base._require_signed(2047, 12) == 2047

    @pytest.mark.parametrize("value", [2048, -2049, 1 << 20])
    def test_require_signed_rejects_out_of_range(self, base, value):
        e = err(base._require_signed, value, 12, "test field")
        assert "test field" in str(e)
        assert str(value) in str(e)
        assert "[-2048..2047]" in str(e)

    def test_require_unsigned_accepts_boundaries(self, base):
        assert base._require_unsigned(0, 5) == 0
        assert base._require_unsigned(31, 5) == 31

    @pytest.mark.parametrize("value", [-1, 32, 1 << 16])
    def test_require_unsigned_rejects_out_of_range(self, base, value):
        e = err(base._require_unsigned, value, 5, "shift amount")
        assert "shift amount" in str(e)
        assert "[0..31]" in str(e)

    def test_require_aligned_accepts_multiples(self, base):
        assert base._require_aligned(8, 4) == 8
        assert base._require_aligned(0, 4) == 0
        assert base._require_aligned(-8, 4) == -8

    @pytest.mark.parametrize("value,align", [(3, 2), (5, 4), (-1, 2)])
    def test_require_aligned_rejects_misaligned(self, base, value, align):
        e = err(base._require_aligned, value, align, "branch offset")
        assert "branch offset" in str(e)
        assert f"multiple of {align}" in str(e)

    def test_mask_wraps_negatives_into_the_field(self, base):
        assert AssemblerBase._mask(-1, 12) == 0xFFF
        assert AssemblerBase._mask(0x1FFF, 12) == 0xFFF


class TestBaseAbstractSurface:
    """The unimplemented hooks + the generic (non-ISA) defaults."""

    def test_encode_is_abstract(self):
        with pytest.raises(NotImplementedError):
            AssemblerBase()._encode("ADD", [], 0, {})

    def test_assemble_on_the_base_class_wraps_notimplemented(self):
        # _pass2 catches the non-AssemblerError and re-raises with line context.
        e = err(AssemblerBase().assemble, "ADD x1, x2, x3")
        assert e.line_number == 1
        assert e.line_text == "ADD x1, x2, x3"
        assert "line 1:" in str(e)

    def test_default_pc_increment_is_four(self):
        assert AssemblerBase()._pc_increment() == 4
        assert AssemblerBase()._estimate_instruction_size("anything") == 4
        assert AssemblerBase()._estimate_instruction_size_at(1, "anything") == 4

    def test_default_split_uses_commas_and_whitespace(self):
        assert AssemblerBase()._split_instruction("ADD x3, x1, x2") == \
            ["ADD", "x3", "x1", "x2"]


class TestEmptyAndCommentOnlyPrograms:
    """A program with nothing to encode assembles to [] rather than raising."""

    @pytest.mark.parametrize("isa", ["riscv", "arm", "x86"])
    @pytest.mark.parametrize("text", ["", "\n\n   \n", "\t\n"])
    def test_empty_program_is_empty_list(self, isa, text):
        assert assemble(isa, text) == []

    @pytest.mark.parametrize("isa,text", [
        ("riscv", "# only a comment\n"),
        ("riscv", "; only a comment\n"),
        ("riscv", "// only a comment\n"),
        ("arm",   "; only a comment\n"),
        ("arm",   "// only a comment\n"),
        ("x86",   "# only a comment\n; and\n// more\n"),
    ])
    def test_comment_only_program_is_empty_list(self, isa, text):
        assert assemble(isa, text) == []

    @pytest.mark.parametrize("isa", ["riscv", "arm", "x86"])
    def test_label_only_program_is_empty_list(self, isa):
        assert assemble(isa, "start:\nend:\n") == []


class TestDuplicateLabels:
    """Pass 1 rejects a redefined label, WITH line context (base.py:77)."""

    @pytest.mark.parametrize("isa,body", [
        ("riscv", "loop:\nNOP\nloop:\nNOP\n"),
        ("arm",   "loop:\nNOP\nloop:\nNOP\n"),
        ("x86",   "loop:\nNOP\nloop:\nNOP\n"),
    ])
    def test_duplicate_label_raises_with_line(self, isa, body):
        with pytest.raises(AssemblerError) as ei:
            assemble(isa, body)
        e = ei.value
        assert "Duplicate label 'loop'" in str(e)
        assert e.line_number == 3
        assert "line 3:" in str(e)

    def test_duplicate_label_on_an_instruction_line_is_detected(self):
        e = err(assemble, "riscv", "loop: NOP\nloop: NOP\n")
        assert "Duplicate label 'loop'" in str(e)
        assert e.line_number == 2

    def test_two_labels_on_one_line_both_register(self):
        # Not an error: the while-loop in _pass1 peels every leading label, and
        # both names resolve to the SAME address (the following NOP at pc=0).
        asm = RISCVAssembler()
        labels, cleaned = asm._pass1(asm._preprocess("a: b: NOP\nNOP\n"))
        assert labels == {"a": 0, "b": 0}
        assert len(cleaned) == 2


class TestUndefinedLabel:
    """An unresolved name falls through to number parsing and fails there."""

    @pytest.mark.parametrize("isa,text", [
        ("riscv", "BEQ x1, x2, nowhere\n"),
        ("riscv", "JAL x1, nowhere\n"),
        ("arm",   "B nowhere\n"),
        ("arm",   "BL nowhere\n"),
        ("arm",   "B.EQ nowhere\n"),
        ("arm",   "CBZ X1, nowhere\n"),
        ("x86",   "JMP nowhere\n"),
        ("x86",   "JE nowhere\n"),
        ("x86",   "CALL nowhere\n"),
    ])
    def test_undefined_label_raises_with_line_context(self, isa, text):
        e = err(assemble, isa, text)
        # Wrapped ValueError -> the wrapper supplies lineno + source text.
        assert e.line_number == 1
        assert "nowhere" in str(e)
        assert "invalid literal" in str(e)


class TestErrorLineContext:
    """Which errors carry the offending line — and which currently do not.

    `_pass2` (base.py:100-105) wraps *non*-AssemblerError exceptions with
    ``(lineno, line)`` but re-raises an AssemblerError UNCHANGED, so every
    error the encoders raise deliberately (arity, range, bad register) loses
    its source line.  The asymmetry is pinned here so it is a noticed change
    if it is ever fixed.
    """

    PROGRAM = (
        "ADDI x1, x0, 1\n"
        "ADDI x2, x0, 2\n"
        "ADD  x3, x1\n"          # <- line 3: arity error
        "ADDI x4, x0, 4\n"
    )

    def test_wrapped_exceptions_do_carry_the_line(self):
        e = err(assemble, "riscv", "ADDI x1, x0, 1\nADDI x2, x0, 0xZZ\n")
        assert e.line_number == 2
        assert e.line_text == "ADDI x2, x0, 0xZZ"
        assert "line 2:" in str(e)

    def test_encoder_raised_errors_name_the_line(self):
        """REGRESSION PIN (fixed 2026-08-08).

        base.py's _pass2 used to re-raise an AssemblerError from _encode()
        unchanged, so encoder diagnostics (wrong arity, out-of-range
        immediate, bad register — the COMMON cases) never named the offending
        line: in a 100-line program the user just got "ADD requires 3
        operands". Bad-literal errors DID get a line, making the gap
        inconsistent as well as unhelpful. Now the except-AssemblerError arm
        fills in (lineno, line) when line_number is None.
        """
        e = err(assemble, "riscv", self.PROGRAM)
        assert "ADD requires 3 operands, got 2" in str(e)
        assert e.line_number == 3
        assert e.line_text == "ADD  x3, x1"
        assert "line 3:" in str(e)

    @pytest.mark.parametrize("src,lineno,needle", [
        ("NOP\nNOP\nADDI x1, x0, 9999999\n", 3, "out of range"),
        ("NOP\nADD x99, x1, x2\n",           2, "out of range"),
        ("NOP\nNOP\nNOP\nSW x1\n",           4, "operands"),
    ])
    def test_every_encoder_error_class_carries_its_line(self, src, lineno, needle):
        """The fix must cover arity, range AND register errors alike."""
        e = err(assemble, "riscv", src)
        assert e.line_number == lineno, str(e)
        assert needle in str(e)


# ════════════════════════════════════════════════════════════════════════════
# riscv.py
# ════════════════════════════════════════════════════════════════════════════

class TestRISCVRegisterParsing:

    @pytest.mark.parametrize("token,expected", [
        ("x0", 0), ("x31", 31), ("X5", 5), (" x7 ", 7),
        ("zero", 0), ("ra", 1), ("sp", 2), ("fp", 8), ("s0", 8), ("t6", 31),
        ("A0", 10),
    ])
    def test_accepted_register_names(self, token, expected):
        assert riscv_parse_reg(token) == expected

    @pytest.mark.parametrize("token", ["xfoo", "x", "x1a", "x1.0", "X0X"])
    def test_malformed_x_register_raises_invalid(self, token):
        e = err(riscv_parse_reg, token)
        assert "Invalid register" in str(e)
        assert token in str(e)

    @pytest.mark.parametrize("token", ["x32", "x99", "x100", "x-1"])
    def test_x_register_index_out_of_range(self, token):
        # 'x-1' parses as int(-1) and is caught by the RANGE check, not the
        # int() guard — pinned so the two messages stay distinguishable.
        e = err(riscv_parse_reg, token)
        assert "Register index out of range" in str(e)
        assert token in str(e)

    @pytest.mark.parametrize("token", ["q1", "eax", "r5", "s12"])
    def test_unknown_register_name(self, token):
        e = err(riscv_parse_reg, token)
        assert "Unknown register" in str(e)
        assert token in str(e)

    def test_bad_register_inside_a_real_instruction(self):
        e = err(assemble, "riscv", "ADD x1, x99, x2")
        assert "Register index out of range" in str(e)


class TestRISCVUnknownMnemonic:

    @pytest.mark.parametrize("text", [
        "BOGUS x1, x2, x3", "MUL x1, x2, x3", "LB x1, 0(x2)",
        "SB x1, 0(x2)", "ECALL", "1BAD: ",
    ])
    def test_unsupported_mnemonic_raises(self, text):
        e = err(assemble, "riscv", text)
        assert "Unknown RISC-V mnemonic" in str(e)

    def test_mnemonics_are_matched_case_insensitively(self):
        # The message reports the UPPER-cased mnemonic (_pass2 upper-cases
        # before dispatch), and lowercase valid mnemonics still assemble.
        e = err(assemble, "riscv", "bogus x1, x2, x3")
        assert "'BOGUS'" in str(e)
        assert assemble("riscv", "addi x1, x0, 5") == assemble("riscv", "ADDI x1, x0, 5")


class TestRISCVOperandArity:

    @pytest.mark.parametrize("text,mnemonic,got", [
        ("ADD x1, x2", "ADD", 2),
        ("SUB x1", "SUB", 1),
        ("AND x1, x2, x3, x4", "AND", 4),
        ("ADDI x1, x0", "ADDI", 2),
        ("SLLI x1", "SLLI", 1),
        ("BEQ x1, x2", "BEQ", 2),
        ("BNE x1, x2, +8, x3", "BNE", 4),
        ("JAL x1", "JAL", 1),
        ("JALR x1, x2", "JALR", 2),
        ("LUI x1", "LUI", 1),
        ("AUIPC x1", "AUIPC", 1),
    ])
    def test_wrong_operand_count(self, text, mnemonic, got):
        e = err(assemble, "riscv", text)
        assert mnemonic in str(e)
        assert f"got {got}" in str(e)


class TestRISCVImmediateRanges:
    """I-type is 12-bit SIGNED; shifts are 5-bit UNSIGNED; U-type truncates."""

    @pytest.mark.parametrize("mn", ["ADDI", "ANDI", "ORI", "XORI", "SLTI", "SLTIU"])
    def test_i_type_boundaries_accepted(self, mn):
        assert assemble("riscv", f"{mn} x1, x0, 2047")
        assert assemble("riscv", f"{mn} x1, x0, -2048")

    @pytest.mark.parametrize("mn", ["ADDI", "ANDI", "ORI", "XORI", "SLTI", "SLTIU"])
    @pytest.mark.parametrize("imm", [2048, -2049])
    def test_i_type_out_of_range(self, mn, imm):
        e = err(assemble, "riscv", f"{mn} x1, x0, {imm}")
        assert f"{mn} immediate {imm} out of range" in str(e)
        assert "signed 12-bit" in str(e)

    @pytest.mark.parametrize("mn", ["SLLI", "SRLI", "SRAI"])
    def test_shamt_boundaries_accepted(self, mn):
        assert assemble("riscv", f"{mn} x1, x0, 0")
        assert assemble("riscv", f"{mn} x1, x0, 31")

    @pytest.mark.parametrize("mn", ["SLLI", "SRLI", "SRAI"])
    @pytest.mark.parametrize("imm", [32, -1, 64])
    def test_shamt_out_of_range(self, mn, imm):
        e = err(assemble, "riscv", f"{mn} x1, x0, {imm}")
        assert "shift amount" in str(e)
        assert "unsigned 5-bit" in str(e)

    def test_jalr_offset_range(self):
        assert assemble("riscv", "JALR x0, x1, 2047")
        assert assemble("riscv", "JALR x0, x1, -2048")
        e = err(assemble, "riscv", "JALR x0, x1, 2048")
        assert "JALR offset 2048 out of range" in str(e)

    @pytest.mark.parametrize("mn", ["LUI", "AUIPC"])
    def test_u_type_silently_masks_to_20_bits(self, mn):
        # Documented behaviour (riscv.py:279 / :288): there is no
        # _require_unsigned on the U-type field, so 0x100000 == 0.
        assert assemble("riscv", f"{mn} x1, 0x100000") == \
            assemble("riscv", f"{mn} x1, 0")
        assert assemble("riscv", f"{mn} x1, 0xFFFFF")   # widest legal value


class TestRISCVMemoryOperands:
    """`LW rd, offset(rs1)` / `SW rs2, offset(rs1)` parsing + offset range."""

    @pytest.mark.parametrize("mn", ["LW", "SW"])
    def test_offset_boundaries_accepted(self, mn):
        assert assemble("riscv", f"{mn} x1, 2047(x0)")
        assert assemble("riscv", f"{mn} x1, -2048(x0)")

    @pytest.mark.parametrize("mn,what", [("LW", "load offset"), ("SW", "store offset")])
    @pytest.mark.parametrize("off", [2048, -2049, 4096])
    def test_offset_out_of_range(self, mn, what, off):
        e = err(assemble, "riscv", f"{mn} x1, {off}({'x0'})")
        assert what in str(e)
        assert "signed 12-bit" in str(e)

    @pytest.mark.parametrize("text", [
        "LW x1, 5",              # no (base)
        "LW x1, x2",             # register instead of offset(base)
        "LW x1, 0 (x0)",         # space breaks the offset(base) regex
        "SW x1, foo",
    ])
    def test_malformed_two_token_memory_operand(self, text):
        e = err(assemble, "riscv", text)
        assert "Invalid memory operand" in str(e)
        assert "Expected 'offset(reg)'" in str(e)

    @pytest.mark.parametrize("text", ["LW x1, 0, x0", "LW x1, zz, x0"])
    def test_malformed_three_token_memory_operand(self, text):
        # riscv.py:314 — the 3-token recombination path.
        e = err(assemble, "riscv", text)
        assert "Invalid memory operand" in str(e)

    def test_three_token_form_recombines_when_valid(self):
        # 'LW x1, 0(x0), x2' splits into 3 tokens; ops[1] already has '(' so
        # ops[1]+ops[2] still matches -> assembles like 'LW x1, 0(x0)'.
        assert assemble("riscv", "LW x1, 0(x0), x2") == assemble("riscv", "LW x1, 0(x0)")

    @pytest.mark.parametrize("text", ["LW x1", "LW", "SW x1"])
    def test_too_few_memory_operands(self, text):
        e = err(assemble, "riscv", text)
        assert "Load/store requires 2 operands" in str(e)

    def test_too_many_memory_operands(self):
        e = err(assemble, "riscv", "LW x1, 0(x0), x2, x3")
        assert "Load/store requires 2 operands" in str(e)
        assert "Got 4" in str(e)

    def test_bad_base_register_in_memory_operand(self):
        e = err(assemble, "riscv", "LW x1, 0(x77)")
        assert "Register index out of range" in str(e)


class TestRISCVBranchAndJumpOffsets:
    """B-type is 13-bit signed & even; J-type is 21-bit signed & even."""

    @pytest.mark.parametrize("mn", ["BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"])
    def test_branch_boundaries_accepted(self, mn):
        assert assemble("riscv", f"{mn} x1, x2, +4094")
        assert assemble("riscv", f"{mn} x1, x2, -4096")

    @pytest.mark.parametrize("mn", ["BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"])
    def test_branch_offset_out_of_range(self, mn):
        e = err(assemble, "riscv", f"{mn} x1, x2, +4096")
        assert f"{mn} branch offset 4096 out of range" in str(e)
        assert "signed 13-bit" in str(e)

    @pytest.mark.parametrize("mn", ["BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"])
    def test_branch_offset_must_be_even(self, mn):
        e = err(assemble, "riscv", f"{mn} x1, x2, +3")
        assert f"{mn} branch offset 3 must be a multiple of 2" in str(e)

    def test_misaligned_branch_via_label_is_impossible_but_numeric_is_caught(self):
        # Labels always land on 4-byte boundaries, so misalignment can only
        # come from an explicit numeric displacement.
        assert assemble("riscv", "loop:\nNOP\nBEQ x0, x0, loop\n")
        err(assemble, "riscv", "BEQ x0, x0, -1")

    def test_jal_boundaries_accepted(self):
        assert assemble("riscv", "JAL x1, +1048574")
        assert assemble("riscv", "JAL x1, -1048576")

    def test_jal_offset_out_of_range(self):
        e = err(assemble, "riscv", "JAL x1, +1048576")
        assert "JAL offset 1048576 out of range" in str(e)
        assert "signed 21-bit" in str(e)

    def test_jal_offset_must_be_even(self):
        e = err(assemble, "riscv", "JAL x1, +3")
        assert "JAL offset 3 must be a multiple of 2" in str(e)


# ════════════════════════════════════════════════════════════════════════════
# arm.py  (the least-covered module)
# ════════════════════════════════════════════════════════════════════════════

class TestARMHashIsAnImmediatePrefixNotAComment:
    """ARM overrides comment stripping because `#` introduces an immediate.

    `ARMAssembler._preprocess` splits on `;` and `//` ONLY.  Reintroducing the
    base class's `#` rule would silently delete every immediate operand.
    """

    def test_hash_immediate_survives_preprocessing(self):
        assert assemble("arm", "MOVZ X1, #10") == [0xD2800141]

    def test_hash_immediate_is_not_stripped_by_the_base_rule(self):
        # Same source through the BASE preprocessor would lose the "#10".
        assert AssemblerBase()._preprocess("MOVZ X1, #10") == [(1, "MOVZ X1,")]
        assert ARMAssembler()._preprocess("MOVZ X1, #10") == [(1, "MOVZ X1, #10")]

    @pytest.mark.parametrize("text", [
        "MOVZ X1, #10 // set to ten",
        "MOVZ X1, #10 ; set to ten",
        "MOVZ X1, #10// tight comment",
    ])
    def test_comment_after_a_hash_immediate(self, text):
        assert assemble("arm", text) == [0xD2800141]

    def test_hash_immediate_inside_an_operand_list(self):
        assert assemble("arm", "ADD X1, X2, #5") == assemble("arm", "ADD X1, X2, #0x5")

    def test_hash_immediate_inside_a_memory_operand(self):
        assert assemble("arm", "LDR X5, [X0, #8] // load")[0] == \
            assemble("arm", "LDR X5, [X0, #8]")[0]

    def test_hex_hash_immediate(self):
        assert assemble("arm", "SUB X1, X2, #0xFF") == assemble("arm", "SUB X1, X2, #255")

    def test_a_leading_hash_line_is_NOT_a_comment(self):
        # Consequence of the override: '#'-style comments are a user error in
        # ARM source and are reported as an unknown mnemonic, not ignored.
        e = err(assemble, "arm", "# this is not a comment\nNOP\n")
        assert "Unknown ARM mnemonic" in str(e)
        assert "'#'" in str(e)

    @pytest.mark.parametrize("text", ["; header\nNOP\n", "// header\nNOP\n"])
    def test_semicolon_and_slashes_are_comments(self, text):
        assert assemble("arm", text) == [0xD503201F]


class TestARMRegisterParsing:

    @pytest.mark.parametrize("token,expected", [
        ("X0", 0), ("X30", 30), ("x7", 7), (" X12 ", 12),
        ("XZR", 31), ("xzr", 31), ("SP", 31), ("sp", 31),
    ])
    def test_accepted_register_names(self, token, expected):
        assert arm_parse_reg(token) == expected

    @pytest.mark.parametrize("token", ["XQ", "X", "X1A"])
    def test_malformed_x_register(self, token):
        e = err(arm_parse_reg, token)
        assert "Invalid register" in str(e)
        assert token in str(e)

    @pytest.mark.parametrize("token", ["X31", "X32", "X99", "X-1"])
    def test_x_index_out_of_range_rejects_31_and_up(self, token):
        # X31 is NOT accepted as a numbered register: XZR is the spelling.
        # 'X-1' parses as -1 and is caught by the same range check.
        e = err(arm_parse_reg, token)
        assert "Register index out of range" in str(e)

    @pytest.mark.parametrize("token", ["W1", "R0", "EAX", "Q3", "zero"])
    def test_unknown_arm_register(self, token):
        e = err(arm_parse_reg, token)
        assert "Unknown ARM register" in str(e)
        assert token in str(e)

    def test_bad_register_inside_a_real_instruction(self):
        e = err(assemble, "arm", "ADD W1, X1, X2")
        assert "Unknown ARM register" in str(e)


class TestARMUnknownMnemonic:

    @pytest.mark.parametrize("text", [
        "BOGUS X1, X2", "MUL X1, X2, X3", "LDP X1, X2, [X0]",
        "ADDS X1, X2, X3", "MOVK X1, #1", "LDUR X1, [X0, #-8]",
    ])
    def test_unsupported_mnemonic_raises(self, text):
        e = err(assemble, "arm", text)
        assert "Unknown ARM mnemonic" in str(e)


class TestARMOperandArity:

    @pytest.mark.parametrize("text,expect", [
        ("ADD X1, X2", "ADD requires 3 register operands, got 2"),
        ("SUB X1", "SUB requires 3 register operands, got 1"),
        ("SUBS X1, X2", "SUBS requires 3 register operands, got 2"),
        ("AND X1, X2", "AND requires 3 register operands, got 2"),
        ("ORR X1", "ORR requires 3 register operands, got 1"),
        ("EOR X1, X2, X3, X4", "EOR requires 3 register operands, got 4"),
        ("MOVZ X1", "MOVZ requires 2 operands, got 1"),
        ("MOV X1", "MOVZ requires 2 operands, got 1"),
        ("B", "B requires 1 operand, got 0"),
        ("BL", "BL requires 1 operand, got 0"),
        ("CBZ X1", "CBZ requires 2 operands, got 1"),
        ("CBNZ X1", "CBNZ requires 2 operands, got 1"),
        ("B.EQ", "B.EQ requires 1 operand, got 0"),
    ])
    def test_wrong_operand_count(self, text, expect):
        assert expect in str(err(assemble, "arm", text))

    def test_cmp_expands_to_subs_and_reports_as_subs(self):
        # CMP is an alias for `SUBS XZR, ...`, so a 1-operand CMP surfaces as
        # a 2-operand SUBS complaint (arm.py:147).
        e = err(assemble, "arm", "CMP X1")
        assert "SUBS requires 3 register operands, got 2" in str(e)

    def test_add_with_two_operands_is_treated_as_register_form(self):
        # arm.py:156 — only a 3-operand '#'-suffixed form takes the imm path.
        e = err(assemble, "arm", "ADD X1, #5")
        assert "ADD requires 3 register operands, got 2" in str(e)

    def test_the_immediate_forms_own_arity_check_is_unreachable_via_assemble(self):
        """arm.py:214 is dead code — pinned so a dispatch change is noticed.

        `_encode()` only routes to `_encode_dp_imm_addsub` when
        ``len(operands) == 3``, so its own "imm requires 3 operands" guard can
        never fire through the public API. It still works when called directly.
        """
        asm = ARMAssembler()
        e = err(asm._encode_dp_imm_addsub, "ADD", ["X1", "#5"])
        assert "ADD imm requires 3 operands, got 2" in str(e)
        # ...and no source string can reach it:
        assert "register operands" in str(err(assemble, "arm", "ADD X1, #5"))

    def test_b_with_extra_operand(self):
        assert "B requires 1 operand, got 2" in str(err(assemble, "arm", "B l1, l2"))


class TestARMImmediateAddSub:
    """`ADD/SUB Xd, Xn, #imm12` — non-negative, 12-bit, or <<12 shifted."""

    def test_boundaries_accepted(self):
        assert assemble("arm", "ADD X1, X2, #0")
        assert assemble("arm", "ADD X1, X2, #4095")

    def test_shifted_by_twelve_is_accepted(self):
        # 0x1000 has a zero low nibble-triple, so sh=1 with imm12=1.
        word = assemble("arm", "ADD X1, X2, #4096")[0]
        assert (word >> 22) & 1 == 1          # sh bit set
        assert (word >> 10) & 0xFFF == 1

    @pytest.mark.parametrize("mn,other", [("ADD", "SUB"), ("SUB", "ADD")])
    def test_negative_immediate_is_rejected_with_a_hint(self, mn, other):
        e = err(assemble, "arm", f"{mn} X1, X2, #-1")
        assert f"{mn} immediate -1 must be non-negative" in str(e)
        assert f"use {other} for negatives" in str(e)

    @pytest.mark.parametrize("imm", [4097, 0x1001, 0x1000000, 1 << 36])
    def test_unrepresentable_immediate_is_rejected(self, imm):
        e = err(assemble, "arm", f"ADD X1, X2, #{imm}")
        assert "too large for ADD/SUB immediate" in str(e)
        assert str(imm) in str(e)

    def test_largest_shifted_immediate_accepted(self):
        assert assemble("arm", "ADD X1, X2, #0xFFF000")


class TestARMMovz:
    """`MOVZ Xd, #imm16` — non-negative, 16-bit, or one 16-bit shift (hw=1)."""

    def test_boundaries_accepted(self):
        assert assemble("arm", "MOVZ X1, #0")
        assert assemble("arm", "MOVZ X1, #0xFFFF")

    def test_negative_immediate_rejected(self):
        e = err(assemble, "arm", "MOVZ X1, #-1")
        assert "MOVZ immediate -1 must be non-negative" in str(e)

    def test_hw_shift_form_accepted(self):
        word = assemble("arm", "MOVZ X1, #0x10000")[0]
        assert (word >> 21) & 0x3 == 1        # hw == 1
        assert (word >> 5) & 0xFFFF == 1

    @pytest.mark.parametrize("imm", [0x10001, 0xFFFFF, 0x12345])
    def test_unrepresentable_immediate_rejected(self, imm):
        e = err(assemble, "arm", f"MOVZ X1, #{imm}")
        assert "MOVZ immediate too large" in str(e)
        assert str(imm) in str(e)

    def test_bits_above_32_are_silently_dropped(self):
        # arm.py:257-259 masks (val >> 16) & 0xFFFF, so anything above bit 31
        # vanishes rather than raising. Pinned as observed behaviour.
        assert assemble("arm", "MOVZ X1, #0x1FFFF0000") == \
            assemble("arm", "MOVZ X1, #0xFFFF0000")


class TestARMLoadStore:
    """`LDR/STR Xt, [Xn{, #off}]` — bracketed, 8-scaled, non-negative, imm12."""

    def test_no_offset_form_accepted(self):
        assert assemble("arm", "LDR X1, [X0]")

    def test_offset_boundaries_accepted(self):
        assert assemble("arm", "LDR X1, [X0, #0]")
        assert assemble("arm", "LDR X1, [X0, #32760]")     # imm12 == 4095

    @pytest.mark.parametrize("mn", ["LDR", "STR"])
    def test_too_few_operands(self, mn):
        e = err(assemble, "arm", f"{mn} X1")
        assert f"{mn} requires at least 2 operands" in str(e)

    @pytest.mark.parametrize("mn", ["LDR", "STR"])
    def test_unbracketed_memory_operand(self, mn):
        e = err(assemble, "arm", f"{mn} X1, X2")
        assert "Expected memory operand in brackets" in str(e)
        assert "'X2'" in str(e)

    @pytest.mark.parametrize("off", [1, 4, 7, 12])
    def test_unscaled_offset_rejected(self, off):
        e = err(assemble, "arm", f"LDR X1, [X0, #{off}]")
        assert f"Offset {off} must be a multiple of 8" in str(e)

    def test_negative_offset_rejected_with_ldur_hint(self):
        e = err(assemble, "arm", "LDR X1, [X0, #-8]")
        assert "must be non-negative" in str(e)
        assert "LDUR" in str(e)

    def test_offset_beyond_imm12_rejected(self):
        e = err(assemble, "arm", "LDR X1, [X0, #32768]")
        assert "LDR/STR scaled offset 4096 out of range" in str(e)
        assert "unsigned 12-bit" in str(e)

    def test_bad_base_register_in_brackets(self):
        e = err(assemble, "arm", "LDR X1, [W0, #0]")
        assert "Unknown ARM register" in str(e)

    def test_extra_operands_after_the_bracket_are_ignored(self):
        # arm.py:273 only checks `len(ops) < 2`, so trailing junk is dropped.
        assert assemble("arm", "STR X1, [X0, #8], X3") == assemble("arm", "STR X1, [X0, #8]")


class TestARMUnclosedBracket:
    """REGRESSION PINS (fixed 2026-08-08).

    `_split_instruction` used `rest.index(']')` with no guard, so a stray '['
    raised a raw `ValueError: substring not found`. Because
    `_split_instruction` was ALSO called outside pass-2's try/except, the
    wrapper could not rescue it — `/assemble` caught broad Exception and
    returned 400, but the user saw "substring not found" for a simple typo.

    Fixed two ways: arm.py raises a real AssemblerError naming the operand,
    and base.py moved `_split_instruction` inside the try so tokenizer errors
    get the same line attribution as encoder errors.
    """

    @pytest.mark.parametrize("text", ["LDR X1, [X0", "STR X1, [X0\nNOP\n"])
    def test_unclosed_bracket_raises_an_assembler_error(self, text):
        e = err(assemble, "arm", text)
        assert isinstance(e, AssemblerError)
        assert "Unclosed '['" in str(e)
        assert "substring not found" not in str(e)

    def test_unclosed_bracket_names_the_offending_line(self):
        e = err(assemble, "arm", "MOVZ X1, #1\nMOVZ X2, #2\nLDR X3, [X0\n")
        assert e.line_number == 3
        assert e.line_text == "LDR X3, [X0"

    def test_closed_brackets_still_assemble(self):
        """Guard must not reject valid bracketed operands."""
        for text in ("LDR X1, [X0]", "LDR X1, [X0, #8]", "STR X1, [X0, #16]"):
            assemble("arm", text)      # must not raise

    def test_unclosed_bracket_should_be_an_assembler_error(self):
        err(assemble, "arm", "LDR X1, [X0")


class TestARMBranches:

    @pytest.mark.parametrize("mn", ["B", "BL"])
    def test_branch_offset_must_be_word_aligned(self, mn):
        e = err(assemble, "arm", f"{mn} +2")
        assert "Branch offset 2 must be a multiple of 4" in str(e)

    @pytest.mark.parametrize("mn", ["B", "BL"])
    def test_aligned_offsets_accepted(self, mn):
        assert assemble("arm", f"{mn} +4")
        assert assemble("arm", f"{mn} -8")

    @pytest.mark.parametrize("cond", ["EQ", "NE", "LT", "GE", "GT", "LE"])
    def test_supported_conditions_assemble(self, cond):
        assert assemble("arm", f"B.{cond} +8")

    @pytest.mark.parametrize("cond", ["HS", "CS", "LO", "CC", "MI", "PL", "HI", "LS"])
    def test_decodable_but_unsupported_conditions_are_rejected(self, cond):
        # arm.py:347 — these map to real A64 condition codes but the branch
        # unit would resolve them as "never taken", so they are rejected at
        # assemble time rather than silently never branching.
        e = err(assemble, "arm", f"B.{cond} +8")
        assert f"B.{cond} is not supported by the simulator's branch unit" in str(e)
        assert "EQ, NE, LT, GE, GT, LE" in str(e)

    @pytest.mark.parametrize("cond", ["ZZ", "XX", "AL", "NV", "Q"])
    def test_unknown_condition_code(self, cond):
        e = err(assemble, "arm", f"B.{cond} +8")
        assert f"Unknown condition code '{cond}'" in str(e)

    def test_bcond_offset_must_be_word_aligned(self):
        e = err(assemble, "arm", "B.EQ +2")
        assert "Branch offset 2 must be a multiple of 4" in str(e)

    @pytest.mark.parametrize("mn", ["CBZ", "CBNZ"])
    def test_cbz_offset_must_be_word_aligned(self, mn):
        e = err(assemble, "arm", f"{mn} X1, +2")
        assert "Branch offset 2 must be a multiple of 4" in str(e)

    @pytest.mark.parametrize("mn", ["CBZ", "CBNZ"])
    def test_cbz_bad_register(self, mn):
        e = err(assemble, "arm", f"{mn} W1, +4")
        assert "Unknown ARM register" in str(e)

    def test_branch_offsets_are_masked_not_range_checked(self):
        # Observed: B/BL/B.cond/CBZ mask the shifted offset (imm26/imm19) with
        # no _require_signed, so a wildly out-of-range target wraps instead of
        # raising. Only alignment is enforced.
        assert assemble("arm", "B 1073741824") == assemble("arm", "B 0")
        assert assemble("arm", "B -1073741824") == assemble("arm", "B 0")
        assert assemble("arm", "B.EQ 2097152") == assemble("arm", "B.EQ 0")


class TestARMRet:

    def test_bare_ret_defaults_to_x30(self):
        assert assemble("arm", "RET") == [0xD65F03C0]

    def test_ret_with_explicit_register(self):
        # arm.py:124 — the `if operands:` branch.
        assert assemble("arm", "RET X5") == [0xD65F00A0]
        assert assemble("arm", "RET XZR") == [0xD65F03E0]

    def test_ret_with_a_bad_register(self):
        e = err(assemble, "arm", "RET W5")
        assert "Unknown ARM register" in str(e)


# ════════════════════════════════════════════════════════════════════════════
# x86.py
# ════════════════════════════════════════════════════════════════════════════

class TestX86Helpers:

    @pytest.mark.parametrize("token,expected", [
        ("EAX", 0), ("ECX", 1), ("EDX", 2), ("EBX", 3),
        ("ESP", 4), ("EBP", 5), ("ESI", 6), ("EDI", 7),
        ("eax", 0), (" edi ", 7),
    ])
    def test_accepted_registers(self, token, expected):
        assert x86_parse_reg(token) == expected

    @pytest.mark.parametrize("token", ["RAX", "R8", "AX", "AL", "XMM0", "x0", ""])
    def test_unknown_register(self, token):
        e = err(x86_parse_reg, token)
        assert "Unknown x86 register" in str(e)

    def test_is_reg_and_is_mem(self):
        assert _is_reg("EAX") and _is_reg(" ebx ")
        assert not _is_reg("[EAX]") and not _is_reg("42")
        assert _is_mem("[EAX]") and _is_mem("[EBX+4]")
        assert not _is_mem("EAX")

    def test_modrm_packs_and_masks(self):
        assert _modrm(0b11, 1, 0) == 0xC8
        assert _modrm(0b00, 0, 3) == 0x03
        assert _modrm(0b111, 0b1111, 0b1111) == 0xFF     # each field masked

    def test_imm8_boundaries(self):
        assert _imm8_byte(127) == 0x7F
        assert _imm8_byte(-128) == 0x80
        assert _imm8_byte(0) == 0x00
        assert _imm8_byte(-1) == 0xFF

    @pytest.mark.parametrize("value", [128, 255, -129, 300, -300])
    def test_imm8_out_of_range(self, value):
        e = err(_imm8_byte, value)
        assert f"value {value} out of range" in str(e)
        assert "[-128..127]" in str(e)
        assert "disp8/imm8/rel8" in str(e)


class TestX86UnknownMnemonic:

    @pytest.mark.parametrize("text", [
        "BOGUS EAX, ECX", "IMUL EAX, ECX", "LEA EAX, [EBX]",
        "INC EAX", "SHL EAX, 1", "TEST EAX, EAX", "MOVZX EAX, AL",
    ])
    def test_unsupported_mnemonic_raises(self, text):
        e = err(assemble, "x86", text)
        assert "Unknown x86 mnemonic" in str(e)


class TestX86OperandArity:

    @pytest.mark.parametrize("text,expect", [
        ("PUSH", "PUSH requires 1 operand, got 0"),
        ("PUSH EAX, ECX", "PUSH requires 1 operand, got 2"),
        ("POP", "POP requires 1 operand, got 0"),
        ("POP EAX, ECX", "POP requires 1 operand, got 2"),
        ("JMP", "JMP requires 1 operand, got 0"),
        ("JMP l1, l2", "JMP requires 1 operand, got 2"),
        ("JE", "JE requires 1 operand, got 0"),
        ("JNE l1, l2", "JNE requires 1 operand, got 2"),
        ("CALL", "CALL requires 1 operand, got 0"),
        ("CALL l1, l2", "CALL requires 1 operand, got 2"),
        ("MOV EAX", "MOV requires 2 operands, got 1"),
        ("MOV EAX, ECX, EDX", "MOV requires 2 operands, got 3"),
        ("ADD EAX", "ADD requires 2 operands, got 1"),
        ("SUB EAX, ECX, EDX", "SUB requires 2 operands, got 3"),
    ])
    def test_wrong_operand_count(self, text, expect):
        assert expect in str(err(assemble, "x86", text))

    @pytest.mark.parametrize("text", ["PUSH R8", "POP RAX"])
    def test_bad_register_operand(self, text):
        e = err(assemble, "x86", text)
        assert "Unknown x86 register" in str(e)

    @pytest.mark.parametrize("text,expect", [
        ("MOV R0, 5", "Invalid MOV operands: R0, 5"),
        ("ADD RAX, RCX", "Invalid ADD operands: RAX, RCX"),
    ])
    def test_unrecognised_register_falls_through_to_the_form_error(self, text, expect):
        # MOV/ALU dispatch on _is_reg()/_is_mem() BEFORE calling _parse_reg,
        # so an unknown register name reads as "no matching operand form"
        # rather than "unknown register".
        assert expect in str(err(assemble, "x86", text))


class TestX86OperandForms:
    """Combinations the encoder has no form for."""

    def test_memory_to_memory_mov(self):
        e = err(assemble, "x86", "MOV [EAX], [EBX]")
        assert "Invalid MOV operands" in str(e)

    def test_immediate_destination_mov(self):
        e = err(assemble, "x86", "MOV 5, EAX")
        assert "Invalid MOV operands: 5, EAX" in str(e)

    @pytest.mark.parametrize("mn", ["ADD", "SUB", "AND", "OR", "XOR", "CMP"])
    def test_alu_with_a_memory_destination(self, mn):
        e = err(assemble, "x86", f"{mn} [EAX], ECX")
        assert f"Invalid {mn} operands" in str(e)

    @pytest.mark.parametrize("mn", ["ADD", "SUB", "AND", "OR", "XOR", "CMP"])
    def test_alu_with_a_memory_source(self, mn):
        # There is no ALU-with-memory form; a bracketed source reaches the
        # imm8 path and dies parsing "[ECX]" as a number.
        e = err(assemble, "x86", f"{mn} EAX, [ECX]")
        assert e.line_number == 1
        assert "invalid literal" in str(e)
        assert "[ECX]" in str(e)

    def test_bad_register_inside_a_memory_operand(self):
        e = err(assemble, "x86", "MOV EAX, [R9]")
        assert "Unknown x86 register" in str(e)

    def test_bad_displacement_inside_a_memory_operand(self):
        e = err(assemble, "x86", "MOV EAX, [EBX+zz]")
        assert e.line_number == 1                # wrapped ValueError
        assert "invalid literal" in str(e)


class TestX86ImmediateRanges:
    """imm32 wraps; every 8-bit field (imm8 / disp8 / rel8) is range-checked."""

    def test_mov_imm32_accepts_the_full_word(self):
        assert assemble("x86", "MOV EAX, 0x7FFFFFFF")[1:] == [0xFF, 0xFF, 0xFF, 0x7F]
        assert assemble("x86", "MOV EAX, -1")[1:] == [0xFF, 0xFF, 0xFF, 0xFF]

    def test_mov_imm32_silently_wraps_above_32_bits(self):
        # _imm32_bytes masks with 0xFFFFFFFF; no range guard. Pinned.
        assert assemble("x86", "MOV EAX, 0x100000000") == assemble("x86", "MOV EAX, 0")

    @pytest.mark.parametrize("mn", ["ADD", "SUB", "AND", "OR", "XOR", "CMP"])
    def test_alu_imm8_boundaries(self, mn):
        assert assemble("x86", f"{mn} EAX, 127")[2] == 0x7F
        assert assemble("x86", f"{mn} EAX, -128")[2] == 0x80

    @pytest.mark.parametrize("mn", ["ADD", "SUB", "AND", "OR", "XOR", "CMP"])
    @pytest.mark.parametrize("imm", [128, 255, -129])
    def test_alu_imm8_out_of_range(self, mn, imm):
        e = err(assemble, "x86", f"{mn} EAX, {imm}")
        assert "out of range for an 8-bit sign-extended field" in str(e)

    def test_unsigned_looking_byte_masks_must_be_written_signed(self):
        # 0xFF is out of range as a POSITIVE imm8; the same bit pattern is
        # reachable via -1 (documented in test_assemblers.py::test_xor_imm8).
        err(assemble, "x86", "XOR EAX, 0xFF")
        assert assemble("x86", "XOR EAX, -1") == [0x83, 0xF0, 0xFF]

    def test_disp8_boundaries(self):
        assert assemble("x86", "MOV EAX, [EBX+127]")[2] == 0x7F
        assert assemble("x86", "MOV EAX, [EBX-128]")[2] == 0x80

    @pytest.mark.parametrize("text", [
        "MOV EAX, [EBX+128]", "MOV EAX, [EBX-129]",
        "MOV [EBX+200], EAX", "MOV [EBX-200], EAX",
    ])
    def test_disp8_out_of_range(self, text):
        e = err(assemble, "x86", text)
        assert "out of range for an 8-bit sign-extended field" in str(e)


class TestX86NumericDisplacementsStayRel8:
    """A NUMERIC jump displacement is an explicit byte offset — never relaxed."""

    def test_short_numeric_jmp_encodes_rel8(self):
        assert assemble("x86", "JMP +4\n" + "NOP\n" * 4)[:2] == [0xEB, 0x02]

    def test_short_numeric_jcc_encodes_rel8(self):
        assert assemble("x86", "JE +4\n" + "NOP\n" * 4)[:2] == [0x74, 0x02]

    def test_backward_numeric_jmp_encodes_rel8(self):
        assert assemble("x86", "NOP\nJMP -1\n")[1:3] == [0xEB, 0xFD]

    @pytest.mark.parametrize("text", ["JMP +300\n", "JMP -200\n", "JE +300\n", "JG -500\n"])
    def test_out_of_range_numeric_displacement_still_raises(self, text):
        e = err(assemble, "x86", text)
        assert "out of range for an 8-bit sign-extended field" in str(e)

    def test_numeric_boundary_values(self):
        # rel8 is measured from the NEXT instruction (pc+2), so +129 is the
        # largest encodable forward numeric displacement.
        assert assemble("x86", "JMP +129\n")[:2] == [0xEB, 0x7F]
        err(assemble, "x86", "JMP +130\n")
        assert assemble("x86", "JMP -126\n")[:2] == [0xEB, 0x80]
        err(assemble, "x86", "JMP -127\n")


class TestX86BranchRelaxationBoundary:
    """Label jumps start rel8 and widen to rel32 EXACTLY at the ±127 edge.

    Displacements are relative to the NEXT instruction, so for a forward jump
    over N one-byte NOPs the rel8 value is N; N=127 fits, N=128 does not.
    """

    @staticmethod
    def _forward(mn, nops):
        return assemble("x86", f"{mn} skip\n" + "NOP\n" * nops + "skip:\nNOP\n")

    @staticmethod
    def _backward(mn, nops):
        return assemble("x86", "loop:\n" + "NOP\n" * nops + f"{mn} loop\nNOP\n")

    def test_forward_jmp_at_the_edge_stays_rel8(self):
        prog = self._forward("JMP", 127)
        assert prog[:2] == [0xEB, 0x7F]
        assert len(prog) == 2 + 127 + 1

    def test_forward_jmp_one_past_the_edge_widens(self):
        prog = self._forward("JMP", 128)
        assert prog[:5] == [0xE9, 0x80, 0x00, 0x00, 0x00]
        assert len(prog) == 5 + 128 + 1
        # disp32 counts from pc+5, and widening pushed the label 3 bytes out:
        # (5 + 128) - 5 == 128.
        assert int.from_bytes(bytes(prog[1:5]), "little") == 128

    def test_forward_jcc_at_the_edge_stays_rel8(self):
        prog = self._forward("JE", 127)
        assert prog[:2] == [0x74, 0x7F]

    def test_forward_jcc_one_past_the_edge_widens_to_0f_8x(self):
        prog = self._forward("JE", 128)
        assert prog[:3] == [0x0F, 0x84, 0x80]       # 0x74 + 0x10 == 0x84
        assert prog[3:6] == [0x00, 0x00, 0x00]
        assert len(prog) == 6 + 128 + 1

    @pytest.mark.parametrize("mn,rel8,rel32", [
        ("JE", 0x74, 0x84), ("JNE", 0x75, 0x85), ("JL", 0x7C, 0x8C),
        ("JGE", 0x7D, 0x8D), ("JLE", 0x7E, 0x8E), ("JG", 0x7F, 0x8F),
    ])
    def test_every_jcc_widens_to_its_two_byte_opcode(self, mn, rel8, rel32):
        assert self._forward(mn, 10)[0] == rel8
        wide = self._forward(mn, 200)
        assert wide[:2] == [0x0F, rel32]

    def test_backward_jmp_at_the_edge_stays_rel8(self):
        # rel8 = -(nops + 2); nops=126 -> -128, the most negative rel8.
        prog = self._backward("JMP", 126)
        assert prog[126:128] == [0xEB, 0x80]

    def test_backward_jmp_one_past_the_edge_widens(self):
        prog = self._backward("JMP", 127)
        assert prog[127] == 0xE9
        disp = int.from_bytes(bytes(prog[128:132]), "little", signed=True)
        assert disp == -(127 + 5)

    def test_backward_jcc_one_past_the_edge_widens(self):
        prog = self._backward("JNE", 127)
        assert prog[127:129] == [0x0F, 0x85]
        disp = int.from_bytes(bytes(prog[129:133]), "little", signed=True)
        assert disp == -(127 + 6)

    def test_short_jumps_are_byte_identical_to_the_unrelaxed_encoding(self):
        # Regression guard: relaxation must not change ANY short encoding.
        assert assemble("x86", "JMP skip\nNOP\nskip:\nNOP\n")[:3] == [0xEB, 0x01, 0x90]
        assert assemble("x86", "CMP EAX, 0\nJE skip\nNOP\nskip:\nNOP\n")[3:5] == \
            [0x74, 0x01]

    def test_widening_is_monotonic_and_converges(self):
        # A jump pushed out of range by ANOTHER jump's widening must also widen
        # (the fixed-point loop), and the result must be stable.
        text = "JMP b\nJMP a\n" + "NOP\n" * 125 + "b:\n" + "NOP\n" * 200 + "a:\nNOP\n"
        prog = assemble("x86", text)
        assert prog[0] == 0xE9                       # first jump widened too
        assert prog[5] == 0xE9                       # second jump widened
        assert assemble("x86", text) == prog         # deterministic

    def test_relaxation_state_does_not_leak_between_assembles(self):
        # _wide_jumps is reset at the top of assemble(); a long program must
        # not make a subsequent short one emit rel32.
        asm = X86Assembler()
        asm.assemble("JMP far\n" + "NOP\n" * 300 + "far:\nNOP\n")
        assert asm.assemble("JMP s\nNOP\ns:\nNOP\n") == [0xEB, 0x01, 0x90, 0x90]


class TestX86SizeEstimation:
    """Pass-1 sizing must agree with pass-2 encoding or every label drifts."""

    @pytest.mark.parametrize("line,size", [
        ("NOP", 1), ("RET", 1), ("PUSH EAX", 1), ("POP EBX", 1),
        ("JMP loop", 2), ("JE loop", 2), ("JG loop", 2),
        ("CALL func", 5),
        ("MOV EAX, 42", 5), ("MOV EAX, ECX", 2),
        ("MOV EAX, [EBX]", 2), ("MOV [EBX], EAX", 2),
        ("MOV EAX, [EBX+4]", 3), ("MOV [EBX + 4], EAX", 3),
        ("MOV EAX, [EBX-4]", 3),
        ("ADD EAX, ECX", 2), ("ADD EAX, 5", 3),
        ("ADD EAX", 2),                 # malformed -> fallback
        ("BOGUS EAX", 2),               # unknown  -> fallback
        ("MOV EAX", 5),                 # malformed MOV -> default guess
    ])
    def test_estimate_matches_the_documented_table(self, line, size):
        assert X86Assembler()._estimate_instruction_size(line) == size

    @pytest.mark.parametrize("line", [
        "MOV [EBX + 4], EAX", "MOV EAX, [EBX + 4]", "MOV [EBX+0], EAX",
        "ADD EAX, 5", "MOV EAX, 0x11223344", "PUSH ESI", "CALL f",
    ])
    def test_estimate_equals_the_encoded_length(self, line):
        asm = X86Assembler()
        text = line + "\nf:\nNOP\n"
        encoded = asm.assemble(text)
        # total encoded length minus the trailing NOP
        assert asm._estimate_instruction_size(line) == len(encoded) - 1

    def test_spaced_memory_operand_is_sized_as_disp8(self):
        # Regression (docstring in x86.py:144): a naive whitespace split would
        # size '[EBX + 4]' as 2 bytes and every later label would drift.
        prog = assemble("x86", "MOV [EBX + 4], EAX\nJMP done\nNOP\ndone:\nNOP\n")
        assert prog[:3] == [0x89, 0x43, 0x04]
        assert prog[3:5] == [0xEB, 0x01]

    def test_empty_line_estimation_is_never_reached_via_assemble(self):
        # _estimate_instruction_size("") indexes parts[0] on an empty list, but
        # _preprocess drops blank lines so assemble() can never feed it one.
        with pytest.raises(IndexError):
            X86Assembler()._estimate_instruction_size("")
        assert assemble("x86", "\n\n  \n") == []

    def test_the_empty_parts_guard_is_dead_code(self):
        """x86.py:153-154 (`if not parts: return 0`) cannot fire.

        `_split_instruction` either returns a non-empty list or raises
        IndexError on the empty string, so the guard is unreachable — pinned
        so a future splitter that CAN return [] is a noticed change.
        """
        asm = X86Assembler()
        for line in ("NOP", "MOV EAX, 1", "  spaced  "):
            assert asm._split_instruction(line)
        with pytest.raises(IndexError):
            asm._split_instruction("")

    def test_pc_increment_is_one_byte_for_a_byte_addressed_isa(self):
        # x86.py:108 — a fallback only; real sizing goes through
        # _estimate_instruction_size (variable-length encoding).
        assert X86Assembler()._pc_increment() == 1
        assert RISCVAssembler()._pc_increment() == 4
        assert ARMAssembler()._pc_increment() == 4


# ════════════════════════════════════════════════════════════════════════════
# Dispatch (`sim/assembler/__init__.py`)
# ════════════════════════════════════════════════════════════════════════════

class TestDispatchErrors:

    @pytest.mark.parametrize("isa", ["mips", "sparc", "", "RISC-V", "x64"])
    def test_unknown_isa_raises_valueerror_listing_the_supported_set(self, isa):
        with pytest.raises(ValueError) as ei:
            assemble(isa, "NOP")
        assert "riscv" in str(ei.value)
        assert "arm" in str(ei.value)
        assert "x86" in str(ei.value)

    @pytest.mark.parametrize("isa", ["riscv", "RISCV", "  RiscV  ", "ARM", "X86"])
    def test_isa_name_is_case_and_whitespace_insensitive(self, isa):
        assert assemble(isa, "NOP") != []

    def test_dispatch_creates_a_fresh_assembler_per_call(self):
        # No shared state between calls (relevant for x86's _wide_jumps).
        long_prog = "JMP far\n" + "NOP\n" * 300 + "far:\nNOP\n"
        assemble("x86", long_prog)
        assert assemble("x86", "JMP s\nNOP\ns:\nNOP\n") == [0xEB, 0x01, 0x90, 0x90]


class TestErrorsAreNeverSilentTruncation:
    """A value that does not fit must RAISE, not encode as something else.

    This is the whole point of the `_require_*` guards: silently masking an
    out-of-range immediate produces a valid-looking instruction that computes
    the wrong thing at runtime.
    """

    @pytest.mark.parametrize("isa,bad,good", [
        ("riscv", "ADDI x1, x0, 2048", "ADDI x1, x0, 2047"),
        ("riscv", "LW x1, 2048(x0)", "LW x1, 2047(x0)"),
        ("riscv", "SLLI x1, x0, 32", "SLLI x1, x0, 31"),
        ("riscv", "BEQ x1, x2, +4096", "BEQ x1, x2, +4094"),
        ("riscv", "JAL x1, +1048576", "JAL x1, +1048574"),
        ("arm",   "ADD X1, X2, #4097", "ADD X1, X2, #4095"),
        ("arm",   "MOVZ X1, #0x10001", "MOVZ X1, #0xFFFF"),
        ("arm",   "LDR X1, [X0, #32768]", "LDR X1, [X0, #32760]"),
        ("x86",   "ADD EAX, 128", "ADD EAX, 127"),
        ("x86",   "MOV EAX, [EBX+128]", "MOV EAX, [EBX+127]"),
    ])
    def test_one_past_the_limit_raises_while_the_limit_encodes(self, isa, bad, good):
        err(assemble, isa, bad)
        assert assemble(isa, good)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
