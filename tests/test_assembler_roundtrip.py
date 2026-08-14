"""
ASSEMBLER <-> DECODER round-trip + cheatsheet-vs-assembler agreement.

WHY THIS FILE EXISTS
--------------------
`tests/test_examples.py` already walks `programs/**/*.asm` and asserts each
file "assembles to a non-empty list".  That is a crash test, not a
correctness test: an assembler that emitted zeros for everything would pass.
And `tests/test_assemblers.py::TestCheatsheet` only checks that the cheatsheet
DICTS have the right KEYS — never that the syntax the UI advertises actually
assembles.

Here every example program and every cheatsheet entry is pushed all the way
through the pipeline the simulator itself uses:

  1. TYPE / SHAPE   — riscv+arm produce 32-bit WORDS (one per instruction
     line), x86 produces a flat BYTE list; nothing is out of range.
  2. DECODER ROUND-TRIP — feed each encoded instruction to the ISA's real
     decoder component and assert the operands come back out.  This closes the
     loop: `assemble()` and the decoder are written independently, so an
     agreed-upon field placement is strong evidence both are right.
  3. x86 SELF-CONSISTENCY — walk the byte stream with the decoder's own
     `instr_len` and require it to land EXACTLY on the end of the program.
     Any encoder/decoder length disagreement desynchronises the whole stream.
  4. CHEATSHEET — every `example` in `sim/assembler/cheatsheet.py` must
     assemble.  A cheatsheet entry the assembler rejects is a user-facing bug:
     the UI tells the user to type something that then errors.

Nothing here builds a CPU or runs a simulation — it is all pure encode/decode.

Run from repo root:  python3 -m pytest tests/test_assembler_roundtrip.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.assembler import assemble
from sim.assembler.arm import ARMAssembler
from sim.assembler.base import AssemblerError
from sim.assembler.cheatsheet import get_cheatsheet
from sim.assembler.riscv import RISCVAssembler
from sim.assembler.x86 import X86Assembler
from sim.isa.arm.decoder_component import ARMDecoder
from sim.isa.riscv.decoder_component import RISCVDecoder
from sim.isa.x86.decoder_component import X86Decoder
from sim.core.signals import AluOp, BranchCond, WBSel

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROGRAMS_DIR = os.path.join(ROOT, "programs")

ISAS = ("riscv", "arm", "x86")
ASSEMBLER_CLASSES = {
    "riscv": RISCVAssembler,
    "arm": ARMAssembler,
    "x86": X86Assembler,
}

# NOP encodings — skipped when checking "the decoder recognised this".
RISCV_NOP = 0x00000013      # ADDI x0, x0, 0
ARM_NOP = 0xD503201F

# Labels the cheatsheet examples reference (`B loop`, `JE done`, ...).
CHEATSHEET_LABELS = ("loop", "func", "done", "neg", "pos", "big", "small")


# ── program discovery ────────────────────────────────────────────────────────

def _discover_programs():
    """[(isa, relpath, abspath), ...] for every programs/<isa>/**/*.asm."""
    found = []
    for isa in ISAS:
        isa_dir = os.path.join(PROGRAMS_DIR, isa)
        if not os.path.isdir(isa_dir):
            continue
        for dirpath, _dirs, files in os.walk(isa_dir):
            for fname in sorted(files):
                if fname.endswith(".asm"):
                    path = os.path.join(dirpath, fname)
                    found.append((isa, os.path.relpath(path, PROGRAMS_DIR), path))
    return sorted(found)


PROGRAMS = _discover_programs()
PROGRAM_IDS = [rel for _isa, rel, _p in PROGRAMS]


def _read(path):
    with open(path, "r") as fh:
        return fh.read()


def test_program_corpus_is_not_empty():
    """Guard: if discovery breaks, every parametrized test below vanishes."""
    assert len(PROGRAMS) >= 40, f"only found {len(PROGRAMS)} example programs"
    for isa in ISAS:
        assert any(i == isa for i, _r, _p in PROGRAMS), f"no {isa} programs found"


# ════════════════════════════════════════════════════════════════════════════
# 1. Output type / shape
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("isa,rel,path", PROGRAMS, ids=PROGRAM_IDS)
def test_every_example_assembles_to_the_right_shape(isa, rel, path):
    program = assemble(isa, _read(path))
    assert isinstance(program, list), f"{rel}: expected list"
    assert program, f"{rel}: assembled to an empty program"
    assert all(isinstance(v, int) for v in program), f"{rel}: non-int entries"
    if isa == "x86":
        # program_format == "bytes"
        assert all(0 <= v <= 0xFF for v in program), f"{rel}: byte out of range"
    else:
        # program_format == "words"
        assert all(0 <= v <= 0xFFFF_FFFF for v in program), f"{rel}: word out of range"


@pytest.mark.parametrize(
    "isa,rel,path",
    [p for p in PROGRAMS if p[0] != "x86"],
    ids=[r for i, r, _p in PROGRAMS if i != "x86"],
)
def test_fixed_width_isas_emit_exactly_one_word_per_instruction_line(isa, rel, path):
    """riscv/arm are 4-byte fixed width: #words == #non-label source lines.

    This is what makes label arithmetic (`pc += 4`) correct; a mnemonic that
    expanded to two words would silently shift every later branch target.
    """
    asm = ASSEMBLER_CLASSES[isa]()
    text = _read(path)
    _labels, cleaned = asm._pass1(asm._preprocess(text))
    assert len(asm.assemble(text)) == len(cleaned)


@pytest.mark.parametrize(
    "isa,rel,path",
    [p for p in PROGRAMS if p[0] == "x86"],
    ids=[r for i, r, _p in PROGRAMS if i == "x86"],
)
def test_x86_pass1_size_estimate_matches_the_encoded_length(isa, rel, path):
    """Pass-1 sizing must total the real byte count or labels drift.

    x86 is variable-length: `_pass1` lays labels out from
    `_estimate_instruction_size_at`, so if that disagrees with what `_encode`
    actually emits, every branch in the file points at the wrong byte.
    """
    asm = X86Assembler()
    text = _read(path)
    program = asm.assemble(text)        # also populates _wide_jumps
    _labels, cleaned = asm._pass1(asm._preprocess(text))
    estimated = sum(asm._estimate_instruction_size_at(ln, line)
                    for ln, _pc, line in cleaned)
    assert estimated == len(program)


@pytest.mark.parametrize("isa,rel,path", PROGRAMS, ids=PROGRAM_IDS)
def test_assembly_is_deterministic(isa, rel, path):
    """Two runs (and two assembler instances) must agree byte for byte."""
    text = _read(path)
    first = assemble(isa, text)
    assert assemble(isa, text) == first
    assert ASSEMBLER_CLASSES[isa]().assemble(text) == first


# ════════════════════════════════════════════════════════════════════════════
# 2. Decoder round-trip — every example program
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "rel,path",
    [(r, p) for i, r, p in PROGRAMS if i == "riscv"],
    ids=[r for i, r, _p in PROGRAMS if i == "riscv"],
)
def test_riscv_examples_are_recognised_by_the_decoder(rel, path):
    """Every non-NOP word must assert at least one control signal.

    A word the decoder does not understand produces all-zero control (a
    silent no-op at runtime) — exactly the failure that "it assembled" cannot
    catch.
    """
    dec = RISCVDecoder()
    words = assemble("riscv", _read(path))
    for i, word in enumerate(words):
        if word == RISCV_NOP:
            continue
        dec._ports["instr_in"] = word
        dec.evaluate()
        active = any(dec[p] for p in ("reg_write", "mem_read", "mem_write",
                                      "branch", "jal", "jalr"))
        assert active, f"{rel}[{i}] = 0x{word:08X} decodes to no control signals"


@pytest.mark.parametrize(
    "rel,path",
    [(r, p) for i, r, p in PROGRAMS if i == "arm"],
    ids=[r for i, r, _p in PROGRAMS if i == "arm"],
)
def test_arm_examples_are_recognised_by_the_decoder(rel, path):
    dec = ARMDecoder()
    words = assemble("arm", _read(path))
    for i, word in enumerate(words):
        if word == ARM_NOP:
            continue
        dec._ports["instr_in"] = word
        dec.evaluate()
        active = any(dec[p] for p in ("reg_write", "mem_read", "mem_write",
                                      "branch", "jal", "jalr", "sets_flags"))
        assert active, f"{rel}[{i}] = 0x{word:08X} decodes to no control signals"


@pytest.mark.parametrize(
    "rel,path",
    [(r, p) for i, r, p in PROGRAMS if i == "x86"],
    ids=[r for i, r, _p in PROGRAMS if i == "x86"],
)
def test_x86_examples_decode_with_self_consistent_lengths(rel, path):
    """Walk the byte stream by the DECODER's instr_len and land exactly at EOF.

    This is the x86 equivalent of the word-count check: the assembler's
    variable-length encoding and the decoder's length computation are written
    separately, and any disagreement desynchronises fetch for the rest of the
    program (an off-by-one lands mid-instruction and decodes garbage).
    """
    dec = X86Decoder()
    program = assemble("x86", _read(path))
    pc = 0
    seen = 0
    while pc < len(program):
        stream = 0
        for off in range(8):
            if pc + off < len(program):
                stream |= program[pc + off] << (8 * off)
        dec._ports["instr_in"] = stream
        dec._ports["pc_in"] = pc
        dec.evaluate()
        length = dec["instr_len"]
        assert length > 0, f"{rel}: zero-length instruction at byte {pc}"
        pc += length
        seen += 1
        assert seen <= len(program), f"{rel}: decode loop did not terminate"
    assert pc == len(program), (
        f"{rel}: decoder walked to byte {pc} but the program is "
        f"{len(program)} bytes — encoder/decoder length disagreement")
    assert seen > 0


# ════════════════════════════════════════════════════════════════════════════
# 3. Decoder round-trip — operand recovery, per mnemonic
# ════════════════════════════════════════════════════════════════════════════

def _decode_riscv(source):
    dec = RISCVDecoder()
    words = assemble("riscv", source)
    assert len(words) == 1, f"{source!r} assembled to {len(words)} words"
    dec._ports["instr_in"] = words[0]
    dec.evaluate()
    return dec


def _decode_arm(source):
    dec = ARMDecoder()
    words = assemble("arm", source)
    assert len(words) == 1, f"{source!r} assembled to {len(words)} words"
    dec._ports["instr_in"] = words[0]
    dec.evaluate()
    return dec


def _decode_x86(source, pc=0):
    dec = X86Decoder()
    program = assemble("x86", source)
    stream = 0
    for off, byte in enumerate(program[:8]):
        stream |= byte << (8 * off)
    dec._ports["instr_in"] = stream
    dec._ports["pc_in"] = pc
    dec.evaluate()
    return dec, program


class TestRISCVOperandRoundTrip:
    """`ADD x3, x1, x2` -> word -> decoder must say rd=3, rs1=1, rs2=2."""

    @pytest.mark.parametrize("mn", ["ADD", "SUB", "AND", "OR", "XOR",
                                    "SLT", "SLTU", "SLL", "SRL", "SRA"])
    def test_r_type_registers(self, mn):
        dec = _decode_riscv(f"{mn} x7, x21, x13")
        assert (dec["rd"], dec["rs1"], dec["rs2"]) == (7, 21, 13)
        assert dec["reg_write"] == 1
        assert dec["alu_src"] == 0
        assert dec["wb_sel"] == WBSel.ALU

    @pytest.mark.parametrize("mn", ["ADDI", "ANDI", "ORI", "XORI", "SLTI", "SLTIU"])
    @pytest.mark.parametrize("imm", [0, 5, 2047, -1, -2048])
    def test_i_type_immediate_sign_extends(self, mn, imm):
        dec = _decode_riscv(f"{mn} x9, x3, {imm}")
        assert (dec["rd"], dec["rs1"]) == (9, 3)
        assert dec["reg_write"] == 1
        assert dec["alu_src"] == 1
        # imm port is 32-bit unsigned; compare in two's complement.
        assert dec["imm"] == (imm & 0xFFFF_FFFF)

    @pytest.mark.parametrize("mn", ["SLLI", "SRLI", "SRAI"])
    @pytest.mark.parametrize("shamt", [0, 1, 31])
    def test_shift_immediate_shamt_survives(self, mn, shamt):
        dec = _decode_riscv(f"{mn} x9, x3, {shamt}")
        assert (dec["rd"], dec["rs1"]) == (9, 3)
        assert dec["imm"] & 0x1F == shamt
        assert dec["reg_write"] == 1

    @pytest.mark.parametrize("offset", [0, 4, 2047, -4, -2048])
    def test_load_offset_and_registers(self, offset):
        dec = _decode_riscv(f"LW x5, {offset}(x2)")
        assert dec["rd"] == 5
        assert dec["rs1"] == 2
        assert dec["imm"] == (offset & 0xFFFF_FFFF)
        assert (dec["mem_read"], dec["mem_write"], dec["reg_write"]) == (1, 0, 1)
        assert dec["wb_sel"] == WBSel.MEMORY

    @pytest.mark.parametrize("offset", [0, 8, 2047, -8, -2048])
    def test_store_offset_and_registers(self, offset):
        dec = _decode_riscv(f"SW x3, {offset}(x4)")
        assert dec["rs1"] == 4          # base
        assert dec["rs2"] == 3          # data
        assert dec["imm"] == (offset & 0xFFFF_FFFF)
        assert (dec["mem_read"], dec["mem_write"], dec["reg_write"]) == (0, 1, 0)

    @pytest.mark.parametrize("mn,cond", [
        ("BEQ", BranchCond.EQ), ("BNE", BranchCond.NEQ),
        ("BLT", BranchCond.LT), ("BGE", BranchCond.GE),
    ])
    @pytest.mark.parametrize("offset", [0, 8, 4094, -8, -4096])
    def test_branch_offset_and_condition(self, mn, cond, offset):
        dec = _decode_riscv(f"{mn} x1, x2, {offset}")
        assert (dec["rs1"], dec["rs2"]) == (1, 2)
        assert dec["branch"] == 1
        assert dec["branch_cond"] == cond
        assert dec["imm"] == (offset & 0xFFFF_FFFF)

    @pytest.mark.parametrize("mn", ["BLTU", "BGEU"])
    def test_unsigned_branches_round_trip_their_offset(self, mn):
        dec = _decode_riscv(f"{mn} x1, x2, 8")
        assert dec["branch"] == 1
        assert dec["imm"] == 8

    @pytest.mark.parametrize("offset", [0, 8, 1048574, -8, -1048576])
    def test_jal_offset(self, offset):
        dec = _decode_riscv(f"JAL x1, {offset}")
        assert dec["rd"] == 1
        assert dec["jal"] == 1
        assert dec["reg_write"] == 1
        assert dec["wb_sel"] == WBSel.PC4
        assert dec["imm"] == (offset & 0xFFFF_FFFF)

    @pytest.mark.parametrize("imm", [0, 4, 2047, -2048])
    def test_jalr_offset(self, imm):
        dec = _decode_riscv(f"JALR x0, x1, {imm}")
        assert (dec["rd"], dec["rs1"]) == (0, 1)
        assert dec["jalr"] == 1
        assert dec["imm"] == (imm & 0xFFFF_FFFF)

    @pytest.mark.parametrize("mn", ["LUI", "AUIPC"])
    @pytest.mark.parametrize("imm20", [0, 1, 0x12345, 0xFFFFF])
    def test_u_type_immediate_is_shifted_left_twelve(self, mn, imm20):
        dec = _decode_riscv(f"{mn} x1, {imm20}")
        assert dec["rd"] == 1
        assert dec["imm"] == ((imm20 << 12) & 0xFFFF_FFFF)
        assert dec["reg_write"] == 1

    def test_nop_pseudo_instruction_decodes_as_addi_x0_x0_0(self):
        assert assemble("riscv", "NOP") == assemble("riscv", "ADDI x0, x0, 0")
        dec = _decode_riscv("NOP")
        assert (dec["rd"], dec["rs1"], dec["imm"]) == (0, 0, 0)

    def test_abi_register_names_round_trip_to_their_indices(self):
        dec = _decode_riscv("ADD a0, sp, ra")
        assert (dec["rd"], dec["rs1"], dec["rs2"]) == (10, 2, 1)

    def test_every_register_index_round_trips(self):
        for idx in range(32):
            dec = _decode_riscv(f"ADD x{idx}, x0, x0")
            assert dec["rd"] == idx


class TestARMOperandRoundTrip:

    @pytest.mark.parametrize("mn", ["ADD", "SUB", "SUBS"])
    def test_dp_register_arithmetic(self, mn):
        dec = _decode_arm(f"{mn} X7, X21, X13")
        assert (dec["rd"], dec["rs1"], dec["rs2"]) == (7, 21, 13)
        assert dec["reg_write"] == 1
        assert dec["sets_flags"] == (1 if mn == "SUBS" else 0)

    @pytest.mark.parametrize("mn,alu_op", [
        ("AND", AluOp.AND), ("ORR", AluOp.OR), ("EOR", AluOp.XOR),
    ])
    def test_dp_register_logical(self, mn, alu_op):
        dec = _decode_arm(f"{mn} X3, X1, X2")
        assert (dec["rd"], dec["rs1"], dec["rs2"]) == (3, 1, 2)
        assert dec["alu_op"] == alu_op
        assert dec["reg_write"] == 1

    def test_cmp_is_subs_into_xzr(self):
        dec = _decode_arm("CMP X1, X2")
        assert dec["rd"] == 31                 # XZR
        assert (dec["rs1"], dec["rs2"]) == (1, 2)
        assert dec["sets_flags"] == 1
        assert dec["reg_write"] == 0
        assert assemble("arm", "CMP X1, X2") == assemble("arm", "SUBS XZR, X1, X2")

    @pytest.mark.parametrize("mn", ["ADD", "SUB"])
    @pytest.mark.parametrize("imm", [0, 1, 5, 4095])
    def test_dp_immediate(self, mn, imm):
        dec = _decode_arm(f"{mn} X2, X1, #{imm}")
        assert (dec["rd"], dec["rs1"]) == (2, 1)
        assert dec["imm"] == imm
        assert dec["alu_src"] == 1
        assert dec["reg_write"] == 1

    @pytest.mark.parametrize("imm", [0x1000, 0x2000, 0xFFF000])
    def test_dp_immediate_shifted_by_twelve(self, imm):
        dec = _decode_arm(f"ADD X2, X1, #{imm}")
        assert dec["imm"] == imm               # decoder re-applies the <<12

    @pytest.mark.parametrize("imm", [0, 1, 10, 42, 0xFFFF])
    def test_movz_immediate(self, imm):
        dec = _decode_arm(f"MOVZ X1, #{imm}")
        assert dec["rd"] == 1
        assert dec["rs1"] == 31                # XZR: 0 + imm
        assert dec["imm"] == imm
        assert dec["reg_write"] == 1

    def test_mov_is_an_alias_for_movz(self):
        assert assemble("arm", "MOV X1, #42") == assemble("arm", "MOVZ X1, #42")

    @pytest.mark.parametrize("imm", [0x10000, 0xFFFF0000])
    def test_movz_hw_shifted_immediate(self, imm):
        dec = _decode_arm(f"MOVZ X1, #{imm}")
        assert dec["imm"] == imm

    @pytest.mark.parametrize("offset", [0, 8, 16, 32760])
    def test_ldr_offset_and_registers(self, offset):
        dec = _decode_arm(f"LDR X5, [X3, #{offset}]")
        assert dec["rd"] == 5
        assert dec["rs1"] == 3
        assert dec["imm"] == offset
        assert (dec["mem_read"], dec["mem_write"], dec["reg_write"]) == (1, 0, 1)
        assert dec["wb_sel"] == WBSel.MEMORY

    @pytest.mark.parametrize("offset", [0, 8, 16, 32760])
    def test_str_offset_and_registers(self, offset):
        dec = _decode_arm(f"STR X5, [X3, #{offset}]")
        assert dec["rs1"] == 3                 # base
        assert dec["rs2"] == 5                 # data
        assert dec["imm"] == offset
        assert (dec["mem_read"], dec["mem_write"], dec["reg_write"]) == (0, 1, 0)

    def test_ldr_without_an_offset_is_offset_zero(self):
        assert assemble("arm", "LDR X1, [X0]") == assemble("arm", "LDR X1, [X0, #0]")

    @pytest.mark.parametrize("offset", [0, 4, 8, 4096, -4, -4096])
    def test_b_offset(self, offset):
        dec = _decode_arm(f"B {offset}")
        assert dec["jal"] == 1
        assert dec["reg_write"] == 0           # B does not link
        assert dec["imm"] == (offset & 0xFFFF_FFFF)

    @pytest.mark.parametrize("offset", [0, 4, 8, -8])
    def test_bl_offset_and_link_register(self, offset):
        dec = _decode_arm(f"BL {offset}")
        assert dec["jal"] == 1
        assert dec["rd"] == 30                 # X30 == LR
        assert dec["reg_write"] == 1
        assert dec["wb_sel"] == WBSel.PC4
        assert dec["imm"] == (offset & 0xFFFF_FFFF)

    @pytest.mark.parametrize("cond,expected", [
        ("EQ", BranchCond.EQ), ("NE", BranchCond.NEQ),
        ("LT", BranchCond.LT), ("GE", BranchCond.GE),
        ("GT", BranchCond.GT), ("LE", BranchCond.LE),
    ])
    def test_bcond_condition_and_offset(self, cond, expected):
        dec = _decode_arm(f"B.{cond} 8")
        assert dec["branch"] == 1
        assert dec["branch_cond"] == expected
        assert dec["imm"] == 8

    @pytest.mark.parametrize("mn,cond", [
        ("CBZ", BranchCond.EQ), ("CBNZ", BranchCond.NEQ),
    ])
    def test_cbz_cbnz(self, mn, cond):
        dec = _decode_arm(f"{mn} X9, 12")
        assert dec["branch"] == 1
        assert dec["rs1"] == 9
        assert dec["branch_cond"] == cond
        assert dec["imm"] == 12

    def test_ret_defaults_to_x30(self):
        dec = _decode_arm("RET")
        assert dec["jalr"] == 1
        assert dec["rs1"] == 30

    def test_ret_with_explicit_register(self):
        dec = _decode_arm("RET X5")
        assert dec["jalr"] == 1
        assert dec["rs1"] == 5

    def test_nop_decodes_to_no_control_signals(self):
        assert assemble("arm", "NOP") == [ARM_NOP]
        dec = _decode_arm("NOP")
        assert not any(dec[p] for p in ("reg_write", "mem_read", "mem_write",
                                        "branch", "jal", "jalr", "sets_flags"))

    def test_every_register_index_round_trips(self):
        for idx in range(31):
            dec = _decode_arm(f"ADD X{idx}, XZR, XZR")
            assert dec["rd"] == idx

    def test_xzr_and_sp_both_map_to_index_31(self):
        assert assemble("arm", "ADD X1, XZR, XZR") == assemble("arm", "ADD X1, SP, SP")


class TestX86OperandRoundTrip:

    def test_nop(self):
        dec, program = _decode_x86("NOP")
        assert program == [0x90]
        assert dec["instr_len"] == 1

    def test_ret(self):
        dec, program = _decode_x86("RET")
        assert program == [0xC3]
        assert dec["instr_len"] == 1
        assert dec["jalr"] == 1

    @pytest.mark.parametrize("reg,idx", [
        ("EAX", 0), ("ECX", 1), ("EDX", 2), ("EBX", 3),
        ("ESP", 4), ("EBP", 5), ("ESI", 6), ("EDI", 7),
    ])
    def test_push_encodes_50_plus_r_and_reads_that_register(self, reg, idx):
        dec, program = _decode_x86(f"PUSH {reg}")
        assert program == [0x50 + idx]
        assert dec["instr_len"] == 1
        assert dec["rs2"] == idx            # data source
        assert dec["mem_write"] == 1

    @pytest.mark.parametrize("reg,idx", [
        ("EAX", 0), ("ECX", 1), ("EDX", 2), ("EBX", 3),
        ("EBP", 5), ("ESI", 6), ("EDI", 7),
    ])
    def test_pop_encodes_58_plus_r_and_writes_that_register(self, reg, idx):
        dec, program = _decode_x86(f"POP {reg}")
        assert program == [0x58 + idx]
        assert dec["rd"] == idx
        assert (dec["mem_read"], dec["reg_write"]) == (1, 1)

    @pytest.mark.parametrize("imm", [0, 1, 42, 0x7FFFFFFF, -1, -42])
    def test_mov_imm32(self, imm):
        dec, program = _decode_x86(f"MOV EAX, {imm}")
        assert len(program) == 5 == dec["instr_len"]
        assert dec["rd"] == 0
        assert dec["imm"] == (imm & 0xFFFF_FFFF)
        assert dec["reg_write"] == 1

    @pytest.mark.parametrize("dst,src", [("EDX", "EAX"), ("ESI", "EDI"), ("EBX", "ECX")])
    def test_mov_reg_reg(self, dst, src):
        dec, program = _decode_x86(f"MOV {dst}, {src}")
        assert len(program) == 2 == dec["instr_len"]
        from sim.assembler.x86 import _REG_MAP
        assert dec["rd"] == _REG_MAP[dst]
        assert dec["rs1"] == _REG_MAP[src]
        assert dec["reg_write"] == 1

    def test_mov_load_no_displacement(self):
        dec, program = _decode_x86("MOV EAX, [EBX]")
        assert program == [0x8B, 0x03]
        assert dec["instr_len"] == 2
        assert (dec["rd"], dec["rs1"], dec["imm"]) == (0, 3, 0)
        assert (dec["mem_read"], dec["reg_write"]) == (1, 1)
        assert dec["wb_sel"] == WBSel.MEMORY

    @pytest.mark.parametrize("disp", [0, 4, 127, -4, -128])
    def test_mov_load_with_disp8(self, disp):
        sign = "+" if disp >= 0 else "-"
        dec, program = _decode_x86(f"MOV EAX, [EBX{sign}{abs(disp)}]")
        assert len(program) == 3 == dec["instr_len"]
        assert dec["rs1"] == 3
        assert dec["imm"] == (disp & 0xFFFF_FFFF)
        assert dec["mem_read"] == 1

    def test_mov_store_no_displacement(self):
        dec, program = _decode_x86("MOV [EBX], EAX")
        assert program == [0x89, 0x03]
        assert (dec["rs1"], dec["rs2"]) == (3, 0)
        assert dec["mem_write"] == 1
        assert dec["reg_write"] == 0

    @pytest.mark.parametrize("disp", [4, 127, -128])
    def test_mov_store_with_disp8(self, disp):
        sign = "+" if disp >= 0 else "-"
        dec, program = _decode_x86(f"MOV [EBX{sign}{abs(disp)}], ECX")
        assert len(program) == 3 == dec["instr_len"]
        assert (dec["rs1"], dec["rs2"]) == (3, 1)
        assert dec["imm"] == (disp & 0xFFFF_FFFF)
        assert dec["mem_write"] == 1

    @pytest.mark.parametrize("mn,opcode,alu_op,writes", [
        ("ADD", 0x01, AluOp.ADD, 1), ("OR", 0x09, AluOp.OR, 1),
        ("AND", 0x21, AluOp.AND, 1), ("SUB", 0x29, AluOp.SUB, 1),
        ("XOR", 0x31, AluOp.XOR, 1), ("CMP", 0x39, AluOp.SUB, 0),
    ])
    def test_alu_reg_reg(self, mn, opcode, alu_op, writes):
        dec, program = _decode_x86(f"{mn} EAX, ECX")
        assert program == [opcode, 0xC8]
        assert dec["instr_len"] == 2
        assert (dec["rs1"], dec["rs2"]) == (0, 1)
        assert dec["alu_op"] == alu_op
        assert dec["reg_write"] == writes
        assert dec["write_flags"] == 1

    @pytest.mark.parametrize("mn,ext", [
        ("ADD", 0), ("OR", 1), ("AND", 4), ("SUB", 5), ("XOR", 6), ("CMP", 7),
    ])
    @pytest.mark.parametrize("imm", [0, 5, 127, -1, -128])
    def test_alu_reg_imm8(self, mn, ext, imm):
        dec, program = _decode_x86(f"{mn} EBX, {imm}")
        assert program[0] == 0x83
        assert (program[1] >> 3) & 7 == ext
        assert program[1] & 7 == 3               # EBX
        assert len(program) == 3 == dec["instr_len"]
        assert dec["imm"] == (imm & 0xFFFF_FFFF)
        assert dec["alu_src"] == 1
        assert dec["write_flags"] == 1

    @pytest.mark.parametrize("disp", [0, 4, 126, -4, -126])
    def test_jmp_rel8_target(self, disp):
        # rel8 is measured from the NEXT instruction, so the DECODER's imm
        # (which it reports relative to the jump's own PC) must equal `disp`.
        dec, program = _decode_x86(f"JMP {disp:+d}")
        assert program[0] == 0xEB
        assert dec["instr_len"] == 2
        assert dec["jal"] == 1
        assert dec["imm"] == (disp & 0xFFFF_FFFF)

    @pytest.mark.parametrize("mn,opcode,cond", [
        ("JE", 0x74, BranchCond.EQ), ("JZ", 0x74, BranchCond.EQ),
        ("JNE", 0x75, BranchCond.NEQ), ("JNZ", 0x75, BranchCond.NEQ),
        ("JL", 0x7C, BranchCond.LT), ("JNGE", 0x7C, BranchCond.LT),
        ("JGE", 0x7D, BranchCond.GE), ("JNL", 0x7D, BranchCond.GE),
        ("JLE", 0x7E, BranchCond.LE), ("JNG", 0x7E, BranchCond.LE),
        ("JG", 0x7F, BranchCond.GT), ("JNLE", 0x7F, BranchCond.GT),
    ])
    def test_every_jcc_alias_encodes_and_decodes(self, mn, opcode, cond):
        dec, program = _decode_x86(f"{mn} +4")
        assert program[0] == opcode
        assert dec["branch"] == 1
        assert dec["branch_cond"] == cond
        assert dec["imm"] == 4

    def test_call_rel32(self):
        dec, program = _decode_x86("CALL +16")
        assert program[0] == 0xE8
        assert len(program) == 5 == dec["instr_len"]
        assert dec["jal"] == 1
        assert dec["imm"] == 16

    def test_widened_jmp_rel32_round_trips(self):
        # Branch relaxation output must still decode: E9 + imm32, len 5.
        dec = X86Decoder()
        program = assemble("x86", "JMP far\n" + "NOP\n" * 200 + "far:\nNOP\n")
        assert program[0] == 0xE9
        stream = int.from_bytes(bytes(program[:8]), "little")
        dec._ports["instr_in"] = stream
        dec._ports["pc_in"] = 0
        dec.evaluate()
        assert dec["instr_len"] == 5
        assert dec["jal"] == 1
        assert dec["imm"] == 5 + 200            # jump lands on the 'far' NOP

    def test_widened_jcc_rel32_round_trips(self):
        dec = X86Decoder()
        program = assemble("x86", "JE far\n" + "NOP\n" * 200 + "far:\nNOP\n")
        assert program[:2] == [0x0F, 0x84]
        stream = int.from_bytes(bytes(program[:8]), "little")
        dec._ports["instr_in"] = stream
        dec._ports["pc_in"] = 0
        dec.evaluate()
        assert dec["instr_len"] == 6
        assert dec["branch"] == 1
        assert dec["branch_cond"] == BranchCond.EQ
        assert dec["imm"] == 6 + 200


# ════════════════════════════════════════════════════════════════════════════
# 4. Label round-trip — a label target resolves to the labelled instruction
# ════════════════════════════════════════════════════════════════════════════

class TestLabelResolution:
    """A `<branch> label` must produce the offset from the branch to the label."""

    def test_riscv_forward_branch_offset(self):
        program = assemble("riscv", "NOP\nBEQ x0, x0, skip\nNOP\nskip:\nNOP\n")
        dec = RISCVDecoder()
        dec._ports["instr_in"] = program[1]
        dec.evaluate()
        assert dec["imm"] == 8              # from pc=4 to pc=12

    def test_riscv_backward_branch_offset(self):
        program = assemble("riscv", "loop:\nNOP\nNOP\nBEQ x0, x0, loop\n")
        dec = RISCVDecoder()
        dec._ports["instr_in"] = program[2]
        dec.evaluate()
        assert dec["imm"] == (-8 & 0xFFFF_FFFF)

    def test_riscv_jal_label(self):
        program = assemble("riscv", "JAL x1, func\nNOP\nfunc:\nNOP\n")
        dec = RISCVDecoder()
        dec._ports["instr_in"] = program[0]
        dec.evaluate()
        assert dec["imm"] == 8

    def test_arm_forward_branch_offset(self):
        program = assemble("arm", "NOP\nB skip\nNOP\nskip:\nNOP\n")
        dec = ARMDecoder()
        dec._ports["instr_in"] = program[1]
        dec.evaluate()
        assert dec["imm"] == 8

    def test_arm_backward_bcond_offset(self):
        program = assemble("arm", "loop:\nNOP\nNOP\nB.NE loop\n")
        dec = ARMDecoder()
        dec._ports["instr_in"] = program[2]
        dec.evaluate()
        assert dec["imm"] == (-8 & 0xFFFF_FFFF)

    def test_arm_cbz_label(self):
        program = assemble("arm", "CBZ X1, done\nNOP\ndone:\nNOP\n")
        dec = ARMDecoder()
        dec._ports["instr_in"] = program[0]
        dec.evaluate()
        assert dec["imm"] == 8

    def test_x86_forward_jmp_label(self):
        program = assemble("x86", "JMP skip\nNOP\nskip:\nNOP\n")
        assert program[:2] == [0xEB, 0x01]      # rel8 from pc+2 == 1
        dec, _ = _decode_x86("JMP skip\nNOP\nskip:\nNOP\n")
        assert dec["imm"] == 3                  # decoder reports pc-relative

    def test_x86_backward_jcc_label(self):
        program = assemble("x86", "loop:\nNOP\nNOP\nJNE loop\n")
        assert program[2] == 0x75
        assert program[3] == 0xFC               # rel8 = -4 (from pc+2 == 4)


# ════════════════════════════════════════════════════════════════════════════
# 5. Cheatsheet <-> assembler agreement
# ════════════════════════════════════════════════════════════════════════════

def _cheatsheet_entries(isa):
    return [(isa, e) for e in get_cheatsheet(isa)]


# Every cheatsheet example must assemble. This set was non-empty until
# 2026-08-08 (x86 "XOR EAX, 0xFF" advertised an out-of-range imm8); the
# cheatsheet was corrected to 0x0F. Kept as an explicit empty set so the sweep
# below stays a hard assertion and a newly-broken example is a failure, not a
# silently-tolerated entry.
KNOWN_BAD_CHEATSHEET_EXAMPLES = set()


def _cheatsheet_param(isa, entry):
    marks = []
    if (isa, entry["example"]) in KNOWN_BAD_CHEATSHEET_EXAMPLES:
        marks.append(pytest.mark.xfail(strict=True, reason="known-bad cheatsheet example"))
    return pytest.param(isa, entry, marks=marks,
                        id=f"{isa}-{entry['mnemonic']}-{entry['syntax']}")


CHEATSHEET_CASES = [(isa, e) for isa in ISAS for _i, e in _cheatsheet_entries(isa)]
CHEATSHEET_PARAMS = [_cheatsheet_param(isa, e) for isa, e in CHEATSHEET_CASES]


def _with_labels(example):
    """Append definitions for every label a cheatsheet example might jump to."""
    return example + "\nNOP\n" + "".join(f"{n}:\nNOP\n" for n in CHEATSHEET_LABELS)


class TestCheatsheetStructure:

    @pytest.mark.parametrize("isa", ISAS)
    def test_entries_have_the_documented_keys(self, isa):
        entries = get_cheatsheet(isa)
        assert len(entries) > 10
        for e in entries:
            assert set(e) == {"category", "mnemonic", "syntax", "description", "example"}
            assert all(isinstance(v, str) and v.strip() for v in e.values())

    @pytest.mark.parametrize("isa", ISAS)
    def test_syntax_and_example_start_with_the_entrys_mnemonic(self, isa):
        for e in get_cheatsheet(isa):
            assert e["syntax"].upper().startswith(e["mnemonic"].upper()), e
            assert e["example"].upper().startswith(e["mnemonic"].upper()), e

    @pytest.mark.parametrize("isa", ISAS)
    def test_isa_name_is_normalised(self, isa):
        assert get_cheatsheet(isa.upper()) == get_cheatsheet(f"  {isa}  ")

    @pytest.mark.parametrize("isa", ["mips", "sparc", "", "x64"])
    def test_unknown_isa_raises(self, isa):
        with pytest.raises(ValueError, match="Unknown ISA"):
            get_cheatsheet(isa)


class TestCheatsheetExamplesAssemble:
    """Every advertised example must actually assemble.

    The cheatsheet is what the UI's slide-out reference shows the user; an
    entry the assembler rejects is a broken instruction in the docs.
    """

    @pytest.mark.parametrize("isa,entry", CHEATSHEET_PARAMS)
    def test_example_assembles_with_labels_defined(self, isa, entry):
        program = assemble(isa, _with_labels(entry["example"]))
        assert program, f"{isa} {entry['example']!r} produced no output"

    @pytest.mark.parametrize("isa,entry", CHEATSHEET_CASES,
                             ids=[f"{i}-{e['mnemonic']}-{e['syntax']}"
                                  for i, e in CHEATSHEET_CASES])
    def test_examples_mnemonic_is_accepted_by_the_assembler(self, isa, entry):
        """Whatever else fails, the MNEMONIC must not be 'unknown'."""
        try:
            assemble(isa, _with_labels(entry["example"]))
        except AssemblerError as exc:
            assert "mnemonic" not in str(exc), \
                f"{isa} cheatsheet advertises an unsupported mnemonic: {exc}"

    @pytest.mark.parametrize("isa", ISAS)
    def test_every_cheatsheet_example_assembles_except_the_known_gap(self, isa):
        """Sweep: exactly the documented set of examples fails, no more."""
        broken = []
        for entry in get_cheatsheet(isa):
            try:
                assemble(isa, _with_labels(entry["example"]))
            except AssemblerError:
                broken.append(entry["example"])
        expected = sorted(ex for i, ex in KNOWN_BAD_CHEATSHEET_EXAMPLES if i == isa)
        assert sorted(broken) == expected

    def test_x86_xor_with_a_positive_byte_mask_is_out_of_imm8_range(self):
        """`XOR EAX, 0xFF` does not assemble — and that is CORRECT.

        The 0x83 Group-1 imm8 field is SIGN-EXTENDED, so 255 is genuinely out
        of the [-128..127] range the encoder enforces. The assembler is right;
        the CHEATSHEET was wrong to advertise it (fixed 2026-08-08 -> 0x0F).
        Pinned so nobody "fixes" the encoder to accept 255.
        """
        with pytest.raises(AssemblerError, match="out of range"):
            assemble("x86", "XOR EAX, 0xFF")
        # The intended bit pattern IS reachable, just spelled signed:
        assert assemble("x86", "XOR EAX, -1") == [0x83, 0xF0, 0xFF]

    def test_x86_cheatsheet_xor_example_now_assembles(self):
        """The corrected cheatsheet entry must actually work."""
        entry = next(e for e in get_cheatsheet("x86")
                     if e["mnemonic"] == "XOR" and "imm8" in e["syntax"])
        assert entry["example"] == "XOR EAX, 0x0F"
        assert assemble("x86", entry["example"]) == [0x83, 0xF0, 0x0F]


class TestCheatsheetCoversTheAssembler:
    """The cheatsheet should not omit a supported mnemonic (documentation gap)."""

    def test_riscv_cheatsheet_lists_every_supported_mnemonic(self):
        documented = {e["mnemonic"].upper() for e in get_cheatsheet("riscv")}
        supported = {
            "ADD", "SUB", "AND", "OR", "XOR", "SLT", "SLTU", "SLL", "SRL", "SRA",
            "ADDI", "ANDI", "ORI", "XORI", "SLTI", "SLTIU", "SLLI", "SRLI", "SRAI",
            "LW", "SW", "BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU",
            "JAL", "JALR", "LUI", "AUIPC",
        }
        assert supported - documented == set()

    def test_arm_cheatsheet_lists_every_supported_mnemonic(self):
        documented = {e["mnemonic"].upper() for e in get_cheatsheet("arm")}
        supported = {
            "ADD", "SUB", "SUBS", "CMP", "AND", "ORR", "EOR", "MOVZ", "MOV",
            "LDR", "STR", "B", "BL", "CBZ", "CBNZ", "RET", "NOP",
            "B.EQ", "B.NE", "B.LT", "B.GE", "B.GT", "B.LE",
        }
        assert supported - documented == set()

    def test_x86_cheatsheet_lists_every_supported_mnemonic(self):
        documented = {e["mnemonic"].upper() for e in get_cheatsheet("x86")}
        supported = {
            "MOV", "ADD", "SUB", "AND", "OR", "XOR", "CMP",
            "JMP", "JE", "JNE", "JL", "JGE", "JLE", "JG",
            "CALL", "RET", "PUSH", "POP", "NOP",
        }
        assert supported - documented == set()

    def test_no_cheatsheet_entry_names_an_unsupported_mnemonic(self):
        """Every documented mnemonic must survive dispatch in _encode()."""
        probes = {
            "riscv": lambda m: f"{m} x1, x2, x3",
            "arm": lambda m: f"{m} X1, X2, X3",
            "x86": lambda m: f"{m} EAX, ECX",
        }
        for isa in ISAS:
            for entry in get_cheatsheet(isa):
                mn = entry["mnemonic"]
                try:
                    assemble(isa, probes[isa](mn))
                except AssemblerError as exc:
                    # Arity / operand-form complaints are fine; "unknown
                    # mnemonic" means the cheatsheet documents something the
                    # assembler cannot encode at all.
                    assert "mnemonic" not in str(exc), f"{isa} {mn}: {exc}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
