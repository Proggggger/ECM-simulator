"""
Direct unit tests for the RISC-V and ARM instruction decoders.

The decoder is described in CLAUDE.md as "the single hot-swap point between
ISAs": every execution model (single-cycle / multicycle / configurable /
pipeline / OoO / superscalar) consumes the SAME set of control ports, and only
the decoder changes.  Until now that contract was only verified indirectly by
whole-program runs, which cannot distinguish "the decoder is right" from "two
bugs cancelled out".

These tests drive the decoders DIRECTLY:

    d = RISCVDecoder()
    d["instr_in"] = word
    d.evaluate()
    assert d["alu_op"] == AluOp.SUB      # ... and every other control port

Everything asserted here was verified against the real source
(sim/isa/<arch>/decoder_component.py) — this project models a SUBSET of each
ISA, so several assertions pin deliberately-simplified or unhandled behaviour
rather than real-hardware behaviour.  Those are called out in comments.

Instruction words come from two sources on purpose:
  * the project's own assembler (sim/assembler) — so each test doubles as an
    assembler <-> decoder round-trip check, and
  * hand-written hex literals for the tricky immediate cases — so a shared
    assembler bug cannot hide a decoder bug.

The x86 decoder gets its own file (tests/test_decoders_x86.py); it appears here
only in the cross-ISA port-contract test, which by design must see all three.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.assembler import assemble
from sim.component.base import Port
from sim.core.signals import AluOp, BranchCond, WBSel
from sim.isa.arm.constants import A64Cond, REGISTER_NAMES as ARM_REGS, bits
from sim.isa.arm.decoder_component import ARMDecoder
from sim.isa.riscv.constants import Funct3, Funct7, Opcode
from sim.isa.riscv.decoder_component import RISCVDecoder
from sim.isa.x86.decoder_component import X86Decoder


# ════════════════════════════════════════════════════════════════════════════
#  THE SHARED DECODER CONTRACT  (all three ISAs)
# ════════════════════════════════════════════════════════════════════════════
# CLAUDE.md "Decoder output contract": all three decoders expose a common
# superset of control ports so the execution models never change.  This is the
# single most valuable assertion in this file — every preset in the 18-way
# matrix wires these by name, so silently renaming/dropping one breaks 6
# presets at once.

CONTRACT_PORTS = {
    # port name  -> (width, direction)
    "rd":          (5,  "out"),
    "rs1":         (5,  "out"),
    "rs2":         (5,  "out"),
    "imm":         (32, "out"),
    "alu_op":      (4,  "out"),
    "alu_src":     (1,  "out"),
    "use_pc":      (1,  "out"),
    "mem_read":    (1,  "out"),
    "mem_write":   (1,  "out"),
    "reg_write":   (1,  "out"),
    "branch":      (1,  "out"),
    "branch_cond": (3,  "out"),
    "jal":         (1,  "out"),
    "jalr":        (1,  "out"),
    "wb_sel":      (2,  "out"),
}

ALL_DECODERS = [
    pytest.param(RISCVDecoder, id="riscv"),
    pytest.param(ARMDecoder, id="arm"),
    pytest.param(X86Decoder, id="x86"),
]

# Ports each decoder legitimately adds ON TOP of the contract.
EXTRA_PORTS = {
    RISCVDecoder: {"instr_in", "opcode", "funct3", "funct7"},
    ARMDecoder:   {"instr_in", "sets_flags"},
    X86Decoder:   {"instr_in", "pc_in", "instr_len", "write_flags"},
}

# A "do nothing" encoding per ISA, used to prove the reset-to-benign property.
QUIESCENT = {
    RISCVDecoder: 0x00000000,   # illegal (opcode 0)
    ARMDecoder:   0xD503201F,   # A64 NOP (hint #0)
    X86Decoder:   0x90,         # x86 NOP
}


@pytest.mark.parametrize("decoder_cls", ALL_DECODERS)
def test_decoder_declares_full_common_port_superset(decoder_cls):
    """Every decoder must declare the whole contract, at the same widths."""
    spec = decoder_cls.ports_spec
    missing = [p for p in CONTRACT_PORTS if p not in spec]
    assert not missing, (
        f"{decoder_cls.__name__} is missing contract port(s) {missing}; "
        f"the shared presets wire these by name")

    for name, (width, direction) in CONTRACT_PORTS.items():
        port = spec[name]
        assert isinstance(port, Port)
        assert port.width == width, (
            f"{decoder_cls.__name__}.{name} width {port.width} != {width}")
        assert port.direction == direction, (
            f"{decoder_cls.__name__}.{name} must be an OUTPUT")


@pytest.mark.parametrize("decoder_cls", ALL_DECODERS)
def test_decoder_declares_an_instruction_input(decoder_cls):
    spec = decoder_cls.ports_spec
    assert "instr_in" in spec
    assert spec["instr_in"].direction == "in"


@pytest.mark.parametrize("decoder_cls", ALL_DECODERS)
def test_decoder_declares_no_unexpected_extra_ports(decoder_cls):
    """Extra ports are allowed but enumerated, so new ones are a deliberate act."""
    extra = set(decoder_cls.ports_spec) - set(CONTRACT_PORTS)
    assert extra == EXTRA_PORTS[decoder_cls]


@pytest.mark.parametrize("decoder_cls", ALL_DECODERS)
def test_decoder_ui_metadata(decoder_cls):
    """Every decoder is a 'decode'-category node so the UI can place it."""
    assert decoder_cls.ui_category == "decode"
    assert decoder_cls.ui_label
    assert decoder_cls.name


@pytest.mark.parametrize("decoder_cls", ALL_DECODERS)
def test_decoder_get_state_reports_the_common_signals(decoder_cls):
    d = decoder_cls()
    d["instr_in"] = QUIESCENT[decoder_cls]
    d.evaluate()
    state = d.get_state()
    for key in ("rd", "rs1", "rs2", "imm", "alu_op", "alu_src",
                "mem_read", "mem_write", "reg_write", "branch", "jal", "jalr"):
        assert key in state, f"{decoder_cls.__name__}.get_state() lost '{key}'"
    # alu_op is rendered as a NAME string for the UI, not a raw int.
    assert isinstance(state["alu_op"], str)


@pytest.mark.parametrize("decoder_cls", ALL_DECODERS)
def test_quiescent_encoding_asserts_no_side_effecting_control(decoder_cls):
    """A NOP / no-op word must not write a register, memory, or redirect the PC.

    This is a SAFETY property: the reset-then-decode structure in every
    evaluate() means a missing `else` would leak the previous instruction's
    control signals into the next cycle.
    """
    d = decoder_cls()
    d["instr_in"] = QUIESCENT[decoder_cls]
    d.evaluate()
    for port in ("mem_read", "mem_write", "reg_write", "branch", "jal",
                 "jalr", "use_pc"):
        assert d[port] == 0, f"{decoder_cls.__name__}: {port} asserted on a NOP"
    assert d["branch_cond"] == BranchCond.NEVER
    assert d["wb_sel"] == WBSel.ALU


@pytest.mark.parametrize("decoder_cls", ALL_DECODERS)
def test_decoder_is_stateless_across_evaluations(decoder_cls):
    """A load followed by a NOP must not leave mem_read / wb_sel asserted.

    Combinational decode has no memory: the presets re-drive `instr_in` every
    cycle and call evaluate(), so any sticky output is a latch bug.
    """
    loads = {
        RISCVDecoder: 0x00002283,   # LW x5, 0(x0)
        ARMDecoder:   0xF9400402,   # LDR X2, [X0, #8]
        X86Decoder:   0x038B,       # 8B 03 = MOV EAX, [EBX]  (LE byte stream)
    }
    d = decoder_cls()
    d["instr_in"] = loads[decoder_cls]
    d.evaluate()
    assert d["mem_read"] == 1 and d["wb_sel"] == WBSel.MEMORY

    d["instr_in"] = QUIESCENT[decoder_cls]
    d.evaluate()
    assert d["mem_read"] == 0
    assert d["reg_write"] == 0
    assert d["wb_sel"] == WBSel.ALU


# ════════════════════════════════════════════════════════════════════════════
#  RISC-V RV32I
# ════════════════════════════════════════════════════════════════════════════

RV_PORTS = [p for p in RISCVDecoder.ports_spec if p != "instr_in"]


def rv(word):
    """Decode one RV32I word; return {port: value} for every output port."""
    d = RISCVDecoder()
    d["instr_in"] = word
    d.evaluate()
    return {p: d[p] for p in RV_PORTS}


def rv_asm(text):
    """Assemble one RISC-V instruction and decode it (round-trip check)."""
    words = assemble("riscv", text)
    assert len(words) == 1, f"expected 1 word from {text!r}, got {words}"
    return rv(words[0]), words[0]


def u32(value):
    """Two's-complement 32-bit view — the decoder masks `imm` to 32 bits."""
    return value & 0xFFFF_FFFF


# ── raw field extraction ────────────────────────────────────────────────────

def test_riscv_raw_field_extraction():
    """opcode/rd/funct3/rs1/rs2/funct7 are plain bit slices of the word."""
    # funct7=0x7F rs2=0x1F rs1=0x1F funct3=0x7 rd=0x1F opcode=0x7F
    out = rv(0xFFFFFFFF)
    assert out["opcode"] == 0x7F
    assert out["rd"] == 0x1F
    assert out["funct3"] == 0x7
    assert out["rs1"] == 0x1F
    assert out["rs2"] == 0x1F
    assert out["funct7"] == 0x7F


def test_riscv_field_extraction_is_positional_not_semantic():
    """rs2 / rd are decoded from fixed bit positions even when meaningless.

    SW's `rd` field really holds imm[4:0], and ADDI's `rs2` field really holds
    imm[4:0].  The decoder exposes them raw; the datapath is expected to gate
    on `reg_write` / `alu_src`.  (This is the "phantom rs2" the superscalar
    hazard detector had to be taught about — see CLAUDE.md.)
    """
    out, word = rv_asm("SW x3, 20(x2)")
    assert word == 0x00312A23
    assert out["rs1"] == 2 and out["rs2"] == 3
    assert out["rd"] == 20          # == imm[4:0], NOT a real destination
    assert out["reg_write"] == 0    # ... and it is correctly gated off

    out, _ = rv_asm("ADDI x1, x2, 5")
    assert out["rs2"] == 5          # == imm[4:0], NOT a real source
    assert out["alu_src"] == 1      # ... and the ALU takes the immediate


# ── R-type ──────────────────────────────────────────────────────────────────

RV_R_TYPE = [
    ("ADD  x1, x2, x3", AluOp.ADD,  Funct3.ADD_SUB, Funct7.NORMAL),
    ("SUB  x1, x2, x3", AluOp.SUB,  Funct3.ADD_SUB, Funct7.ALT),
    ("AND  x1, x2, x3", AluOp.AND,  Funct3.AND,     Funct7.NORMAL),
    ("OR   x1, x2, x3", AluOp.OR,   Funct3.OR,      Funct7.NORMAL),
    ("XOR  x1, x2, x3", AluOp.XOR,  Funct3.XOR,     Funct7.NORMAL),
    ("SLT  x1, x2, x3", AluOp.SLT,  Funct3.SLT,     Funct7.NORMAL),
    ("SLTU x1, x2, x3", AluOp.SLTU, Funct3.SLTU,    Funct7.NORMAL),
    ("SLL  x1, x2, x3", AluOp.SLL,  Funct3.SLL,     Funct7.NORMAL),
    ("SRL  x1, x2, x3", AluOp.SRL,  Funct3.SRL_SRA, Funct7.NORMAL),
    ("SRA  x1, x2, x3", AluOp.SRA,  Funct3.SRL_SRA, Funct7.ALT),
]


@pytest.mark.parametrize("text,alu_op,funct3,funct7", RV_R_TYPE)
def test_riscv_r_type_control_bundle(text, alu_op, funct3, funct7):
    out, word = rv_asm(text)
    assert word & 0x7F == Opcode.R_TYPE
    assert out["funct3"] == funct3
    assert out["funct7"] == funct7
    assert out["alu_op"] == alu_op
    # R-type: register operands, writes rd from the ALU, no memory, no control flow.
    assert out["rd"] == 1 and out["rs1"] == 2 and out["rs2"] == 3
    assert out["alu_src"] == 0
    assert out["use_pc"] == 0
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.ALU
    assert out["imm"] == 0
    assert out["mem_read"] == out["mem_write"] == 0
    assert out["branch"] == out["jal"] == out["jalr"] == 0
    assert out["branch_cond"] == BranchCond.NEVER


def test_riscv_r_type_unmapped_funct3_funct7_leaves_alu_op_at_add():
    """RV32M MUL (funct7=0000001, funct3=000) is NOT modelled.

    _decode_r_alu only inspects funct3 (and funct7 for ADD/SUB and SRL/SRA), so
    MUL decodes as a register-write of the *ADD* result.  Pinned so a future
    RV32M addition is a deliberate change, not a silent one.
    """
    mul = (0b0000001 << 25) | (3 << 20) | (2 << 15) | (0 << 12) | (1 << 7) | Opcode.R_TYPE
    out = rv(mul)
    assert out["funct7"] == 0b0000001
    assert out["alu_op"] == AluOp.ADD   # falls through the ALT check
    assert out["reg_write"] == 1


def test_riscv_r_type_funct7_alt_only_affects_add_and_srl():
    """funct7=0100000 must NOT turn AND into something else."""
    and_alt = (Funct7.ALT << 25) | (3 << 20) | (2 << 15) | \
              (Funct3.AND << 12) | (1 << 7) | Opcode.R_TYPE
    assert rv(and_alt)["alu_op"] == AluOp.AND


# ── I-type ALU ──────────────────────────────────────────────────────────────

RV_I_ALU = [
    ("ADDI  x1, x2, 5",  AluOp.ADD),
    ("ANDI  x1, x2, 5",  AluOp.AND),
    ("ORI   x1, x2, 5",  AluOp.OR),
    ("XORI  x1, x2, 5",  AluOp.XOR),
    ("SLTI  x1, x2, 5",  AluOp.SLT),
    ("SLTIU x1, x2, 5",  AluOp.SLTU),
    ("SLLI  x1, x2, 5",  AluOp.SLL),
    ("SRLI  x1, x2, 5",  AluOp.SRL),
    ("SRAI  x1, x2, 5",  AluOp.SRA),
]


@pytest.mark.parametrize("text,alu_op", RV_I_ALU)
def test_riscv_i_type_alu_control_bundle(text, alu_op):
    out, word = rv_asm(text)
    assert word & 0x7F == Opcode.I_ALU
    assert out["alu_op"] == alu_op
    assert out["rd"] == 1 and out["rs1"] == 2
    assert out["alu_src"] == 1          # operand B is the immediate
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.ALU
    assert out["use_pc"] == 0
    assert out["mem_read"] == out["mem_write"] == 0
    assert out["branch"] == out["jal"] == out["jalr"] == 0


def test_riscv_addi_never_becomes_subi():
    """_decode_i_alu ignores funct7 for ADD_SUB — there is no SUBI in RV32I.

    ADDI x1, x2, -2048 has funct7 bits = 0b1000000, which for an R-type would
    NOT be ALT anyway; the sharper case is imm=0x400 (funct7 == Funct7.ALT).
    """
    addi_alt = (Funct7.ALT << 25) | (0 << 20) | (2 << 15) | \
               (Funct3.ADD_SUB << 12) | (1 << 7) | Opcode.I_ALU
    out = rv(addi_alt)
    assert out["funct7"] == Funct7.ALT
    assert out["alu_op"] == AluOp.ADD    # still ADD, not SUB
    assert out["imm"] == 1024


def test_riscv_srai_shamt_immediate_carries_the_funct7_bits():
    """SRAI's imm is the RAW 12-bit I-field, so it includes funct7<<5."""
    out, word = rv_asm("SRAI x1, x2, 31")
    assert word == 0x41F15093
    assert out["alu_op"] == AluOp.SRA
    assert out["imm"] == (Funct7.ALT << 5) | 31 == 1055
    # SRLI, by contrast, is a clean shamt.
    out, _ = rv_asm("SRLI x1, x2, 31")
    assert out["alu_op"] == AluOp.SRL
    assert out["imm"] == 31


# ── loads / stores ──────────────────────────────────────────────────────────

def test_riscv_load_control_bundle():
    out, word = rv_asm("LW x5, 8(x2)")
    assert word & 0x7F == Opcode.LOAD
    assert out["rd"] == 5 and out["rs1"] == 2
    assert out["imm"] == 8
    assert out["alu_op"] == AluOp.ADD    # address = rs1 + imm
    assert out["alu_src"] == 1
    assert out["mem_read"] == 1
    assert out["mem_write"] == 0
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.MEMORY
    assert out["branch"] == out["jal"] == out["jalr"] == out["use_pc"] == 0


def test_riscv_store_control_bundle():
    out, word = rv_asm("SW x3, 8(x2)")
    assert word & 0x7F == Opcode.STORE
    assert out["rs1"] == 2 and out["rs2"] == 3
    assert out["imm"] == 8
    assert out["alu_op"] == AluOp.ADD
    assert out["alu_src"] == 1
    assert out["mem_write"] == 1
    assert out["mem_read"] == 0
    assert out["reg_write"] == 0         # a store must NOT write a register
    assert out["wb_sel"] == WBSel.ALU
    assert out["branch"] == out["jal"] == out["jalr"] == 0


def test_riscv_load_and_store_widths_share_one_decode_path():
    """funct3 selects LB/LH/LW/LBU/LHU in real RV32I; this model ignores it.

    All LOAD funct3 values produce the identical word-sized control bundle.
    Pinned so adding sub-word support is a deliberate change.
    """
    bundles = set()
    for f3 in (Funct3.BYTE, Funct3.HALF, Funct3.WORD, Funct3.BYTE_U, Funct3.HALF_U):
        word = (8 << 20) | (2 << 15) | (f3 << 12) | (5 << 7) | Opcode.LOAD
        out = rv(word)
        bundles.add((out["mem_read"], out["reg_write"], out["wb_sel"],
                     out["alu_op"], out["imm"]))
    assert bundles == {(1, 1, WBSel.MEMORY, AluOp.ADD, 8)}


# ── branches ────────────────────────────────────────────────────────────────

RV_BRANCH = [
    ("BEQ",  Funct3.BEQ,  BranchCond.EQ,  AluOp.SUB),
    ("BNE",  Funct3.BNE,  BranchCond.NEQ, AluOp.SUB),
    ("BLT",  Funct3.BLT,  BranchCond.LT,  AluOp.SLT),
    ("BGE",  Funct3.BGE,  BranchCond.GE,  AluOp.SLT),
    ("BLTU", Funct3.BLTU, BranchCond.LT,  AluOp.SLTU),
    ("BGEU", Funct3.BGEU, BranchCond.GE,  AluOp.SLTU),
]


@pytest.mark.parametrize("mnemonic,funct3,cond,alu_op", RV_BRANCH)
def test_riscv_branch_condition_and_alu_op(mnemonic, funct3, cond, alu_op):
    out, word = rv_asm(f"{mnemonic} x1, x2, +8")
    assert word & 0x7F == Opcode.BRANCH
    assert out["funct3"] == funct3
    assert out["branch"] == 1
    assert out["branch_cond"] == cond
    assert out["alu_op"] == alu_op
    assert out["rs1"] == 1 and out["rs2"] == 2
    assert out["imm"] == 8
    assert out["alu_src"] == 0           # compare two registers
    assert out["reg_write"] == 0
    assert out["mem_read"] == out["mem_write"] == 0
    assert out["jal"] == out["jalr"] == 0


def test_riscv_unsigned_branches_reuse_the_signed_condition_codes():
    """BLTU/BGEU differ from BLT/BGE only in alu_op (SLTU vs SLT).

    The shared BranchCond enum has no unsigned variants; the ALU does the
    unsigned compare and the branch unit just tests its result.
    """
    blt = rv(assemble("riscv", "BLT  x1, x2, +8")[0])
    bltu = rv(assemble("riscv", "BLTU x1, x2, +8")[0])
    assert blt["branch_cond"] == bltu["branch_cond"] == BranchCond.LT
    assert (blt["alu_op"], bltu["alu_op"]) == (AluOp.SLT, AluOp.SLTU)


@pytest.mark.parametrize("funct3", [0b010, 0b011])
def test_riscv_reserved_branch_funct3_is_branch_but_never_taken(funct3):
    """funct3 = 010/011 are reserved. `branch` is asserted but cond = NEVER.

    Safety property: an unknown branch flavour must not become an unconditional
    jump, and must not fall back to some other condition.
    """
    word = (funct3 << 12) | Opcode.BRANCH
    out = rv(word)
    assert out["branch"] == 1
    assert out["branch_cond"] == BranchCond.NEVER
    assert out["alu_op"] == AluOp.ADD     # reset value; _decode_branch no-ops
    assert out["reg_write"] == 0
    assert out["jal"] == out["jalr"] == 0


# ── LUI vs AUIPC ────────────────────────────────────────────────────────────

def test_riscv_lui_passes_the_immediate_through_without_pc():
    out, word = rv_asm("LUI x1, 0x12345")
    assert word == 0x123450B7
    assert out["rd"] == 1
    assert out["imm"] == 0x12345000
    assert out["alu_op"] == AluOp.PASS   # ALU forwards operand B unchanged
    assert out["alu_src"] == 1
    assert out["use_pc"] == 0            # <-- the LUI/AUIPC discriminator
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.ALU


def test_riscv_auipc_adds_the_immediate_to_the_pc():
    out, word = rv_asm("AUIPC x5, 0x12345")
    assert word == 0x12345297
    assert out["rd"] == 5
    assert out["imm"] == 0x12345000
    assert out["alu_op"] == AluOp.ADD    # PC + imm, not PASS
    assert out["alu_src"] == 1
    assert out["use_pc"] == 1            # <-- the LUI/AUIPC discriminator
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.ALU


def test_riscv_lui_and_auipc_differ_only_in_use_pc_and_alu_op():
    lui = rv(assemble("riscv", "LUI   x5, 0x12345")[0])
    auipc = rv(assemble("riscv", "AUIPC x5, 0x12345")[0])
    differing = {k for k in lui if lui[k] != auipc[k]}
    assert differing == {"opcode", "use_pc", "alu_op"}


# ── JAL vs JALR ─────────────────────────────────────────────────────────────

def test_riscv_jal_control_bundle():
    out, word = rv_asm("JAL x1, +8")
    assert word == 0x008000EF
    assert out["rd"] == 1
    assert out["imm"] == 8
    assert out["jal"] == 1
    assert out["jalr"] == 0
    assert out["branch"] == 0
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.PC4     # link register gets PC+4
    # JAL is PC-relative but does NOT go through the ALU's use_pc path.
    assert out["use_pc"] == 0
    assert out["alu_src"] == 0
    assert out["alu_op"] == AluOp.ADD     # reset value; unused for JAL


def test_riscv_jalr_control_bundle():
    out, word = rv_asm("JALR x1, x2, -8")
    assert word == 0xFF8100E7
    assert out["rd"] == 1 and out["rs1"] == 2
    assert out["imm"] == u32(-8)
    assert out["jalr"] == 1
    assert out["jal"] == 0
    assert out["branch"] == 0
    assert out["alu_src"] == 1            # target = rs1 + imm
    assert out["alu_op"] == AluOp.ADD
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.PC4


def test_riscv_jal_and_jalr_are_mutually_exclusive():
    for text in ("JAL x1, +8", "JALR x1, x2, 0"):
        out, _ = rv_asm(text)
        assert out["jal"] + out["jalr"] == 1


# ── immediate assembly + sign extension (the classic decoder break) ─────────

@pytest.mark.parametrize("word,expected,note", [
    (0x00500093, 5,          "ADDI x1, x0, 5"),
    (0xFFF00093, u32(-1),    "ADDI x1, x0, -1"),
    (0x7FF00093, 2047,       "ADDI x1, x0, 2047  (max positive I-imm)"),
    (0x80000093, u32(-2048), "ADDI x1, x0, -2048 (min negative I-imm)"),
])
def test_riscv_i_type_immediate_sign_extension(word, expected, note):
    assert rv(word)["imm"] == expected, note


@pytest.mark.parametrize("word,expected,note", [
    (0x00312023, 0,          "SW x3, 0(x2)"),
    (0x00312A23, 20,         "SW x3, 20(x2)  (imm split 11:5 / 4:0)"),
    (0xFE312E23, u32(-4),    "SW x3, -4(x2)"),
    (0x7E312FA3, 2047,       "SW x3, 2047(x2)  (max positive S-imm)"),
    (0x80312023, u32(-2048), "SW x3, -2048(x2) (min negative S-imm)"),
])
def test_riscv_s_type_immediate_reassembly_and_sign_extension(word, expected, note):
    """S-type imm is split across bits [31:25] and [11:7] — reassemble + extend."""
    assert rv(word)["imm"] == expected, note


# B-type scrambles the offset as imm[12|10:5|4:1|11].  One test per bit
# position proves each field lands in the right place (a swapped 11/12 pair is
# the classic bug, and it only shows up on large or negative offsets).
RV_B_SINGLE_BIT = [
    (0x00000163, 1 << 1),
    (0x00000263, 1 << 2),
    (0x00000463, 1 << 3),
    (0x00000863, 1 << 4),
    (0x02000063, 1 << 5),
    (0x04000063, 1 << 6),
    (0x08000063, 1 << 7),
    (0x10000063, 1 << 8),
    (0x20000063, 1 << 9),
    (0x40000063, 1 << 10),
    (0x000000E3, 1 << 11),      # imm[11] lives in bit 7 of the word
    (0x80000063, u32(-4096)),   # imm[12] is the sign bit
]


@pytest.mark.parametrize("word,expected", RV_B_SINGLE_BIT)
def test_riscv_b_type_immediate_bit_scrambling(word, expected):
    out = rv(word)
    assert out["opcode"] == Opcode.BRANCH
    assert out["imm"] == expected, f"0x{word:08X} -> 0x{out['imm']:08X}"


@pytest.mark.parametrize("word,expected,note", [
    (0x00208463, 8,          "BEQ x1, x2, +8"),
    (0xFE208EE3, u32(-4),    "BEQ x1, x2, -4"),
    (0x7E208FE3, 4094,       "BEQ x1, x2, +4094 (max B-imm)"),
    (0x80208063, u32(-4096), "BEQ x1, x2, -4096 (min B-imm)"),
])
def test_riscv_b_type_immediate_extremes(word, expected, note):
    assert rv(word)["imm"] == expected, note


def test_riscv_b_type_immediate_is_always_even():
    """Bit 0 of the B-offset is implicit-zero — it is never taken from the word."""
    for word, _ in RV_B_SINGLE_BIT:
        assert rv(word)["imm"] & 1 == 0


# J-type scrambles as imm[20|10:1|11|19:12].
RV_J_SINGLE_BIT = [
    (0x0020006F, 1 << 1),
    (0x0040006F, 1 << 2),
    (0x0080006F, 1 << 3),
    (0x0100006F, 1 << 4),
    (0x0200006F, 1 << 5),
    (0x0400006F, 1 << 6),
    (0x0800006F, 1 << 7),
    (0x1000006F, 1 << 8),
    (0x2000006F, 1 << 9),
    (0x4000006F, 1 << 10),
    (0x0010006F, 1 << 11),        # imm[11] lives in word bit 20
    (0x0000106F, 1 << 12),        # imm[19:12] are word bits 19:12 (in order)
    (0x0000206F, 1 << 13),
    (0x0000406F, 1 << 14),
    (0x0000806F, 1 << 15),
    (0x0001006F, 1 << 16),
    (0x0002006F, 1 << 17),
    (0x0004006F, 1 << 18),
    (0x0008006F, 1 << 19),
    (0x8000006F, u32(-1048576)),  # imm[20] is the sign bit
]


@pytest.mark.parametrize("word,expected", RV_J_SINGLE_BIT)
def test_riscv_j_type_immediate_bit_scrambling(word, expected):
    out = rv(word)
    assert out["opcode"] == Opcode.JAL
    assert out["imm"] == expected, f"0x{word:08X} -> 0x{out['imm']:08X}"


@pytest.mark.parametrize("word,expected,note", [
    (0x008000EF, 8,             "JAL x1, +8"),
    (0xFFDFF0EF, u32(-4),       "JAL x1, -4"),
    (0x7FFFF0EF, 1048574,       "JAL x1, +1048574 (max J-imm)"),
    (0x800000EF, u32(-1048576), "JAL x1, -1048576 (min J-imm)"),
])
def test_riscv_j_type_immediate_extremes(word, expected, note):
    assert rv(word)["imm"] == expected, note


def test_riscv_u_type_immediate_is_not_sign_extended_it_is_shifted():
    """U-type takes bits [31:12] verbatim — the low 12 bits are cleared."""
    assert rv(0xFFFFF0B7)["imm"] == 0xFFFFF000     # LUI x1, 0xFFFFF
    assert rv(0x80000297)["imm"] == 0x80000000     # AUIPC x5, 0x80000
    assert rv(0x000000B7)["imm"] == 0x00000000     # LUI x1, 0


def test_riscv_immediate_is_zero_for_r_type_and_unknown_opcodes():
    """Opcodes with no immediate field must emit imm = 0, not stale bits."""
    assert rv(0x002081B3)["imm"] == 0    # ADD x3, x1, x2
    assert rv(0x00000073)["imm"] == 0    # ECALL (SYSTEM, not modelled)
    assert rv(0xFFFFFFFF)["imm"] == 0    # opcode 0x7F is not an RV32I opcode


# ── illegal / unmodelled opcodes ────────────────────────────────────────────

RV_CONTROL_PORTS = ("alu_op", "alu_src", "use_pc", "mem_read", "mem_write",
                    "reg_write", "branch", "jal", "jalr", "imm")


@pytest.mark.parametrize("word,note", [
    (0x00000000, "all zeros (opcode 0x00 — not an RV32I opcode)"),
    (0x0000007F, "opcode 0x7F — reserved / >32-bit encoding escape"),
    (0xFFFFFFFF, "all ones"),
    (0x00000073, "ECALL — SYSTEM opcode is declared but NOT decoded"),
    (0x00100073, "EBREAK — likewise"),
    (0x0000000F, "FENCE — MISC-MEM opcode, not modelled"),
])
def test_riscv_unknown_opcode_asserts_no_control_signals(word, note):
    """SAFETY: an unrecognised opcode must be inert, not a random instruction.

    The decoder resets every control output before the opcode `if` chain and
    has no `else`, so an unmatched opcode falls through with everything low.
    Verified, not assumed — this is what stops a garbage word from writing a
    register or storing to memory.
    """
    out = rv(word)
    for port in RV_CONTROL_PORTS:
        assert out[port] == 0, f"{note}: {port} = {out[port]}"
    assert out["branch_cond"] == BranchCond.NEVER
    assert out["wb_sel"] == WBSel.ALU


def test_riscv_unknown_opcode_still_reports_the_raw_fields():
    """Field extraction is unconditional — the UI shows it even for junk."""
    out = rv(0xFFFFFFFF)
    assert (out["rd"], out["rs1"], out["rs2"]) == (31, 31, 31)
    assert out["opcode"] == 0x7F


# ── assembler <-> decoder round trip on a whole program ─────────────────────

def test_riscv_assembler_decoder_round_trip_program():
    """Assemble a mixed program and confirm the decoder recovers each intent."""
    words = assemble("riscv", """
        ADDI x1, x0, 5
        ADD  x3, x1, x2
        SW   x3, 0(x0)
        LW   x5, 0(x0)
        BEQ  x1, x2, +8
        JAL  x1, +8
        LUI  x6, 0x1
        AUIPC x7, 0x1
    """)
    kinds = []
    for w in words:
        out = rv(w)
        if out["jal"]:
            kinds.append("jal")
        elif out["branch"]:
            kinds.append("branch")
        elif out["mem_read"]:
            kinds.append("load")
        elif out["mem_write"]:
            kinds.append("store")
        elif out["use_pc"]:
            kinds.append("auipc")
        elif out["alu_op"] == AluOp.PASS:
            kinds.append("lui")
        elif out["alu_src"]:
            kinds.append("alu_imm")
        else:
            kinds.append("alu_reg")
    assert kinds == ["alu_imm", "alu_reg", "store", "load",
                     "branch", "jal", "lui", "auipc"]


def test_riscv_get_state_snapshot_shape():
    d = RISCVDecoder()
    d["instr_in"] = assemble("riscv", "SUB x1, x2, x3")[0]
    d.evaluate()
    assert d.get_state() == {
        "rd": 1, "rs1": 2, "rs2": 3, "imm": 0,
        "alu_op": "SUB", "alu_src": 0,
        "mem_read": 0, "mem_write": 0, "reg_write": 1,
        "branch": 0, "jal": 0, "jalr": 0,
    }


# ════════════════════════════════════════════════════════════════════════════
#  ARM AArch64 (A64) — only the subset the ARMDecoder actually models
# ════════════════════════════════════════════════════════════════════════════
# Modelled (per the module docstring + source):
#   DP-register arith  ADD / SUB / SUBS (CMP when Rd == XZR)
#   DP-register logical AND / ORR / EOR
#   DP-immediate       ADD / SUB (with optional <<12)
#   Move wide          MOVZ
#   Load/store         LDR / STR, unsigned scaled offset
#   Branch             B, BL, B.cond, CBZ, CBNZ, RET
# Everything else (including all sf=0 / 32-bit W-register forms) decodes to
# "no control asserted".

ARM_PORTS = [p for p in ARMDecoder.ports_spec if p != "instr_in"]


def arm(word):
    d = ARMDecoder()
    d["instr_in"] = word
    d.evaluate()
    return {p: d[p] for p in ARM_PORTS}


def arm_asm(text):
    words = assemble("arm", text)
    assert len(words) == 1, f"expected 1 word from {text!r}, got {words}"
    return arm(words[0]), words[0]


# ── constants module ────────────────────────────────────────────────────────

def test_arm_bits_helper_extracts_inclusive_ranges():
    assert bits(0b1011, 3, 0) == 0b1011
    assert bits(0b1011, 3, 3) == 1
    assert bits(0b1011, 1, 0) == 0b11
    assert bits(0xFFFFFFFF, 31, 0) == 0xFFFFFFFF
    assert bits(0xD503201F, 31, 24) == 0xD5
    assert bits(0x8B020023, 4, 0) == 3       # Rd
    assert bits(0x8B020023, 9, 5) == 1       # Rn
    assert bits(0x8B020023, 20, 16) == 2     # Rm


def test_arm_condition_codes_are_the_real_a64_encodings():
    assert (A64Cond.EQ, A64Cond.NE) == (0x0, 0x1)
    assert (A64Cond.GE, A64Cond.LT) == (0xA, 0xB)
    assert (A64Cond.GT, A64Cond.LE) == (0xC, 0xD)
    # All 12 declared codes must be distinct 4-bit values.
    codes = [v for k, v in vars(A64Cond).items() if not k.startswith("_")]
    assert len(codes) == len(set(codes)) == 12
    assert all(0 <= c <= 0xF for c in codes)


# ── NOP ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("word,note", [
    (0x00000000, "all-zero word is treated as a NOP by an explicit early return"),
    (0xD503201F, "the real A64 NOP (hint #0)"),
])
def test_arm_nop_encodings_short_circuit(word, note):
    out = arm(word)
    assert all(out[p] == 0 for p in ARM_PORTS if p != "wb_sel"), note
    assert out["wb_sel"] == WBSel.ALU


# ── DP register: arithmetic ─────────────────────────────────────────────────

def test_arm_add_register():
    out, word = arm_asm("ADD X3, X1, X2")
    assert word == 0x8B020023
    assert (out["rd"], out["rs1"], out["rs2"]) == (3, 1, 2)
    assert out["alu_op"] == AluOp.ADD
    assert out["alu_src"] == 0
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.ALU
    assert out["sets_flags"] == 0
    assert out["imm"] == 0
    assert out["mem_read"] == out["mem_write"] == 0
    assert out["branch"] == out["jal"] == out["jalr"] == 0


def test_arm_sub_register_does_not_set_flags():
    out, word = arm_asm("SUB X4, X1, X2")
    assert word == 0xCB020024
    assert out["alu_op"] == AluOp.SUB
    assert out["reg_write"] == 1
    assert out["sets_flags"] == 0     # plain SUB (opc=10) is flag-transparent


def test_arm_subs_sets_flags_and_still_writes_rd():
    out, word = arm_asm("SUBS X5, X1, X2")
    assert word == 0xEB020025
    assert out["rd"] == 5
    assert out["alu_op"] == AluOp.SUB
    assert out["sets_flags"] == 1
    assert out["reg_write"] == 1


def test_arm_cmp_is_subs_to_xzr_and_suppresses_the_register_write():
    """CMP Xn, Xm == SUBS XZR, Xn, Xm.  Rd == 31 forces reg_write low."""
    out, word = arm_asm("CMP X1, X2")
    assert word == 0xEB02003F
    assert out["rd"] == 31           # XZR
    assert (out["rs1"], out["rs2"]) == (1, 2)
    assert out["alu_op"] == AluOp.SUB
    assert out["sets_flags"] == 1
    assert out["reg_write"] == 0     # <-- the CMP/SUBS discriminator
    assert out["branch"] == 0


def test_arm_subs_to_xzr_and_cmp_are_the_same_encoding():
    cmp_word = assemble("arm", "CMP  X1, X2")[0]
    subs_word = assemble("arm", "SUBS XZR, X1, X2")[0]
    assert cmp_word == subs_word
    assert arm(cmp_word) == arm(subs_word)


def test_arm_dp_reg_arith_unmapped_opc_leaves_alu_op_at_add():
    """opc = 01 (ADDS) is NOT modelled — it decodes as a flag-free ADD.

    _decode_dp_reg_arith only handles opc 00/10/11, so ADDS silently loses its
    flag write.  Pinned as a known modelling gap.
    """
    adds = (1 << 31) | (0b01 << 29) | (0b01011 << 24) | (2 << 16) | (1 << 5) | 3
    out = arm(adds)
    assert out["alu_op"] == AluOp.ADD
    assert out["reg_write"] == 1
    assert out["sets_flags"] == 0    # <-- the gap: ADDS should set flags


# ── DP register: logical ────────────────────────────────────────────────────

ARM_LOGICAL = [
    ("AND X3, X1, X2", 0x8A020023, AluOp.AND),
    ("ORR X4, X1, X2", 0xAA020024, AluOp.OR),
    ("EOR X5, X1, X2", 0xCA020025, AluOp.XOR),
]


@pytest.mark.parametrize("text,word_expected,alu_op", ARM_LOGICAL)
def test_arm_logical_register_forms(text, word_expected, alu_op):
    out, word = arm_asm(text)
    assert word == word_expected
    assert out["alu_op"] == alu_op
    assert out["rs1"] == 1 and out["rs2"] == 2
    assert out["alu_src"] == 0
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.ALU
    assert out["sets_flags"] == 0     # ANDS is not modelled
    assert out["mem_read"] == out["mem_write"] == 0


def test_arm_dp_reg_logical_unmapped_opc_leaves_alu_op_at_add():
    """opc = 11 (ANDS / BIC family) is not modelled -> alu_op stays ADD."""
    ands = (1 << 31) | (0b11 << 29) | (0b01010 << 24) | (2 << 16) | (1 << 5) | 3
    out = arm(ands)
    assert out["alu_op"] == AluOp.ADD
    assert out["reg_write"] == 1
    assert out["sets_flags"] == 0


# ── DP immediate ────────────────────────────────────────────────────────────

def test_arm_add_immediate():
    out, word = arm_asm("ADD X2, X1, #5")
    assert word == 0x91001422
    assert (out["rd"], out["rs1"]) == (2, 1)
    assert out["imm"] == 5
    assert out["alu_op"] == AluOp.ADD
    assert out["alu_src"] == 1        # <-- register vs immediate discriminator
    assert out["rs2"] == 0            # no Rm field in the immediate form
    assert out["reg_write"] == 1
    assert out["sets_flags"] == 0


def test_arm_sub_immediate():
    out, word = arm_asm("SUB X1, X1, #1")
    assert word == 0xD1000421
    assert out["imm"] == 1            # magnitude only; alu_op carries the sign
    assert out["alu_op"] == AluOp.SUB
    assert out["alu_src"] == 1
    assert out["reg_write"] == 1


def test_arm_dp_immediate_shift_bit_scales_by_4096():
    """The `sh` bit (word bit 22) means imm12 << 12."""
    out, word = arm_asm("ADD X2, X1, #4096")
    assert word == 0x91400422
    assert bits(word, 22, 22) == 1
    assert out["imm"] == 4096
    # Hand-built: imm12 = 0xFFF with sh = 1 -> 0xFFF000.
    shifted = (1 << 31) | (0b00 << 29) | (0b100010 << 23) | \
              (1 << 22) | (0xFFF << 10) | (1 << 5) | 2
    assert arm(shifted)["imm"] == 0xFFF000


@pytest.mark.parametrize("opc,note", [(0b01, "ADDS imm"), (0b11, "SUBS imm")])
def test_arm_dp_immediate_flag_setting_forms_are_not_modelled(opc, note):
    """Only opc 00 (ADD) / 10 (SUB) are handled; 01/11 lose alu_op AND flags."""
    word = (1 << 31) | (opc << 29) | (0b100010 << 23) | (5 << 10) | (1 << 5) | 2
    out = arm(word)
    assert out["imm"] == 5
    assert out["alu_src"] == 1
    assert out["reg_write"] == 1
    assert out["alu_op"] == AluOp.ADD    # reset value, even for SUBS-imm
    assert out["sets_flags"] == 0, note


# ── MOVZ ────────────────────────────────────────────────────────────────────

def test_arm_movz_reads_xzr_and_adds_the_immediate():
    out, word = arm_asm("MOVZ X1, #10")
    assert word == 0xD2800141
    assert out["rd"] == 1
    assert out["rs1"] == 31          # XZR — MOVZ is modelled as 0 + imm
    assert out["imm"] == 10
    assert out["alu_op"] == AluOp.ADD
    assert out["alu_src"] == 1
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.ALU
    assert out["sets_flags"] == 0


@pytest.mark.parametrize("hw,imm16,expected", [
    (0, 0xFFFF, 0xFFFF),
    (1, 0x0001, 0x0001_0000),
    (2, 0xABCD, 0),          # imm port is 32 bits: ABCD << 32 masks to 0
    (3, 0x0001, 0),          # likewise
])
def test_arm_movz_halfword_shift(hw, imm16, expected):
    """MOVZ shifts imm16 by hw*16.  hw >= 2 shifts past the 32-bit imm port.

    A64 is a 64-bit ISA but this simulator has a 32-bit datapath, so the two
    upper halfword slots decode to zero.  The assembler only ever emits hw 0/1.
    """
    word = (1 << 31) | (0b10 << 29) | (0b100101 << 23) | (hw << 21) | (imm16 << 5) | 1
    out = arm(word)
    assert out["imm"] == expected
    assert out["rd"] == 1 and out["rs1"] == 31
    assert out["reg_write"] == 1


def test_arm_movz_assembler_emits_hw_1_for_high_halfwords():
    out, word = arm_asm("MOVZ X1, #65536")
    assert word == 0xD2A00021
    assert bits(word, 22, 21) == 1
    assert out["imm"] == 65536


# ── Load / Store (unsigned scaled offset) ───────────────────────────────────

def test_arm_ldr_control_bundle():
    out, word = arm_asm("LDR X2, [X1, #8]")
    assert out["rd"] == 2            # Rt is the destination for a load
    assert out["rs1"] == 1           # Rn is the base
    assert out["rs2"] == 0
    assert out["imm"] == 8
    assert out["alu_op"] == AluOp.ADD
    assert out["alu_src"] == 1
    assert out["mem_read"] == 1
    assert out["mem_write"] == 0
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.MEMORY


def test_arm_str_routes_rt_to_rs2_and_writes_no_register():
    out, word = arm_asm("STR X1, [X0, #0]")
    assert word == 0xF9000001
    assert out["rs2"] == 1           # Rt becomes the store DATA source
    assert out["rd"] == 0
    assert out["rs1"] == 0           # X0 base
    assert out["mem_write"] == 1
    assert out["mem_read"] == 0
    assert out["reg_write"] == 0
    assert out["wb_sel"] == WBSel.ALU


@pytest.mark.parametrize("size,imm12,expected_offset,note", [
    (0b00, 5, 5,   "size=0 (byte)     -> scale 1"),
    (0b01, 5, 10,  "size=1 (halfword) -> scale 2"),
    (0b10, 5, 20,  "size=2 (word)     -> scale 4"),
    (0b11, 5, 40,  "size=3 (doubleword) -> scale 8"),
])
def test_arm_ldst_offset_is_scaled_by_the_size_field(size, imm12, expected_offset, note):
    """offset = imm12 << size.  The assembler only emits size=3 (scale 8)."""
    word = (size << 30) | (0b111 << 27) | (0b01 << 24) | (1 << 22) | \
           (imm12 << 10) | (1 << 5) | 2
    assert arm(word)["imm"] == expected_offset, note


def test_arm_ldst_max_scaled_offset():
    out, word = arm_asm("LDR X2, [X1, #32760]")
    assert bits(word, 21, 10) == 0xFFF        # imm12 saturated
    assert out["imm"] == 0xFFF * 8 == 32760


def test_arm_ldst_offset_is_unsigned_there_is_no_negative_form():
    """imm12 is unsigned in the scaled form — the decoder never sign-extends."""
    word = (0b11 << 30) | (0b111 << 27) | (0b01 << 24) | (1 << 22) | \
           (0xFFF << 10) | (1 << 5) | 2
    assert arm(word)["imm"] == 32760          # NOT a negative number


# ── Branches ────────────────────────────────────────────────────────────────

def test_arm_b_unconditional_writes_no_link_register():
    out, word = arm_asm("B +8")
    assert word == 0x14000002
    assert out["imm"] == 8
    assert out["jal"] == 1
    assert out["reg_write"] == 0     # <-- the B/BL discriminator
    assert out["rd"] == 0
    assert out["branch"] == 0
    assert out["jalr"] == 0
    assert out["wb_sel"] == WBSel.ALU


def test_arm_b_backward_offset_is_sign_extended_from_26_bits():
    out, word = arm_asm("B -8")
    assert word == 0x17FFFFFE
    assert out["imm"] == u32(-8)


def test_arm_bl_targets_x30_with_pc4():
    out, word = arm_asm("BL +8")
    assert word == 0x94000002
    assert out["imm"] == 8
    assert out["rd"] == 30           # X30 = link register, implicit in the encoding
    assert out["jal"] == 1
    assert out["reg_write"] == 1     # <-- the B/BL discriminator
    assert out["wb_sel"] == WBSel.PC4
    assert out["branch"] == 0


def test_arm_bl_backward_offset_sign_extended():
    out, _ = arm_asm("BL -8")
    assert out["imm"] == u32(-8)
    assert out["rd"] == 30


@pytest.mark.parametrize("imm26,expected", [
    (0x1FFFFFF, 0x7FFFFFC),          # largest positive: (2^25 - 1) << 2
    (0x2000000, u32(-(1 << 27))),    # sign bit of imm26 set
])
def test_arm_b_immediate_extremes(imm26, expected):
    word = (0b000101 << 26) | imm26
    assert arm(word)["imm"] == expected


ARM_BCOND = [
    ("B.EQ", A64Cond.EQ, BranchCond.EQ,  AluOp.SUB),
    ("B.NE", A64Cond.NE, BranchCond.NEQ, AluOp.SUB),
    ("B.LT", A64Cond.LT, BranchCond.LT,  AluOp.SLT),
    ("B.GE", A64Cond.GE, BranchCond.GE,  AluOp.SLT),
    ("B.GT", A64Cond.GT, BranchCond.GT,  AluOp.SLT),
    ("B.LE", A64Cond.LE, BranchCond.LE,  AluOp.SLT),
]


@pytest.mark.parametrize("mnemonic,a64_cond,branch_cond,alu_op", ARM_BCOND)
def test_arm_bcond_condition_mapping(mnemonic, a64_cond, branch_cond, alu_op):
    out, word = arm_asm(f"{mnemonic} +8")
    assert bits(word, 31, 24) == 0b01010100
    assert bits(word, 3, 0) == a64_cond
    assert out["branch"] == 1
    assert out["branch_cond"] == branch_cond
    assert out["alu_op"] == alu_op
    assert out["imm"] == 8
    assert out["reg_write"] == 0
    assert out["jal"] == out["jalr"] == 0
    assert out["mem_read"] == out["mem_write"] == 0
    # B.cond consumes flags latched by a preceding SUBS/CMP; it sets none itself.
    assert out["sets_flags"] == 0


@pytest.mark.parametrize("cond,name", [
    (A64Cond.HS, "HS"), (A64Cond.LO, "LO"),
    (A64Cond.MI, "MI"), (A64Cond.PL, "PL"),
    (A64Cond.HI, "HI"), (A64Cond.LS, "LS"),
    (0x6, "reserved 0x6"), (0x7, "reserved 0x7"),
    (0xE, "AL"), (0xF, "NV"),
])
def test_arm_unsupported_bcond_is_a_branch_that_is_never_taken(cond, name):
    """SAFETY: the datapath has no C or V flag, so unsigned/overflow conditions
    decode to BranchCond.NEVER rather than to a wrong condition.

    `branch` still asserts (the instruction IS a branch) but the branch unit
    will never take it.  The ARM assembler refuses to emit these (see
    _BCOND_SUPPORTED), so they only arrive from raw hex programs.
    """
    word = (0b01010100 << 24) | (2 << 5) | cond
    out = arm(word)
    assert out["branch"] == 1, name
    assert out["branch_cond"] == BranchCond.NEVER, name
    assert out["alu_op"] == AluOp.ADD    # reset value; no compare is set up
    assert out["imm"] == 8
    assert out["reg_write"] == 0


def test_arm_bcond_imm19_is_sign_extended_from_21_bits():
    out, word = arm_asm("B.NE -8")
    assert word == 0x54FFFFC1
    assert out["imm"] == u32(-8)
    # Hand-built extremes: imm19 = 0x3FFFF (max +) and 0x40000 (sign bit).
    assert arm((0b01010100 << 24) | (0x3FFFF << 5) | A64Cond.EQ)["imm"] == 0xFFFFC
    assert arm((0b01010100 << 24) | (0x40000 << 5) | A64Cond.EQ)["imm"] == u32(-(1 << 20))


# ── CBZ / CBNZ ──────────────────────────────────────────────────────────────

def test_arm_cbz_compares_rt_against_xzr():
    out, word = arm_asm("CBZ X1, +8")
    assert word == 0xB4000041
    assert out["rs1"] == 1           # Rt is the value under test
    assert out["rs2"] == 31          # ... compared against XZR (reads as 0)
    assert out["alu_op"] == AluOp.SUB
    assert out["alu_src"] == 0
    assert out["branch"] == 1
    assert out["branch_cond"] == BranchCond.EQ
    assert out["imm"] == 8
    assert out["reg_write"] == 0
    # CBZ/CBNZ compute their own comparison, so they publish fresh flags.
    assert out["sets_flags"] == 1


def test_arm_cbnz_only_flips_the_condition():
    out, word = arm_asm("CBNZ X1, -8")
    assert word == 0xB5FFFFC1
    assert bits(word, 24, 24) == 1               # op = 1 selects CBNZ
    assert out["branch_cond"] == BranchCond.NEQ
    assert out["imm"] == u32(-8)
    assert out["sets_flags"] == 1

    cbz = arm(assemble("arm", "CBZ  X1, -8")[0])
    cbnz = arm(assemble("arm", "CBNZ X1, -8")[0])
    assert {k for k in cbz if cbz[k] != cbnz[k]} == {"branch_cond"}


# ── RET ─────────────────────────────────────────────────────────────────────

def test_arm_ret_defaults_to_x30():
    out, word = arm_asm("RET")
    assert word == 0xD65F03C0
    assert out["rs1"] == 30
    assert out["jalr"] == 1
    assert out["jal"] == 0
    assert out["branch"] == 0
    assert out["imm"] == 0
    assert out["reg_write"] == 0     # RET does not write a link register back
    assert out["wb_sel"] == WBSel.ALU


def test_arm_ret_honours_an_explicit_register():
    out, word = arm_asm("RET X5")
    assert word == 0xD65F00A0
    assert out["rs1"] == 5
    assert out["jalr"] == 1


def test_arm_br_is_not_modelled():
    """BR Xn (0xD61F....) differs from RET only in bit 21+; it is NOT decoded."""
    out = arm(0xD61F03C0)
    assert out["jalr"] == 0
    assert all(out[p] == 0 for p in ARM_PORTS if p != "wb_sel")


# ── unrecognised encodings ──────────────────────────────────────────────────

ARM_CONTROL_PORTS = ("rd", "rs1", "rs2", "imm", "alu_op", "alu_src", "use_pc",
                     "mem_read", "mem_write", "reg_write", "branch", "jal",
                     "jalr", "sets_flags")


@pytest.mark.parametrize("word,note", [
    (0xFFFFFFFF, "all ones"),
    (0x12345678, "arbitrary junk"),
    (0x0B020023, "ADD W3, W1, W2 — sf=0 (32-bit form) is not modelled"),
    (0x11001422, "ADD W2, W1, #5 — sf=0 immediate form"),
    (0x52800141, "MOVZ W1, #10 — sf=0 move-wide"),
    (0x9B027C20, "MADD X0, X1, X2, XZR — no multiply in this datapath"),
    (0xD4000001, "SVC #0 — no exceptions modelled"),
])
def test_arm_unknown_encoding_asserts_no_control(word, note):
    """SAFETY: unmatched encodings fall off the end of the if/elif chain with
    every control output still at its reset value."""
    out = arm(word)
    for port in ARM_CONTROL_PORTS:
        assert out[port] == 0, f"{note}: {port} = {out[port]}"
    assert out["branch_cond"] == BranchCond.NEVER
    assert out["wb_sel"] == WBSel.ALU


# ── XZR handling ────────────────────────────────────────────────────────────

def test_arm_xzr_is_index_31_in_the_register_name_table():
    assert len(ARM_REGS) == 32
    assert ARM_REGS[0] == "X0"
    assert ARM_REGS[30] == "X30"
    assert ARM_REGS[31] == "XZR"


def test_arm_xzr_as_a_source_decodes_as_register_31():
    """The decoder emits index 31; the REGFILE is what forces it to read 0."""
    out, _ = arm_asm("ADD X2, X3, XZR")
    assert out["rs2"] == 31
    assert out["reg_write"] == 1


def test_arm_xzr_as_a_destination_still_asserts_reg_write_except_for_subs():
    """Only SUBS special-cases Rd == XZR (that is what makes it a CMP).

    MOVZ XZR, #10 keeps reg_write = 1; suppressing the write is the register
    file's job (zero_reg_index = 31), not the decoder's.
    """
    out, _ = arm_asm("MOVZ XZR, #10")
    assert out["rd"] == 31
    assert out["reg_write"] == 1     # decoder does NOT gate this

    out, _ = arm_asm("CMP X1, X2")
    assert out["rd"] == 31
    assert out["reg_write"] == 0     # SUBS DOES gate it


# ── decoder-level state snapshot ────────────────────────────────────────────

def test_arm_get_state_snapshot_shape():
    d = ARMDecoder()
    d["instr_in"] = assemble("arm", "SUB X4, X1, X2")[0]
    d.evaluate()
    assert d.get_state() == {
        "rd": 4, "rs1": 1, "rs2": 2, "imm": 0,
        "alu_op": "SUB", "alu_src": 0,
        "mem_read": 0, "mem_write": 0, "reg_write": 1,
        "branch": 0, "jal": 0, "jalr": 0,
    }


def test_arm_assembler_decoder_round_trip_program():
    words = assemble("arm", """
        MOVZ X1, #10
        ADD  X4, X2, X3
        SUB  X1, X1, #1
        STR  X1, [X0, #0]
        LDR  X2, [X0, #0]
        CMP  X1, XZR
        B.NE -4
        BL   +8
        RET
    """)
    kinds = []
    for w in words:
        out = arm(w)
        if out["jalr"]:
            kinds.append("ret")
        elif out["jal"]:
            kinds.append("bl" if out["reg_write"] else "b")
        elif out["branch"]:
            kinds.append("bcond")
        elif out["mem_read"]:
            kinds.append("ldr")
        elif out["mem_write"]:
            kinds.append("str")
        elif out["sets_flags"]:
            kinds.append("cmp")
        elif out["alu_src"]:
            kinds.append("alu_imm")
        else:
            kinds.append("alu_reg")
    assert kinds == ["alu_imm", "alu_reg", "alu_imm", "str", "ldr",
                     "cmp", "bcond", "bl", "ret"]


def test_arm_decoder_has_internal_flag_state_that_evaluate_never_touches():
    """__init__ creates _flag_z/_flag_n but evaluate() is purely combinational.

    Pinned because a future change that starts USING them would make the
    decoder stateful and break the pipeline/OoO presets, which re-evaluate the
    same decoder for different instructions within one cycle.
    """
    d = ARMDecoder()
    assert d._flag_z == 0 and d._flag_n == 0
    d["instr_in"] = assemble("arm", "CMP X1, X2")[0]
    d.evaluate()
    assert d._flag_z == 0 and d._flag_n == 0
