"""
Direct unit tests for the x86-32 (IA-32) instruction decoder.

x86 is the weakest-covered of the three decoders and by far the trickiest,
because it is the only VARIABLE-LENGTH one: the decoder must publish an
`instr_len` that the fetch unit uses to advance the PC, and a wrong length
desynchronises the whole byte stream (every later instruction decodes garbage).
So these tests assert `instr_len` for every supported encoding length as
aggressively as they assert the control bundle.

The decoder reads a 64-bit LITTLE-ENDIAN packed byte window (8 bytes from
ByteInstructionMemory), so all tests go through `x86([...bytes...])`, which
packs the list the same way the real ByteInstructionMemory does.

Modelled subset (from the module docstring + verified against the source):
  1 byte : NOP (90), RET (C3), PUSH r32 (50+r), POP r32 (58+r)
  2 bytes: ALU r/m32,r32 (01/09/21/29/31/39 with mod=11), MOV r/m32,r32 (89),
           MOV r32,r/m32 (8B), MOV via [reg] (mod=00), JMP rel8 (EB),
           Jcc rel8 (70-7F)
  3 bytes: Group 1 imm8 (83 /ext), MOV with disp8 (mod=01)
  5 bytes: MOV r32,imm32 (B8+r), CALL rel32 (E8), JMP rel32 (E9)
  6 bytes: Jcc rel32 (0F 80-8F)

Everything asserted here was verified against the real source.  Several tests
pin DELIBERATE SIMPLIFICATIONS or known gaps (POP/ESP, CALL not pushing a
return address, memory-destination ALU forms being length-only) — those are
labelled in the docstrings so nobody "fixes" a test instead of the code.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.assembler import assemble
from sim.core.signals import AluOp, BranchCond, JccCond, WBSel
from sim.isa.x86.constants import (
    GRP1_ADD, GRP1_AND, GRP1_CMP, GRP1_OR, GRP1_SUB, GRP1_XOR,
    OP_ADD_R32, OP_AND_R32, OP_CALL_REL32, OP_CMP_R32, OP_IMM_GRP1,
    OP_JMP_REL8, OP_MOV_R_RM, OP_MOV_RM_R, OP_NOP, OP_OR_R32, OP_RET,
    OP_SUB_R32, OP_XOR_R32, REG_EAX, REG_EBP, REG_EBX, REG_ECX, REG_EDI,
    REG_EDX, REG_ESI, REG_ESP, REGISTER_NAMES,
    modrm_mod, modrm_reg, modrm_rm,
)
from sim.isa.x86.decoder_component import X86Decoder, _byte, _modrm_disp_len


X86_PORTS = [p for p in X86Decoder.ports_spec if p not in ("instr_in", "pc_in")]


def pack(byte_list):
    """Pack up to 8 bytes into the 64-bit little-endian window the decoder reads.

    This mirrors what ByteInstructionMemory hands the decoder each cycle.
    """
    value = 0
    for i, b in enumerate(byte_list[:8]):
        value |= (b & 0xFF) << (8 * i)
    return value


def x86(byte_list, pc=0):
    """Decode one x86 instruction from a byte list; return {port: value}."""
    d = X86Decoder()
    d["instr_in"] = pack(byte_list)
    d["pc_in"] = pc
    d.evaluate()
    return {p: d[p] for p in X86_PORTS}


def x86_asm(text, pc=0):
    """Assemble one x86 instruction and decode it (round-trip check)."""
    byte_list = assemble("x86", text)
    return x86(byte_list, pc=pc), byte_list


def u32(value):
    return value & 0xFFFF_FFFF


def modrm(mod, reg, rm):
    return ((mod & 3) << 6) | ((reg & 7) << 3) | (rm & 7)


# ════════════════════════════════════════════════════════════════════════════
#  constants.py + helpers
# ════════════════════════════════════════════════════════════════════════════

def test_x86_register_indices_match_the_architectural_encoding():
    assert (REG_EAX, REG_ECX, REG_EDX, REG_EBX) == (0, 1, 2, 3)
    assert (REG_ESP, REG_EBP, REG_ESI, REG_EDI) == (4, 5, 6, 7)
    assert REGISTER_NAMES == ["EAX", "ECX", "EDX", "EBX",
                              "ESP", "EBP", "ESI", "EDI"]
    for idx, name in enumerate(REGISTER_NAMES):
        assert {"EAX": REG_EAX, "ECX": REG_ECX, "EDX": REG_EDX, "EBX": REG_EBX,
                "ESP": REG_ESP, "EBP": REG_EBP, "ESI": REG_ESI,
                "EDI": REG_EDI}[name] == idx


def test_x86_modrm_field_helpers():
    assert modrm_mod(0b11_010_001) == 0b11
    assert modrm_reg(0b11_010_001) == 0b010
    assert modrm_rm(0b11_010_001) == 0b001
    assert (modrm_mod(0x00), modrm_reg(0x00), modrm_rm(0x00)) == (0, 0, 0)
    assert (modrm_mod(0xFF), modrm_reg(0xFF), modrm_rm(0xFF)) == (3, 7, 7)
    # Round-trip against the builder used throughout this file.
    for mod in range(4):
        for reg in range(8):
            for rm in range(8):
                b = modrm(mod, reg, rm)
                assert (modrm_mod(b), modrm_reg(b), modrm_rm(b)) == (mod, reg, rm)


def test_x86_group1_extension_codes():
    assert (GRP1_ADD, GRP1_OR, GRP1_AND) == (0, 1, 4)
    assert (GRP1_SUB, GRP1_XOR, GRP1_CMP) == (5, 6, 7)


def test_x86_byte_extractor_reads_little_endian_offsets():
    stream = pack([0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88])
    assert [_byte(stream, i) for i in range(8)] == \
        [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
    assert _byte(pack([0x90]), 1) == 0x00   # unread bytes read as zero


@pytest.mark.parametrize("mod,rm,expected,note", [
    (0b11, 0, 0, "register-direct: no displacement"),
    (0b11, 7, 0, "register-direct: no displacement"),
    (0b00, 0, 0, "[reg]: no displacement"),
    (0b00, 3, 0, "[EBX]: no displacement"),
    (0b00, 0b100, 1, "SIB byte follows (best-effort, no further disp)"),
    (0b00, 0b101, 4, "disp32 with no base register"),
    (0b01, 0, 1, "disp8"),
    (0b01, 0b101, 1, "disp8 (rm=101 is [EBP+disp8], NOT the disp32 escape)"),
    (0b10, 0, 4, "disp32"),
    (0b10, 0b100, 4, "disp32 (SIB length is NOT added — best-effort)"),
])
def test_x86_modrm_disp_len_table(mod, rm, expected, note):
    """The addressing-mode length table that keeps the byte stream aligned."""
    assert _modrm_disp_len(mod, rm) == expected, note


# ════════════════════════════════════════════════════════════════════════════
#  1-byte encodings
# ════════════════════════════════════════════════════════════════════════════

def test_x86_nop_is_one_byte_and_completely_inert():
    out, byte_list = x86_asm("NOP")
    assert byte_list == [OP_NOP]
    assert out["instr_len"] == 1
    for port in ("rd", "rs1", "rs2", "imm", "alu_op", "alu_src", "use_pc",
                 "mem_read", "mem_write", "reg_write", "branch", "jal",
                 "jalr", "write_flags"):
        assert out[port] == 0
    assert out["branch_cond"] == BranchCond.NEVER
    assert out["wb_sel"] == WBSel.ALU


def test_x86_ret_loads_from_esp_and_jumps_indirect():
    """RET is modelled as an indirect jump through [ESP] — but note it does NOT
    adjust ESP afterwards (the same single-write-port limit as POP)."""
    out, byte_list = x86_asm("RET")
    assert byte_list == [OP_RET]
    assert out["instr_len"] == 1
    assert out["rs1"] == REG_ESP
    assert out["imm"] == 0
    assert out["alu_op"] == AluOp.ADD      # address = ESP + 0
    assert out["alu_src"] == 1
    assert out["mem_read"] == 1
    assert out["jalr"] == 1
    assert out["jal"] == 0
    assert out["branch"] == 0
    assert out["reg_write"] == 0           # no ESP restore -> known gap
    assert out["write_flags"] == 0


@pytest.mark.parametrize("reg,name", list(enumerate(REGISTER_NAMES)))
def test_x86_push_is_fully_modelled_for_every_register(reg, name):
    """PUSH r32: store rs2 at [ESP-4] AND write the decremented ESP back.

    PUSH is the one stack op this datapath models completely, because the
    single write port lands on ESP and the store data comes from rs2.
    """
    out, byte_list = x86_asm(f"PUSH {name}")
    assert byte_list == [0x50 + reg]
    assert out["instr_len"] == 1
    assert out["rs1"] == REG_ESP           # address base
    assert out["rs2"] == reg               # the value being pushed
    assert out["imm"] == u32(-4)           # ESP - 4
    assert out["alu_op"] == AluOp.ADD
    assert out["alu_src"] == 1
    assert out["mem_write"] == 1
    assert out["mem_read"] == 0
    assert out["rd"] == REG_ESP            # ESP is written back
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.ALU      # ... with the ALU result (ESP-4)
    assert out["write_flags"] == 0
    assert out["branch"] == out["jal"] == out["jalr"] == 0


@pytest.mark.parametrize("reg,name", list(enumerate(REGISTER_NAMES)))
def test_x86_pop_loads_the_destination_but_does_not_restore_esp(reg, name):
    """POP r32 — pins the DOCUMENTED GAP (CLAUDE.md "Known limitations").

    Real POP does two register writes: rd <- [ESP] and ESP <- ESP+4.  This
    datapath has ONE write port and the decoder contract has ONE `rd`, so only
    the first happens.  We pin the CURRENT behaviour so the gap stays visible
    and a future two-write-port datapath is a deliberate, test-updating change.
    """
    out, byte_list = x86_asm(f"POP {name}")
    assert byte_list == [0x58 + reg]
    assert out["instr_len"] == 1
    assert out["rs1"] == REG_ESP           # address base
    assert out["imm"] == 0                 # loads from [ESP+0]
    assert out["alu_op"] == AluOp.ADD
    assert out["alu_src"] == 1
    assert out["mem_read"] == 1
    assert out["mem_write"] == 0
    assert out["rd"] == reg
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.MEMORY
    # THE GAP: the ESP += 4 second write is simply not expressible here.
    assert out["rd"] != REG_ESP or reg == REG_ESP
    assert out["write_flags"] == 0


def test_x86_push_and_pop_opcode_ranges_do_not_overlap():
    """0x50-0x57 is PUSH, 0x58-0x5F is POP — the boundary is a classic off-by-one."""
    assert x86([0x57])["mem_write"] == 1 and x86([0x57])["rs2"] == REG_EDI
    assert x86([0x58])["mem_read"] == 1 and x86([0x58])["rd"] == REG_EAX
    assert x86([0x4F])["instr_len"] == 1 and x86([0x4F])["mem_write"] == 0  # DEC: not modelled
    assert x86([0x60])["mem_read"] == 0 and x86([0x60])["mem_write"] == 0   # PUSHA: not modelled


# ════════════════════════════════════════════════════════════════════════════
#  2-byte encodings: ALU r/m32, r32  (mod = 11)
# ════════════════════════════════════════════════════════════════════════════

X86_ALU_R32 = [
    ("ADD", OP_ADD_R32, AluOp.ADD, True),
    ("OR",  OP_OR_R32,  AluOp.OR,  True),
    ("AND", OP_AND_R32, AluOp.AND, True),
    ("SUB", OP_SUB_R32, AluOp.SUB, True),
    ("XOR", OP_XOR_R32, AluOp.XOR, True),
    ("CMP", OP_CMP_R32, AluOp.SUB, False),   # CMP discards the result
]


@pytest.mark.parametrize("mnemonic,opcode,alu_op,writes", X86_ALU_R32)
def test_x86_alu_register_register_forms(mnemonic, opcode, alu_op, writes):
    """`ALU dst, src` encodes as `opcode /r` with reg=src and rm=dst."""
    out, byte_list = x86_asm(f"{mnemonic} EBX, ESI")
    assert byte_list == [opcode, modrm(0b11, REG_ESI, REG_EBX)]
    assert out["instr_len"] == 2
    assert out["rs1"] == REG_EBX           # rm -> operand A
    assert out["rs2"] == REG_ESI           # reg -> operand B
    assert out["alu_op"] == alu_op
    assert out["alu_src"] == 0             # register operand, not immediate
    assert out["write_flags"] == 1         # every ALU op updates EFLAGS
    assert out["mem_read"] == out["mem_write"] == 0
    assert out["branch"] == out["jal"] == out["jalr"] == 0
    if writes:
        assert out["rd"] == REG_EBX
        assert out["reg_write"] == 1
        assert out["wb_sel"] == WBSel.ALU
    else:
        assert out["reg_write"] == 0       # CMP: flags only
        assert out["rd"] == 0


def test_x86_cmp_and_sub_differ_only_in_the_register_write():
    # rm (the destination) must be NON-zero, or CMP's suppressed rd = 0 is
    # indistinguishable from SUB's rd = 0 and the diff looks smaller than it is.
    sub = x86([OP_SUB_R32, modrm(0b11, REG_ECX, REG_EBX)])
    cmp_ = x86([OP_CMP_R32, modrm(0b11, REG_ECX, REG_EBX)])
    assert sub["alu_op"] == cmp_["alu_op"] == AluOp.SUB
    assert sub["write_flags"] == cmp_["write_flags"] == 1
    assert {k for k in sub if sub[k] != cmp_[k]} == {"rd", "reg_write"}


def test_x86_opcode_to_alu_falls_back_to_add_for_an_unknown_opcode():
    """Defensive default in _opcode_to_alu — unreachable via evaluate(), since
    the caller already filtered to the six ALU opcodes, but pinned anyway."""
    assert X86Decoder()._opcode_to_alu(0xFF) == (AluOp.ADD, False)


# ── ALU with a MEMORY ModRM: length only, no datapath effect ────────────────

@pytest.mark.parametrize("modrm_byte,extra,expected_len,note", [
    (modrm(0b00, REG_ECX, REG_EAX),  [],               2, "[EAX], no disp"),
    (modrm(0b01, REG_ECX, REG_EAX),  [0x08],           3, "[EAX+disp8]"),
    (modrm(0b10, REG_ECX, REG_EAX),  [0, 0, 0, 0],     6, "[EAX+disp32]"),
    (modrm(0b00, REG_ECX, 0b101),    [0, 0, 0, 0],     6, "[disp32], no base"),
    (modrm(0b00, REG_ECX, 0b100),    [0x24],           3, "SIB byte"),
])
def test_x86_alu_with_memory_operand_computes_length_but_asserts_nothing(
        modrm_byte, extra, expected_len, note):
    """Read-modify-write-to-memory is NOT modelled by this single-pass datapath.

    The decoder deliberately computes only the correct `instr_len` so the
    variable-length fetch stream stays aligned; every control signal stays low
    so the instruction is a no-op rather than a wrong operation.  (These forms
    are reachable only from raw byte programs — the assembler never emits them.)
    """
    out = x86([OP_ADD_R32, modrm_byte] + extra)
    assert out["instr_len"] == expected_len, note
    for port in ("rd", "rs1", "rs2", "imm", "alu_src", "mem_read", "mem_write",
                 "reg_write", "branch", "jal", "jalr", "write_flags"):
        assert out[port] == 0, f"{note}: {port} asserted"


# ════════════════════════════════════════════════════════════════════════════
#  MOV r/m32, r32 (0x89) and MOV r32, r/m32 (0x8B)
# ════════════════════════════════════════════════════════════════════════════

def test_x86_mov_register_to_register_via_89():
    """0x89 mod=11: reg is the SOURCE, rm is the DESTINATION."""
    out, byte_list = x86_asm("MOV EBX, ESI")
    assert byte_list == [OP_MOV_RM_R, modrm(0b11, REG_ESI, REG_EBX)]
    assert out["instr_len"] == 2
    assert out["rs1"] == REG_ESI           # source feeds operand A
    assert out["rd"] == REG_EBX
    assert out["imm"] == 0
    assert out["alu_op"] == AluOp.ADD      # dst = src + 0
    assert out["alu_src"] == 1
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.ALU
    assert out["mem_read"] == out["mem_write"] == 0
    assert out["write_flags"] == 0         # MOV never touches EFLAGS


def test_x86_mov_register_to_register_via_8b_swaps_reg_and_rm():
    """0x8B mod=11 is the mirror image of 0x89: reg is the DESTINATION."""
    out = x86([OP_MOV_R_RM, modrm(0b11, REG_ESI, REG_EBX)])
    assert out["instr_len"] == 2
    assert out["rs1"] == REG_EBX           # rm -> source
    assert out["rd"] == REG_ESI            # reg -> destination
    assert out["alu_src"] == 1
    assert out["reg_write"] == 1
    assert out["write_flags"] == 0


def test_x86_mov_load_from_bare_register_indirect_mod00():
    out, byte_list = x86_asm("MOV EAX, [EBX]")
    assert byte_list == [OP_MOV_R_RM, modrm(0b00, REG_EAX, REG_EBX)]
    assert out["instr_len"] == 2
    assert out["rs1"] == REG_EBX           # address base
    assert out["rd"] == REG_EAX
    assert out["imm"] == 0
    assert out["alu_op"] == AluOp.ADD
    assert out["alu_src"] == 1
    assert out["mem_read"] == 1
    assert out["mem_write"] == 0
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.MEMORY


def test_x86_mov_store_to_bare_register_indirect_mod00():
    out, byte_list = x86_asm("MOV [EBX], EAX")
    assert byte_list == [OP_MOV_RM_R, modrm(0b00, REG_EAX, REG_EBX)]
    assert out["instr_len"] == 2
    assert out["rs1"] == REG_EBX           # address base
    assert out["rs2"] == REG_EAX           # store data
    assert out["imm"] == 0
    assert out["mem_write"] == 1
    assert out["mem_read"] == 0
    assert out["reg_write"] == 0
    assert out["wb_sel"] == WBSel.ALU


def test_x86_mov_load_with_positive_disp8_mod01():
    out, byte_list = x86_asm("MOV EAX, [EBX+4]")
    assert byte_list == [OP_MOV_R_RM, modrm(0b01, REG_EAX, REG_EBX), 0x04]
    assert out["instr_len"] == 3
    assert out["rs1"] == REG_EBX
    assert out["rd"] == REG_EAX
    assert out["imm"] == 4
    assert out["mem_read"] == 1
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.MEMORY


def test_x86_mov_load_with_negative_disp8_is_sign_extended():
    out, byte_list = x86_asm("MOV EAX, [EBX-4]")
    assert byte_list == [OP_MOV_R_RM, modrm(0b01, REG_EAX, REG_EBX), 0xFC]
    assert out["instr_len"] == 3
    assert out["imm"] == u32(-4)           # 0xFC sign-extends, not 252


def test_x86_mov_store_with_disp8_mod01():
    out, byte_list = x86_asm("MOV [EBX+8], EAX")
    assert byte_list == [OP_MOV_RM_R, modrm(0b01, REG_EAX, REG_EBX), 0x08]
    assert out["instr_len"] == 3
    assert out["rs1"] == REG_EBX
    assert out["rs2"] == REG_EAX
    assert out["imm"] == 8
    assert out["mem_write"] == 1
    assert out["reg_write"] == 0


@pytest.mark.parametrize("disp8,expected", [
    (0x00, 0), (0x01, 1), (0x7F, 127), (0x80, u32(-128)), (0xFF, u32(-1)),
])
def test_x86_mov_disp8_sign_extension_boundaries(disp8, expected):
    out = x86([OP_MOV_R_RM, modrm(0b01, REG_EAX, REG_EBX), disp8])
    assert out["imm"] == expected
    assert out["instr_len"] == 3


@pytest.mark.parametrize("rm,note", [
    (0b100, "rm=100 is really a SIB escape, decoded here as plain [ESP]"),
    (0b101, "rm=101 with mod=00 is really disp32, decoded here as plain [EBP]"),
])
def test_x86_mov_mod00_escape_encodings_are_treated_as_plain_indirect(rm, note):
    """_decode_mov_rm does NOT consult _modrm_disp_len for mod=00.

    Architecturally, mod=00/rm=100 means "a SIB byte follows" and mod=00/rm=101
    means "a disp32 follows".  This decoder treats both as bare [reg], giving
    instr_len = 2.  Pinned as a deliberate simplification: the project's own
    assembler emits `MOV EAX, [ESP]` / `[EBP]` exactly this way, so decoder and
    assembler agree end-to-end.
    """
    out = x86([OP_MOV_R_RM, modrm(0b00, REG_EAX, rm)])
    assert out["instr_len"] == 2, note
    assert out["rs1"] == rm
    assert out["mem_read"] == 1
    assert out["reg_write"] == 1


def test_x86_mov_assembler_agrees_with_the_mod00_escape_treatment():
    """End-to-end: the assembler's [ESP]/[EBP] forms decode back correctly."""
    for text, base in (("MOV EAX, [ESP]", REG_ESP), ("MOV EAX, [EBP]", REG_EBP)):
        out, byte_list = x86_asm(text)
        assert len(byte_list) == out["instr_len"] == 2
        assert out["rs1"] == base
        assert out["mem_read"] == 1


@pytest.mark.parametrize("opcode", [OP_MOV_RM_R, OP_MOV_R_RM])
def test_x86_mov_with_disp32_mod10_reports_the_right_length(opcode):
    """REGRESSION PIN for a FIXED bug (2026-08-08).

    `_decode_mov_rm` used to handle mod=11/00/01 and then fall off the end,
    leaving instr_len at its reset value of 1. Fetch advanced the PC by 1
    instead of 6, landed mid-instruction, and the whole byte stream
    desynchronised. The 0x83 Group-1 path and the ALU r/m,r path both got
    this right via _modrm_disp_len(); only MOV was missed.
    """
    out = x86([opcode, modrm(0b10, REG_EAX, REG_EBX), 0x10, 0x00, 0x00, 0x00])
    assert out["instr_len"] == 6


@pytest.mark.parametrize("opcode,is_load", [(OP_MOV_R_RM, True),
                                            (OP_MOV_RM_R, False)])
def test_x86_mov_disp32_decodes_operands_and_control(opcode, is_load):
    """mod=10 is base+disp32 — a form the datapath CAN express, so unlike the
    unmodelled memory-ALU forms it asserts full control, not length only."""
    out = x86([opcode, modrm(0b10, REG_ECX, REG_EBX), 0x40, 0x00, 0x00, 0x00])
    assert out["instr_len"] == 6
    assert out["imm"] == 0x40
    assert out["rs1"] == REG_EBX                  # base register
    if is_load:
        assert out["mem_read"] == 1 and out["reg_write"] == 1
        assert out["rd"] == REG_ECX
    else:
        assert out["mem_write"] == 1 and out["reg_write"] == 0
        assert out["rs2"] == REG_ECX


@pytest.mark.parametrize("disp,expected", [
    (0x00000000, 0),
    (0x0000007F, 0x7F),
    (0x00000080, 0x80),          # NOT sign-extended like disp8 would be
    (0x7FFFFFFF, 0x7FFFFFFF),
    (0xFFFFFFFF, 0xFFFFFFFF),    # -1, masked into the 32-bit imm port
    (0xFFFFFFF0, 0xFFFFFFF0),    # -16
])
def test_x86_mov_disp32_is_read_little_endian_full_width(disp, expected):
    """All four displacement bytes must be consumed, LSB first."""
    b = [OP_MOV_R_RM, modrm(0b10, REG_EAX, REG_EBX),
         disp & 0xFF, (disp >> 8) & 0xFF, (disp >> 16) & 0xFF, (disp >> 24) & 0xFF]
    out = x86(b)
    assert out["imm"] == expected
    assert out["instr_len"] == 6


def test_x86_mov_disp32_keeps_a_byte_stream_aligned_end_to_end():
    """The point of the length fix: the PC must land on real instructions.

    MOV EAX,99 (5) | MOV [EBX+0x40],EAX (6) | MOV ECX,7 (5)
    A wrong length here desynced everything after it and lost the store.
    """
    from sim.harness import simulate
    prog = [0xB8, 0x63, 0x00, 0x00, 0x00,
            0x89, 0x83, 0x40, 0x00, 0x00, 0x00,
            0xB9, 0x07, 0x00, 0x00, 0x00]
    r = simulate("x86", "single_cycle", program=prog, cycles=12)
    seen = []
    for pc in r.pc_stream:
        if pc not in seen:
            seen.append(pc)
    assert seen[:3] == ["0x00000000", "0x00000005", "0x0000000b"]
    assert r.reg(0) == 99          # EAX
    assert r.reg(1) == 7           # ECX — reached only if lengths are right
    assert r.mem_at(0x40) == 99    # the store actually committed


# ════════════════════════════════════════════════════════════════════════════
#  5-byte encodings
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("reg,name", list(enumerate(REGISTER_NAMES)))
def test_x86_mov_imm32_is_five_bytes_for_every_register(reg, name):
    out, byte_list = x86_asm(f"MOV {name}, 42")
    assert byte_list == [0xB8 + reg, 0x2A, 0x00, 0x00, 0x00]
    assert out["instr_len"] == 5
    assert out["rd"] == reg
    assert out["imm"] == 42
    assert out["alu_op"] == AluOp.PASS     # ALU forwards operand B unchanged
    assert out["alu_src"] == 1
    assert out["reg_write"] == 1
    assert out["wb_sel"] == WBSel.ALU
    assert out["rs1"] == out["rs2"] == 0
    assert out["mem_read"] == out["mem_write"] == 0
    assert out["write_flags"] == 0


@pytest.mark.parametrize("imm_bytes,expected,note", [
    ([0x00, 0x00, 0x00, 0x00], 0,          "zero"),
    ([0x78, 0x56, 0x34, 0x12], 0x12345678, "little-endian byte order"),
    ([0xFF, 0xFF, 0xFF, 0xFF], 0xFFFFFFFF, "all ones (-1)"),
    ([0x00, 0x00, 0x00, 0x80], 0x80000000, "the sign bit only"),
    ([0xFF, 0x00, 0x00, 0x00], 0x000000FF, "no sign extension from byte 0"),
])
def test_x86_mov_imm32_little_endian_assembly(imm_bytes, expected, note):
    """imm32 is assembled from 4 LE bytes and NOT sign-extended (it is already 32-bit)."""
    out = x86([0xB8] + imm_bytes)
    assert out["imm"] == expected, note
    assert out["instr_len"] == 5


def test_x86_call_rel32_is_modelled_as_a_plain_jump():
    """CALL pushes no return address here — a documented simplification.

    The decoder emits jal=1 with a PC-relative displacement and nothing else;
    no example program uses CALL/RET linkage.
    """
    out = x86([OP_CALL_REL32, 0x00, 0x00, 0x00, 0x00], pc=0)
    assert out["instr_len"] == 5
    assert out["jal"] == 1
    assert out["imm"] == 5                 # target = pc + 5 + 0, imm = target - pc
    assert out["branch"] == 0
    assert out["jalr"] == 0
    assert out["reg_write"] == 0           # <-- no link register
    assert out["mem_write"] == 0           # <-- no push
    assert out["rd"] == 0


def test_x86_call_rel32_negative_displacement():
    out = x86([OP_CALL_REL32, 0xF0, 0xFF, 0xFF, 0xFF], pc=100)
    assert out["instr_len"] == 5
    assert out["imm"] == u32(5 - 16)       # pc + 5 - 16 - pc
    assert out["jal"] == 1


def test_x86_jmp_rel32_e9():
    """The widened form the assembler relaxes to for far label jumps."""
    out = x86([0xE9, 0x0A, 0x00, 0x00, 0x00], pc=0)
    assert out["instr_len"] == 5
    assert out["jal"] == 1
    assert out["imm"] == 15                # 5 (next instr) + 10
    assert out["branch"] == 0
    assert out["reg_write"] == 0


def test_x86_jmp_rel32_negative_displacement():
    out = x86([0xE9, 0xF0, 0xFF, 0xFF, 0xFF], pc=1000)
    assert out["imm"] == u32(5 - 16)
    assert out["instr_len"] == 5


# ════════════════════════════════════════════════════════════════════════════
#  6-byte encodings: Jcc rel32 (0F 80-8F)
# ════════════════════════════════════════════════════════════════════════════

X86_JCC = [
    ("JE",  JccCond.JE,  BranchCond.EQ),
    ("JNE", JccCond.JNE, BranchCond.NEQ),
    ("JL",  JccCond.JL,  BranchCond.LT),
    ("JGE", JccCond.JGE, BranchCond.GE),
    ("JLE", JccCond.JLE, BranchCond.LE),
    ("JG",  JccCond.JG,  BranchCond.GT),
]


@pytest.mark.parametrize("mnemonic,cond_nibble,branch_cond", X86_JCC)
def test_x86_jcc_rel32_six_byte_form(mnemonic, cond_nibble, branch_cond):
    """0F 8x rel32 — the relaxed/widened conditional jump. SIX bytes."""
    out = x86([0x0F, 0x80 | cond_nibble, 0x00, 0x00, 0x00, 0x00], pc=0)
    assert out["instr_len"] == 6
    assert out["branch"] == 1
    assert out["branch_cond"] == branch_cond
    assert out["imm"] == 6                 # pc + 6 + 0 - pc
    assert out["jal"] == out["jalr"] == 0
    assert out["reg_write"] == 0
    assert out["mem_read"] == out["mem_write"] == 0
    # Jcc reads LATCHED flags from FlagsRegister; it publishes none.
    assert out["write_flags"] == 0


def test_x86_jcc_rel32_displacement_is_relative_to_the_next_instruction():
    out = x86([0x0F, 0x84, 0x00, 0x01, 0x00, 0x00], pc=0)
    assert out["imm"] == 6 + 0x100
    out = x86([0x0F, 0x84, 0xFA, 0xFF, 0xFF, 0xFF], pc=0)
    assert out["imm"] == u32(6 - 6) == 0   # jumps back onto itself


@pytest.mark.parametrize("cond_nibble,name", [
    (0x0, "JO"), (0x1, "JNO"), (0x2, "JB"), (0x3, "JAE"),
    (0x6, "JBE"), (0x7, "JA"), (0x8, "JS"), (0x9, "JNS"),
    (0xA, "JP"), (0xB, "JNP"),
])
def test_x86_unsupported_jcc_rel32_conditions_are_never_taken(cond_nibble, name):
    """SAFETY: the datapath has no CF/OF/PF/SF, so the unsigned, overflow,
    sign and parity conditions decode to BranchCond.NEVER — a branch that is
    correctly recognised (right length, right target) but never taken, rather
    than a wrongly-taken branch."""
    out = x86([0x0F, 0x80 | cond_nibble, 0x00, 0x00, 0x00, 0x00], pc=0)
    assert out["instr_len"] == 6, name
    assert out["branch"] == 1, name
    assert out["branch_cond"] == BranchCond.NEVER, name
    assert out["imm"] == 6


@pytest.mark.parametrize("second_byte", [0x00, 0x05, 0x7F, 0x90, 0xA2, 0xAF])
def test_x86_0f_escape_outside_the_jcc_range_is_not_decoded(second_byte):
    """0F 8x is the only two-byte-opcode family modelled; everything else
    (SYSCALL, SETcc, CPUID, IMUL r,r/m ...) falls through as an unknown."""
    out = x86([0x0F, second_byte])
    assert out["branch"] == 0
    assert out["instr_len"] == 1           # reset value; the byte is not consumed
    assert out["reg_write"] == 0


# ════════════════════════════════════════════════════════════════════════════
#  2-byte encodings: JMP rel8 / Jcc rel8
# ════════════════════════════════════════════════════════════════════════════

def test_x86_jmp_rel8_two_byte_form():
    out, byte_list = x86_asm("JMP +2", pc=0)
    assert byte_list == [OP_JMP_REL8, 0x00]
    assert out["instr_len"] == 2
    assert out["jal"] == 1
    assert out["branch"] == 0
    assert out["reg_write"] == 0
    assert out["write_flags"] == 0


@pytest.mark.parametrize("disp8,expected,note", [
    (0x00, 2,           "fall through to the next instruction"),
    (0x02, 4,           "skip 2 bytes past the next instruction"),
    (0x7F, 2 + 127,     "max forward rel8"),
    (0xFE, 0,           "jump back onto this instruction (infinite loop)"),
    (0xF0, u32(2 - 16), "backward"),
    (0x80, u32(2 - 128), "min backward rel8"),
])
def test_x86_jmp_rel8_sign_extension_and_pc_normalisation(disp8, expected, note):
    """`imm` is published as (target - pc), i.e. already normalised so the
    datapath can add it to the CURRENT pc rather than to pc+len."""
    assert x86([OP_JMP_REL8, disp8], pc=0)["imm"] == expected, note


@pytest.mark.parametrize("pc", [0, 2, 4, 100, 0x1000, 0xFFFF])
def test_x86_relative_jump_imm_is_pc_independent(pc):
    """Because imm = (pc + len + disp) - pc, the published value must be the
    SAME at every pc — the pc_in port only matters for the wrap-around mask."""
    assert x86([OP_JMP_REL8, 0x02], pc=pc)["imm"] == 4
    assert x86([0x74, 0xFB], pc=pc)["imm"] == u32(-3)
    assert x86([0xE9, 0x0A, 0, 0, 0], pc=pc)["imm"] == 15


@pytest.mark.parametrize("mnemonic,cond_nibble,branch_cond", X86_JCC)
def test_x86_jcc_rel8_two_byte_form(mnemonic, cond_nibble, branch_cond):
    out, byte_list = x86_asm(f"{mnemonic} +2", pc=0)
    assert byte_list == [0x70 | cond_nibble, 0x00]
    assert out["instr_len"] == 2
    assert out["branch"] == 1
    assert out["branch_cond"] == branch_cond
    assert out["imm"] == 2
    assert out["jal"] == out["jalr"] == 0
    assert out["reg_write"] == 0
    assert out["mem_read"] == out["mem_write"] == 0
    assert out["write_flags"] == 0


def test_x86_jcc_rel8_and_rel32_agree_on_condition_mapping():
    """The two Jcc paths duplicate the condition table; they must not drift."""
    for _mnemonic, nibble, expected in X86_JCC:
        short = x86([0x70 | nibble, 0x00])
        near = x86([0x0F, 0x80 | nibble, 0, 0, 0, 0])
        assert short["branch_cond"] == near["branch_cond"] == expected
        assert short["instr_len"] == 2 and near["instr_len"] == 6


@pytest.mark.parametrize("cond_nibble,name", [
    (0x0, "JO"), (0x1, "JNO"), (0x2, "JB"), (0x3, "JAE"),
    (0x6, "JBE"), (0x7, "JA"), (0x8, "JS"), (0x9, "JNS"),
    (0xA, "JP"), (0xB, "JNP"),
])
def test_x86_unsupported_jcc_rel8_conditions_are_never_taken(cond_nibble, name):
    out = x86([0x70 | cond_nibble, 0x02], pc=0)
    assert out["instr_len"] == 2, name
    assert out["branch"] == 1, name
    assert out["branch_cond"] == BranchCond.NEVER, name


def test_x86_jcc_and_jmp_rel8_backward_wrap_is_masked_to_32_bits():
    """A backward branch from a low pc wraps the target; imm stays 32-bit."""
    out = x86([OP_JMP_REL8, 0xF0], pc=0)
    assert out["imm"] == u32(-14)
    assert 0 <= out["imm"] <= 0xFFFFFFFF
    out = x86([OP_JMP_REL8, 0x04], pc=0xFFFFFFFE)
    assert 0 <= out["imm"] <= 0xFFFFFFFF


# ════════════════════════════════════════════════════════════════════════════
#  3-byte encodings: Group 1 (0x83 /ext imm8)
# ════════════════════════════════════════════════════════════════════════════

X86_GRP1 = [
    ("ADD", GRP1_ADD, AluOp.ADD, True),
    ("OR",  GRP1_OR,  AluOp.OR,  True),
    ("AND", GRP1_AND, AluOp.AND, True),
    ("SUB", GRP1_SUB, AluOp.SUB, True),
    ("XOR", GRP1_XOR, AluOp.XOR, True),
    ("CMP", GRP1_CMP, AluOp.SUB, False),
]


@pytest.mark.parametrize("mnemonic,ext,alu_op,writes", X86_GRP1)
def test_x86_group1_imm8_forms(mnemonic, ext, alu_op, writes):
    out, byte_list = x86_asm(f"{mnemonic} EDI, 5")
    assert byte_list == [OP_IMM_GRP1, modrm(0b11, ext, REG_EDI), 0x05]
    assert out["instr_len"] == 3
    assert out["rs1"] == REG_EDI           # rm is the operand
    assert out["imm"] == 5
    assert out["alu_op"] == alu_op
    assert out["alu_src"] == 1             # <-- immediate, unlike the /r form
    assert out["write_flags"] == 1
    assert out["mem_read"] == out["mem_write"] == 0
    assert out["branch"] == out["jal"] == out["jalr"] == 0
    if writes:
        assert out["rd"] == REG_EDI
        assert out["reg_write"] == 1
        assert out["wb_sel"] == WBSel.ALU
    else:
        assert out["reg_write"] == 0       # CMP: flags only
        assert out["rd"] == 0


@pytest.mark.parametrize("imm8,expected,note", [
    (0x00, 0,            "zero"),
    (0x05, 5,            "small positive"),
    (0x7F, 127,          "max positive imm8"),
    (0x80, u32(-128),    "min negative imm8"),
    (0xFF, u32(-1),      "-1, the classic sign-extension check"),
])
def test_x86_group1_imm8_is_sign_extended_to_32_bits(imm8, expected, note):
    out = x86([OP_IMM_GRP1, modrm(0b11, GRP1_ADD, REG_EAX), imm8])
    assert out["imm"] == expected, note
    assert out["instr_len"] == 3


@pytest.mark.parametrize("ext,name", [(2, "ADC /2"), (3, "SBB /3")])
def test_x86_group1_unmodelled_extensions_write_no_register(ext, name):
    """ADC and SBB need a carry flag this datapath does not have.

    They still consume 3 bytes and set write_flags (the shared prologue runs),
    but no alu_op is selected and reg_write stays low — inert, not wrong.
    """
    out = x86([OP_IMM_GRP1, modrm(0b11, ext, REG_ECX), 0x05])
    assert out["instr_len"] == 3, name
    assert out["rs1"] == REG_ECX
    assert out["imm"] == 5
    assert out["alu_src"] == 1
    assert out["write_flags"] == 1
    assert out["reg_write"] == 0           # <-- no destination selected
    assert out["rd"] == 0
    assert out["alu_op"] == AluOp.ADD      # reset value, never assigned


@pytest.mark.parametrize("modrm_byte,extra,expected_len,note", [
    (modrm(0b00, GRP1_ADD, REG_EAX), [0x05],                   3, "[EAX], imm8"),
    (modrm(0b01, GRP1_ADD, REG_EAX), [0x08, 0x05],             4, "[EAX+disp8], imm8"),
    (modrm(0b10, GRP1_ADD, REG_EAX), [0, 0, 0, 0, 0x05],       7, "[EAX+disp32], imm8"),
    (modrm(0b00, GRP1_ADD, 0b101),   [0, 0, 0, 0, 0x05],       7, "[disp32], imm8"),
    (modrm(0b00, GRP1_ADD, 0b100),   [0x24, 0x05],             4, "SIB, imm8"),
])
def test_x86_group1_with_memory_operand_computes_length_and_bails(
        modrm_byte, extra, expected_len, note):
    """Memory-destination Group 1 is NOT modelled: length only, no side effects.

    Note instr_len can reach 7 here, which still fits the 4-bit port (max 15).
    """
    out = x86([OP_IMM_GRP1, modrm_byte] + extra)
    assert out["instr_len"] == expected_len, note
    for port in ("rd", "rs1", "imm", "alu_src", "reg_write", "mem_read",
                 "mem_write", "write_flags"):
        assert out[port] == 0, f"{note}: {port} asserted"


def test_x86_instr_len_port_is_wide_enough_for_the_longest_form():
    """The 4-bit instr_len port must hold the longest length the decoder emits."""
    assert X86Decoder.ports_spec["instr_len"].width == 4
    longest = x86([OP_IMM_GRP1, modrm(0b10, GRP1_ADD, REG_EAX), 0, 0, 0, 0, 0x05])
    assert longest["instr_len"] == 7 <= (1 << 4) - 1


# ════════════════════════════════════════════════════════════════════════════
#  Unknown opcodes
# ════════════════════════════════════════════════════════════════════════════

X86_CONTROL_PORTS = ("rd", "rs1", "rs2", "imm", "alu_op", "alu_src", "use_pc",
                     "mem_read", "mem_write", "reg_write", "branch", "jal",
                     "jalr", "write_flags")


@pytest.mark.parametrize("byte_list,note", [
    ([0x00], "ADD r/m8, r8 — no 8-bit forms modelled"),
    ([0xF4], "HLT"),
    ([0xCC], "INT3"),
    ([0xCD, 0x80], "INT 0x80"),
    ([0x40], "INC EAX (one-byte form)"),
    ([0x48], "DEC EAX (one-byte form)"),
    ([0x60], "PUSHA"),
    ([0x61], "POPA"),
    ([0xF7, 0xE1], "MUL ECX"),
    ([0x81, 0xC0, 0x00, 0x01, 0x00, 0x00], "Group 1 with imm32 (0x81)"),
    ([0xFF, 0xE0], "JMP r/m32 (indirect)"),
])
def test_x86_unknown_opcode_asserts_no_control_and_defaults_to_length_1(byte_list, note):
    """SAFETY: unrecognised bytes fall off the end of evaluate() with every
    control output at its reset value and instr_len at its reset value of 1.

    A length of 1 is the safe default: it advances the PC by one byte so the
    fetch unit resynchronises on the next byte rather than skipping forward
    into the middle of a real instruction.
    """
    out = x86(byte_list)
    for port in X86_CONTROL_PORTS:
        assert out[port] == 0, f"{note}: {port} = {out[port]}"
    assert out["branch_cond"] == BranchCond.NEVER
    assert out["wb_sel"] == WBSel.ALU
    assert out["instr_len"] == 1, note


def test_x86_reset_signals_defaults():
    """_reset_signals() must leave the decoder in the inert state."""
    d = X86Decoder()
    d["instr_in"] = pack([0x8B, 0x43, 0xFC])   # a load with a negative disp
    d.evaluate()
    assert d["mem_read"] == 1 and d["imm"] == u32(-4)

    d._reset_signals()
    for port in X86_CONTROL_PORTS:
        assert d[port] == 0
    assert d["wb_sel"] == WBSel.ALU
    assert d["instr_len"] == 1


# ════════════════════════════════════════════════════════════════════════════
#  assembler <-> decoder round trips
# ════════════════════════════════════════════════════════════════════════════

def test_x86_instr_len_matches_the_assembler_byte_count_for_every_form():
    """The single most important x86 invariant: what the assembler EMITS and
    what the decoder says it CONSUMED must be the same number of bytes.

    Any mismatch desynchronises the fetch stream for the rest of the program.
    """
    for text in [
        "NOP", "RET",
        "PUSH EAX", "PUSH EDI", "POP EBX", "POP ESP",
        "MOV EAX, 42", "MOV EDI, -1",
        "MOV EBX, ESI", "MOV EAX, [EBX]", "MOV [EBX], EAX",
        "MOV EAX, [EBX+4]", "MOV [EBX-8], ECX",
        "ADD EAX, ECX", "SUB EBX, EDX", "AND ESI, EDI",
        "OR EAX, EBX", "XOR ECX, ECX", "CMP EAX, EBX",
        "ADD EAX, 5", "SUB ECX, 1", "CMP EDX, -1",
        "AND EBX, 7", "OR ESI, 3", "XOR EDI, 15",
        "JMP +2", "JE +2", "JNE -2", "JL +4", "JGE +4", "JLE +4", "JG +4",
        "CALL +5",
    ]:
        byte_list = assemble("x86", text)
        out = x86(byte_list)
        assert out["instr_len"] == len(byte_list), (
            f"{text!r}: assembler emitted {len(byte_list)} byte(s) "
            f"{[hex(b) for b in byte_list]} but the decoder reports "
            f"instr_len = {out['instr_len']}")


def test_x86_relaxed_rel32_jumps_round_trip():
    """Branch relaxation (rel8 -> E9 / 0F 8x) must still decode to the right
    length, or a far loop silently corrupts the PC."""
    source = "JMP far\n" + "NOP\n" * 200 + "far:\nJE back\n" + \
             "NOP\n" * 200 + "back:\nNOP\n"
    byte_list = assemble("x86", source)
    # The leading JMP had to widen to E9 rel32.
    assert byte_list[0] == 0xE9
    head = x86(byte_list[0:8])
    assert head["instr_len"] == 5
    assert head["jal"] == 1
    # ... and the forward JE widened to 0F 8x rel32.
    je_at = 5 + 200
    assert byte_list[je_at] == 0x0F and byte_list[je_at + 1] == 0x84
    je = x86(byte_list[je_at:je_at + 8], pc=je_at)
    assert je["instr_len"] == 6
    assert je["branch"] == 1 and je["branch_cond"] == BranchCond.EQ


def test_x86_decoding_the_demo_program_walks_the_byte_stream_exactly():
    """Walk the whole x86 demo program using ONLY the decoder's instr_len.

    If any length is wrong the walk lands mid-instruction and the recovered
    mnemonic sequence diverges — this is the end-to-end proof that the
    variable-length decode is self-consistent.
    """
    from sim.isa.x86.config import X86

    program = X86().demo_program()
    pc = 0
    kinds = []
    while pc < len(program):
        out = x86(program[pc:pc + 8], pc=pc)
        assert out["instr_len"] >= 1
        if out["branch"]:
            kinds.append("jcc")
        elif out["jal"]:
            kinds.append("jmp")
        elif out["alu_op"] == AluOp.PASS:
            kinds.append("mov_imm")
        elif out["write_flags"] and not out["reg_write"]:
            kinds.append("cmp")
        elif out["write_flags"]:
            kinds.append("alu")
        else:
            kinds.append("mov_reg")
        pc += out["instr_len"]
    assert pc == len(program), "instr_len walk overshot / undershot the program"
    assert kinds == ["mov_imm", "mov_imm", "mov_imm", "mov_reg", "alu",
                     "mov_reg", "mov_reg", "alu", "cmp", "jcc"]


def test_x86_get_state_snapshot_shape():
    d = X86Decoder()
    d["instr_in"] = pack(assemble("x86", "ADD EAX, 5"))
    d.evaluate()
    assert d.get_state() == {
        "rd": REG_EAX, "rs1": REG_EAX, "rs2": 0, "imm": 5,
        "alu_op": "ADD", "alu_src": 1,
        "mem_read": 0, "mem_write": 0, "reg_write": 1,
        "branch": 0, "jal": 0, "jalr": 0,
        "instr_len": 3,
    }


def test_x86_get_state_includes_instr_len_unlike_the_other_two_isas():
    """instr_len is x86-only — the UI uses it to draw the variable-length fetch."""
    d = X86Decoder()
    d["instr_in"] = pack([0xB8, 0x01, 0x00, 0x00, 0x00])
    d.evaluate()
    assert d.get_state()["instr_len"] == 5
