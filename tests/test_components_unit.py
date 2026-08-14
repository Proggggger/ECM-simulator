"""Direct UNIT tests for the basic hardware components.

Every component here is driven in isolation — write its input ports with
``comp["port"] = v``, call ``evaluate()`` / ``rising_edge()`` by hand, and
assert the outputs — rather than through a full preset run. That makes the
component's own contract (not the surrounding datapath) the thing under test,
so a regression points straight at the file that broke.

Covers: StandardALU (every AluOp + boundaries), AluInputMux, FlagsRegister,
SimpleFetch / X86Fetch PC precedence, PcLenAdder, InstructionMemory /
ByteInstructionMemory, DataMemory (+ MultiPortDataMemory), StandardRegisterFile,
WritebackUnit, and the multicycle Controller / gates / Latch / PCLatch /
CycleBudgetController.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.components.alu.base import ALUBase
from sim.components.alu.flags_register import FlagsRegister
from sim.components.alu.input_mux import AluInputMux
from sim.components.alu.standard import StandardALU
from sim.components.fetch.pc_len_adder import PcLenAdder
from sim.components.fetch.simple import SimpleFetch
from sim.components.fetch.x86_fetch import X86Fetch
from sim.components.memory.base import MemoryUnitBase
from sim.components.memory.byte_imem import ByteInstructionMemory
from sim.components.memory.dmem import (
    DEFAULT_MEM_WORDS, STATE_WINDOW, DataMemory, MultiPortDataMemory,
    memory_snapshot,
)
from sim.components.memory.imem import InstructionMemory
from sim.components.multicycle.budget_controller import (
    CLASSES, DEFAULT_COSTS, FETDECEXE_COSTS, WORK, CycleBudgetController,
    clock_period, estimate_time,
)
from sim.components.multicycle.controller import (
    DECODE, EXECUTE, FETCH, MEMORY, MultiCycleController,
)
from sim.components.multicycle.gate import AndNotSignal, GatedSignal, OrSignal
from sim.components.multicycle.latch import Latch
from sim.components.multicycle.pc_latch import PCLatch
from sim.components.regfile.base import RegisterFileBase
from sim.components.regfile.standard import StandardRegisterFile
from sim.components.writeback.standard import WritebackUnit
from sim.core.signals import AluOp, WBSel

STACK_TOP = 65528          # compiler's software-stack base (see CLAUDE.md)
PRINT_ADDR = 2040          # print() console word


# ════════════════════════════════════════════════════════════════════════════
# ALU
# ════════════════════════════════════════════════════════════════════════════

class TestStandardALU(unittest.TestCase):

    def setUp(self):
        self.alu = StandardALU()

    def do(self, a, b, op):
        self.alu["a"] = a
        self.alu["b"] = b
        self.alu["op"] = op
        self.alu.evaluate()
        return self.alu["result"]

    # ── arithmetic ──────────────────────────────────────────────
    def test_add(self):
        self.assertEqual(self.do(2, 3, AluOp.ADD), 5)
        self.assertEqual(self.do(0, 0, AluOp.ADD), 0)
        self.assertEqual(self.do(0x7FFF_FFFF, 0, AluOp.ADD), 0x7FFF_FFFF)

    def test_add_wraps_at_2_32(self):
        self.assertEqual(self.do(0xFFFF_FFFF, 1, AluOp.ADD), 0)
        self.assertEqual(self.do(0xFFFF_FFFF, 2, AluOp.ADD), 1)
        self.assertEqual(self.do(0x8000_0000, 0x8000_0000, AluOp.ADD), 0)
        # signed overflow wraps silently (no V flag in this datapath)
        self.assertEqual(self.do(0x7FFF_FFFF, 1, AluOp.ADD), 0x8000_0000)

    def test_add_with_negative_operands_as_twos_complement(self):
        neg1 = 0xFFFF_FFFF
        self.assertEqual(self.do(5, neg1, AluOp.ADD), 4)
        self.assertEqual(self.do(neg1, neg1, AluOp.ADD), 0xFFFF_FFFE)

    def test_sub(self):
        self.assertEqual(self.do(5, 3, AluOp.SUB), 2)
        self.assertEqual(self.do(3, 3, AluOp.SUB), 0)

    def test_sub_borrows_below_zero(self):
        self.assertEqual(self.do(0, 1, AluOp.SUB), 0xFFFF_FFFF)
        self.assertEqual(self.do(3, 5, AluOp.SUB), 0xFFFF_FFFE)
        self.assertEqual(self.do(0x8000_0000, 1, AluOp.SUB), 0x7FFF_FFFF)

    # ── logic ───────────────────────────────────────────────────
    def test_and_or_xor(self):
        self.assertEqual(self.do(0xF0F0, 0xFF00, AluOp.AND), 0xF000)
        self.assertEqual(self.do(0xF0F0, 0xFF00, AluOp.OR), 0xFFF0)
        self.assertEqual(self.do(0xF0F0, 0xFF00, AluOp.XOR), 0x0FF0)
        self.assertEqual(self.do(0xFFFF_FFFF, 0xFFFF_FFFF, AluOp.XOR), 0)
        self.assertEqual(self.do(0xFFFF_FFFF, 0, AluOp.AND), 0)
        self.assertEqual(self.do(0xFFFF_FFFF, 0, AluOp.OR), 0xFFFF_FFFF)

    # ── comparisons ─────────────────────────────────────────────
    def test_slt_is_signed(self):
        self.assertEqual(self.do(1, 2, AluOp.SLT), 1)
        self.assertEqual(self.do(2, 1, AluOp.SLT), 0)
        self.assertEqual(self.do(2, 2, AluOp.SLT), 0)
        self.assertEqual(self.do(0xFFFF_FFFF, 1, AluOp.SLT), 1, "-1 < 1")
        self.assertEqual(self.do(1, 0xFFFF_FFFF, AluOp.SLT), 0, "1 > -1")
        self.assertEqual(self.do(0x8000_0000, 0x7FFF_FFFF, AluOp.SLT), 1,
                         "INT_MIN < INT_MAX")
        self.assertEqual(self.do(0xFFFF_FFFF, 0xFFFF_FFFF, AluOp.SLT), 0)

    def test_sltu_is_unsigned(self):
        self.assertEqual(self.do(1, 2, AluOp.SLTU), 1)
        self.assertEqual(self.do(0xFFFF_FFFF, 1, AluOp.SLTU), 0, "-1 is huge unsigned")
        self.assertEqual(self.do(1, 0xFFFF_FFFF, AluOp.SLTU), 1)
        self.assertEqual(self.do(0x7FFF_FFFF, 0x8000_0000, AluOp.SLTU), 1)
        self.assertEqual(self.do(0, 0, AluOp.SLTU), 0)

    def test_slt_and_sltu_disagree_exactly_across_the_sign_boundary(self):
        a, b = 0x8000_0000, 0x0000_0001
        self.assertEqual(self.do(a, b, AluOp.SLT), 1)     # signed: INT_MIN < 1
        self.assertEqual(self.do(a, b, AluOp.SLTU), 0)    # unsigned: 2^31 > 1

    # ── shifts ──────────────────────────────────────────────────
    def test_sll(self):
        self.assertEqual(self.do(1, 0, AluOp.SLL), 1)
        self.assertEqual(self.do(1, 1, AluOp.SLL), 2)
        self.assertEqual(self.do(1, 31, AluOp.SLL), 0x8000_0000)
        self.assertEqual(self.do(0xFFFF_FFFF, 4, AluOp.SLL), 0xFFFF_FFF0,
                         "bits shifted out of 32 are masked away")

    def test_srl_is_logical(self):
        self.assertEqual(self.do(0x8000_0000, 31, AluOp.SRL), 1)
        self.assertEqual(self.do(0xFFFF_FFFF, 4, AluOp.SRL), 0x0FFF_FFFF)
        self.assertEqual(self.do(0xFFFF_FFFF, 31, AluOp.SRL), 1)
        self.assertEqual(self.do(0xFFFF_FFFF, 0, AluOp.SRL), 0xFFFF_FFFF)

    def test_sra_is_arithmetic(self):
        self.assertEqual(self.do(0x8000_0000, 31, AluOp.SRA), 0xFFFF_FFFF)
        self.assertEqual(self.do(0xFFFF_FFF0, 4, AluOp.SRA), 0xFFFF_FFFF, "-16 >> 4 == -1")
        self.assertEqual(self.do(0x7FFF_FFFF, 1, AluOp.SRA), 0x3FFF_FFFF)
        self.assertEqual(self.do(0xFFFF_FFFF, 0, AluOp.SRA), 0xFFFF_FFFF)

    def test_shift_amount_uses_only_the_low_five_bits_of_b(self):
        # shamt = b & 0x1F, so 32 behaves as 0 and 33 as 1.
        self.assertEqual(self.do(1, 32, AluOp.SLL), 1)
        self.assertEqual(self.do(1, 33, AluOp.SLL), 2)
        self.assertEqual(self.do(0x8000_0000, 32, AluOp.SRL), 0x8000_0000)
        self.assertEqual(self.do(0x8000_0000, 0xFFFF_FFFF, AluOp.SRL), 1,
                         "0xFFFFFFFF & 0x1F == 31")

    # ── misc ────────────────────────────────────────────────────
    def test_pass_forwards_b(self):
        self.assertEqual(self.do(0xDEAD, 0xBEEF, AluOp.PASS), 0xBEEF)
        self.assertEqual(self.do(0xFFFF_FFFF, 0, AluOp.PASS), 0)

    def test_unknown_op_yields_zero(self):
        # op is a 4-bit port, so 11..15 are reachable but undefined.
        for op in range(AluOp.PASS + 1, 16):
            self.assertEqual(self.do(7, 9, op), 0, f"op {op}")
            self.assertEqual(self.alu["zero"], 1)

    def test_every_named_alu_op_is_implemented(self):
        """No AluOp constant may silently fall into the `else: r = 0` arm.

        Each (a, b) pair is chosen so a CORRECT implementation is non-zero,
        which is exactly what the undefined-op fallback can never produce.
        """
        cases = {
            "ADD":  (0xF0, 0x0F),
            "SUB":  (9, 2),
            "AND":  (0xF0, 0xFF),
            "OR":   (0xF0, 0x0F),
            "XOR":  (0xF0, 0x0F),
            "SLT":  (1, 2),        # true -> 1
            "SLTU": (1, 2),
            "SLL":  (0xF0, 4),
            "SRL":  (0xF0, 4),
            "SRA":  (0xF0, 4),
            "PASS": (0, 0x0F),
        }
        self.assertEqual(
            set(cases),
            {n for n in vars(AluOp) if not n.startswith("_")},
            "a new AluOp was added without a case here")
        for op_name, (a, b) in cases.items():
            r = self.do(a, b, getattr(AluOp, op_name))
            self.assertNotEqual(r, 0, f"{op_name} produced 0 — is it wired up?")

    def test_zero_flag_tracks_the_masked_result(self):
        self.do(3, 3, AluOp.SUB)
        self.assertEqual(self.alu["zero"], 1)
        self.do(3, 2, AluOp.SUB)
        self.assertEqual(self.alu["zero"], 0)
        # A result that is zero only AFTER 32-bit masking still sets zero.
        self.do(0xFFFF_FFFF, 1, AluOp.ADD)
        self.assertEqual(self.alu["result"], 0)
        self.assertEqual(self.alu["zero"], 1)

    def test_zero_flag_after_a_shift_that_pushes_all_bits_out(self):
        self.assertEqual(self.do(0xFFFF_FFFF, 31, AluOp.SLL), 0x8000_0000)
        self.assertEqual(self.alu["zero"], 0)
        self.assertEqual(self.do(1, 31, AluOp.SRL), 0)
        self.assertEqual(self.alu["zero"], 1)

    def test_operand_ports_mask_to_32_bits(self):
        self.alu["a"] = 1 << 32
        self.assertEqual(self.alu["a"], 0)
        self.alu["b"] = -1
        self.assertEqual(self.alu["b"], 0xFFFF_FFFF)

    def test_op_port_is_four_bits_wide(self):
        self.alu["op"] = 16
        self.assertEqual(self.alu["op"], 0, "op wraps mod 16")
        self.alu["op"] = 15
        self.assertEqual(self.alu["op"], 15)

    def test_alu_is_purely_combinational(self):
        """No rising_edge() state: re-evaluating with the same inputs repeats."""
        first = self.do(5, 3, AluOp.ADD)
        self.alu.rising_edge()
        self.alu.evaluate()
        self.assertEqual(self.alu["result"], first)

    def test_get_state_renders_the_op_name(self):
        self.do(5, 3, AluOp.SUB)
        st = self.alu.get_state()
        self.assertEqual(st, {"a": 5, "b": 3, "op": "SUB", "result": 2, "zero": 0})

    def test_get_state_falls_back_to_the_numeric_op(self):
        self.do(0, 0, 13)
        self.assertEqual(self.alu.get_state()["op"], "13")

    def test_conforms_to_the_alu_port_contract(self):
        self.assertIsInstance(self.alu, ALUBase)
        self.assertEqual(set(ALUBase.ports_spec), {"a", "b", "op", "result", "zero"})
        self.assertEqual(StandardALU.ports_spec, ALUBase.ports_spec)


# ════════════════════════════════════════════════════════════════════════════
# ALU input mux
# ════════════════════════════════════════════════════════════════════════════

class TestAluInputMux(unittest.TestCase):

    def setUp(self):
        self.m = AluInputMux()
        self.m["rs1_data"] = 0x1111
        self.m["rs2_data"] = 0x2222
        self.m["pc"] = 0x3333
        self.m["imm"] = 0x4444

    def sel(self, use_pc, alu_src):
        self.m["use_pc"] = use_pc
        self.m["alu_src"] = alu_src
        self.m.evaluate()
        return self.m["alu_a"], self.m["alu_b"]

    def test_all_four_select_combinations(self):
        self.assertEqual(self.sel(0, 0), (0x1111, 0x2222))   # R-type
        self.assertEqual(self.sel(0, 1), (0x1111, 0x4444))   # I-type
        self.assertEqual(self.sel(1, 0), (0x3333, 0x2222))   # PC + reg
        self.assertEqual(self.sel(1, 1), (0x3333, 0x4444))   # AUIPC / branch target

    def test_the_two_selects_are_independent(self):
        a0, _ = self.sel(0, 1)
        a1, _ = self.sel(1, 1)
        self.assertNotEqual(a0, a1)
        _, b0 = self.sel(1, 0)
        _, b1 = self.sel(1, 1)
        self.assertNotEqual(b0, b1)

    def test_selects_are_one_bit_ports(self):
        self.m["use_pc"] = 2
        self.assertEqual(self.m["use_pc"], 0)
        self.m["alu_src"] = 3
        self.assertEqual(self.m["alu_src"], 1)

    def test_passes_full_width_values_through(self):
        self.m["rs1_data"] = 0xFFFF_FFFF
        self.m["rs2_data"] = 0x8000_0000
        self.assertEqual(self.sel(0, 0), (0xFFFF_FFFF, 0x8000_0000))

    def test_is_combinational(self):
        before = self.sel(1, 1)
        self.m.rising_edge()
        self.m.evaluate()
        self.assertEqual((self.m["alu_a"], self.m["alu_b"]), before)


# ════════════════════════════════════════════════════════════════════════════
# Flags register (x86 EFLAGS latch)
# ════════════════════════════════════════════════════════════════════════════

class TestFlagsRegister(unittest.TestCase):

    def setUp(self):
        self.f = FlagsRegister()

    def test_initial_state_is_cleared(self):
        self.f.evaluate()
        self.assertEqual(self.f["zero_out"], 0)
        self.assertEqual(self.f["result_out"], 0)
        self.assertEqual(self.f.get_state(), {"zero": 0, "result": 0, "write_flags": 0})

    def test_write_flags_passes_the_live_alu_outputs_through(self):
        self.f["alu_zero_in"] = 1
        self.f["alu_result_in"] = 0xABCD
        self.f["write_flags"] = 1
        self.f.evaluate()
        self.assertEqual(self.f["zero_out"], 1)
        self.assertEqual(self.f["result_out"], 0xABCD)

    def test_pass_through_happens_before_the_latch_updates(self):
        """evaluate() must not depend on rising_edge() having run first."""
        self.f["alu_zero_in"] = 1
        self.f["alu_result_in"] = 0xABCD
        self.f["write_flags"] = 1
        self.f.evaluate()
        self.assertEqual(self.f.get_state()["zero"], 0, "not latched yet")
        self.f.rising_edge()
        self.assertEqual(self.f.get_state(), {"zero": 1, "result": 0xABCD,
                                              "write_flags": 1})

    def test_a_non_flag_setter_holds_the_last_latched_flags(self):
        self.f["alu_zero_in"] = 1
        self.f["alu_result_in"] = 0xABCD
        self.f["write_flags"] = 1
        self.f.evaluate()
        self.f.rising_edge()
        # Now a Jcc: the ALU is computing something unrelated.
        self.f["write_flags"] = 0
        self.f["alu_zero_in"] = 0
        self.f["alu_result_in"] = 0xFFFF
        self.f.evaluate()
        self.assertEqual(self.f["zero_out"], 1, "CMP's ZF survives")
        self.assertEqual(self.f["result_out"], 0xABCD)

    def test_rising_edge_is_a_no_op_when_write_flags_is_clear(self):
        self.f["alu_zero_in"] = 1
        self.f["write_flags"] = 1
        self.f.rising_edge()
        self.f["alu_zero_in"] = 0
        self.f["write_flags"] = 0
        self.f.rising_edge()
        self.assertEqual(self.f.get_state()["zero"], 1)

    def test_a_second_flag_setter_overwrites_the_latch(self):
        for zero, result in ((1, 0x11), (0, 0x22)):
            self.f["write_flags"] = 1
            self.f["alu_zero_in"] = zero
            self.f["alu_result_in"] = result
            self.f.evaluate()
            self.f.rising_edge()
        self.f["write_flags"] = 0
        self.f.evaluate()
        self.assertEqual((self.f["zero_out"], self.f["result_out"]), (0, 0x22))

    def test_latches_a_zero_result_not_just_truthy_ones(self):
        self.f["write_flags"] = 1
        self.f["alu_zero_in"] = 1
        self.f["alu_result_in"] = 0x99
        self.f.rising_edge()
        self.f["alu_result_in"] = 0
        self.f.rising_edge()
        self.f["write_flags"] = 0
        self.f.evaluate()
        self.assertEqual(self.f["result_out"], 0)

    def test_each_instance_has_its_own_latch(self):
        other = FlagsRegister()
        self.f["write_flags"] = 1
        self.f["alu_zero_in"] = 1
        self.f.rising_edge()
        self.assertEqual(other.get_state()["zero"], 0)


# ════════════════════════════════════════════════════════════════════════════
# Fetch units
# ════════════════════════════════════════════════════════════════════════════

class TestSimpleFetch(unittest.TestCase):

    def test_reset_pc_and_outputs(self):
        f = SimpleFetch(pc_reset=0x100)
        f.evaluate()
        self.assertEqual(f["pc_out"], 0x100)
        self.assertEqual(f["pc4_out"], 0x104)
        self.assertEqual(f.get_state(), {"pc": "0x00000100", "pc4": "0x00000104"})

    def test_default_reset_pc_is_zero(self):
        f = SimpleFetch()
        f.evaluate()
        self.assertEqual(f["pc_out"], 0)

    def test_sequential_advance_is_plus_four(self):
        f = SimpleFetch()
        seen = []
        for _ in range(4):
            f.evaluate()
            seen.append(f["pc_out"])
            f.rising_edge()
        self.assertEqual(seen, [0, 4, 8, 12])

    # ── the documented precedence: branch_taken > stall > predict_taken > +4 ──
    def _all_asserted(self):
        f = SimpleFetch(pc_reset=0x100)
        f["next_pc"] = 0x200
        f["predict_target"] = 0x300
        f["branch_taken"] = 1
        f["stall"] = 1
        f["predict_taken"] = 1
        return f

    def test_precedence_1_resolved_branch_beats_everything(self):
        f = self._all_asserted()
        f.rising_edge()
        self.assertEqual(f._pc, 0x200)

    def test_precedence_2_stall_beats_prediction_and_plus4(self):
        f = self._all_asserted()
        f["branch_taken"] = 0
        f.rising_edge()
        self.assertEqual(f._pc, 0x100, "stall holds the PC")

    def test_precedence_3_prediction_beats_plus4(self):
        f = self._all_asserted()
        f["branch_taken"] = 0
        f["stall"] = 0
        f.rising_edge()
        self.assertEqual(f._pc, 0x300)

    def test_precedence_4_fallthrough_is_plus4(self):
        f = self._all_asserted()
        f["branch_taken"] = 0
        f["stall"] = 0
        f["predict_taken"] = 0
        f.rising_edge()
        self.assertEqual(f._pc, 0x104)

    def test_full_precedence_ladder_in_one_sequence(self):
        f = self._all_asserted()
        f.rising_edge()                       # branch -> 0x200
        f["branch_taken"] = 0
        f.rising_edge()                       # stall  -> held
        f["stall"] = 0
        f.rising_edge()                       # predict -> 0x300
        f["predict_taken"] = 0
        f.rising_edge()                       # +4     -> 0x304
        f.evaluate()
        self.assertEqual(f["pc_out"], 0x304)

    def test_branch_taken_can_redirect_while_stalled(self):
        """Multicycle holds `stall` all through F/D/E yet must still redirect."""
        f = SimpleFetch(pc_reset=0x40)
        f["stall"] = 1
        f["branch_taken"] = 1
        f["next_pc"] = 0x80
        f.rising_edge()
        self.assertEqual(f._pc, 0x80)

    def test_branch_to_address_zero_is_honoured(self):
        f = SimpleFetch(pc_reset=0x40)
        f["branch_taken"] = 1
        f["next_pc"] = 0
        f.rising_edge()
        self.assertEqual(f._pc, 0)

    def test_predict_taken_to_address_zero_is_honoured(self):
        f = SimpleFetch(pc_reset=0x40)
        f["predict_taken"] = 1
        f["predict_target"] = 0
        f.rising_edge()
        self.assertEqual(f._pc, 0)

    def test_pc_wraps_at_2_32(self):
        f = SimpleFetch(pc_reset=0xFFFF_FFFC)
        f.evaluate()
        self.assertEqual(f["pc4_out"], 0, "PC+4 wraps")
        f.rising_edge()
        self.assertEqual(f._pc, 0)

    def test_evaluate_reflects_the_pc_written_at_the_previous_edge(self):
        f = SimpleFetch()
        f.evaluate()
        self.assertEqual(f["pc_out"], 0)
        f.rising_edge()
        self.assertEqual(f["pc_out"], 0, "pc_out is stale until re-evaluated")
        f.evaluate()
        self.assertEqual(f["pc_out"], 4)

    def test_stall_holds_the_pc_indefinitely(self):
        f = SimpleFetch(pc_reset=0x20)
        f["stall"] = 1
        for _ in range(10):
            f.rising_edge()
        self.assertEqual(f._pc, 0x20)


class TestX86Fetch(unittest.TestCase):

    def test_reset_and_default_length(self):
        f = X86Fetch(pc_reset=0)
        f["instr_len"] = 0
        f.evaluate()
        self.assertEqual(f["pc_out"], 0)
        self.assertEqual(f["pc_plus_len"], 1, "len 0 is treated as 1")
        self.assertEqual(f.get_state(),
                         {"pc": "0x00000000", "pc_plus_len": "0x00000001",
                          "instr_len": 0})

    def test_advances_by_instruction_length(self):
        f = X86Fetch()
        for length, expected in ((1, 1), (5, 6), (2, 8), (6, 14)):
            f["instr_len"] = length
            f.rising_edge()
            f.evaluate()
            self.assertEqual(f["pc_out"], expected)

    def test_zero_length_still_advances_by_one(self):
        f = X86Fetch()
        f["instr_len"] = 0
        f.rising_edge()
        self.assertEqual(f._pc, 1, "a 0-length instruction must not hang the PC")

    def test_instr_len_is_a_four_bit_port(self):
        f = X86Fetch()
        f["instr_len"] = 15
        self.assertEqual(f["instr_len"], 15)
        f["instr_len"] = 16
        self.assertEqual(f["instr_len"], 0)

    def _all_asserted(self):
        f = X86Fetch(pc_reset=0x100)
        f["instr_len"] = 5
        f["next_pc"] = 0x200
        f["predict_target"] = 0x300
        f["branch_taken"] = 1
        f["stall"] = 1
        f["predict_taken"] = 1
        return f

    def test_precedence_matches_simple_fetch(self):
        f = self._all_asserted()
        f.rising_edge()
        self.assertEqual(f._pc, 0x200, "branch_taken wins")
        f["branch_taken"] = 0
        f.rising_edge()
        self.assertEqual(f._pc, 0x200, "stall beats prediction")
        f["stall"] = 0
        f.rising_edge()
        self.assertEqual(f._pc, 0x300, "prediction beats +len")
        f["predict_taken"] = 0
        f.rising_edge()
        self.assertEqual(f._pc, 0x305, "+len fall-through")

    def test_pc_plus_len_wraps(self):
        f = X86Fetch(pc_reset=0xFFFF_FFFF)
        f["instr_len"] = 2
        f.evaluate()
        self.assertEqual(f["pc_plus_len"], 1)
        f.rising_edge()
        self.assertEqual(f._pc, 1)

    def test_stall_holds_the_pc(self):
        f = X86Fetch(pc_reset=0x10)
        f["instr_len"] = 3
        f["stall"] = 1
        for _ in range(5):
            f.rising_edge()
        self.assertEqual(f._pc, 0x10)


class TestPcLenAdder(unittest.TestCase):

    def test_adds_the_instruction_length(self):
        p = PcLenAdder()
        p["pc"] = 0x100
        for length in range(1, 16):
            p["len"] = length
            p.evaluate()
            self.assertEqual(p["out"], 0x100 + length)

    def test_zero_length_is_treated_as_one(self):
        p = PcLenAdder()
        p["pc"] = 0x100
        p["len"] = 0
        p.evaluate()
        self.assertEqual(p["out"], 0x101)

    def test_wraps_at_2_32(self):
        p = PcLenAdder()
        p["pc"] = 0xFFFF_FFFF
        p["len"] = 3
        p.evaluate()
        self.assertEqual(p["out"], 2)

    def test_get_state(self):
        p = PcLenAdder()
        p["pc"] = 0x10
        p["len"] = 5
        p.evaluate()
        self.assertEqual(p.get_state(),
                         {"pc": "0x00000010", "len": 5, "out": "0x00000015"})

    def test_is_combinational(self):
        p = PcLenAdder()
        p["pc"] = 0x10
        p["len"] = 2
        p.evaluate()
        p.rising_edge()
        p.evaluate()
        self.assertEqual(p["out"], 0x12)

    def test_matches_x86_fetch_fall_through_for_every_length(self):
        """The adder exists to recompute what X86Fetch can't; they must agree."""
        for length in range(0, 16):
            f = X86Fetch(pc_reset=0x400)
            f["instr_len"] = length
            f.evaluate()
            p = PcLenAdder()
            p["pc"] = 0x400
            p["len"] = length
            p.evaluate()
            self.assertEqual(p["out"], f["pc_plus_len"], f"len={length}")


# ════════════════════════════════════════════════════════════════════════════
# Instruction memories
# ════════════════════════════════════════════════════════════════════════════

NOP_WORD = 0x0000_0013     # ADDI x0, x0, 0


class TestInstructionMemory(unittest.TestCase):

    def setUp(self):
        self.im = InstructionMemory([0xAAAA_AAAA, 0xBBBB_BBBB, 0xCCCC_CCCC])

    def read(self, addr):
        self.im["addr"] = addr
        self.im.evaluate()
        return self.im["data"]

    def test_word_addressing(self):
        self.assertEqual(self.read(0), 0xAAAA_AAAA)
        self.assertEqual(self.read(4), 0xBBBB_BBBB)
        self.assertEqual(self.read(8), 0xCCCC_CCCC)

    def test_unaligned_addresses_truncate_to_the_containing_word(self):
        for addr in (4, 5, 6, 7):
            self.assertEqual(self.read(addr), 0xBBBB_BBBB, f"addr {addr}")

    def test_out_of_range_fetch_returns_a_nop(self):
        self.assertEqual(self.read(12), NOP_WORD, "one past the end")
        self.assertEqual(self.read(0x1000), NOP_WORD)
        self.assertEqual(self.read(0xFFFF_FFFF), NOP_WORD)

    def test_last_valid_word_then_first_invalid(self):
        self.assertEqual(self.read(8), 0xCCCC_CCCC)
        self.assertEqual(self.read(11), 0xCCCC_CCCC)
        self.assertEqual(self.read(12), NOP_WORD)

    def test_empty_program_is_all_nops(self):
        im = InstructionMemory([])
        im["addr"] = 0
        im.evaluate()
        self.assertEqual(im["data"], NOP_WORD)
        self.assertEqual(im.get_state()["size"], 0)

    def test_program_words_are_masked_to_32_bits(self):
        im = InstructionMemory([(1 << 32) | 1, -1])
        self.assertEqual(im._program, [1, 0xFFFF_FFFF])

    def test_is_read_only(self):
        before = list(self.im._program)
        self.read(0)
        self.im.rising_edge()
        self.assertEqual(self.im._program, before)

    def test_get_state(self):
        self.read(4)
        st = self.im.get_state()
        self.assertEqual(st["addr"], "0x00000004")
        self.assertEqual(st["data"], "0xbbbbbbbb")
        self.assertEqual(st["size"], 3)
        self.assertEqual(st["program"], [0xAAAA_AAAA, 0xBBBB_BBBB, 0xCCCC_CCCC])

    def test_get_state_program_is_a_copy(self):
        st = self.im.get_state()
        st["program"][0] = 0
        self.assertEqual(self.im._program[0], 0xAAAA_AAAA)

    def test_addr_port_masks_to_32_bits(self):
        self.im["addr"] = 1 << 32
        self.assertEqual(self.im["addr"], 0)
        self.im.evaluate()
        self.assertEqual(self.im["data"], 0xAAAA_AAAA)


class TestByteInstructionMemory(unittest.TestCase):

    # MOV EAX, 1  =  B8 01 00 00 00
    PROG = [0xB8, 0x01, 0x00, 0x00, 0x00]

    def setUp(self):
        self.bm = ByteInstructionMemory(self.PROG)

    def read(self, addr):
        self.bm["addr"] = addr
        self.bm.evaluate()
        return self.bm["data"]

    def test_packs_eight_bytes_little_endian(self):
        # bytes B8 01 00 00 00 then 3 NOP (0x90) pads
        self.assertEqual(self.read(0), 0x9090_9000_0000_01B8)

    def test_every_byte_offset_is_addressable(self):
        self.assertEqual(self.read(1) & 0xFF, 0x01)
        self.assertEqual(self.read(2) & 0xFF, 0x00)
        self.assertEqual(self.read(4) & 0xFF, 0x00)

    def test_reads_past_the_end_pad_with_nop_0x90(self):
        self.assertEqual(self.read(5), 0x9090_9090_9090_9090)
        self.assertEqual(self.read(1000), 0x9090_9090_9090_9090)

    def test_a_read_straddling_the_end_pads_only_the_missing_bytes(self):
        # addr 3 -> bytes[3]=0x00, bytes[4]=0x00, then six 0x90 pads
        self.assertEqual(self.read(3), 0x9090_9090_9090_0000)

    def test_program_bytes_are_masked(self):
        bm = ByteInstructionMemory([0x1FF, -1, 256])
        self.assertEqual(list(bm._program), [0xFF, 0xFF, 0x00])

    def test_empty_program_is_all_nop_padding(self):
        bm = ByteInstructionMemory([])
        bm["addr"] = 0
        bm.evaluate()
        self.assertEqual(bm["data"], 0x9090_9090_9090_9090)
        self.assertEqual(bm.get_state()["size"], 0)

    def test_data_port_is_64_bits_wide(self):
        self.assertEqual(ByteInstructionMemory.ports_spec["data"].width, 64,
                         "5-6 byte x86 instructions must survive the fetch bus")
        self.assertEqual(self.read(5).bit_length(), 64)

    def test_get_state_renders_the_byte_window(self):
        self.read(0)
        st = self.bm.get_state()
        self.assertEqual(st["addr"], "0x00000000")
        self.assertEqual(st["bytes"], "B8 01 00 00 00 90 90 90")
        self.assertEqual(st["size"], 5)
        self.assertEqual(st["program_bytes"], self.PROG)

    def test_get_state_pads_out_of_range_windows_with_90(self):
        self.read(100)
        self.assertEqual(self.bm.get_state()["bytes"], "90 90 90 90 90 90 90 90")

    def test_is_read_only(self):
        before = bytes(self.bm._program)
        self.read(0)
        self.bm.rising_edge()
        self.assertEqual(self.bm._program, before)


# ════════════════════════════════════════════════════════════════════════════
# Data memory
# ════════════════════════════════════════════════════════════════════════════

class TestDataMemory(unittest.TestCase):

    def test_default_capacity_is_64kb(self):
        d = DataMemory()
        self.assertEqual(d._size, DEFAULT_MEM_WORDS)
        self.assertEqual(DEFAULT_MEM_WORDS * 4, 65536)
        self.assertEqual(d.get_state()["size"], DEFAULT_MEM_WORDS)

    def test_word_addressing_reads_and_writes(self):
        d = DataMemory(size=16)
        d["addr"] = 4
        d["wdata"] = 0xDEAD
        d["wen"] = 1
        d.rising_edge()
        self.assertEqual(d._mem[1], 0xDEAD, "byte 4 == word 1")
        d.evaluate()
        self.assertEqual(d["rdata"], 0xDEAD)

    def test_unaligned_addresses_hit_the_containing_word(self):
        d = DataMemory(size=16)
        d._mem[1] = 0x55
        for addr in (4, 5, 6, 7):
            d["addr"] = addr
            d.evaluate()
            self.assertEqual(d["rdata"], 0x55, f"addr {addr}")

    def test_read_is_combinational_and_write_is_synchronous(self):
        d = DataMemory(size=16)
        d._mem[2] = 0x55
        d["addr"] = 8
        d["wdata"] = 0x99
        d["wen"] = 1
        d.evaluate()
        self.assertEqual(d["rdata"], 0x55, "read-before-write in the same cycle")
        d.rising_edge()
        d.evaluate()
        self.assertEqual(d["rdata"], 0x99)

    def test_wen_low_blocks_the_write(self):
        d = DataMemory(size=16)
        d["addr"] = 0
        d["wdata"] = 0x77
        d["wen"] = 0
        d.rising_edge()
        self.assertEqual(d._mem[0], 0)

    def test_waddr_valid_selects_the_dedicated_write_address(self):
        d = DataMemory(size=16)
        d["addr"] = 40          # word 10 (the read bus)
        d["waddr"] = 8          # word 2  (the OoO store-commit bus)
        d["waddr_valid"] = 1
        d["wdata"] = 0x1234
        d["wen"] = 1
        d.rising_edge()
        self.assertEqual(d._mem[2], 0x1234)
        self.assertEqual(d._mem[10], 0, "the read address must not be written")

    def test_waddr_valid_allows_a_store_to_byte_address_zero(self):
        """The reason waddr_valid exists: waddr==0 is a legitimate target."""
        d = DataMemory(size=16)
        d["addr"] = 40
        d["waddr"] = 0
        d["waddr_valid"] = 1
        d["wdata"] = 0xABC
        d["wen"] = 1
        d.rising_edge()
        self.assertEqual(d._mem[0], 0xABC)

    def test_waddr_ignored_when_waddr_valid_is_clear(self):
        d = DataMemory(size=16)
        d["addr"] = 0
        d["waddr"] = 8
        d["waddr_valid"] = 0
        d["wdata"] = 0x7
        d["wen"] = 1
        d.rising_edge()
        self.assertEqual(d._mem[0], 0x7, "falls back to the shared addr bus")
        self.assertEqual(d._mem[2], 0)

    def test_get_state_write_addr_mirrors_the_waddr_fallback(self):
        d = DataMemory(size=16)
        d["addr"] = 12
        d["waddr"] = 4
        d["waddr_valid"] = 0
        self.assertEqual(d.get_state()["write_addr"], 12)
        d["waddr_valid"] = 1
        self.assertEqual(d.get_state()["write_addr"], 4)

    def test_print_addr_write_is_visible_via_write_addr(self):
        """How the /lab console detects print() stores."""
        d = DataMemory()
        d["addr"] = PRINT_ADDR
        d["wdata"] = 42
        d["wen"] = 1
        d.rising_edge()
        st = d.get_state()
        self.assertEqual(st["write_addr"], PRINT_ADDR)
        self.assertEqual(st["wen"], 1)
        self.assertEqual(st["memory"][PRINT_ADDR >> 2], 42)

    def test_addresses_wrap_modulo_the_size(self):
        d = DataMemory(size=4)
        d["addr"] = 16          # word 4 -> wraps to word 0
        d["wdata"] = 7
        d["wen"] = 1
        d.rising_edge()
        self.assertEqual(d._mem, [7, 0, 0, 0])
        d["addr"] = 0
        d.evaluate()
        self.assertEqual(d["rdata"], 7)

    def test_top_of_the_address_space_wraps_rather_than_raising(self):
        d = DataMemory()
        d["addr"] = 0xFFFF_FFFC
        d["wdata"] = 5
        d["wen"] = 1
        d.rising_edge()
        d.evaluate()
        self.assertEqual(d["rdata"], 5)
        self.assertEqual(d._mem[(0xFFFF_FFFC >> 2) % DEFAULT_MEM_WORDS], 5)

    # ── the dense window / sparse high map split ─────────────────
    def test_state_window_constant(self):
        self.assertEqual(STATE_WINDOW, 2048)

    def test_dense_window_covers_the_low_region_in_full(self):
        d = DataMemory()
        st = d.get_state()
        self.assertEqual(len(st["memory"]), STATE_WINDOW)
        self.assertEqual(st["window"], STATE_WINDOW)
        self.assertEqual(st["memory_hi"], {}, "zeros above the window are omitted")

    def test_low_writes_land_in_the_dense_window(self):
        d = DataMemory()
        d["addr"] = (STATE_WINDOW - 1) * 4
        d["wdata"] = 0x1234
        d["wen"] = 1
        d.rising_edge()
        st = d.get_state()
        self.assertEqual(st["memory"][STATE_WINDOW - 1], 0x1234)
        self.assertEqual(st["memory_hi"], {})

    def test_high_writes_land_in_the_sparse_map_only(self):
        d = DataMemory()
        d["addr"] = STATE_WINDOW * 4     # first word above the window
        d["wdata"] = 0x99
        d["wen"] = 1
        d.rising_edge()
        st = d.get_state()
        self.assertEqual(len(st["memory"]), STATE_WINDOW)
        self.assertEqual(st["memory_hi"], {STATE_WINDOW: 0x99})

    def test_stack_top_word_is_reachable_and_reported_high(self):
        d = DataMemory()
        word = STACK_TOP >> 2
        self.assertLess(word, d._size, "STACK_TOP must fit in the 64 KB memory")
        d["addr"] = STACK_TOP
        d["wdata"] = 0xFEED
        d["wen"] = 1
        d.rising_edge()
        d.evaluate()
        self.assertEqual(d["rdata"], 0xFEED)
        self.assertEqual(d.get_state()["memory_hi"], {word: 0xFEED})

    def test_sparse_map_keys_are_ints(self):
        d = DataMemory()
        d["addr"] = STACK_TOP
        d["wdata"] = 1
        d["wen"] = 1
        d.rising_edge()
        for k in d.get_state()["memory_hi"]:
            self.assertIsInstance(k, int)

    def test_full_word_reconstruction_rule(self):
        """`w < len(memory) ? memory[w] : memory_hi[w]` must recover every word."""
        d = DataMemory()
        targets = {0: 1, 5: 2, STATE_WINDOW - 1: 3, STATE_WINDOW: 4,
                   STACK_TOP >> 2: 5}
        for word, val in targets.items():
            d["addr"] = word * 4
            d["wdata"] = val
            d["wen"] = 1
            d.rising_edge()
        st = d.get_state()
        for word, val in targets.items():
            got = st["memory"][word] if word < len(st["memory"]) \
                else st["memory_hi"].get(word, 0)
            self.assertEqual(got, val, f"word {word}")

    def test_memory_smaller_than_the_window_reports_a_shrunken_window(self):
        d = DataMemory(size=4)
        st = d.get_state()
        self.assertEqual(st["window"], 4)
        self.assertEqual(len(st["memory"]), 4)
        self.assertEqual(st["memory_hi"], {})

    def test_memory_snapshot_helper(self):
        dense, hi = memory_snapshot([1, 0, 5, 0, 9], 5, window=2)
        self.assertEqual(dense, [1, 0])
        self.assertEqual(hi, {2: 5, 4: 9}, "zeros above the window are dropped")

    def test_memory_snapshot_window_larger_than_size(self):
        dense, hi = memory_snapshot([1, 2], 2, window=99)
        self.assertEqual((dense, hi), ([1, 2], {}))

    def test_wdata_port_masks_to_32_bits(self):
        d = DataMemory(size=4)
        d["wdata"] = -1
        d["wen"] = 1
        d["addr"] = 0
        d.rising_edge()
        self.assertEqual(d._mem[0], 0xFFFF_FFFF)


class TestMultiPortDataMemory(unittest.TestCase):

    def test_declares_one_port_group_per_lane(self):
        m = MultiPortDataMemory(num_lanes=3, size=16)
        for i in range(3):
            for p in ("addr", "wdata", "mem_read", "mem_write", "rdata"):
                self.assertIn(f"{p}_{i}", m.ports_spec)
        self.assertNotIn("addr_3", m.ports_spec)

    def test_lanes_read_independently(self):
        m = MultiPortDataMemory(num_lanes=2, size=16)
        m._mem[0], m._mem[1] = 0xAA, 0xBB
        m["addr_0"], m["addr_1"] = 0, 4
        m["mem_read_0"] = m["mem_read_1"] = 1
        m.evaluate()
        self.assertEqual((m["rdata_0"], m["rdata_1"]), (0xAA, 0xBB))

    def test_older_lane_store_forwards_to_a_younger_lane_load(self):
        m = MultiPortDataMemory(num_lanes=2, size=16)
        m._mem[1] = 0xAA
        m["addr_0"] = 4
        m["mem_write_0"] = 1
        m["wdata_0"] = 0xBB
        m["addr_1"] = 4
        m["mem_read_1"] = 1
        m.evaluate()
        self.assertEqual(m["rdata_1"], 0xBB, "lane 1 sees lane 0's store")

    def test_a_younger_lane_store_never_forwards_backwards(self):
        m = MultiPortDataMemory(num_lanes=2, size=16)
        m._mem[1] = 0xAA
        m["addr_0"] = 4
        m["mem_read_0"] = 1
        m["addr_1"] = 4
        m["mem_write_1"] = 1
        m["wdata_1"] = 0xBB
        m.evaluate()
        self.assertEqual(m["rdata_0"], 0xAA, "lane 0 is OLDER: no back-forwarding")

    def test_same_word_stores_commit_in_lane_order(self):
        m = MultiPortDataMemory(num_lanes=3, size=16)
        for i, val in enumerate((0x11, 0x22, 0x33)):
            m[f"addr_{i}"] = 4
            m[f"mem_write_{i}"] = 1
            m[f"wdata_{i}"] = val
        m.rising_edge()
        self.assertEqual(m._mem[1], 0x33, "the youngest lane wins")

    def test_get_state_reports_the_active_lanes(self):
        m = MultiPortDataMemory(num_lanes=3, size=16)
        m["addr_1"] = 8
        m["mem_write_1"] = 1
        m["wdata_1"] = 0x42
        m.evaluate()
        st = m.get_state()
        self.assertEqual(st["active_lanes"], [1])
        self.assertEqual(st["wen"], 1)
        self.assertEqual(st["write_addr"], 8)
        self.assertEqual(st["addr"], "0x00000008")

    def test_get_state_with_no_active_lane(self):
        m = MultiPortDataMemory(num_lanes=2, size=16)
        st = m.get_state()
        self.assertEqual(st["active_lanes"], [])
        self.assertEqual(st["wen"], 0)
        self.assertEqual(st["addr"], "0x00000000")

    def test_shares_the_dense_window_snapshot_shape_with_datamemory(self):
        m = MultiPortDataMemory(num_lanes=2)
        st = m.get_state()
        self.assertEqual(len(st["memory"]), STATE_WINDOW)
        self.assertEqual(st["size"], DEFAULT_MEM_WORDS)
        self.assertEqual(st["memory_hi"], {})


class TestMemoryUnitBase(unittest.TestCase):

    def test_port_contract(self):
        self.assertEqual(set(MemoryUnitBase.ports_spec), {"addr", "rdata"})
        self.assertEqual(MemoryUnitBase.ui_category, "memory")


# ════════════════════════════════════════════════════════════════════════════
# Register file
# ════════════════════════════════════════════════════════════════════════════

class TestStandardRegisterFile(unittest.TestCase):

    def riscv(self, **kw):
        return StandardRegisterFile(num_regs=32, zero_reg=True, **kw)

    def write(self, rf, rd, data):
        rf["rd_addr"] = rd
        rf["rd_data"] = data
        rf["wen"] = 1
        rf.rising_edge()

    def read(self, rf, rs1=0, rs2=0, rs3=0):
        rf["rs1_addr"], rf["rs2_addr"], rf["rs3_addr"] = rs1, rs2, rs3
        rf.evaluate()
        return rf["rs1_data"], rf["rs2_data"], rf["rs3_data"]

    def test_conforms_to_the_port_contract(self):
        self.assertIsInstance(self.riscv(), RegisterFileBase)
        self.assertEqual(set(RegisterFileBase.ports_spec),
                         {"rs1_addr", "rs2_addr", "rs3_addr", "rs1_data",
                          "rs2_data", "rs3_data", "rd_addr", "rd_data", "wen"})

    def test_starts_all_zero(self):
        rf = self.riscv()
        self.assertEqual(rf.regs, [0] * 32)

    def test_write_then_read(self):
        rf = self.riscv()
        self.write(rf, 5, 0xDEAD)
        self.assertEqual(self.read(rf, rs1=5)[0], 0xDEAD)

    def test_three_read_ports_are_independent(self):
        rf = self.riscv()
        self.write(rf, 1, 0x11)
        self.write(rf, 2, 0x22)
        self.write(rf, 3, 0x33)
        self.assertEqual(self.read(rf, 1, 2, 3), (0x11, 0x22, 0x33))

    def test_wen_low_blocks_the_write(self):
        rf = self.riscv()
        rf["rd_addr"] = 5
        rf["rd_data"] = 0xBEEF
        rf["wen"] = 0
        rf.rising_edge()
        self.assertEqual(rf.regs[5], 0)

    # ── zero register ────────────────────────────────────────────
    def test_riscv_x0_is_hardwired_to_zero(self):
        rf = self.riscv()
        self.write(rf, 0, 0xFFFF_FFFF)
        self.assertEqual(rf.regs[0], 0, "the write must be dropped")
        self.assertEqual(self.read(rf, rs1=0)[0], 0)

    def test_arm_zero_register_is_x31(self):
        rf = StandardRegisterFile(num_regs=32, zero_reg=True, zero_reg_index=31)
        self.write(rf, 31, 0xFFFF_FFFF)
        self.assertEqual(rf.regs[31], 0)
        self.assertEqual(self.read(rf, rs1=31)[0], 0)
        # ...and X0 is a NORMAL register on ARM.
        self.write(rf, 0, 0x1234)
        self.assertEqual(rf.regs[0], 0x1234)
        self.assertEqual(self.read(rf, rs1=0)[0], 0x1234)

    def test_x86_has_no_zero_register(self):
        rf = StandardRegisterFile(num_regs=8, zero_reg=False)
        self.write(rf, 0, 0x1234)
        self.assertEqual(rf.regs[0], 0x1234, "EAX (index 0) is a real GPR")
        self.assertEqual(self.read(rf, rs1=0)[0], 0x1234)

    def test_zero_reg_false_ignores_zero_reg_index(self):
        rf = StandardRegisterFile(num_regs=32, zero_reg=False, zero_reg_index=31)
        self.write(rf, 31, 0x99)
        self.assertEqual(rf.regs[31], 0x99)

    # ── address wrapping ─────────────────────────────────────────
    def test_addresses_wrap_modulo_num_regs(self):
        rf = StandardRegisterFile(num_regs=8, zero_reg=False)
        self.write(rf, 9, 0x42)         # 9 % 8 == 1
        self.assertEqual(rf.regs[1], 0x42)
        self.assertEqual(self.read(rf, rs1=1)[0], 0x42)
        self.assertEqual(self.read(rf, rs1=9)[0], 0x42)

    def test_address_ports_are_five_bits_wide(self):
        rf = self.riscv()
        rf["rd_addr"] = 33
        self.assertEqual(rf["rd_addr"], 1, "5-bit port wraps mod 32")
        rf["rs1_addr"] = 31
        self.assertEqual(rf["rs1_addr"], 31)

    # ── write-through ────────────────────────────────────────────
    def test_write_through_off_by_default(self):
        rf = self.riscv()
        rf["rd_addr"] = 6
        rf["rd_data"] = 0x77
        rf["wen"] = 1
        self.assertEqual(self.read(rf, rs1=6)[0], 0,
                         "without write-through the read sees the OLD value")

    def test_write_through_forwards_within_the_same_cycle(self):
        rf = self.riscv(write_through=True)
        rf["rd_addr"] = 6
        rf["rd_data"] = 0x77
        rf["wen"] = 1
        self.assertEqual(self.read(rf, rs1=6, rs2=6, rs3=6), (0x77, 0x77, 0x77))

    def test_write_through_only_affects_the_matching_register(self):
        rf = self.riscv(write_through=True)
        rf.regs[7] = 0x11
        rf["rd_addr"] = 6
        rf["rd_data"] = 0x77
        rf["wen"] = 1
        self.assertEqual(self.read(rf, rs1=6, rs2=7), (0x77, 0x11, 0))

    def test_write_through_requires_wen(self):
        rf = self.riscv(write_through=True)
        rf["rd_addr"] = 6
        rf["rd_data"] = 0x77
        rf["wen"] = 0
        self.assertEqual(self.read(rf, rs1=6)[0], 0)

    def test_write_through_never_defeats_the_zero_register(self):
        rf = self.riscv(write_through=True)
        rf["rd_addr"] = 0
        rf["rd_data"] = 0x77
        rf["wen"] = 1
        self.assertEqual(self.read(rf, rs1=0)[0], 0)

    def test_write_through_matches_on_the_wrapped_address(self):
        rf = StandardRegisterFile(num_regs=8, zero_reg=False, write_through=True)
        rf["rd_addr"] = 9        # -> reg 1
        rf["rd_data"] = 0x42
        rf["wen"] = 1
        self.assertEqual(self.read(rf, rs1=1)[0], 0x42)

    def test_write_through_and_the_edge_agree(self):
        rf = self.riscv(write_through=True)
        rf["rd_addr"] = 6
        rf["rd_data"] = 0x77
        rf["wen"] = 1
        forwarded = self.read(rf, rs1=6)[0]
        rf.rising_edge()
        rf["wen"] = 0
        self.assertEqual(self.read(rf, rs1=6)[0], forwarded)

    # ── misc ────────────────────────────────────────────────────
    def test_full_width_values_round_trip(self):
        rf = self.riscv()
        for val in (0, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF):
            self.write(rf, 10, val)
            self.assertEqual(self.read(rf, rs1=10)[0], val, hex(val))

    def test_negative_write_data_is_masked_by_the_port(self):
        rf = self.riscv()
        self.write(rf, 10, -1)
        self.assertEqual(rf.regs[10], 0xFFFF_FFFF)

    def test_get_state(self):
        rf = self.riscv()
        self.write(rf, 3, 0x30)
        self.read(rf, rs1=3, rs2=3)
        st = rf.get_state()
        self.assertEqual(st["rs1_data"], 0x30)
        self.assertEqual(st["rs2_data"], 0x30)
        self.assertEqual(len(st["registers"]), 32)
        st["registers"][3] = 0
        self.assertEqual(rf.regs[3], 0x30, "get_state returns a copy")

    def test_x86_file_has_eight_registers(self):
        rf = StandardRegisterFile(num_regs=8, zero_reg=False)
        self.assertEqual(len(rf.get_state()["registers"]), 8)


# ════════════════════════════════════════════════════════════════════════════
# Writeback mux
# ════════════════════════════════════════════════════════════════════════════

class TestWritebackUnit(unittest.TestCase):

    def setUp(self):
        self.w = WritebackUnit()
        self.w["alu_in"] = 0xAAAA
        self.w["mem_in"] = 0xBBBB
        self.w["pc4_in"] = 0xCCCC

    def out(self, sel):
        self.w["sel"] = sel
        self.w.evaluate()
        return self.w["data_out"]

    def test_selects_alu_memory_and_pc4(self):
        self.assertEqual(self.out(WBSel.ALU), 0xAAAA)
        self.assertEqual(self.out(WBSel.MEMORY), 0xBBBB)
        self.assertEqual(self.out(WBSel.PC4), 0xCCCC)

    def test_reserved_select_yields_zero(self):
        self.assertEqual(self.out(3), 0)

    def test_sel_is_a_two_bit_port(self):
        self.w["sel"] = 4
        self.assertEqual(self.w["sel"], 0)
        self.w["sel"] = 5
        self.assertEqual(self.w["sel"], 1)

    def test_forwards_a_zero_source_value(self):
        self.w["mem_in"] = 0
        self.assertEqual(self.out(WBSel.MEMORY), 0)

    def test_full_width_values_pass_through(self):
        self.w["alu_in"] = 0xFFFF_FFFF
        self.assertEqual(self.out(WBSel.ALU), 0xFFFF_FFFF)

    def test_get_state_names_the_selection(self):
        self.out(WBSel.MEMORY)
        self.assertEqual(self.w.get_state(), {"sel": "MEMORY", "data_out": 0xBBBB})
        self.out(WBSel.PC4)
        self.assertEqual(self.w.get_state()["sel"], "PC4")
        self.out(3)
        self.assertEqual(self.w.get_state()["sel"], "3", "unknown sel prints numerically")

    def test_is_combinational(self):
        self.out(WBSel.ALU)
        self.w.rising_edge()
        self.w.evaluate()
        self.assertEqual(self.w["data_out"], 0xAAAA)


# ════════════════════════════════════════════════════════════════════════════
# Multicycle FSM + support components
# ════════════════════════════════════════════════════════════════════════════

class TestMultiCycleController(unittest.TestCase):
    """Drives the FSM one phase at a time and asserts the enable vector."""

    ENABLES = ("ir_latch", "ab_latch", "alu_out_latch", "fetch_stall",
               "reg_write_en", "mem_read_en", "mem_write_en",
               "branch_resolve_en")

    def step(self, c, mem_read=0, mem_write=0, reg_write=0):
        """One cycle: drive the decoder ports, evaluate, snapshot, clock."""
        c["dec_mem_read"] = mem_read
        c["dec_mem_write"] = mem_write
        c["dec_reg_write"] = reg_write
        c.evaluate()
        snap = {k: c[k] for k in self.ENABLES}
        snap["phase"] = c["phase"]
        c.rising_edge()
        return snap

    def test_starts_in_fetch(self):
        c = MultiCycleController()
        c.evaluate()
        self.assertEqual(c["phase"], FETCH)
        self.assertEqual(c.get_state(), {"phase": FETCH, "phase_name": "FETCH"})

    def test_phase_constants(self):
        self.assertEqual((FETCH, DECODE, EXECUTE, MEMORY), (0, 1, 2, 3))

    def test_alu_op_is_a_three_phase_sequence(self):
        c = MultiCycleController()
        f, d, e = (self.step(c, reg_write=1) for _ in range(3))
        self.assertEqual((f["phase"], d["phase"], e["phase"]),
                         (FETCH, DECODE, EXECUTE))
        # FETCH: latch IR and let the PC advance.
        self.assertEqual(f["ir_latch"], 1)
        self.assertEqual(f["fetch_stall"], 0)
        # DECODE: latch A/B, hold the PC.
        self.assertEqual(d["ab_latch"], 1)
        self.assertEqual(d["fetch_stall"], 1)
        # EXECUTE: latch the ALU output, resolve branches, write back.
        self.assertEqual(e["alu_out_latch"], 1)
        self.assertEqual(e["branch_resolve_en"], 1)
        self.assertEqual(e["reg_write_en"], 1)
        self.assertEqual(e["mem_read_en"], 0)
        self.assertEqual(e["mem_write_en"], 0)
        # ...and back to FETCH.
        self.assertEqual(self.step(c, reg_write=1)["phase"], FETCH)

    def test_a_non_writing_alu_op_never_asserts_reg_write_en(self):
        c = MultiCycleController()
        for _ in range(3):
            snap = self.step(c, reg_write=0)
            self.assertEqual(snap["reg_write_en"], 0)

    def test_load_takes_four_phases_and_writes_back_in_memory(self):
        c = MultiCycleController()
        phases = [self.step(c, mem_read=1, reg_write=1) for _ in range(4)]
        self.assertEqual([p["phase"] for p in phases],
                         [FETCH, DECODE, EXECUTE, MEMORY])
        self.assertEqual(phases[EXECUTE]["reg_write_en"], 0,
                         "a load must NOT write back in EXECUTE")
        mem = phases[MEMORY]
        self.assertEqual(mem["mem_read_en"], 1)
        self.assertEqual(mem["mem_write_en"], 0)
        self.assertEqual(mem["reg_write_en"], 1)
        self.assertEqual(mem["fetch_stall"], 1)
        self.assertEqual(self.step(c, mem_read=1, reg_write=1)["phase"], FETCH)

    def test_store_takes_four_phases_and_writes_memory_only(self):
        c = MultiCycleController()
        phases = [self.step(c, mem_write=1) for _ in range(4)]
        mem = phases[MEMORY]
        self.assertEqual(mem["mem_write_en"], 1)
        self.assertEqual(mem["mem_read_en"], 0)
        self.assertEqual(mem["reg_write_en"], 0)

    def test_x86_push_writes_memory_and_a_register_in_the_memory_phase(self):
        """Regression: a store that also writes a register (PUSH updates ESP)."""
        c = MultiCycleController()
        phases = [self.step(c, mem_write=1, reg_write=1) for _ in range(4)]
        mem = phases[MEMORY]
        self.assertEqual(mem["mem_write_en"], 1)
        self.assertEqual(mem["reg_write_en"], 1)

    def test_decoder_signals_are_sampled_in_decode_only(self):
        """Later changes on the decoder ports must not re-steer the FSM."""
        c = MultiCycleController()
        self.step(c)                                   # FETCH
        self.step(c, mem_read=1, reg_write=1)          # DECODE latches load
        # From here the decoder ports go quiet (e.g. the IR changed) — the FSM
        # must still take the MEMORY phase it committed to.
        self.assertEqual(self.step(c)["phase"], EXECUTE)
        mem = self.step(c)
        self.assertEqual(mem["phase"], MEMORY)
        self.assertEqual(mem["mem_read_en"], 1)
        self.assertEqual(mem["reg_write_en"], 1)

    def test_only_fetch_releases_the_pc(self):
        c = MultiCycleController()
        stalls = [self.step(c, mem_read=1, reg_write=1)["fetch_stall"]
                  for _ in range(4)]
        self.assertEqual(stalls, [0, 1, 1, 1])

    def test_branch_resolution_is_enabled_in_execute_only(self):
        c = MultiCycleController()
        flags = [self.step(c, mem_read=1)["branch_resolve_en"] for _ in range(4)]
        self.assertEqual(flags, [0, 0, 1, 0])

    def test_exactly_one_latch_enable_is_hot_per_phase(self):
        c = MultiCycleController()
        for _ in range(4):
            s = self.step(c, mem_read=1, reg_write=1)
            hot = s["ir_latch"] + s["ab_latch"] + s["alu_out_latch"]
            self.assertLessEqual(hot, 1, f"phase {s['phase']}")

    def test_repeats_the_cycle_for_consecutive_instructions(self):
        c = MultiCycleController()
        seen = [self.step(c, reg_write=1)["phase"] for _ in range(9)]
        self.assertEqual(seen, [FETCH, DECODE, EXECUTE] * 3)

    def test_get_state_names_every_phase(self):
        c = MultiCycleController()
        names = []
        for _ in range(4):
            c["dec_mem_read"] = 1
            c.evaluate()
            names.append(c.get_state()["phase_name"])
            c.rising_edge()
        self.assertEqual(names, ["FETCH", "DECODE", "EXECUTE", "MEMORY"])


class TestGates(unittest.TestCase):

    def truth(self, cls):
        g = cls()
        out = {}
        for a in (0, 1):
            for b in (0, 1):
                g["a"], g["b"] = a, b
                g.evaluate()
                out[(a, b)] = g["out"]
        return out

    def test_and_gate(self):
        self.assertEqual(self.truth(GatedSignal),
                         {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1})

    def test_or_gate(self):
        self.assertEqual(self.truth(OrSignal),
                         {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1})

    def test_and_not_gate(self):
        self.assertEqual(self.truth(AndNotSignal),
                         {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 0})

    def test_gates_are_one_bit_and_combinational(self):
        for cls in (GatedSignal, OrSignal, AndNotSignal):
            g = cls()
            for p in ("a", "b", "out"):
                self.assertEqual(g.ports_spec[p].width, 1, f"{cls.__name__}.{p}")
            g["a"] = g["b"] = 1
            g.evaluate()
            before = g["out"]
            g.rising_edge()
            g.evaluate()
            self.assertEqual(g["out"], before)

    def test_inputs_wider_than_one_bit_are_masked_by_the_port(self):
        g = GatedSignal()
        g["a"] = 2      # -> 0
        g["b"] = 1
        g.evaluate()
        self.assertEqual(g["out"], 0)

    def test_outputs_are_normalised_to_0_or_1(self):
        for cls in (GatedSignal, OrSignal, AndNotSignal):
            g = cls()
            for a in (0, 1):
                for b in (0, 1):
                    g["a"], g["b"] = a, b
                    g.evaluate()
                    self.assertIn(g["out"], (0, 1))


class TestLatch(unittest.TestCase):

    def test_starts_at_zero(self):
        l = Latch()
        l.evaluate()
        self.assertEqual(l["data_out"], 0)
        self.assertEqual(l.get_state(), {"value": "0x00000000"})

    def test_captures_only_when_enabled(self):
        l = Latch()
        l["data_in"] = 0xABCD
        l["enable"] = 0
        l.rising_edge()
        l.evaluate()
        self.assertEqual(l["data_out"], 0)
        l["enable"] = 1
        l.rising_edge()
        l.evaluate()
        self.assertEqual(l["data_out"], 0xABCD)

    def test_holds_its_value_after_enable_drops(self):
        l = Latch()
        l["data_in"] = 0xABCD
        l["enable"] = 1
        l.rising_edge()
        l["enable"] = 0
        l["data_in"] = 0xFFFF
        for _ in range(3):
            l.rising_edge()
        l.evaluate()
        self.assertEqual(l["data_out"], 0xABCD)

    def test_output_is_registered_not_transparent(self):
        l = Latch()
        l["data_in"] = 0x1234
        l["enable"] = 1
        l.evaluate()
        self.assertEqual(l["data_out"], 0, "the new value appears only after the edge")
        l.rising_edge()
        l.evaluate()
        self.assertEqual(l["data_out"], 0x1234)

    def test_can_latch_zero_over_a_non_zero_value(self):
        l = Latch()
        l["enable"] = 1
        l["data_in"] = 0x99
        l.rising_edge()
        l["data_in"] = 0
        l.rising_edge()
        l.evaluate()
        self.assertEqual(l["data_out"], 0)

    def test_is_64_bits_wide_for_x86_instructions(self):
        """A 32-bit IR silently truncated 5-6 byte x86 instructions."""
        for p in ("data_in", "data_out"):
            self.assertEqual(Latch.ports_spec[p].width, 64, p)
        l = Latch()
        l["data_in"] = 0xDEAD_BEEF_CAFE_1234
        l["enable"] = 1
        l.rising_edge()
        l.evaluate()
        self.assertEqual(l["data_out"], 0xDEAD_BEEF_CAFE_1234)

    def test_data_in_masks_above_64_bits(self):
        l = Latch()
        l["data_in"] = 1 << 64
        self.assertEqual(l["data_in"], 0)

    def test_get_state_renders_at_least_eight_hex_digits(self):
        l = Latch()
        l["data_in"] = 0xDEAD_BEEF_CAFE
        l["enable"] = 1
        l.rising_edge()
        self.assertEqual(l.get_state()["value"], "0xdeadbeefcafe")

    def test_instances_are_independent(self):
        a, b = Latch(), Latch()
        a["data_in"] = 0x11
        a["enable"] = 1
        a.rising_edge()
        b.evaluate()
        self.assertEqual(b["data_out"], 0)


class TestPCLatch(unittest.TestCase):

    def test_latches_the_pc_and_derives_pc4(self):
        p = PCLatch()
        p["data_in"] = 0x100
        p["enable"] = 1
        p.rising_edge()
        p.evaluate()
        self.assertEqual(p["data_out"], 0x100)
        self.assertEqual(p["pc4_out"], 0x104)

    def test_pc4_wraps_at_2_32(self):
        p = PCLatch()
        p["data_in"] = 0xFFFF_FFFC
        p["enable"] = 1
        p.rising_edge()
        p.evaluate()
        self.assertEqual(p["pc4_out"], 0)

    def test_holds_when_disabled(self):
        p = PCLatch()
        p["data_in"] = 0x40
        p["enable"] = 1
        p.rising_edge()
        p["enable"] = 0
        p["data_in"] = 0x80
        p.rising_edge()
        p.evaluate()
        self.assertEqual(p["data_out"], 0x40)

    def test_pc4_tracks_the_latched_value_not_the_input(self):
        p = PCLatch()
        p["data_in"] = 0x40
        p["enable"] = 1
        p.rising_edge()
        p["data_in"] = 0xF00
        p.evaluate()
        self.assertEqual(p["pc4_out"], 0x44)

    def test_get_state_exposes_a_pc_field_for_the_ui_badge(self):
        p = PCLatch()
        p["data_in"] = 0x2C
        p["enable"] = 1
        p.rising_edge()
        self.assertEqual(p.get_state(), {"pc": "0x0000002c", "value": "0x0000002c"})

    def test_is_a_32_bit_latch(self):
        for name in ("data_in", "data_out", "pc4_out"):
            self.assertEqual(PCLatch.ports_spec[name].width, 32, name)


class TestCycleBudgetController(unittest.TestCase):

    def run_instr(self, bc, cycles):
        """Evaluate/clock `cycles` times, returning (commit, stall) per cycle."""
        out = []
        for _ in range(cycles):
            bc.evaluate()
            out.append((bc["commit"], bc["stall"]))
            bc.rising_edge()
        return out

    def test_default_costs_are_all_one(self):
        self.assertEqual(DEFAULT_COSTS, {"alu": 1, "load": 1, "store": 1, "branch": 1})
        self.assertEqual(CycleBudgetController()._costs, DEFAULT_COSTS)

    def test_all_ones_commits_every_cycle_like_single_cycle(self):
        bc = CycleBudgetController()
        self.assertEqual(self.run_instr(bc, 4), [(1, 0)] * 4)

    def test_a_three_cycle_budget_commits_on_the_last_cycle_only(self):
        bc = CycleBudgetController(costs={"alu": 3})
        self.assertEqual(self.run_instr(bc, 6),
                         [(0, 1), (0, 1), (1, 0)] * 2)

    def test_fetdecexe_profile_matches_the_multicycle_shape(self):
        self.assertEqual(FETDECEXE_COSTS, {"alu": 3, "load": 4, "store": 4, "branch": 3})
        bc = CycleBudgetController(costs=FETDECEXE_COSTS)
        bc["mem_read"] = 1
        self.assertEqual(self.run_instr(bc, 4), [(0, 1), (0, 1), (0, 1), (1, 0)])

    def test_stall_is_the_inverse_of_commit(self):
        bc = CycleBudgetController(costs={"alu": 4})
        for commit, stall in self.run_instr(bc, 9):
            self.assertEqual(commit + stall, 1)

    def test_instruction_classification(self):
        bc = CycleBudgetController()
        self.assertEqual(bc._class(), "alu", "nothing asserted -> alu")
        for port, expected in (("mem_read", "load"), ("mem_write", "store"),
                               ("branch", "branch"), ("jal", "branch"),
                               ("jalr", "branch")):
            bc = CycleBudgetController()
            bc[port] = 1
            self.assertEqual(bc._class(), expected, port)

    def test_load_outranks_store_and_branch_in_classification(self):
        bc = CycleBudgetController()
        bc["mem_read"] = bc["mem_write"] = bc["branch"] = 1
        self.assertEqual(bc._class(), "load")
        bc["mem_read"] = 0
        self.assertEqual(bc._class(), "store")

    def test_each_class_uses_its_own_budget(self):
        costs = {"alu": 2, "load": 5, "store": 3, "branch": 4}
        for port, expected in ((None, 2), ("mem_read", 5), ("mem_write", 3),
                               ("branch", 4), ("jal", 4), ("jalr", 4)):
            bc = CycleBudgetController(costs=costs)
            if port:
                bc[port] = 1
            self.assertEqual(bc._cost(), expected, str(port))

    def test_budgets_are_clamped_to_at_least_one(self):
        bc = CycleBudgetController(costs={"alu": 0, "load": -5, "store": 1})
        self.assertEqual(bc._costs, {"alu": 1, "load": 1, "store": 1, "branch": 1})
        self.assertEqual(self.run_instr(bc, 2), [(1, 0), (1, 0)])

    def test_unknown_cost_keys_are_ignored(self):
        bc = CycleBudgetController(costs={"bogus": 9, "alu": 2})
        self.assertEqual(bc._costs, {"alu": 2, "load": 1, "store": 1, "branch": 1})

    def test_string_costs_are_coerced_to_int(self):
        self.assertEqual(CycleBudgetController(costs={"alu": "3"})._costs["alu"], 3)

    def test_none_costs_falls_back_to_the_defaults(self):
        self.assertEqual(CycleBudgetController(costs=None)._costs, DEFAULT_COSTS)

    def test_the_counter_resets_between_instructions(self):
        bc = CycleBudgetController(costs={"alu": 2})
        self.assertEqual(self.run_instr(bc, 6), [(0, 1), (1, 0)] * 3)

    def test_switching_class_mid_flight_still_terminates(self):
        """The PC is held while stalled, so _cost is stable in practice; even if
        it is not, the elapsed counter must never wedge."""
        bc = CycleBudgetController(costs={"alu": 4, "load": 1})
        bc.evaluate()
        self.assertEqual(bc["commit"], 0)
        bc.rising_edge()
        bc["mem_read"] = 1        # now a 1-cycle class
        bc.evaluate()
        self.assertEqual(bc["commit"], 1, "budget already exceeded -> commit now")
        bc.rising_edge()
        self.assertEqual(bc._elapsed, 0)

    def test_get_state(self):
        bc = CycleBudgetController(costs={"alu": 3})
        bc.evaluate()
        st = bc.get_state()
        self.assertEqual(st["class"], "alu")
        self.assertEqual(st["cycle"], 1)
        self.assertEqual(st["budget"], 3)
        self.assertEqual(st["commit"], 0)
        self.assertEqual(st["costs"], {"alu": 3, "load": 1, "store": 1, "branch": 1})
        bc.rising_edge()
        bc.evaluate()
        self.assertEqual(bc.get_state()["cycle"], 2)

    def test_get_state_costs_is_a_copy(self):
        bc = CycleBudgetController(costs={"alu": 3})
        bc.get_state()["costs"]["alu"] = 99
        self.assertEqual(bc._costs["alu"], 3)

    def test_ports_are_one_bit(self):
        for p in CycleBudgetController.ports_spec.values():
            self.assertEqual(p.width, 1)


class TestTimeModel(unittest.TestCase):

    def test_work_units_are_declared_for_every_class(self):
        self.assertEqual(set(WORK), set(CLASSES))
        self.assertEqual(WORK, {"alu": 5, "load": 6, "store": 5, "branch": 4})

    def test_single_cycle_has_the_longest_clock_period(self):
        """all-1 budgets => the busiest class (load, 6 units) sets the period."""
        self.assertEqual(clock_period(DEFAULT_COSTS), 6.0)
        self.assertEqual(clock_period(None), 6.0, "None == the defaults")
        self.assertEqual(clock_period({}), 6.0)

    def test_more_cycles_per_instruction_shortens_the_clock(self):
        self.assertAlmostEqual(clock_period(FETDECEXE_COSTS), 5 / 3)
        fast = clock_period({"alu": 10, "load": 10, "store": 10, "branch": 10})
        self.assertLess(fast, clock_period(FETDECEXE_COSTS))
        self.assertAlmostEqual(fast, 0.6)

    def test_clock_period_is_set_by_the_busiest_class(self):
        # load is slow (1 cycle for 6 units) even though everything else is deep
        costs = {"alu": 6, "load": 1, "store": 6, "branch": 6}
        self.assertEqual(clock_period(costs), 6.0)

    def test_clock_period_clamps_zero_and_missing_budgets_to_one(self):
        self.assertEqual(clock_period({"alu": 0, "load": 0, "store": 0, "branch": 0}), 6.0)
        self.assertEqual(clock_period({"alu": 3}), 6.0, "missing keys default to 1")

    def test_estimate_time_is_cycles_times_period(self):
        self.assertEqual(estimate_time(10, DEFAULT_COSTS), 60.0)
        self.assertAlmostEqual(estimate_time(30, FETDECEXE_COSTS), 50.0)
        self.assertEqual(estimate_time(0, DEFAULT_COSTS), 0.0)

    def test_cycles_are_not_time(self):
        """The headline lesson: fewer cycles can still mean more total time."""
        single = estimate_time(10, DEFAULT_COSTS)        # 10 cycles, slow clock
        multi = estimate_time(33, FETDECEXE_COSTS)       # 33 cycles, fast clock
        self.assertLess(multi, single)


if __name__ == "__main__":
    unittest.main()
