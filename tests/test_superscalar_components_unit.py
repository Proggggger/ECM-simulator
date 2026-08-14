"""
UNIT tests for the superscalar hardware components (`sim/components/superscalar/`)
and the multi-port data memory (`sim/components/memory/dmem.py`).

WHY THIS FILE EXISTS
--------------------
The N-wide datapath was only ever exercised end-to-end through
`sim/isa/*/presets/superscalar.py` (see tests/test_riscv_superscalar.py). A
full-CPU run is a poor probe of these parts: lane-0-only branch resolution and
group-alignment sensitivity mean an internal path can be wrong while the
architectural end state still matches, and per-lane behaviour (which lane won a
same-word store, which lane forwarded) is simply not observable from registers.
`wide_regfile`, `wide_imem`, `wide_pipeline_regs`, `superscalar_fetch` and
`wide_flags_select` had ZERO direct tests.

Here each component is driven DIRECTLY: construct it with an explicit
``num_lanes``, write input ports, call ``evaluate()`` / ``rising_edge()``, and
assert on output ports and ``get_state()``. Lane counts 1..6 are swept where the
logic is width-dependent.

CLOCKING CONTRACT: mirrors ``CPU.tick()`` — evaluate() for every component,
then rising_edge() for every component. Tests call them in that order.

DOCUMENTED-GAP PINS (deliberately NOT bug reports; see CLAUDE.md
"Known limitations")
------------------------------------------------------------------
* ``test_a_branch_in_lane_zero_squashes_its_whole_shadow`` /
  ``test_a_branch_past_lane_zero_squashes_itself_for_refetch`` — pins the
  lane-0-only branch-resolution policy (conservative: never speculate past a
  branch within a group).
* ``test_partial_squash_never_retreats_past_one_executed_lane`` — pins
  ``squash_from == 0`` being clamped to 1 (a zero-progress refetch would loop
  forever).
* ``TestMemoryLaneMuxIsDeadCode`` — asserts nothing imports the retired
  single-port ``memory_lane_mux``, so the file can be deleted without ceremony
  and cannot silently come back.

Run from repo root:  python3 -m pytest tests/test_superscalar_components_unit.py -q
"""
import ast
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.components.memory.dmem import (STATE_WINDOW, DataMemory,
                                        MultiPortDataMemory)
from sim.components.superscalar.cross_lane_hazard import CrossLaneHazardDetector
from sim.components.superscalar.superscalar_fetch import SuperscalarFetch
from sim.components.superscalar.wide_flags_select import WideFlagsSelect
from sim.components.superscalar.wide_forwarding import WideForwardingUnit
from sim.components.superscalar.wide_imem import WideInstructionMemory
from sim.components.superscalar.wide_pipeline_regs import (WideEX_MEM,
                                                           WideID_EX,
                                                           WideIF_ID,
                                                           WideMEM_WB)
from sim.components.superscalar.wide_regfile import WideRegisterFile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Lane widths swept by the width-parametrized tests. 1 is the degenerate
# scalar case, 6 is past every preset's default and catches off-by-one loops.
LANE_WIDTHS = [1, 2, 3, 4, 5, 6]

RISCV_NOP = 0x00000013


def _cycle(comp):
    """Run one simulated clock on a single component: evaluate then latch."""
    comp.evaluate()
    comp.rising_edge()


# ═══════════════════════════════════════════════════════════════════
# CrossLaneHazardDetector
# ═══════════════════════════════════════════════════════════════════

def _hazard(num_lanes=2, zero_reg_index=0, **inputs):
    """Build a detector, drive the named input ports, and evaluate once."""
    det = CrossLaneHazardDetector(num_lanes=num_lanes, zero_reg_index=zero_reg_index)
    for port, value in inputs.items():
        det[port] = value
    det.evaluate()
    return det


def _lane_valids(det):
    return [det[f"lane_valid_{i}"] for i in range(det.num_lanes)]


class TestCrossLaneHazardDetectorBaseline:

    @pytest.mark.parametrize("n", LANE_WIDTHS)
    def test_independent_lanes_all_execute_at_every_width(self, n):
        det = _hazard(n, **{f"alu_src_{i}": 1 for i in range(n)})
        assert det["stall"] == 0
        assert det["squash_from"] == n
        assert det["partial_squash"] == 0
        assert _lane_valids(det) == [1] * n

    @pytest.mark.parametrize("n", LANE_WIDTHS)
    def test_get_state_reports_one_row_per_lane(self, n):
        det = _hazard(n, **{f"alu_src_{i}": 1 for i in range(n)})
        state = det.get_state()
        assert state["stall"] == "OK"
        assert state["squash_from"] == n
        for i in range(n):
            assert state[f"lane_{i}_valid"] == 1

    def test_two_writers_of_different_registers_do_not_interfere(self):
        det = _hazard(2,
                      rd_0=1, rs1_0=0, alu_src_0=1, reg_write_0=1,
                      rd_1=2, rs1_1=0, alu_src_1=1, reg_write_1=1)
        assert det["squash_from"] == 2
        assert _lane_valids(det) == [1, 1]


class TestCrossLaneHazardDetectorIntraGroupRAW:

    def test_a_younger_lane_reading_an_older_lanes_destination_is_squashed(self):
        det = _hazard(2,
                      rd_0=1, rs1_0=0, alu_src_0=1, reg_write_0=1,
                      rd_1=2, rs1_1=1, alu_src_1=1, reg_write_1=1)
        assert det["stall"] == 0
        assert det["squash_from"] == 1
        assert det["partial_squash"] == 1
        assert _lane_valids(det) == [1, 0]

    def test_squashing_a_lane_squashes_every_later_lane_too(self):
        det = _hazard(4,
                      rd_0=1, alu_src_0=1, reg_write_0=1,
                      rs1_1=1, alu_src_1=1,
                      alu_src_2=1, alu_src_3=1)
        assert det["squash_from"] == 1
        assert _lane_valids(det) == [1, 0, 0, 0]

    def test_the_first_dependent_lane_sets_the_refetch_boundary(self):
        det = _hazard(4,
                      rd_0=9, alu_src_0=1, reg_write_0=1,
                      alu_src_1=1,
                      rs1_2=9, alu_src_2=1,
                      rs1_3=9, alu_src_3=1)
        assert det["squash_from"] == 2
        assert _lane_valids(det) == [1, 1, 0, 0]

    def test_a_later_producers_hazard_cannot_widen_an_earlier_boundary(self):
        """Lane 0->1 already squashes from lane 1; the lane 2->3 dependency sits
        entirely inside the squashed region and must not move the boundary."""
        det = _hazard(4,
                      rd_0=1, alu_src_0=1, reg_write_0=1,
                      rs1_1=1, alu_src_1=1,
                      rd_2=5, alu_src_2=1, reg_write_2=1,
                      rs1_3=5, alu_src_3=1)
        assert det["squash_from"] == 1
        assert _lane_valids(det) == [1, 0, 0, 0]

    def test_a_producer_that_does_not_write_a_register_creates_no_hazard(self):
        det = _hazard(2,
                      rd_0=1, alu_src_0=1, reg_write_0=0,
                      rs1_1=1, alu_src_1=1)
        assert det["squash_from"] == 2

    @pytest.mark.parametrize("zero_reg_index,zero_idx,label", [
        (0, 0, "riscv-x0"),
        (31, 31, "arm-XZR"),
    ])
    def test_writes_to_the_hardwired_zero_register_create_no_hazard(
            self, zero_reg_index, zero_idx, label):
        det = _hazard(2, zero_reg_index=zero_reg_index,
                      rd_0=zero_idx, alu_src_0=1, reg_write_0=1,
                      rs1_1=zero_idx, alu_src_1=1)
        assert det["squash_from"] == 2, label

    def test_x86_has_no_zero_register_so_index_zero_does_create_a_hazard(self):
        det = _hazard(2, zero_reg_index=None,
                      rd_0=0, alu_src_0=1, reg_write_0=1,
                      rs1_1=0, alu_src_1=1)
        assert det["squash_from"] == 1
        assert _lane_valids(det) == [1, 0]


class TestCrossLaneHazardDetectorReadsRs2Gating:
    """PINS THE FIXED "phantom rs2" BUG. For I-type ALU ops, loads, LUI/AUIPC
    and JAL the decoder's rs2 FIELD carries immediate bits (RISC-V) or a stale
    0 (ARM/x86), not a real source register. Treating it as one squashed
    independent lanes and made wider configurations SLOWER on hazard-free code.
    rs2 is a genuine read only for register-ALU ops (alu_src==0, not JAL/JALR),
    stores (rs2 == store data) and branches (rs2 == compare RHS)."""

    def test_an_immediate_form_rs2_field_does_not_trigger_a_raw_squash(self):
        det = _hazard(2,
                      rd_0=1, alu_src_0=1, reg_write_0=1,
                      rd_1=2, rs1_1=0, rs2_1=1, alu_src_1=1, reg_write_1=1)
        assert det["squash_from"] == 2, "alu_src=1 means rs2 holds immediate bits"
        assert _lane_valids(det) == [1, 1]

    def test_a_register_form_rs2_does_trigger_a_raw_squash(self):
        det = _hazard(2,
                      rd_0=1, alu_src_0=1, reg_write_0=1,
                      rd_1=2, rs1_1=0, rs2_1=1, alu_src_1=0, reg_write_1=1)
        assert det["squash_from"] == 1
        assert _lane_valids(det) == [1, 0]

    def test_a_stores_rs2_counts_as_a_real_read_despite_alu_src(self):
        """A store computes its address with an immediate (alu_src=1) but rs2
        is genuinely the data register."""
        det = _hazard(2,
                      rd_0=1, alu_src_0=1, reg_write_0=1,
                      rs1_1=0, rs2_1=1, alu_src_1=1, mem_write_1=1)
        assert det["squash_from"] == 1

    def test_a_branchs_rs2_counts_as_a_real_read(self):
        det = _hazard(2,
                      rd_0=1, alu_src_0=1, reg_write_0=1,
                      rs1_1=0, rs2_1=1, alu_src_1=1, branch_1=1)
        assert det["squash_from"] == 1

    @pytest.mark.parametrize("jump_port", ["jal_0", "jalr_0"])
    def test_a_jumps_rs2_field_is_not_a_register_read(self, jump_port):
        """JAL/JALR are excluded from the alu_src==0 register-read rule (their
        rs2 field holds encoding bits). Probed via the load-use path, which is
        independent of the control-shadow squash rule."""
        common = dict(alu_src_0=0, rs1_0=1, rs2_0=7, alu_src_1=1,
                      id_ex_mem_read_0=1, id_ex_rd_0=7)
        assert _hazard(2, **common)["stall"] == 1, \
            "sanity: for a non-jump, alu_src=0 rs2 IS a register read"
        assert _hazard(2, **common, **{jump_port: 1})["stall"] == 0

    @pytest.mark.parametrize("jump_port", ["jal_0", "jalr_0"])
    def test_a_jumps_rs1_is_still_a_real_register_read(self, jump_port):
        """Only rs2 is excluded — JALR's rs1 is the genuine target base."""
        det = _hazard(2, alu_src_0=0, rs1_0=7, alu_src_1=1,
                      id_ex_mem_read_0=1, id_ex_rd_0=7, **{jump_port: 1})
        assert det["stall"] == 1


class TestCrossLaneHazardDetectorLoadUseStall:

    def test_a_load_in_id_ex_feeding_any_current_lane_stalls_the_whole_group(self):
        det = _hazard(2, rs1_0=7, alu_src_0=1, id_ex_mem_read_0=1, id_ex_rd_0=7)
        assert det["stall"] == 1
        assert _lane_valids(det) == [0, 0], "a stall squashes ALL lanes"
        assert det["squash_from"] == det.num_lanes
        assert det["partial_squash"] == 0, "a stall is not a partial squash"
        assert det.get_state()["stall"] == "STALL"

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_a_load_in_any_ex_lane_can_stall_any_current_lane(self, n):
        for ex_lane in range(n):
            for cur_lane in range(n):
                det = _hazard(n,
                              **{f"alu_src_{i}": 1 for i in range(n)},
                              **{f"rs1_{cur_lane}": 12,
                                 f"id_ex_mem_read_{ex_lane}": 1,
                                 f"id_ex_rd_{ex_lane}": 12})
                assert det["stall"] == 1, f"ex_lane={ex_lane} cur_lane={cur_lane}"

    def test_a_load_whose_destination_nobody_reads_does_not_stall(self):
        det = _hazard(2, rs1_0=1, rs1_1=2, alu_src_0=1, alu_src_1=1,
                      id_ex_mem_read_0=1, id_ex_rd_0=7)
        assert det["stall"] == 0

    def test_the_load_use_check_also_respects_reads_rs2(self):
        det = _hazard(2, rs2_0=7, alu_src_0=1,     # immediate form -> not a read
                      id_ex_mem_read_0=1, id_ex_rd_0=7)
        assert det["stall"] == 0

        det = _hazard(2, rs2_0=7, alu_src_0=0,     # register form -> a real read
                      id_ex_mem_read_0=1, id_ex_rd_0=7)
        assert det["stall"] == 1

    @pytest.mark.parametrize("zero_reg_index,zero_idx,label", [
        (0, 0, "riscv-x0"),
        (31, 31, "arm-XZR"),
    ])
    def test_a_load_into_the_hardwired_zero_register_never_stalls(
            self, zero_reg_index, zero_idx, label):
        det = _hazard(2, zero_reg_index=zero_reg_index,
                      rs1_0=zero_idx, alu_src_0=1,
                      id_ex_mem_read_0=1, id_ex_rd_0=zero_idx)
        assert det["stall"] == 0, label

    def test_x86_load_into_index_zero_does_stall_because_eax_is_real(self):
        """Contrast with the narrow HazardDetector, which hardcodes rd != 0 —
        the WIDE detector is correctly parameterized."""
        det = _hazard(2, zero_reg_index=None,
                      rs1_0=0, alu_src_0=1,
                      id_ex_mem_read_0=1, id_ex_rd_0=0)
        assert det["stall"] == 1

    def test_a_stall_outranks_an_intra_group_raw_squash(self):
        det = _hazard(2,
                      rd_0=1, alu_src_0=1, reg_write_0=1,
                      rs1_1=1, alu_src_1=1,
                      rs1_0=7, id_ex_mem_read_0=1, id_ex_rd_0=7)
        assert det["stall"] == 1
        assert det["partial_squash"] == 0
        assert det["squash_from"] == 2


class TestCrossLaneHazardDetectorControlHazard:
    """Pins the DOCUMENTED lane-0-only branch-resolution policy: a control
    transfer must be the LAST instruction that executes in its group."""

    @pytest.mark.parametrize("control_port", ["branch_0", "jal_0", "jalr_0"])
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_a_branch_in_lane_zero_squashes_its_whole_shadow(self, control_port, n):
        det = _hazard(n, **{f"alu_src_{i}": 1 for i in range(n)},
                      **{control_port: 1})
        assert det["squash_from"] == 1, "the branch executes in lane 0 and resolves"
        assert _lane_valids(det) == [1] + [0] * (n - 1)
        assert det["partial_squash"] == (1 if n > 1 else 0)

    @pytest.mark.parametrize("control_port", ["branch_2", "jal_2", "jalr_2"])
    def test_a_branch_past_lane_zero_squashes_itself_for_refetch(self, control_port):
        """The branch itself is squashed so it restarts in lane 0 next cycle,
        where lane-0-only resolution can see it."""
        det = _hazard(4, **{f"alu_src_{i}": 1 for i in range(4)},
                      **{control_port: 1})
        assert det["squash_from"] == 2
        assert _lane_valids(det) == [1, 1, 0, 0]

    def test_a_control_transfer_in_lane_one_still_leaves_lane_zero_running(self):
        det = _hazard(3, **{f"alu_src_{i}": 1 for i in range(3)}, jal_1=1)
        assert det["squash_from"] == 1
        assert _lane_valids(det) == [1, 0, 0]

    def test_only_the_first_control_transfer_in_the_group_matters(self):
        det = _hazard(4, **{f"alu_src_{i}": 1 for i in range(4)},
                      branch_1=1, jal_3=1)
        assert det["squash_from"] == 1

    def test_a_raw_squash_earlier_than_the_branch_wins(self):
        det = _hazard(4,
                      rd_0=5, alu_src_0=1, reg_write_0=1,
                      rs1_1=5, alu_src_1=1,
                      alu_src_2=1, branch_2=1,
                      alu_src_3=1)
        assert det["squash_from"] == 1, "min() of the two boundaries"

    def test_a_single_lane_machine_never_partially_squashes_a_branch(self):
        det = _hazard(1, alu_src_0=1, branch_0=1)
        assert det["squash_from"] == 1
        assert det["partial_squash"] == 0
        assert _lane_valids(det) == [1]


# ═══════════════════════════════════════════════════════════════════
# WideForwardingUnit
# ═══════════════════════════════════════════════════════════════════

class TestWideForwardingUnit:

    def _unit(self, num_lanes=3, **kw):
        fu = WideForwardingUnit(num_lanes=num_lanes, **kw)
        for i in range(num_lanes):
            fu[f"rs1_data_in_{i}"] = 0x100 + i
            fu[f"rs2_data_in_{i}"] = 0x200 + i
        return fu

    def test_no_producer_leaves_every_lane_on_its_register_file_value(self):
        fu = self._unit()
        for i in range(3):
            fu[f"id_ex_rs1_{i}"] = 5 + i
        fu.evaluate()
        for i in range(3):
            assert fu[f"rs1_data_out_{i}"] == 0x100 + i
            assert fu[f"rs2_data_out_{i}"] == 0x200 + i

    def test_a_producer_in_any_ex_mem_lane_forwards_to_any_consumer_lane(self):
        for producer in range(3):
            for consumer in range(3):
                fu = self._unit()
                fu[f"id_ex_rs1_{consumer}"] = 5
                fu[f"ex_mem_rd_{producer}"] = 5
                fu[f"ex_mem_reg_write_{producer}"] = 1
                fu[f"ex_mem_alu_result_{producer}"] = 0xE0 + producer
                fu.evaluate()
                assert fu[f"rs1_data_out_{consumer}"] == 0xE0 + producer

    def test_the_youngest_ex_mem_writer_of_a_register_wins(self):
        """PINS THE FIXED BUG: this used to return the OLDEST matching lane.
        Within a group lane order IS program order, so the highest-index writer
        holds the architecturally-correct value."""
        fu = self._unit()
        fu["id_ex_rs1_0"] = 5
        fu["ex_mem_rd_1"] = 5
        fu["ex_mem_reg_write_1"] = 1
        fu["ex_mem_alu_result_1"] = 0xE1
        fu["ex_mem_rd_2"] = 5
        fu["ex_mem_reg_write_2"] = 1
        fu["ex_mem_alu_result_2"] = 0xE2
        fu.evaluate()
        assert fu["rs1_data_out_0"] == 0xE2

    def test_the_youngest_mem_wb_writer_of_a_register_wins(self):
        fu = self._unit()
        fu["id_ex_rs1_0"] = 5
        fu["mem_wb_rd_0"] = 5
        fu["mem_wb_reg_write_0"] = 1
        fu["mem_wb_data_0"] = 0xF0
        fu["mem_wb_rd_2"] = 5
        fu["mem_wb_reg_write_2"] = 1
        fu["mem_wb_data_2"] = 0xF2
        fu.evaluate()
        assert fu["rs1_data_out_0"] == 0xF2

    def test_ex_mem_beats_mem_wb_even_from_an_older_lane(self):
        fu = self._unit()
        fu["id_ex_rs1_0"] = 5
        fu["ex_mem_rd_0"] = 5
        fu["ex_mem_reg_write_0"] = 1
        fu["ex_mem_alu_result_0"] = 0xE0
        fu["mem_wb_rd_2"] = 5
        fu["mem_wb_reg_write_2"] = 1
        fu["mem_wb_data_2"] = 0xF2
        fu.evaluate()
        assert fu["rs1_data_out_0"] == 0xE0, "the nearer stage holds the newer value"

    def test_reg_write_low_never_forwards(self):
        fu = self._unit()
        fu["id_ex_rs1_0"] = 5
        fu["ex_mem_rd_1"] = 5
        fu["ex_mem_reg_write_1"] = 0
        fu["ex_mem_alu_result_1"] = 0xE1
        fu["mem_wb_rd_1"] = 5
        fu["mem_wb_reg_write_1"] = 0
        fu["mem_wb_data_1"] = 0xF1
        fu.evaluate()
        assert fu["rs1_data_out_0"] == 0x100

    def test_rs1_and_rs2_forward_independently(self):
        fu = self._unit()
        fu["id_ex_rs1_1"] = 5
        fu["id_ex_rs2_1"] = 6
        fu["ex_mem_rd_0"] = 5
        fu["ex_mem_reg_write_0"] = 1
        fu["ex_mem_alu_result_0"] = 0xAA
        fu["mem_wb_rd_0"] = 6
        fu["mem_wb_reg_write_0"] = 1
        fu["mem_wb_data_0"] = 0xBB
        fu.evaluate()
        assert fu["rs1_data_out_1"] == 0xAA
        assert fu["rs2_data_out_1"] == 0xBB

    @pytest.mark.parametrize("zero_reg_index,zero_idx,real_idx,label", [
        (0, 0, 31, "riscv: x0 hardwired, x31 real"),
        (31, 31, 0, "arm: XZR=31 hardwired, x0 real"),
    ])
    def test_the_hardwired_zero_register_is_never_forwarded(
            self, zero_reg_index, zero_idx, real_idx, label):
        fu = WideForwardingUnit(num_lanes=2, zero_reg_index=zero_reg_index)
        fu["rs1_data_in_0"] = 0
        fu["id_ex_rs1_0"] = zero_idx
        fu["ex_mem_rd_0"] = zero_idx
        fu["ex_mem_reg_write_0"] = 1
        fu["ex_mem_alu_result_0"] = 0x77
        fu.evaluate()
        assert fu["rs1_data_out_0"] == 0, label

        fu["id_ex_rs1_0"] = real_idx
        fu["ex_mem_rd_0"] = real_idx
        fu.evaluate()
        assert fu["rs1_data_out_0"] == 0x77, label

    def test_x86_forwards_index_zero_because_eax_is_a_real_register(self):
        fu = WideForwardingUnit(num_lanes=2, zero_reg_index=None)
        fu["rs1_data_in_0"] = 0
        fu["id_ex_rs1_0"] = 0
        fu["ex_mem_rd_0"] = 0
        fu["ex_mem_reg_write_0"] = 1
        fu["ex_mem_alu_result_0"] = 0x77
        fu.evaluate()
        assert fu["rs1_data_out_0"] == 0x77

    @pytest.mark.parametrize("n", LANE_WIDTHS)
    def test_every_lane_has_a_full_port_set_at_every_width(self, n):
        fu = WideForwardingUnit(num_lanes=n)
        for i in range(n):
            for port in (f"id_ex_rs1_{i}", f"id_ex_rs2_{i}",
                         f"ex_mem_rd_{i}", f"mem_wb_rd_{i}",
                         f"rs1_data_out_{i}", f"rs2_data_out_{i}"):
                assert port in fu.ports_spec
        assert f"rs1_data_out_{n}" not in fu.ports_spec
        assert fu.get_state() == {f"lane_{i}": "active" for i in range(n)}


# ═══════════════════════════════════════════════════════════════════
# WideRegisterFile
# ═══════════════════════════════════════════════════════════════════

class TestWideRegisterFile:

    def test_reads_are_zero_until_something_is_written(self):
        rf = WideRegisterFile(num_lanes=2)
        rf["rs1_addr_0"] = 3
        rf.evaluate()
        assert rf["rs1_data_0"] == 0
        assert rf.get_state()["registers"] == [0] * 32

    def test_each_lane_writes_its_own_register(self):
        rf = WideRegisterFile(num_lanes=3)
        for i in range(3):
            rf[f"rd_addr_{i}"] = 5 + i
            rf[f"rd_data_{i}"] = 100 + i
            rf[f"wen_{i}"] = 1
        rf.rising_edge()
        for i in range(3):
            rf[f"wen_{i}"] = 0
            rf[f"rs1_addr_{i}"] = 5 + i
        rf.evaluate()
        for i in range(3):
            assert rf[f"rs1_data_{i}"] == 100 + i

    def test_each_lane_has_two_independent_read_ports(self):
        rf = WideRegisterFile(num_lanes=2)
        rf.regs[4] = 44
        rf.regs[9] = 99
        rf["rs1_addr_1"] = 4
        rf["rs2_addr_1"] = 9
        rf.evaluate()
        assert (rf["rs1_data_1"], rf["rs2_data_1"]) == (44, 99)

    def test_a_same_cycle_write_is_visible_on_the_read_ports(self):
        """Write-through: a writeback coinciding with a younger instruction's
        ID read (a RAW across fetch groups) must return the NEW value."""
        rf = WideRegisterFile(num_lanes=2)
        rf["rd_addr_0"] = 1
        rf["rd_data_0"] = 11
        rf["wen_0"] = 1
        rf["rd_addr_1"] = 2
        rf["rd_data_1"] = 22
        rf["wen_1"] = 1
        rf["rs1_addr_0"] = 1
        rf["rs2_addr_0"] = 2
        rf.evaluate()
        assert (rf["rs1_data_0"], rf["rs2_data_0"]) == (11, 22)

    def test_when_two_lanes_write_the_same_register_the_youngest_wins(self):
        rf = WideRegisterFile(num_lanes=3)
        for i, value in enumerate((5, 6, 7)):
            rf[f"rd_addr_{i}"] = 4
            rf[f"rd_data_{i}"] = value
            rf[f"wen_{i}"] = 1
        rf["rs1_addr_0"] = 4
        rf.evaluate()
        assert rf["rs1_data_0"] == 7, "write-through must agree with the edge"
        rf.rising_edge()
        assert rf.regs[4] == 7

    def test_write_enable_low_writes_nothing(self):
        rf = WideRegisterFile(num_lanes=2)
        rf["rd_addr_0"] = 3
        rf["rd_data_0"] = 42
        rf["wen_0"] = 0
        rf["rs1_addr_0"] = 3
        rf.evaluate()
        assert rf["rs1_data_0"] == 0
        rf.rising_edge()
        assert rf.regs[3] == 0

    @pytest.mark.parametrize("zero_idx,real_idx,label", [
        (0, 31, "riscv-x0"),
        (31, 0, "arm-XZR"),
    ])
    def test_the_hardwired_zero_register_reads_zero_and_swallows_writes(
            self, zero_idx, real_idx, label):
        rf = WideRegisterFile(num_lanes=2, zero_reg=True, zero_reg_index=zero_idx)
        rf["rd_addr_0"] = zero_idx
        rf["rd_data_0"] = 99
        rf["wen_0"] = 1
        rf["rd_addr_1"] = real_idx
        rf["rd_data_1"] = 42
        rf["wen_1"] = 1
        rf["rs1_addr_0"] = zero_idx
        rf["rs2_addr_0"] = real_idx
        rf.evaluate()
        assert rf["rs1_data_0"] == 0, label
        assert rf["rs2_data_0"] == 42, f"{label}: the other index is a real GPR"
        rf.rising_edge()
        assert rf.regs[zero_idx] == 0
        assert rf.regs[real_idx] == 42

    def test_x86_regfile_has_eight_real_registers_including_index_zero(self):
        rf = WideRegisterFile(num_regs=8, zero_reg=False, num_lanes=2)
        assert len(rf.regs) == 8
        rf["rd_addr_0"] = 0
        rf["rd_data_0"] = 88
        rf["wen_0"] = 1
        rf["rs1_addr_0"] = 0
        rf.evaluate()
        assert rf["rs1_data_0"] == 88
        rf.rising_edge()
        assert rf.regs[0] == 88

    def test_addresses_wrap_at_the_register_count(self):
        rf = WideRegisterFile(num_regs=8, zero_reg=False, num_lanes=1)
        rf.regs[0] = 77
        rf["rs1_addr_0"] = 8       # wraps to 0
        rf.evaluate()
        assert rf["rs1_data_0"] == 77

    @pytest.mark.parametrize("n", LANE_WIDTHS)
    def test_port_set_scales_with_lane_count(self, n):
        rf = WideRegisterFile(num_lanes=n)
        for i in range(n):
            for port in (f"rs1_addr_{i}", f"rs2_addr_{i}", f"rs1_data_{i}",
                         f"rs2_data_{i}", f"rd_addr_{i}", f"rd_data_{i}", f"wen_{i}"):
                assert port in rf.ports_spec
        assert f"wen_{n}" not in rf.ports_spec


# ═══════════════════════════════════════════════════════════════════
# WideInstructionMemory
# ═══════════════════════════════════════════════════════════════════

class TestWideInstructionMemory:

    def test_n_consecutive_words_are_fetched_from_the_base_address(self):
        imem = WideInstructionMemory([0x11, 0x22, 0x33, 0x44], num_lanes=3)
        imem["addr"] = 0
        imem.evaluate()
        assert [imem[f"data_{i}"] for i in range(3)] == [0x11, 0x22, 0x33]

    def test_the_base_address_is_a_byte_address(self):
        imem = WideInstructionMemory([0x11, 0x22, 0x33, 0x44], num_lanes=2)
        imem["addr"] = 8       # word 2
        imem.evaluate()
        assert [imem[f"data_{i}"] for i in range(2)] == [0x33, 0x44]

    def test_lanes_past_the_end_of_the_program_read_nop(self):
        imem = WideInstructionMemory([0x11, 0x22, 0x33], num_lanes=3)
        imem["addr"] = 8
        imem.evaluate()
        assert [imem[f"data_{i}"] for i in range(3)] == [0x33, RISCV_NOP, RISCV_NOP]

    def test_a_fully_out_of_range_group_reads_all_nops(self):
        imem = WideInstructionMemory([0x11], num_lanes=2)
        imem["addr"] = 100
        imem.evaluate()
        assert [imem[f"data_{i}"] for i in range(2)] == [RISCV_NOP, RISCV_NOP]

    def test_program_words_are_masked_to_32_bits(self):
        imem = WideInstructionMemory([0x1_FFFF_FFFF], num_lanes=1)
        imem["addr"] = 0
        imem.evaluate()
        assert imem["data_0"] == 0xFFFF_FFFF

    def test_get_state_reports_the_address_and_every_lane(self):
        imem = WideInstructionMemory([0xABCD], num_lanes=2)
        imem["addr"] = 0
        imem.evaluate()
        assert imem.get_state() == {
            "addr": "0x00000000",
            "data_0": "0x0000abcd",
            "data_1": f"0x{RISCV_NOP:08x}",
        }

    @pytest.mark.parametrize("n", LANE_WIDTHS)
    def test_lane_count_determines_the_fetch_group_size(self, n):
        program = list(range(1, 17))
        imem = WideInstructionMemory(program, num_lanes=n)
        imem["addr"] = 4       # word 1
        imem.evaluate()
        assert [imem[f"data_{i}"] for i in range(n)] == program[1:1 + n]


# ═══════════════════════════════════════════════════════════════════
# SuperscalarFetch
# ═══════════════════════════════════════════════════════════════════

class TestSuperscalarFetch:

    @pytest.mark.parametrize("n", LANE_WIDTHS)
    def test_lane_pcs_and_link_values_are_derived_from_the_base_pc(self, n):
        fetch = SuperscalarFetch(num_lanes=n, pc_reset=0x40)
        fetch.evaluate()
        assert fetch["pc_out"] == 0x40
        for i in range(n):
            assert fetch[f"pc_lane_{i}"] == 0x40 + i * 4
            assert fetch[f"pc4_lane_{i}"] == 0x40 + i * 4 + 4
        assert fetch["pc_next_group"] == 0x40 + n * 4

    @pytest.mark.parametrize("n", LANE_WIDTHS)
    def test_the_pc_advances_by_one_whole_group_each_cycle(self, n):
        fetch = SuperscalarFetch(num_lanes=n)
        for step in range(4):
            fetch.evaluate()
            assert fetch["pc_out"] == step * n * 4
            fetch.rising_edge()

    def test_stall_holds_the_pc(self):
        fetch = SuperscalarFetch(num_lanes=2)
        _cycle(fetch)
        fetch["stall"] = 1
        for _ in range(3):
            _cycle(fetch)
        fetch.evaluate()
        assert fetch["pc_out"] == 8

    def test_a_resolved_branch_redirects_the_pc(self):
        fetch = SuperscalarFetch(num_lanes=2)
        fetch["branch_taken"] = 1
        fetch["next_pc"] = 0x100
        _cycle(fetch)
        fetch.evaluate()
        assert fetch["pc_out"] == 0x100

    def test_a_prediction_redirects_the_pc(self):
        fetch = SuperscalarFetch(num_lanes=2)
        fetch["predict_taken"] = 1
        fetch["predict_target"] = 0x200
        _cycle(fetch)
        fetch.evaluate()
        assert fetch["pc_out"] == 0x200

    def test_a_stall_outranks_a_prediction(self):
        """Otherwise the held group would be dropped (see SimpleFetch)."""
        fetch = SuperscalarFetch(num_lanes=2)
        fetch["stall"] = 1
        fetch["predict_taken"] = 1
        fetch["predict_target"] = 0x300
        _cycle(fetch)
        fetch.evaluate()
        assert fetch["pc_out"] == 0

    def test_a_resolved_branch_outranks_a_stall(self):
        fetch = SuperscalarFetch(num_lanes=2)
        fetch["stall"] = 1
        fetch["branch_taken"] = 1
        fetch["next_pc"] = 0x400
        _cycle(fetch)
        fetch.evaluate()
        assert fetch["pc_out"] == 0x400

    def test_a_stall_outranks_a_partial_squash(self):
        fetch = SuperscalarFetch(num_lanes=4, pc_reset=0x40)
        fetch["stall"] = 1
        fetch["partial_squash"] = 1
        fetch["squash_from"] = 1
        _cycle(fetch)
        fetch.evaluate()
        assert fetch["pc_out"] == 0x40

    @pytest.mark.parametrize("executed,expected", [
        (1, 0x40 - 3 * 4),
        (2, 0x40 - 2 * 4),
        (3, 0x40 - 1 * 4),
        (4, 0x40),
    ])
    def test_a_partial_squash_retreats_by_the_lanes_that_did_not_execute(
            self, executed, expected):
        """The group in decode already advanced the PC by a FULL group, so the
        refetch retreats by (num_lanes - executed) * 4."""
        fetch = SuperscalarFetch(num_lanes=4, pc_reset=0x40)
        fetch["partial_squash"] = 1
        fetch["squash_from"] = executed
        _cycle(fetch)
        fetch.evaluate()
        assert fetch["pc_out"] == expected

    def test_partial_squash_never_retreats_past_one_executed_lane(self):
        """squash_from == 0 is clamped to 1: a zero-progress refetch would spin."""
        fetch = SuperscalarFetch(num_lanes=4, pc_reset=0x40)
        fetch["partial_squash"] = 1
        fetch["squash_from"] = 0
        _cycle(fetch)
        fetch.evaluate()
        assert fetch["pc_out"] == 0x40 - 3 * 4

    def test_the_pc_wraps_at_32_bits(self):
        fetch = SuperscalarFetch(num_lanes=2, pc_reset=0xFFFF_FFF8)
        _cycle(fetch)
        fetch.evaluate()
        assert fetch["pc_out"] == 0

    def test_get_state_reports_the_pc_and_every_lane_pc(self):
        fetch = SuperscalarFetch(num_lanes=3, pc_reset=0x20)
        fetch.evaluate()
        assert fetch.get_state() == {
            "pc": "0x00000020",
            "pc_lane_0": "0x00000020",
            "pc_lane_1": "0x00000024",
            "pc_lane_2": "0x00000028",
        }


# ═══════════════════════════════════════════════════════════════════
# Wide pipeline registers
# ═══════════════════════════════════════════════════════════════════

class TestWideIF_ID:

    def _reg(self, num_lanes=2, **kw):
        reg = WideIF_ID(num_lanes=num_lanes, **kw)
        for i in range(num_lanes):
            reg[f"pc_in_{i}"] = 0x10 + i * 4
            reg[f"pc4_in_{i}"] = 0x14 + i * 4
            reg[f"instr_in_{i}"] = 0xAAA0 + i
            reg[f"predicted_taken_in_{i}"] = i % 2
        return reg

    def test_reset_lanes_are_nop_bubbles(self):
        reg = WideIF_ID(num_lanes=2)
        reg.evaluate()
        for i in range(2):
            assert reg[f"instr_out_{i}"] == RISCV_NOP
            assert reg[f"valid_{i}"] == 0
        assert reg.get_state()["lane_0"]["valid"] == "BUBBLE"

    def test_a_custom_nop_encoding_is_used_for_bubbles(self):
        reg = WideIF_ID(num_lanes=1, nop_encoding=0xD503201F)   # ARM NOP
        reg.evaluate()
        assert reg["instr_out_0"] == 0xD503201F

    def test_every_lane_latches_pc_pc4_instruction_and_prediction(self):
        reg = self._reg(num_lanes=2)
        reg.rising_edge()
        reg.evaluate()
        for i in range(2):
            assert reg[f"pc_out_{i}"] == 0x10 + i * 4
            assert reg[f"pc4_out_{i}"] == 0x14 + i * 4
            assert reg[f"instr_out_{i}"] == 0xAAA0 + i
            assert reg[f"valid_{i}"] == 1
            assert reg[f"predicted_taken_out_{i}"] == i % 2
        assert reg.get_state()["lane_1"] == {
            "pc": "0x00000014", "instr": "0x0000aaa1", "valid": "VALID"}

    def test_stall_holds_every_lane(self):
        reg = self._reg()
        reg.rising_edge()
        reg["stall"] = 1
        for i in range(2):
            reg[f"instr_in_{i}"] = 0xBBB0 + i
        reg.rising_edge()
        reg.evaluate()
        assert [reg[f"instr_out_{i}"] for i in range(2)] == [0xAAA0, 0xAAA1]

    def test_predict_flush_bubbles_every_lane(self):
        reg = self._reg()
        reg.rising_edge()
        reg["predict_flush"] = 1
        reg.rising_edge()
        reg.evaluate()
        for i in range(2):
            assert reg[f"valid_{i}"] == 0
            assert reg[f"instr_out_{i}"] == RISCV_NOP
            assert reg[f"predicted_taken_out_{i}"] == 0

    def test_flush_bubbles_every_lane(self):
        reg = self._reg()
        reg.rising_edge()
        reg["flush"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert [reg[f"valid_{i}"] for i in range(2)] == [0, 0]

    def test_a_partial_squash_discards_the_stale_fetch_in_flight(self):
        """The PC is being corrected, so whatever IF holds is from the wrong
        group and must not enter decode."""
        reg = self._reg()
        reg.rising_edge()
        reg["partial_squash"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert [reg[f"valid_{i}"] for i in range(2)] == [0, 0]
        assert reg["instr_out_0"] == RISCV_NOP

    def test_a_partial_squash_outranks_a_stall(self):
        reg = self._reg()
        reg.rising_edge()
        reg["stall"] = 1
        reg["partial_squash"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["valid_0"] == 0

    @pytest.mark.parametrize("n", LANE_WIDTHS)
    def test_port_set_and_latch_scale_with_lane_count(self, n):
        reg = self._reg(num_lanes=n)
        reg.rising_edge()
        reg.evaluate()
        assert len(reg.get_state()) == n
        for i in range(n):
            assert reg[f"instr_out_{i}"] == 0xAAA0 + i
        assert f"instr_out_{n}" not in reg.ports_spec


class TestWideID_EX:

    def _reg(self, num_lanes=2):
        reg = WideID_EX(num_lanes=num_lanes)
        for i in range(num_lanes):
            reg[f"rd_in_{i}"] = 3 + i
            reg[f"alu_op_in_{i}"] = 2
            reg[f"pc_in_{i}"] = 0x20 + i * 4
            reg[f"lane_valid_in_{i}"] = 1
            reg[f"valid_in_{i}"] = 1
        return reg

    def test_reset_lanes_are_bubbles(self):
        reg = WideID_EX(num_lanes=2)
        reg.evaluate()
        assert [reg[f"valid_{i}"] for i in range(2)] == [0, 0]

    def test_every_lane_latches_its_own_context(self):
        reg = self._reg()
        reg.rising_edge()
        reg.evaluate()
        assert [reg[f"rd_out_{i}"] for i in range(2)] == [3, 4]
        assert [reg[f"valid_{i}"] for i in range(2)] == [1, 1]
        assert reg.get_state()["lane_0"] == {
            "pc": "0x00000020", "rd": 3, "alu_op": 2, "valid": "VALID"}

    def test_a_squashed_lane_becomes_a_bubble_while_its_peers_advance(self):
        """lane_valid_in comes from the cross-lane hazard detector."""
        reg = self._reg()
        reg["lane_valid_in_1"] = 0
        reg.rising_edge()
        reg.evaluate()
        assert (reg["rd_out_0"], reg["valid_0"]) == (3, 1)
        assert (reg["rd_out_1"], reg["valid_1"]) == (0, 0)

    def test_an_upstream_invalid_lane_becomes_a_bubble(self):
        reg = self._reg()
        reg["valid_in_0"] = 0
        reg.rising_edge()
        reg.evaluate()
        assert reg["valid_0"] == 0
        assert reg["rd_out_0"] == 0, "the whole context is cleared, not just valid"
        assert reg["valid_1"] == 1

    def test_stall_bubbles_every_lane_rather_than_holding(self):
        reg = self._reg()
        reg.rising_edge()
        reg["stall"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert [reg[f"valid_{i}"] for i in range(2)] == [0, 0]
        assert [reg[f"rd_out_{i}"] for i in range(2)] == [0, 0]

    def test_flush_bubbles_every_lane(self):
        reg = self._reg()
        reg.rising_edge()
        reg["flush"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert [reg[f"valid_{i}"] for i in range(2)] == [0, 0]

    @pytest.mark.parametrize("n", LANE_WIDTHS)
    def test_all_declared_fields_round_trip_for_every_lane(self, n):
        from sim.components.superscalar.wide_pipeline_regs import _ID_EX_FIELDS
        reg = WideID_EX(num_lanes=n)
        for i in range(n):
            for field in _ID_EX_FIELDS:
                reg[f"{field}_in_{i}"] = 1
            reg[f"lane_valid_in_{i}"] = 1
            reg[f"valid_in_{i}"] = 1
        reg.rising_edge()
        reg.evaluate()
        for i in range(n):
            for field in _ID_EX_FIELDS:
                assert reg[f"{field}_out_{i}"] == 1, f"lane {i} field {field}"


class TestWideEX_MEM:

    def _reg(self, num_lanes=2):
        reg = WideEX_MEM(num_lanes=num_lanes)
        for i in range(num_lanes):
            reg[f"alu_result_in_{i}"] = 0x50 + i
            reg[f"rd_in_{i}"] = 6 + i
            reg[f"pc_in_{i}"] = 0x60 + i * 4
            reg[f"valid_in_{i}"] = 1
        return reg

    def test_every_lane_latches_its_own_result(self):
        reg = self._reg()
        reg.rising_edge()
        reg.evaluate()
        assert [reg[f"alu_result_out_{i}"] for i in range(2)] == [0x50, 0x51]
        assert [reg[f"valid_{i}"] for i in range(2)] == [1, 1]
        assert reg.get_state()["lane_1"] == {
            "pc": "0x00000064", "rd": 7, "alu_result": 0x51, "valid": "VALID"}

    def test_valid_propagates_per_lane_from_id_ex(self):
        reg = self._reg()
        reg["valid_in_1"] = 0
        reg.rising_edge()
        reg.evaluate()
        assert [reg[f"valid_{i}"] for i in range(2)] == [1, 0]

    def test_flush_clears_every_lane(self):
        reg = self._reg()
        reg.rising_edge()
        reg["flush"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert [reg[f"valid_{i}"] for i in range(2)] == [0, 0]
        assert [reg[f"alu_result_out_{i}"] for i in range(2)] == [0, 0]

    def test_ex_mem_has_no_stall_port(self):
        """The load-use bubble is inserted upstream, at wide ID/EX."""
        reg = WideEX_MEM(num_lanes=2)
        assert "stall" not in reg.ports_spec
        assert "flush" in reg.ports_spec

    @pytest.mark.parametrize("n", LANE_WIDTHS)
    def test_all_declared_fields_round_trip_for_every_lane(self, n):
        from sim.components.superscalar.wide_pipeline_regs import _EX_MEM_FIELDS
        reg = WideEX_MEM(num_lanes=n)
        for i in range(n):
            for field in _EX_MEM_FIELDS:
                reg[f"{field}_in_{i}"] = 1
            reg[f"valid_in_{i}"] = 1
        reg.rising_edge()
        reg.evaluate()
        for i in range(n):
            for field in _EX_MEM_FIELDS:
                assert reg[f"{field}_out_{i}"] == 1, f"lane {i} field {field}"


class TestWideMEM_WB:

    def _reg(self, num_lanes=2):
        reg = WideMEM_WB(num_lanes=num_lanes)
        for i in range(num_lanes):
            reg[f"mem_data_in_{i}"] = 0x70 + i
            reg[f"alu_result_in_{i}"] = 0x80 + i
            reg[f"rd_in_{i}"] = 8 + i
            reg[f"pc_in_{i}"] = 0x90 + i * 4
            reg[f"wb_sel_in_{i}"] = 1
            reg[f"valid_in_{i}"] = 1
        return reg

    def test_every_lane_latches_its_writeback_context(self):
        reg = self._reg()
        reg.rising_edge()
        reg.evaluate()
        assert [reg[f"mem_data_out_{i}"] for i in range(2)] == [0x70, 0x71]
        assert [reg[f"alu_result_out_{i}"] for i in range(2)] == [0x80, 0x81]
        assert [reg[f"valid_{i}"] for i in range(2)] == [1, 1]
        assert reg.get_state()["lane_0"] == {
            "pc": "0x00000090", "rd": 8, "wb_sel": 1, "valid": "VALID"}

    def test_valid_propagates_per_lane_from_ex_mem(self):
        reg = self._reg()
        reg["valid_in_1"] = 0
        reg.rising_edge()
        reg.evaluate()
        assert [reg[f"valid_{i}"] for i in range(2)] == [1, 0]

    def test_mem_wb_is_the_final_stage_so_it_has_no_flush_or_stall(self):
        reg = WideMEM_WB(num_lanes=2)
        assert "flush" not in reg.ports_spec
        assert "stall" not in reg.ports_spec

    @pytest.mark.parametrize("n", LANE_WIDTHS)
    def test_all_declared_fields_round_trip_for_every_lane(self, n):
        from sim.components.superscalar.wide_pipeline_regs import _MEM_WB_FIELDS
        reg = WideMEM_WB(num_lanes=n)
        for i in range(n):
            for field in _MEM_WB_FIELDS:
                reg[f"{field}_in_{i}"] = 1
            reg[f"valid_in_{i}"] = 1
        reg.rising_edge()
        reg.evaluate()
        for i in range(n):
            for field in _MEM_WB_FIELDS:
                assert reg[f"{field}_out_{i}"] == 1, f"lane {i} field {field}"


class TestWidePipelineRegistersEndToEnd:

    def test_a_context_walks_lane_1_from_if_id_through_mem_wb(self):
        """Stitch the four wide registers together and clock a value along the
        lane-1 slot, exactly as the superscalar preset wires them."""
        if_id = WideIF_ID(num_lanes=2)
        id_ex = WideID_EX(num_lanes=2)
        ex_mem = WideEX_MEM(num_lanes=2)
        mem_wb = WideMEM_WB(num_lanes=2)

        if_id["pc_in_1"] = 0x24
        if_id["instr_in_1"] = 0xFEED
        if_id.rising_edge()
        if_id.evaluate()
        assert if_id["valid_1"] == 1

        id_ex["pc_in_1"] = if_id["pc_out_1"]
        id_ex["rd_in_1"] = 12
        id_ex["lane_valid_in_1"] = 1
        id_ex["valid_in_1"] = if_id["valid_1"]
        id_ex.rising_edge()
        id_ex.evaluate()
        assert (id_ex["pc_out_1"], id_ex["rd_out_1"], id_ex["valid_1"]) == (0x24, 12, 1)

        ex_mem["pc_in_1"] = id_ex["pc_out_1"]
        ex_mem["rd_in_1"] = id_ex["rd_out_1"]
        ex_mem["alu_result_in_1"] = 0xDEAD
        ex_mem["valid_in_1"] = id_ex["valid_1"]
        ex_mem.rising_edge()
        ex_mem.evaluate()
        assert (ex_mem["alu_result_out_1"], ex_mem["valid_1"]) == (0xDEAD, 1)

        mem_wb["pc_in_1"] = ex_mem["pc_out_1"]
        mem_wb["rd_in_1"] = ex_mem["rd_out_1"]
        mem_wb["alu_result_in_1"] = ex_mem["alu_result_out_1"]
        mem_wb["valid_in_1"] = ex_mem["valid_1"]
        mem_wb.rising_edge()
        mem_wb.evaluate()
        assert (mem_wb["pc_out_1"], mem_wb["rd_out_1"]) == (0x24, 12)
        assert mem_wb["alu_result_out_1"] == 0xDEAD
        assert mem_wb["valid_1"] == 1
        assert mem_wb["valid_0"] == 0, "the untouched lane stays a bubble"


# ═══════════════════════════════════════════════════════════════════
# WideFlagsSelect
# ═══════════════════════════════════════════════════════════════════

class TestWideFlagsSelect:

    def test_no_flag_setter_drives_nothing(self):
        sel = WideFlagsSelect(num_lanes=3)
        sel.evaluate()
        assert (sel["write_flags_out"], sel["alu_zero_out"], sel["alu_result_out"]) == (0, 0, 0)
        assert sel.get_state() == {"write_flags": 0, "zero": 0}

    @pytest.mark.parametrize("lane", [0, 1, 2])
    def test_a_lone_flag_setter_in_any_lane_drives_the_flags_register(self, lane):
        sel = WideFlagsSelect(num_lanes=3)
        sel[f"write_flags_{lane}"] = 1
        sel[f"alu_zero_{lane}"] = 1
        sel[f"alu_result_{lane}"] = 0x33
        sel.evaluate()
        assert sel["write_flags_out"] == 1
        assert sel["alu_zero_out"] == 1
        assert sel["alu_result_out"] == 0x33

    def test_the_youngest_flag_setter_in_the_group_wins(self):
        """Lane order is program order, so the highest-index setter is latest."""
        sel = WideFlagsSelect(num_lanes=3)
        sel["write_flags_0"] = 1
        sel["alu_zero_0"] = 1
        sel["alu_result_0"] = 0
        sel["write_flags_2"] = 1
        sel["alu_zero_2"] = 0
        sel["alu_result_2"] = 0x55
        sel.evaluate()
        assert (sel["alu_zero_out"], sel["alu_result_out"]) == (0, 0x55)
        assert sel.get_state() == {"write_flags": 1, "zero": 0}

    def test_squashed_lanes_carry_write_flags_low_and_are_ignored(self):
        sel = WideFlagsSelect(num_lanes=3)
        sel["write_flags_0"] = 1
        sel["alu_zero_0"] = 1
        sel["alu_result_0"] = 0
        sel["write_flags_2"] = 0        # squashed lane: bubble
        sel["alu_zero_2"] = 0
        sel["alu_result_2"] = 0x55
        sel.evaluate()
        assert (sel["alu_zero_out"], sel["alu_result_out"]) == (1, 0)

    @pytest.mark.parametrize("n", LANE_WIDTHS)
    def test_the_last_lane_always_wins_at_every_width(self, n):
        sel = WideFlagsSelect(num_lanes=n)
        for i in range(n):
            sel[f"write_flags_{i}"] = 1
            sel[f"alu_result_{i}"] = 0x10 + i
        sel.evaluate()
        assert sel["alu_result_out"] == 0x10 + (n - 1)


# ═══════════════════════════════════════════════════════════════════
# MultiPortDataMemory
# ═══════════════════════════════════════════════════════════════════

class TestMultiPortDataMemory:

    def test_every_lane_can_store_to_an_independent_address_in_one_cycle(self):
        """This is the whole point of the multi-port memory: no structural
        stall, so memory-bound code scales with lane width."""
        mem = MultiPortDataMemory(num_lanes=4, size=64)
        for i in range(4):
            mem[f"addr_{i}"] = i * 4
            mem[f"wdata_{i}"] = 100 + i
            mem[f"mem_write_{i}"] = 1
        _cycle(mem)
        assert mem._mem[:4] == [100, 101, 102, 103]

    def test_every_lane_can_load_from_an_independent_address_in_one_cycle(self):
        mem = MultiPortDataMemory(num_lanes=4, size=64)
        mem._mem[:4] = [10, 20, 30, 40]
        for i in range(4):
            mem[f"addr_{i}"] = (3 - i) * 4
            mem[f"mem_read_{i}"] = 1
        mem.evaluate()
        assert [mem[f"rdata_{i}"] for i in range(4)] == [40, 30, 20, 10]

    def test_two_lanes_storing_the_same_word_commit_in_program_order(self):
        """Lane order IS program order, so the YOUNGER (higher-index) lane wins."""
        mem = MultiPortDataMemory(num_lanes=3, size=64)
        mem["addr_0"] = 16
        mem["wdata_0"] = 111
        mem["mem_write_0"] = 1
        mem["addr_2"] = 16
        mem["wdata_2"] = 333
        mem["mem_write_2"] = 1
        _cycle(mem)
        assert mem._mem[4] == 333

    def test_an_older_store_forwards_to_a_younger_load_in_the_same_group(self):
        """Otherwise a load in lane 1 would read the pre-cycle memory and miss
        the store its own group's lane 0 just performed."""
        mem = MultiPortDataMemory(num_lanes=3, size=64)
        mem._mem[4] = 7
        mem["addr_0"] = 16
        mem["wdata_0"] = 42
        mem["mem_write_0"] = 1
        mem["addr_1"] = 16
        mem["mem_read_1"] = 1
        mem["addr_2"] = 16
        mem["mem_read_2"] = 1
        mem.evaluate()
        assert mem["rdata_1"] == 42
        assert mem["rdata_2"] == 42

    def test_a_younger_store_never_forwards_backwards_to_an_older_load(self):
        mem = MultiPortDataMemory(num_lanes=3, size=64)
        mem._mem[4] = 7
        mem["addr_0"] = 16
        mem["mem_read_0"] = 1
        mem["addr_1"] = 16
        mem["wdata_1"] = 42
        mem["mem_write_1"] = 1
        mem.evaluate()
        assert mem["rdata_0"] == 7, "lane 0 is OLDER and must not see lane 1's store"

    def test_the_youngest_older_store_wins_the_forward(self):
        mem = MultiPortDataMemory(num_lanes=4, size=64)
        mem["addr_0"] = 16
        mem["wdata_0"] = 1
        mem["mem_write_0"] = 1
        mem["addr_1"] = 16
        mem["wdata_1"] = 2
        mem["mem_write_1"] = 1
        mem["addr_3"] = 16
        mem["mem_read_3"] = 1
        mem.evaluate()
        assert mem["rdata_3"] == 2

    def test_a_store_to_a_different_word_does_not_forward(self):
        mem = MultiPortDataMemory(num_lanes=2, size=64)
        mem._mem[4] = 7
        mem["addr_0"] = 20            # word 5
        mem["wdata_0"] = 42
        mem["mem_write_0"] = 1
        mem["addr_1"] = 16            # word 4
        mem["mem_read_1"] = 1
        mem.evaluate()
        assert mem["rdata_1"] == 7

    def test_forwarding_is_word_granular_not_byte_granular(self):
        """Byte addresses inside the same word alias to the same slot."""
        mem = MultiPortDataMemory(num_lanes=2, size=64)
        mem["addr_0"] = 16
        mem["wdata_0"] = 0xABCD
        mem["mem_write_0"] = 1
        mem["addr_1"] = 18            # still word 4
        mem["mem_read_1"] = 1
        mem.evaluate()
        assert mem["rdata_1"] == 0xABCD

    def test_a_lane_with_no_memory_op_still_reads_its_address(self):
        """rdata is combinational and unconditional; mem_read only tags intent."""
        mem = MultiPortDataMemory(num_lanes=2, size=64)
        mem._mem[4] = 55
        mem["addr_1"] = 16
        mem["mem_read_1"] = 0
        mem.evaluate()
        assert mem["rdata_1"] == 55

    def test_mem_write_low_writes_nothing(self):
        mem = MultiPortDataMemory(num_lanes=2, size=64)
        mem["addr_0"] = 16
        mem["wdata_0"] = 99
        mem["mem_write_0"] = 0
        _cycle(mem)
        assert mem._mem[4] == 0

    def test_a_store_to_byte_address_zero_is_honoured(self):
        mem = MultiPortDataMemory(num_lanes=2, size=64)
        mem["addr_0"] = 0
        mem["wdata_0"] = 0xBEEF
        mem["mem_write_0"] = 1
        _cycle(mem)
        assert mem._mem[0] == 0xBEEF

    def test_addresses_wrap_at_the_memory_size(self):
        mem = MultiPortDataMemory(num_lanes=1, size=8)
        mem["addr_0"] = 8 * 4         # word 8 wraps to word 0
        mem["wdata_0"] = 5
        mem["mem_write_0"] = 1
        _cycle(mem)
        assert mem._mem[0] == 5

    def test_read_data_is_masked_to_32_bits(self):
        mem = MultiPortDataMemory(num_lanes=1, size=8)
        mem._mem[0] = 0x1_FFFF_FFFF
        mem.evaluate()
        assert mem["rdata_0"] == 0xFFFF_FFFF

    @pytest.mark.parametrize("n", LANE_WIDTHS)
    def test_port_set_scales_with_lane_count(self, n):
        mem = MultiPortDataMemory(num_lanes=n, size=16)
        for i in range(n):
            for port in (f"addr_{i}", f"wdata_{i}", f"mem_read_{i}",
                         f"mem_write_{i}", f"rdata_{i}"):
                assert port in mem.ports_spec
        assert f"rdata_{n}" not in mem.ports_spec


class TestMultiPortDataMemorySnapshot:
    """The per-cycle snapshot must stay shape-compatible with DataMemory's, or
    the harness (SimResult.memory) and the UI break."""

    def test_get_state_matches_the_single_port_snapshot_shape(self):
        multi = MultiPortDataMemory(num_lanes=2, size=64).get_state()
        single = DataMemory(size=64).get_state()
        shared = {"addr", "rdata", "wdata", "wen", "write_addr",
                  "memory", "memory_hi", "size", "window"}
        assert shared <= set(multi)
        assert shared <= set(single)

    def test_the_snapshot_splits_into_a_dense_window_and_a_sparse_high_map(self):
        mem = MultiPortDataMemory(num_lanes=1)
        mem._mem[10] = 1
        mem._mem[3000] = 99
        state = mem.get_state()
        assert len(state["memory"]) == STATE_WINDOW
        assert state["memory"][10] == 1
        assert state["memory_hi"] == {3000: 99}
        assert state["window"] == STATE_WINDOW
        assert state["size"] == len(mem._mem)

    def test_an_idle_memory_reports_no_active_lanes_and_no_write(self):
        mem = MultiPortDataMemory(num_lanes=2, size=16)
        state = mem.get_state()
        assert state["active_lanes"] == []
        assert state["wen"] == 0
        assert state["write_addr"] == 0
        assert state["addr"] == "0x00000000"

    def test_the_snapshot_names_every_lane_doing_a_memory_op(self):
        mem = MultiPortDataMemory(num_lanes=4, size=64)
        mem["addr_1"] = 16
        mem["mem_read_1"] = 1
        mem["addr_3"] = 32
        mem["wdata_3"] = 7
        mem["mem_write_3"] = 1
        mem.evaluate()
        state = mem.get_state()
        assert state["active_lanes"] == [1, 3]
        assert state["addr"] == "0x00000010", "reports the FIRST active lane"
        assert state["wen"] == 1
        assert state["write_addr"] == 32, "reports the YOUNGEST store"

    def test_write_addr_reports_the_youngest_store_when_several_lanes_store(self):
        mem = MultiPortDataMemory(num_lanes=3, size=64)
        for i, addr in enumerate((4, 8, 12)):
            mem[f"addr_{i}"] = addr
            mem[f"wdata_{i}"] = i
            mem[f"mem_write_{i}"] = 1
        mem.evaluate()
        assert mem.get_state()["write_addr"] == 12


# ═══════════════════════════════════════════════════════════════════
# Dead-code guard
# ═══════════════════════════════════════════════════════════════════

class TestMemoryLaneMuxIsDeadCode:
    """`sim/components/superscalar/memory_lane_mux.py` is the RETIRED
    single-port memory arbiter. It was replaced by MultiPortDataMemory (see
    the CrossLaneHazardDetector comment "3. (removed) Structural memory
    hazard"), and nothing imports it any more — it is 0% covered by design.

    These tests assert that it is genuinely unreachable, so the file can be
    deleted, and so it cannot silently be re-wired without the guard firing.
    """

    def test_no_module_in_the_repo_imports_memory_lane_mux(self):
        offenders = []
        for path in REPO_ROOT.rglob("*.py"):
            if "memory_lane_mux" in path.name or "__pycache__" in path.parts:
                continue
            if path.resolve() == pathlib.Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "memory_lane_mux" in text or "MemoryLaneMux" in text:
                offenders.append(str(path.relative_to(REPO_ROOT)))
        assert offenders == [], (
            "memory_lane_mux is documented as dead code; if it is being used "
            f"again, update this guard. Referencing files: {offenders}")

    def test_the_superscalar_package_does_not_re_export_it(self):
        import sim.components.superscalar as pkg
        assert not hasattr(pkg, "MemoryLaneMux")

    def test_the_module_still_parses_so_deletion_is_the_only_cleanup_needed(self):
        """Sanity: it is dead, not broken — this documents that removing it is a
        pure deletion with no dangling references to repair."""
        source = (REPO_ROOT / "sim" / "components" / "superscalar"
                  / "memory_lane_mux.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert classes == ["MemoryLaneMux"]
