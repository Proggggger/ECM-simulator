"""
UNIT tests for the out-of-order (`sim/components/ooo/`) and 5-stage-pipeline
(`sim/components/pipeline/`) hardware components.

WHY THIS FILE EXISTS
--------------------
Every component below was previously exercised ONLY end-to-end, through a full
preset run (`tests/test_riscv_ooo.py`, `tests/test_riscv_pipeline.py`, ...). A
whole-CPU run is a weak probe of a state machine: an internal path can be dead,
inverted, or silently swallowed and the architectural end state still comes out
right (or the failure is attributed to some other component). Several of these
files had ZERO direct tests at all (`cdb_value_select`, `committed_flags`,
`branch_flag_mux`, `pc_operand_select`, `forwarding_unit`, `hazard_detector`).

Here each component is driven DIRECTLY: construct it, write input ports via
``comp["port"] = v``, call ``evaluate()`` / ``rising_edge()`` by hand, and assert
on output ports and ``get_state()``. That makes the ROB's circular-buffer
bookkeeping, the RS's wakeup/selection policy, the RAT's rename/free protocol,
and the forwarding priority network independently falsifiable.

CLOCKING CONTRACT (mirrors ``CPU.tick()`` in sim/component/wire.py)
-------------------------------------------------------------------
``tick()`` runs (1) evaluate() over the eval order, (2) a final wire
propagation, (3) rising_edge() on every component. So a sequential component's
``rising_edge()`` legitimately reads OUTPUT ports that its own ``evaluate()``
computed this cycle (e.g. the ROB commits the entry its evaluate() exposed on
``commit_en``). Tests therefore always call ``evaluate()`` before
``rising_edge()`` for one simulated cycle, via the ``_cycle`` helpers.

DOCUMENTED-GAP PINS (deliberately NOT bug reports)
--------------------------------------------------
* ``test_rob_commit_of_taken_branch_always_flushes`` — the ROB's mispredict
  check is the SIMPLIFIED model documented in rob.py: any committed taken
  branch flushes (no predicted-vs-actual compare). Pinned so a future real
  mispredict comparison is a deliberate, noticed change.
* ``test_hazard_detector_does_not_gate_rs2_on_reads_rs2`` — HazardDetector, unlike
  CrossLaneHazardDetector, has no ``reads_rs2`` gate, so an I-type instruction
  whose rs2 FIELD happens to hold matching immediate bits over-stalls. That is
  conservative (correct, just slower), so it is pinned, not filed.

BUG FOUND HERE AND SINCE FIXED (2026-08-08)
-------------------------------------------
``TestHazardDetectorZeroRegisterGap`` — HazardDetector used to hardcode
``rd != 0`` instead of taking the ``zero_reg_index`` parameter every other
zero-reg-aware component takes. On ARM (zero reg is XZR=31) and x86 (NO zero
reg) register index 0 is a REAL GPR, so a load into ARM X0 / x86 EAX skipped
the load-use stall and the consumer read the load's ADDRESS instead of its
data. Fixed by parameterising the component and passing 31 / None from the
ARM / x86 pipeline presets; those tests are now regression pins.

Run from repo root:  python3 -m pytest tests/test_ooo_components_unit.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.components.ooo.branch_flag_mux import BranchFlagMux
from sim.components.ooo.cdb_value_select import CdbValueSelect
from sim.components.ooo.committed_flags import CommittedFlags
from sim.components.ooo.pc_operand_select import PcOperandSelect
from sim.components.ooo.rat import RegisterAliasTable
from sim.components.ooo.reservation_station import RS_SIZE, ReservationStation
from sim.components.ooo.rob import ROB_SIZE, ReorderBuffer
from sim.components.ooo.store_commit import StoreCommitUnit
from sim.components.pipeline.forwarding_unit import ForwardingUnit
from sim.components.pipeline.hazard_detector import HazardDetector
from sim.components.pipeline.registers import EX_MEM, ID_EX, IF_ID, MEM_WB
from sim.harness import simulate


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _cycle(comp):
    """Run one simulated clock on a single component: evaluate then latch."""
    comp.evaluate()
    comp.rising_edge()


def _rob_dispatch(rob, rd=0, pc=0, is_branch=0, is_load=0, is_store=0,
                  store_data_reg=0, reg_write=1, write_flags=0):
    """Drive the ROB's dispatch input group (does NOT clock)."""
    rob["dispatch_en"] = 1
    rob["dispatch_rd"] = rd
    rob["dispatch_pc"] = pc
    rob["dispatch_is_branch"] = is_branch
    rob["dispatch_is_load"] = is_load
    rob["dispatch_is_store"] = is_store
    rob["dispatch_store_data_reg"] = store_data_reg
    rob["dispatch_reg_write"] = reg_write
    rob["dispatch_write_flags"] = write_flags


def _rob_dispatch_off(rob):
    rob["dispatch_en"] = 0


def _rob_complete(rob, tag, value=0, mem_value=0, branch_taken=0):
    rob["complete_en"] = 1
    rob["complete_tag"] = tag
    rob["complete_value"] = value
    rob["complete_mem_value"] = mem_value
    rob["complete_branch_taken"] = branch_taken


def _rob_complete_off(rob):
    rob["complete_en"] = 0
    rob["complete_branch_taken"] = 0


def _rs_issue(rs, op=0, src1_ready=1, src1_tag=0, src1_val=0,
              src2_ready=1, src2_tag=0, src2_val=0, rob_tag=0,
              imm=0, alu_src=0, is_load=0):
    """Drive the reservation station's issue input group (does NOT clock)."""
    rs["issue_en"] = 1
    rs["issue_op"] = op
    rs["issue_src1_ready"] = src1_ready
    rs["issue_src1_tag"] = src1_tag
    rs["issue_src1_val"] = src1_val
    rs["issue_src2_ready"] = src2_ready
    rs["issue_src2_tag"] = src2_tag
    rs["issue_src2_val"] = src2_val
    rs["issue_rob_tag"] = rob_tag
    rs["issue_imm"] = imm
    rs["issue_alu_src"] = alu_src
    rs["issue_is_load"] = is_load


def _rs_issue_off(rs):
    rs["issue_en"] = 0


def _rs_cdb(rs, tag, value):
    rs["cdb_en"] = 1
    rs["cdb_tag"] = tag
    rs["cdb_value"] = value


def _rs_cdb_off(rs):
    rs["cdb_en"] = 0


# ═══════════════════════════════════════════════════════════════════
# ReorderBuffer
# ═══════════════════════════════════════════════════════════════════

class TestReorderBufferAllocateCompleteCommit:
    """The dispatch -> complete -> in-order-commit lifecycle."""

    def test_fresh_rob_is_empty_not_full_and_advertises_tail_tag_zero(self):
        rob = ReorderBuffer()
        rob.evaluate()
        assert rob["empty"] == 1
        assert rob["full"] == 0
        assert rob["dispatch_tag"] == 0
        assert rob["commit_en"] == 0
        assert rob["store_pending_mask"] == 0
        assert rob["head_ptr"] == 0

    def test_dispatch_tag_advertises_current_tail_before_the_clock_edge(self):
        """dispatch_tag is combinational so rename can consume it same-cycle."""
        rob = ReorderBuffer()
        for expected_tag in range(4):
            _rob_dispatch(rob, rd=expected_tag + 1)
            rob.evaluate()
            assert rob["dispatch_tag"] == expected_tag
            rob.rising_edge()

    def test_allocated_entry_is_not_committable_until_completed(self):
        rob = ReorderBuffer()
        _rob_dispatch(rob, rd=3)
        _cycle(rob)
        _rob_dispatch_off(rob)

        rob.evaluate()
        assert rob["empty"] == 0
        assert rob["commit_en"] == 0, "an incomplete head entry must not commit"

    def test_complete_then_commit_exposes_rd_value_and_tag(self):
        rob = ReorderBuffer()
        _rob_dispatch(rob, rd=3)
        _cycle(rob)
        _rob_dispatch_off(rob)

        _rob_complete(rob, tag=0, value=0xDEAD)
        _cycle(rob)
        _rob_complete_off(rob)

        rob.evaluate()
        assert rob["commit_en"] == 1
        assert rob["commit_rd"] == 3
        assert rob["commit_value"] == 0xDEAD
        assert rob["commit_tag"] == 0
        assert rob["commit_reg_write"] == 1
        assert rob["commit_is_store"] == 0

        rob.rising_edge()
        rob.evaluate()
        assert rob["empty"] == 1
        assert rob.get_state()["head"] == 1

    def test_commit_is_strictly_in_order_even_when_completion_is_out_of_order(self):
        """The whole point of the ROB: younger-first completion, oldest-first commit."""
        rob = ReorderBuffer()
        _rob_dispatch(rob, rd=1)
        _cycle(rob)
        _rob_dispatch(rob, rd=2)
        _cycle(rob)
        _rob_dispatch_off(rob)

        # Complete the YOUNGER entry (tag 1) first.
        _rob_complete(rob, tag=1, value=0xB)
        _cycle(rob)
        _rob_complete_off(rob)

        rob.evaluate()
        assert rob["commit_en"] == 0, "younger-ready must not overtake the head"

        _rob_complete(rob, tag=0, value=0xA)
        _cycle(rob)
        _rob_complete_off(rob)

        rob.evaluate()
        assert (rob["commit_rd"], rob["commit_value"]) == (1, 0xA)
        rob.rising_edge()
        rob.evaluate()
        assert (rob["commit_rd"], rob["commit_value"]) == (2, 0xB)

    def test_completing_a_load_latches_the_memory_value_not_the_alu_address(self):
        """For a load, complete_value is the ADDRESS; the ROB must keep mem data."""
        rob = ReorderBuffer()
        _rob_dispatch(rob, rd=5, is_load=1)
        _cycle(rob)
        _rob_dispatch_off(rob)

        _rob_complete(rob, tag=0, value=0xAAAA, mem_value=0xBBBB)
        _cycle(rob)
        _rob_complete_off(rob)

        rob.evaluate()
        assert rob["commit_value"] == 0xBBBB

    def test_complete_targeting_an_invalid_slot_is_ignored(self):
        rob = ReorderBuffer()
        _rob_complete(rob, tag=4, value=0x1234)
        _cycle(rob)
        _rob_complete_off(rob)

        rob.evaluate()
        assert rob["empty"] == 1
        assert rob["commit_en"] == 0

    def test_get_state_reports_entry_table_head_tail_and_count(self):
        rob = ReorderBuffer()
        _rob_dispatch(rob, rd=9, pc=0x40, is_load=1)
        _cycle(rob)
        _rob_dispatch_off(rob)
        rob.evaluate()

        state = rob.get_state()
        assert state["head"] == 0
        assert state["tail"] == 1
        assert state["count"] == 1
        assert state["full"] is False
        assert len(state["entries"]) == ROB_SIZE
        entry = state["entries"][0]
        assert entry["valid"] == 1
        assert entry["ready"] == 0
        assert entry["rd"] == 9
        assert entry["is_load"] == 1
        assert entry["pc"] == "0x00000040"


class TestReorderBufferCapacityAndWrap:

    def test_rob_reports_full_after_exactly_ROB_SIZE_dispatches(self):
        rob = ReorderBuffer()
        for i in range(ROB_SIZE):
            _rob_dispatch(rob, rd=i + 1)
            _cycle(rob)
        _rob_dispatch_off(rob)

        rob.evaluate()
        assert rob["full"] == 1
        assert rob.get_state()["count"] == ROB_SIZE

    def test_dispatch_into_a_full_rob_is_dropped_not_wrapped_over(self):
        rob = ReorderBuffer()
        for i in range(ROB_SIZE):
            _rob_dispatch(rob, rd=i + 1)
            _cycle(rob)

        _rob_dispatch(rob, rd=30)          # one too many
        rob.evaluate()
        assert rob["full"] == 1
        rob.rising_edge()

        assert rob.get_state()["count"] == ROB_SIZE
        # Slot 0 still holds the FIRST instruction (rd=1), not the overflow.
        assert rob.get_state()["entries"][0]["rd"] == 1

    def test_head_and_tail_wrap_through_the_circular_buffer(self):
        """Steady-state dispatch+complete for >2 full laps: tags must recycle."""
        rob = ReorderBuffer()
        commit_tags = []
        pending = []

        for i in range(30):
            _rob_dispatch(rob, rd=(i % 20) + 1, pc=i * 4)
            if pending:
                _rob_complete(rob, tag=pending[0], value=0x1000 + pending[0])
            else:
                _rob_complete_off(rob)

            rob.evaluate()
            if rob["commit_en"]:
                commit_tags.append(rob["commit_tag"])
            was_full = rob["full"]
            allocated = rob["dispatch_tag"]
            rob.rising_edge()

            if rob["complete_en"] and pending:
                pending.pop(0)
            if not was_full:
                pending.append(allocated)

        assert len(commit_tags) > 2 * ROB_SIZE, "must lap the buffer more than twice"
        # Tags recycle 0..7,0..7,... — that IS the wrap.
        assert commit_tags[:ROB_SIZE] == list(range(ROB_SIZE))
        assert commit_tags[ROB_SIZE:2 * ROB_SIZE] == list(range(ROB_SIZE))
        assert 0 <= rob.get_state()["head"] < ROB_SIZE
        assert 0 <= rob.get_state()["tail"] < ROB_SIZE


class TestReorderBufferStoreAndRegWriteGating:

    def test_store_pending_mask_sets_one_bit_per_uncommitted_store_slot(self):
        rob = ReorderBuffer()
        _rob_dispatch(rob, rd=0, is_store=1, reg_write=0)    # slot 0: store
        _cycle(rob)
        _rob_dispatch(rob, rd=4)                              # slot 1: ALU op
        _cycle(rob)
        _rob_dispatch(rob, rd=0, is_store=1, reg_write=0)    # slot 2: store
        _cycle(rob)
        _rob_dispatch_off(rob)

        rob.evaluate()
        assert rob["store_pending_mask"] == 0b101
        assert rob["head_ptr"] == 0

    def test_store_pending_mask_clears_when_the_store_commits(self):
        rob = ReorderBuffer()
        _rob_dispatch(rob, rd=0, is_store=1, reg_write=0)
        _cycle(rob)
        _rob_dispatch_off(rob)

        _rob_complete(rob, tag=0, value=0x80)
        _cycle(rob)
        _rob_complete_off(rob)

        rob.evaluate()
        assert rob["store_pending_mask"] == 0b1, "still pending until it retires"
        rob.rising_edge()
        rob.evaluate()
        assert rob["store_pending_mask"] == 0

    def test_head_ptr_tracks_the_oldest_entry_for_rs_age_comparison(self):
        rob = ReorderBuffer()
        for i in range(3):
            _rob_dispatch(rob, rd=i + 1)
            _cycle(rob)
        _rob_dispatch_off(rob)

        for expected_head in range(3):
            rob.evaluate()
            assert rob["head_ptr"] == expected_head
            _rob_complete(rob, tag=expected_head, value=0)
            rob.rising_edge()
            _rob_complete_off(rob)
            rob.evaluate()      # entry now ready -> commit_en high
            rob.rising_edge()   # retire it

    def test_committing_a_store_drives_is_store_and_the_store_data_register(self):
        rob = ReorderBuffer()
        _rob_dispatch(rob, rd=0, is_store=1, reg_write=0, store_data_reg=7)
        _cycle(rob)
        _rob_dispatch_off(rob)

        _rob_complete(rob, tag=0, value=0x80)   # value = the store ADDRESS
        _cycle(rob)
        _rob_complete_off(rob)

        rob.evaluate()
        assert rob["commit_en"] == 1
        assert rob["commit_is_store"] == 1
        assert rob["commit_store_data_reg"] == 7
        assert rob["commit_value"] == 0x80
        assert rob["commit_reg_write"] == 0, "a pure store must not enable regfile write"

    def test_commit_reg_write_mirrors_the_per_entry_reg_write_flag(self):
        """commit_reg_write drives regfile.wen, so a store must not raise it."""
        rob = ReorderBuffer()
        _rob_dispatch(rob, rd=2, reg_write=1)                 # slot 0: writes a reg
        _cycle(rob)
        _rob_dispatch(rob, rd=0, is_store=1, reg_write=0)     # slot 1: pure store
        _cycle(rob)
        _rob_dispatch_off(rob)

        # Complete the YOUNGER entry first so no commit fires while we set up.
        for tag in (1, 0):
            _rob_complete(rob, tag=tag, value=0x10 + tag)
            _cycle(rob)
            _rob_complete_off(rob)

        rob.evaluate()
        assert (rob["commit_reg_write"], rob["commit_is_store"]) == (1, 0)
        rob.rising_edge()
        rob.evaluate()
        assert (rob["commit_reg_write"], rob["commit_is_store"]) == (0, 1)

    def test_commit_write_flags_propagates_the_flag_setter_bit(self):
        rob = ReorderBuffer()
        _rob_dispatch(rob, rd=0, reg_write=0, write_flags=1)
        _cycle(rob)
        _rob_dispatch_off(rob)

        _rob_complete(rob, tag=0, value=0)
        _cycle(rob)
        _rob_complete_off(rob)

        rob.evaluate()
        assert rob["commit_write_flags"] == 1
        assert rob["commit_reg_write"] == 0


class TestReorderBufferFlush:

    def test_rob_commit_of_taken_branch_always_flushes(self):
        """PINS THE DOCUMENTED SIMPLIFICATION (rob.py docstring): there is no
        predicted-vs-actual compare — ANY committed taken branch raises flush,
        with the correct target carried in the entry's value field."""
        rob = ReorderBuffer()
        _rob_dispatch(rob, rd=0, is_branch=1, reg_write=0, pc=0x40)
        _cycle(rob)
        _rob_dispatch_off(rob)

        _rob_complete(rob, tag=0, value=0x100, branch_taken=1)
        _cycle(rob)
        _rob_complete_off(rob)

        rob.evaluate()
        assert rob["commit_en"] == 1
        assert rob["flush"] == 1
        assert rob["flush_target_pc"] == 0x100
        assert rob.get_state()["flush"] == 1

    def test_committed_not_taken_branch_does_not_flush(self):
        rob = ReorderBuffer()
        _rob_dispatch(rob, rd=0, is_branch=1, reg_write=0)
        _cycle(rob)
        _rob_dispatch_off(rob)

        _rob_complete(rob, tag=0, value=0x50, branch_taken=0)
        _cycle(rob)
        _rob_complete_off(rob)

        rob.evaluate()
        assert rob["commit_en"] == 1
        assert rob["flush"] == 0

    def test_non_branch_entry_never_flushes_even_with_branch_taken_asserted(self):
        rob = ReorderBuffer()
        _rob_dispatch(rob, rd=6, is_branch=0)
        _cycle(rob)
        _rob_dispatch_off(rob)

        _rob_complete(rob, tag=0, value=0x77, branch_taken=1)
        _cycle(rob)
        _rob_complete_off(rob)

        rob.evaluate()
        assert rob["flush"] == 0

    def test_flush_wipes_every_entry_and_resets_head_tail_count(self):
        rob = ReorderBuffer()
        for i in range(5):
            _rob_dispatch(rob, rd=i + 1)
            _cycle(rob)
        _rob_dispatch(rob, rd=0, is_branch=1, reg_write=0)
        _cycle(rob)
        _rob_dispatch_off(rob)

        # Complete the HEAD so it commits, and make it a taken branch by
        # completing the branch slot too — the head commits first though, so
        # instead flush directly from the branch by draining to it.
        for tag in range(5):
            _rob_complete(rob, tag=tag, value=0)
            _cycle(rob)
            _rob_complete_off(rob)
            rob.evaluate()
            rob.rising_edge()

        _rob_complete(rob, tag=5, value=0x200, branch_taken=1)
        _cycle(rob)
        _rob_complete_off(rob)

        rob.evaluate()
        assert rob["flush"] == 1
        rob.rising_edge()

        state = rob.get_state()
        assert state["head"] == 0
        assert state["tail"] == 0
        assert state["count"] == 0
        assert all(e["valid"] == 0 for e in state["entries"])
        rob.evaluate()
        assert rob["empty"] == 1
        assert rob["store_pending_mask"] == 0

    def test_flush_outranks_a_same_cycle_dispatch(self):
        """On the flush edge the ROB returns early — no new entry is allocated."""
        rob = ReorderBuffer()
        _rob_dispatch(rob, rd=0, is_branch=1, reg_write=0)
        _cycle(rob)
        _rob_complete(rob, tag=0, value=0x300, branch_taken=1)
        _cycle(rob)
        _rob_complete_off(rob)

        rob.evaluate()
        assert rob["flush"] == 1
        _rob_dispatch(rob, rd=12)   # a younger instruction arriving on the flush edge
        rob.rising_edge()

        assert rob.get_state()["count"] == 0


# ═══════════════════════════════════════════════════════════════════
# ReservationStation
# ═══════════════════════════════════════════════════════════════════

class TestReservationStationIssueAndSelect:

    def test_empty_station_offers_nothing_and_is_not_full(self):
        rs = ReservationStation()
        rs.evaluate()
        assert rs["exec_valid"] == 0
        assert rs["full"] == 0
        assert rs.get_state()["full"] is False

    def test_entry_with_both_operands_ready_is_offered_for_execution(self):
        rs = ReservationStation()
        _rs_issue(rs, op=1, src1_val=10, src2_val=20, rob_tag=3)
        _cycle(rs)
        _rs_issue_off(rs)

        rs.evaluate()
        assert rs["exec_valid"] == 1
        assert rs["exec_op"] == 1
        assert rs["exec_src1"] == 10
        assert rs["exec_src2"] == 20
        assert rs["exec_rob_tag"] == 3
        assert rs["exec_is_load"] == 0

    def test_alu_src_selects_the_immediate_as_src2(self):
        rs = ReservationStation()
        _rs_issue(rs, src1_val=100, src2_val=200, imm=42, alu_src=1, rob_tag=2)
        _cycle(rs)
        _rs_issue_off(rs)

        rs.evaluate()
        assert rs["exec_src2"] == 42, "alu_src=1 must override the register src2"

    def test_an_entry_waiting_on_a_tag_is_not_offered(self):
        rs = ReservationStation()
        _rs_issue(rs, src1_ready=0, src1_tag=5, src2_val=7, rob_tag=1)
        _cycle(rs)
        _rs_issue_off(rs)

        rs.evaluate()
        assert rs["exec_valid"] == 0

    def test_the_selected_entry_is_freed_on_the_clock_edge(self):
        rs = ReservationStation()
        _rs_issue(rs, src1_val=1, src2_val=2, rob_tag=1)
        _cycle(rs)
        _rs_issue_off(rs)

        rs.evaluate()
        assert rs["exec_valid"] == 1
        rs.rising_edge()

        rs.evaluate()
        assert rs["exec_valid"] == 0
        assert all(e["valid"] == 0 for e in rs.get_state()["entries"])

    def test_oldest_ready_entry_wins_selection(self):
        """Insertion order (age), not slot index, decides who executes."""
        rs = ReservationStation()
        _rs_issue(rs, src1_ready=0, src1_tag=9, src2_val=1, rob_tag=1)   # age 0
        _cycle(rs)
        _rs_issue(rs, src1_val=5, src2_val=6, rob_tag=2)                 # age 1
        _rs_cdb(rs, tag=9, value=0x77)     # wakes the OLDER entry this same edge
        _cycle(rs)
        _rs_issue_off(rs)
        _rs_cdb_off(rs)

        rs.evaluate()
        assert rs["exec_rob_tag"] == 1, "oldest ready entry must go first"
        assert rs["exec_src1"] == 0x77
        rs.rising_edge()

        rs.evaluate()
        assert rs["exec_rob_tag"] == 2

    def test_get_state_exposes_the_entry_table(self):
        rs = ReservationStation()
        _rs_issue(rs, op=6, src1_val=0x11, src2_ready=0, src2_tag=4,
                  rob_tag=2, imm=0x99, alu_src=1)
        _cycle(rs)
        _rs_issue_off(rs)
        rs.evaluate()

        state = rs.get_state()
        assert len(state["entries"]) == RS_SIZE
        e = state["entries"][0]
        assert e["valid"] == 1
        assert e["op"] == 6
        assert e["src1_ready"] == 1
        assert e["src1_val"] == "0x00000011"
        assert e["src2_ready"] == 0
        assert e["src2_tag"] == 4
        assert e["rob_tag"] == 2
        assert e["imm"] == "0x00000099"
        assert e["alu_src"] == 1


class TestReservationStationCdbWakeup:

    def test_cdb_broadcast_resolves_a_pending_src1(self):
        rs = ReservationStation()
        _rs_issue(rs, op=2, src1_ready=0, src1_tag=5, src2_val=7, rob_tag=1)
        _cycle(rs)
        _rs_issue_off(rs)

        rs.evaluate()
        assert rs["exec_valid"] == 0

        _rs_cdb(rs, tag=5, value=0x1234)
        rs.rising_edge()
        _rs_cdb_off(rs)

        rs.evaluate()
        assert rs["exec_valid"] == 1
        assert rs["exec_src1"] == 0x1234
        assert rs["exec_src2"] == 7

    def test_cdb_broadcast_resolves_a_pending_src2(self):
        rs = ReservationStation()
        _rs_issue(rs, src1_val=2, src2_ready=0, src2_tag=8, rob_tag=1)
        _cycle(rs)
        _rs_issue_off(rs)

        _rs_cdb(rs, tag=8, value=0x66)
        rs.rising_edge()
        _rs_cdb_off(rs)

        rs.evaluate()
        assert rs["exec_valid"] == 1
        assert rs["exec_src2"] == 0x66

    def test_one_broadcast_resolves_both_sources_when_they_share_a_tag(self):
        rs = ReservationStation()
        _rs_issue(rs, src1_ready=0, src1_tag=8, src2_ready=0, src2_tag=8, rob_tag=1)
        _cycle(rs)
        _rs_issue_off(rs)

        _rs_cdb(rs, tag=8, value=5)
        rs.rising_edge()
        _rs_cdb_off(rs)

        rs.evaluate()
        assert (rs["exec_src1"], rs["exec_src2"]) == (5, 5)

    def test_non_matching_broadcast_leaves_the_entry_waiting(self):
        rs = ReservationStation()
        _rs_issue(rs, src1_ready=0, src1_tag=5, src2_val=1, rob_tag=1)
        _cycle(rs)
        _rs_issue_off(rs)

        _rs_cdb(rs, tag=6, value=0xDEAD)
        rs.rising_edge()
        _rs_cdb_off(rs)

        rs.evaluate()
        assert rs["exec_valid"] == 0

    def test_entry_issued_on_the_same_edge_as_its_broadcast_captures_it_inline(self):
        """Without the inline snoop the new entry would miss its own wakeup and
        deadlock until the (never-repeated) tag was rebroadcast."""
        rs = ReservationStation()
        _rs_issue(rs, src1_ready=0, src1_tag=12, src2_val=3, rob_tag=4)
        _rs_cdb(rs, tag=12, value=0xABC)
        _cycle(rs)
        _rs_issue_off(rs)
        _rs_cdb_off(rs)

        rs.evaluate()
        assert rs["exec_valid"] == 1
        assert rs["exec_src1"] == 0xABC

    def test_the_inline_snoop_also_covers_src2_of_the_newly_issued_entry(self):
        rs = ReservationStation()
        _rs_issue(rs, src1_val=3, src2_ready=0, src2_tag=13, rob_tag=6)
        _rs_cdb(rs, tag=13, value=0x5A5A)
        _cycle(rs)
        _rs_issue_off(rs)
        _rs_cdb_off(rs)

        rs.evaluate()
        assert rs["exec_valid"] == 1
        assert rs["exec_src1"] == 3
        assert rs["exec_src2"] == 0x5A5A


class TestReservationStationBackpressure:

    def test_station_reports_full_after_RS_SIZE_unresolved_entries(self):
        rs = ReservationStation()
        for i in range(RS_SIZE):
            _rs_issue(rs, src1_ready=0, src1_tag=20, src2_ready=0, src2_tag=20, rob_tag=i)
            _cycle(rs)
        _rs_issue_off(rs)

        rs.evaluate()
        assert rs["full"] == 1
        assert rs.get_state()["full"] is True
        assert rs._free_slot() is None

    def test_issue_into_a_full_station_is_dropped(self):
        rs = ReservationStation()
        for i in range(RS_SIZE):
            _rs_issue(rs, src1_ready=0, src1_tag=20, src2_ready=0, src2_tag=20, rob_tag=i)
            _cycle(rs)

        _rs_issue(rs, src1_ready=0, src1_tag=21, src2_ready=0, src2_tag=21, rob_tag=9)
        _cycle(rs)
        _rs_issue_off(rs)

        tags = [e["rob_tag"] for e in rs.get_state()["entries"]]
        assert sorted(tags) == list(range(RS_SIZE)), "the overflow entry must be dropped"

    def test_freeing_an_entry_clears_full(self):
        rs = ReservationStation()
        for i in range(RS_SIZE - 1):
            _rs_issue(rs, src1_ready=0, src1_tag=20, src2_ready=0, src2_tag=20, rob_tag=i)
            _cycle(rs)
        _rs_issue(rs, src1_val=1, src2_val=2, rob_tag=RS_SIZE - 1)   # this one is ready
        _cycle(rs)
        _rs_issue_off(rs)

        rs.evaluate()
        assert rs["full"] == 1
        rs.rising_edge()      # the ready entry issues out and frees its slot

        rs.evaluate()
        assert rs["full"] == 0


class TestReservationStationLoadStoreOrdering:
    """Loads must not read dmem before an OLDER store has committed to it
    (there is no store->load forwarding in the OoO datapath)."""

    def _load_entry(self, rob_tag):
        rs = ReservationStation()
        _rs_issue(rs, rob_tag=rob_tag, is_load=1, src1_val=8)
        _cycle(rs)
        _rs_issue_off(rs)
        return rs

    def test_load_is_held_while_an_older_store_is_uncommitted(self):
        rs = self._load_entry(rob_tag=1)
        rs["rob_head"] = 0
        rs["store_mask"] = 1 << 0          # store in slot 0 == older than tag 1
        rs.evaluate()
        assert rs["exec_valid"] == 0

    def test_load_issues_once_the_older_store_commits(self):
        rs = self._load_entry(rob_tag=1)
        rs["rob_head"] = 0
        rs["store_mask"] = 1 << 0
        rs.evaluate()
        assert rs["exec_valid"] == 0

        rs["store_mask"] = 0
        rs.evaluate()
        assert rs["exec_valid"] == 1
        assert rs["exec_is_load"] == 1

    def test_a_younger_store_never_blocks_a_load(self):
        """Otherwise the pair would deadlock: neither could reach the head."""
        rs = self._load_entry(rob_tag=1)
        rs["rob_head"] = 0
        rs["store_mask"] = 1 << 3          # slot 3 is YOUNGER than the load at 1
        rs.evaluate()
        assert rs["exec_valid"] == 1

    def test_age_comparison_is_relative_to_the_rob_head_across_the_wrap(self):
        rs = self._load_entry(rob_tag=1)
        rs["rob_head"] = 6
        # head=6 -> slot 7 has age 1, the load (tag 1) has age (1-6)%8 = 3.
        rs["store_mask"] = 1 << 7
        rs.evaluate()
        assert rs["exec_valid"] == 0, "wrapped-older store must still block"

        # slot 2 has age (2-6)%8 = 4, i.e. YOUNGER than the load.
        rs["store_mask"] = 1 << 2
        rs.evaluate()
        assert rs["exec_valid"] == 1

    def test_a_store_in_the_loads_own_slot_does_not_block_it(self):
        rs = self._load_entry(rob_tag=0)
        rs["rob_head"] = 0
        rs["store_mask"] = 1 << 0          # same age as the load
        rs.evaluate()
        assert rs["exec_valid"] == 1

    def test_a_non_load_entry_ignores_pending_stores_entirely(self):
        rs = ReservationStation()
        _rs_issue(rs, rob_tag=1, is_load=0, src1_val=8)
        _cycle(rs)
        _rs_issue_off(rs)

        rs["rob_head"] = 0
        rs["store_mask"] = 1 << 0
        rs.evaluate()
        assert rs["exec_valid"] == 1, "only loads observe store ordering"

    def test_an_empty_store_mask_short_circuits(self):
        rs = self._load_entry(rob_tag=3)
        rs["rob_head"] = 0
        rs["store_mask"] = 0
        rs.evaluate()
        assert rs["exec_valid"] == 1


class TestReservationStationFlush:

    def test_flush_invalidates_every_entry_and_resets_the_age_counter(self):
        rs = ReservationStation()
        _rs_issue(rs, rob_tag=2, src1_val=1, src2_val=2)
        _cycle(rs)
        _rs_issue_off(rs)
        rs.evaluate()
        assert rs["exec_valid"] == 1

        rs["flush"] = 1
        rs.rising_edge()
        rs["flush"] = 0

        rs.evaluate()
        assert rs["exec_valid"] == 0
        assert rs["full"] == 0
        assert all(e["valid"] == 0 for e in rs.get_state()["entries"])
        assert rs._age_counter == 0
        assert rs._selected_idx is None

    def test_flush_outranks_a_same_edge_issue_and_cdb_broadcast(self):
        rs = ReservationStation()
        rs["flush"] = 1
        _rs_issue(rs, rob_tag=3, src1_val=1, src2_val=2)
        _rs_cdb(rs, tag=1, value=0xFF)
        rs.rising_edge()
        rs["flush"] = 0
        _rs_issue_off(rs)
        _rs_cdb_off(rs)

        rs.evaluate()
        assert rs["exec_valid"] == 0

    def test_station_is_reusable_after_a_flush(self):
        rs = ReservationStation()
        _rs_issue(rs, rob_tag=2, src1_val=1, src2_val=2)
        _cycle(rs)
        _rs_issue_off(rs)
        rs["flush"] = 1
        rs.rising_edge()
        rs["flush"] = 0

        _rs_issue(rs, rob_tag=5, src1_val=8, src2_val=9)
        _cycle(rs)
        _rs_issue_off(rs)

        rs.evaluate()
        assert rs["exec_valid"] == 1
        assert rs["exec_rob_tag"] == 5


# ═══════════════════════════════════════════════════════════════════
# RegisterAliasTable
# ═══════════════════════════════════════════════════════════════════

class TestRegisterAliasTableRename:

    def test_unmapped_registers_read_ready_with_no_tag(self):
        rat = RegisterAliasTable()
        rat["rs1_arch"] = 1
        rat["rs2_arch"] = 2
        rat.evaluate()
        assert (rat["rs1_ready"], rat["rs1_tag"]) == (1, 0)
        assert (rat["rs2_ready"], rat["rs2_tag"]) == (1, 0)

    def test_allocating_an_alias_makes_the_register_not_ready_with_its_tag(self):
        rat = RegisterAliasTable()
        rat["rd_arch"] = 1
        rat["alloc_en"] = 1
        rat["alloc_tag"] = 5
        rat.rising_edge()
        rat["alloc_en"] = 0

        rat["rs1_arch"] = 1
        rat["rs2_arch"] = 2
        rat.evaluate()
        assert (rat["rs1_ready"], rat["rs1_tag"]) == (0, 5)
        assert rat["rs2_ready"] == 1, "an unrelated register is unaffected"
        assert rat.get_state()["active_aliases"] == {"x1": 5}

    def test_alloc_en_low_does_not_create_a_mapping(self):
        rat = RegisterAliasTable()
        rat["rd_arch"] = 4
        rat["alloc_en"] = 0
        rat["alloc_tag"] = 3
        rat.rising_edge()

        rat["rs1_arch"] = 4
        rat.evaluate()
        assert rat["rs1_ready"] == 1

    def test_reallocating_the_same_register_overwrites_the_alias(self):
        rat = RegisterAliasTable()
        rat["rd_arch"] = 3
        rat["alloc_en"] = 1
        rat["alloc_tag"] = 2
        rat.rising_edge()
        rat["alloc_tag"] = 6
        rat.rising_edge()
        rat["alloc_en"] = 0

        rat["rs1_arch"] = 3
        rat.evaluate()
        assert rat["rs1_tag"] == 6, "the newest writer owns the register"

    def test_get_state_lists_one_mapping_row_per_architectural_register(self):
        rat = RegisterAliasTable()
        rat["rd_arch"] = 7
        rat["alloc_en"] = 1
        rat["alloc_tag"] = 3
        rat.rising_edge()
        rat["alloc_en"] = 0
        rat["rs1_arch"] = 7
        rat.evaluate()

        state = rat.get_state()
        assert len(state["mappings"]) == 32
        assert state["mappings"][7] == {"reg": "x7", "pending": True, "tag": 3}
        assert state["mappings"][8] == {"reg": "x8", "pending": False, "tag": None}
        assert state["rs1_ready"] == 0
        assert state["rs1_tag"] == 3


class TestRegisterAliasTableCommit:

    def test_commit_frees_the_mapping_so_lookups_fall_back_to_the_regfile(self):
        rat = RegisterAliasTable()
        rat["rd_arch"] = 1
        rat["alloc_en"] = 1
        rat["alloc_tag"] = 5
        rat.rising_edge()
        rat["alloc_en"] = 0

        rat["commit_en"] = 1
        rat["commit_rd"] = 1
        rat["commit_tag"] = 5
        rat.rising_edge()
        rat["commit_en"] = 0

        rat["rs1_arch"] = 1
        rat.evaluate()
        assert rat["rs1_ready"] == 1
        assert rat.get_state()["active_aliases"] == {}

    def test_a_same_cycle_commit_is_visible_to_the_combinational_lookup(self):
        """The regfile is write-through, so a register being retired THIS cycle
        must read ready — otherwise the dispatching instruction would wait on a
        tag that is already gone."""
        rat = RegisterAliasTable()
        rat["rd_arch"] = 1
        rat["alloc_en"] = 1
        rat["alloc_tag"] = 5
        rat.rising_edge()
        rat["alloc_en"] = 0

        rat["rs1_arch"] = 1
        rat.evaluate()
        assert rat["rs1_ready"] == 0

        rat["commit_en"] = 1
        rat["commit_rd"] = 1
        rat["commit_tag"] = 5
        rat.evaluate()
        assert rat["rs1_ready"] == 1, "same-cycle commit must clear the dependency"

    def test_a_stale_commit_must_not_free_a_newer_alias(self):
        """x3 renamed to tag 2 then re-renamed to tag 6; tag 2 retiring must
        leave the tag-6 alias intact or the younger consumer reads garbage."""
        rat = RegisterAliasTable()
        rat["rd_arch"] = 3
        rat["alloc_en"] = 1
        rat["alloc_tag"] = 2
        rat.rising_edge()
        rat["alloc_tag"] = 6
        rat.rising_edge()
        rat["alloc_en"] = 0

        rat["commit_en"] = 1
        rat["commit_rd"] = 3
        rat["commit_tag"] = 2       # the STALE tag
        rat["rs1_arch"] = 3
        rat.evaluate()
        assert (rat["rs1_ready"], rat["rs1_tag"]) == (0, 6)
        rat.rising_edge()
        rat["commit_en"] = 0

        rat.evaluate()
        assert (rat["rs1_ready"], rat["rs1_tag"]) == (0, 6)
        assert rat.get_state()["active_aliases"] == {"x3": 6}

    def test_commit_of_an_unmapped_register_is_a_no_op(self):
        rat = RegisterAliasTable()
        rat["commit_en"] = 1
        rat["commit_rd"] = 9
        rat["commit_tag"] = 1
        rat.rising_edge()
        rat["commit_en"] = 0

        rat["rs1_arch"] = 9
        rat.evaluate()
        assert rat["rs1_ready"] == 1


class TestRegisterAliasTableZeroRegisterParameterization:
    """RISC-V x0 and ARM XZR(31) are hardwired zero and must never be renamed;
    x86 has NO zero register, so index 0 (EAX) is a real renameable GPR."""

    @pytest.mark.parametrize("kwargs,zero_idx,label", [
        ({}, 0, "riscv-x0"),
        ({"zero_reg_index": 31}, 31, "arm-XZR"),
        ({"has_zero_reg": False}, None, "x86-none"),
    ])
    def test_only_the_configured_zero_register_resists_renaming(self, kwargs, zero_idx, label):
        rat = RegisterAliasTable(**kwargs)
        for reg in (0, 31):
            rat["rd_arch"] = reg
            rat["alloc_en"] = 1
            rat["alloc_tag"] = 4
            rat.rising_edge()
        rat["alloc_en"] = 0

        rat["rs1_arch"] = 0
        rat["rs2_arch"] = 31
        rat.evaluate()

        expected_ready = {0: rat["rs1_ready"], 31: rat["rs2_ready"]}
        for reg in (0, 31):
            if reg == zero_idx:
                assert expected_ready[reg] == 1, f"{label}: x{reg} must stay unrenamed"
            else:
                assert expected_ready[reg] == 0, f"{label}: x{reg} is a real GPR"

        aliases = rat.get_state()["active_aliases"]
        if zero_idx is None:
            assert aliases == {"x0": 4, "x31": 4}, label
        else:
            assert f"x{zero_idx}" not in aliases, label

    def test_x86_rat_renames_index_zero_because_eax_is_a_real_register(self):
        rat = RegisterAliasTable(has_zero_reg=False)
        rat["rd_arch"] = 0
        rat["alloc_en"] = 1
        rat["alloc_tag"] = 7
        rat.rising_edge()
        rat["alloc_en"] = 0

        rat["rs1_arch"] = 0
        rat.evaluate()
        assert (rat["rs1_ready"], rat["rs1_tag"]) == (0, 7)

    def test_commit_to_the_zero_register_is_ignored(self):
        rat = RegisterAliasTable(zero_reg_index=31)
        rat["rd_arch"] = 5
        rat["alloc_en"] = 1
        rat["alloc_tag"] = 3
        rat.rising_edge()
        rat["alloc_en"] = 0

        rat["commit_en"] = 1
        rat["commit_rd"] = 31
        rat["commit_tag"] = 3
        rat["rs1_arch"] = 31
        rat.evaluate()
        assert rat["rs1_ready"] == 1
        rat.rising_edge()

        rat["rs1_arch"] = 5
        rat.evaluate()
        assert rat["rs1_tag"] == 3, "x5's alias must be untouched"


class TestRegisterAliasTableFlush:

    def test_flush_clears_every_mapping(self):
        rat = RegisterAliasTable()
        for reg, tag in ((9, 7), (12, 2), (30, 5)):
            rat["rd_arch"] = reg
            rat["alloc_en"] = 1
            rat["alloc_tag"] = tag
            rat.rising_edge()
        rat["alloc_en"] = 0
        assert len(rat.get_state()["active_aliases"]) == 3

        rat["flush"] = 1
        rat.rising_edge()
        rat["flush"] = 0

        assert rat.get_state()["active_aliases"] == {}
        for reg in (9, 12, 30):
            rat["rs1_arch"] = reg
            rat.evaluate()
            assert rat["rs1_ready"] == 1

    def test_flush_outranks_a_same_edge_allocation(self):
        rat = RegisterAliasTable()
        rat["flush"] = 1
        rat["rd_arch"] = 4
        rat["alloc_en"] = 1
        rat["alloc_tag"] = 6
        rat.rising_edge()
        rat["flush"] = 0
        rat["alloc_en"] = 0

        rat["rs1_arch"] = 4
        rat.evaluate()
        assert rat["rs1_ready"] == 1


# ═══════════════════════════════════════════════════════════════════
# Small OoO datapath selectors (previously ZERO tests)
# ═══════════════════════════════════════════════════════════════════

class TestCdbValueSelect:
    """Pins the FIXED bug: the CDB used to broadcast a load's ALU result (its
    ADDRESS) to waiting RS entries instead of the loaded DATA."""

    def test_non_load_broadcasts_the_alu_result(self):
        sel = CdbValueSelect()
        sel["is_load"] = 0
        sel["alu_value"] = 0x1000
        sel["mem_value"] = 0xBEEF
        sel.evaluate()
        assert sel["value"] == 0x1000

    def test_load_broadcasts_the_memory_data_not_the_computed_address(self):
        sel = CdbValueSelect()
        sel["is_load"] = 1
        sel["alu_value"] = 0x1000       # the load's address
        sel["mem_value"] = 0xBEEF       # the loaded data
        sel.evaluate()
        assert sel["value"] == 0xBEEF, "a dependent must wake with DATA, not the address"

    def test_get_state_reports_the_selected_value(self):
        sel = CdbValueSelect()
        sel["is_load"] = 1
        sel["mem_value"] = 0xBEEF
        sel.evaluate()
        assert sel.get_state() == {"is_load": 1, "value": "0x0000beef"}

    def test_load_data_reaches_a_dependent_through_a_real_rs_wakeup(self):
        """Wire CdbValueSelect into the RS snoop path, as the OoO preset does."""
        sel = CdbValueSelect()
        rs = ReservationStation()
        _rs_issue(rs, src1_ready=0, src1_tag=2, src2_val=1, rob_tag=3)
        _cycle(rs)
        _rs_issue_off(rs)

        sel["is_load"] = 1
        sel["alu_value"] = 0x2000       # address computed by the ALU
        sel["mem_value"] = 0x4242       # data returned by dmem
        sel.evaluate()

        _rs_cdb(rs, tag=2, value=sel["value"])
        rs.rising_edge()
        _rs_cdb_off(rs)

        rs.evaluate()
        assert rs["exec_src1"] == 0x4242


class TestCommittedFlags:

    def test_flags_start_cleared(self):
        cf = CommittedFlags()
        cf.evaluate()
        assert (cf["zero_out"], cf["result_out"]) == (0, 0)
        assert cf.get_state() == {"zero": 0, "result": "0x00000000"}

    def test_committing_a_zero_result_sets_the_zero_flag(self):
        cf = CommittedFlags()
        cf["commit_write_flags"] = 1
        cf["commit_value"] = 0
        cf.rising_edge()
        cf.evaluate()
        assert (cf["zero_out"], cf["result_out"]) == (1, 0)

    def test_committing_a_nonzero_result_clears_the_zero_flag(self):
        cf = CommittedFlags()
        cf["commit_write_flags"] = 1
        cf["commit_value"] = 5
        cf.rising_edge()
        cf.evaluate()
        assert (cf["zero_out"], cf["result_out"]) == (0, 5)

    def test_a_negative_difference_is_latched_as_its_32_bit_pattern(self):
        """BranchResolutionUnit in compare_mode='sub' reads the sign of result."""
        cf = CommittedFlags()
        cf["commit_write_flags"] = 1
        cf["commit_value"] = (-3) & 0xFFFFFFFF
        cf.rising_edge()
        cf.evaluate()
        assert cf["result_out"] == 0xFFFFFFFD
        assert cf["zero_out"] == 0

    def test_a_non_flag_setting_commit_leaves_the_latched_flags_alone(self):
        cf = CommittedFlags()
        cf["commit_write_flags"] = 1
        cf["commit_value"] = 5
        cf.rising_edge()

        cf["commit_write_flags"] = 0
        cf["commit_value"] = 0          # would set zero if it were latched
        for _ in range(3):
            cf.rising_edge()
        cf.evaluate()
        assert (cf["zero_out"], cf["result_out"]) == (0, 5)
        assert cf.get_state() == {"zero": 0, "result": "0x00000005"}

    def test_the_latest_flag_setter_wins(self):
        cf = CommittedFlags()
        for value, expect_zero in ((7, 0), (0, 1), (9, 0)):
            cf["commit_write_flags"] = 1
            cf["commit_value"] = value
            cf.rising_edge()
            cf.evaluate()
            assert cf["zero_out"] == expect_zero
            assert cf["result_out"] == value


class TestBranchFlagMux:

    def test_select_high_passes_the_self_comparing_branch_inputs(self):
        """select == the decoder's sets_flags bit: 1 => CBNZ/CBZ comparator."""
        mux = BranchFlagMux()
        mux["a_zero"], mux["a_result"] = 1, 0xAAA
        mux["b_zero"], mux["b_result"] = 0, 0xBBB
        mux["select"] = 1
        mux.evaluate()
        assert (mux["zero_out"], mux["result_out"]) == (1, 0xAAA)
        assert mux.get_state() == {"select": 1, "zero": 1, "result": "0x00000aaa"}

    def test_select_low_passes_the_committed_flags(self):
        mux = BranchFlagMux()
        mux["a_zero"], mux["a_result"] = 1, 0xAAA
        mux["b_zero"], mux["b_result"] = 0, 0xBBB
        mux["select"] = 0
        mux.evaluate()
        assert (mux["zero_out"], mux["result_out"]) == (0, 0xBBB)
        assert mux.get_state() == {"select": 0, "zero": 0, "result": "0x00000bbb"}

    def test_the_mux_is_purely_combinational(self):
        mux = BranchFlagMux()
        mux["a_zero"], mux["b_zero"] = 1, 0
        for select, expected in ((1, 1), (0, 0), (1, 1)):
            mux["select"] = select
            mux.evaluate()
            assert mux["zero_out"] == expected


class TestPcOperandSelect:

    def test_use_pc_drives_the_pc_as_a_ready_operand_with_no_tag(self):
        """AUIPC's src1 is the instruction PC — a constant with no producer, so
        it must be marked ready even when the RAT reports a pending writer."""
        sel = PcOperandSelect()
        sel["pc"] = 0x400
        sel["reg_val"] = 0x99
        sel["reg_ready"] = 0
        sel["reg_tag"] = 6
        sel["use_pc"] = 1
        sel.evaluate()
        assert (sel["src_val"], sel["src_ready"], sel["src_tag"]) == (0x400, 1, 0)
        assert sel.get_state() == {"use_pc": 1, "src_val": "0x00000400", "src_ready": 1}

    def test_use_pc_low_passes_the_regfile_value_and_rat_readiness_through(self):
        sel = PcOperandSelect()
        sel["pc"] = 0x400
        sel["reg_val"] = 0x99
        sel["reg_ready"] = 0
        sel["reg_tag"] = 6
        sel["use_pc"] = 0
        sel.evaluate()
        assert (sel["src_val"], sel["src_ready"], sel["src_tag"]) == (0x99, 0, 6)

    def test_use_pc_low_with_a_ready_register_passes_ready_through(self):
        sel = PcOperandSelect()
        sel["use_pc"] = 0
        sel["reg_val"] = 0x99
        sel["reg_ready"] = 1
        sel["reg_tag"] = 0
        sel.evaluate()
        assert (sel["src_val"], sel["src_ready"], sel["src_tag"]) == (0x99, 1, 0)


class TestStoreCommitUnit:

    def test_idle_unit_drives_no_write(self):
        scu = StoreCommitUnit()
        scu["commit_addr"] = 0x40
        scu["commit_data"] = 0x1234
        scu.evaluate()
        assert scu["dmem_wen"] == 0
        assert scu["dmem_waddr"] == 0
        assert scu["dmem_wdata"] == 0
        assert scu.get_state() == {"wen": 0, "addr": "0x00000000", "data": "0x00000000"}

    def test_a_commit_that_is_not_a_store_drives_no_write(self):
        scu = StoreCommitUnit()
        scu["commit_en"] = 1
        scu["commit_is_store"] = 0
        scu["commit_addr"] = 0x40
        scu["commit_data"] = 0x1234
        scu.evaluate()
        assert scu["dmem_wen"] == 0

    def test_committing_a_store_drives_address_data_and_enable(self):
        scu = StoreCommitUnit()
        scu["commit_en"] = 1
        scu["commit_is_store"] = 1
        scu["commit_addr"] = 0x40
        scu["commit_data"] = 0x1234
        scu.evaluate()
        assert scu["dmem_wen"] == 1
        assert scu["dmem_waddr"] == 0x40
        assert scu["dmem_wdata"] == 0x1234
        assert scu.get_state() == {"wen": 1, "addr": "0x00000040", "data": "0x00001234"}

    def test_a_store_to_byte_address_zero_still_asserts_write_enable(self):
        """dmem.waddr_valid exists precisely so address 0 is not mistaken for
        'no write' — a regression that used to silently drop the store."""
        scu = StoreCommitUnit()
        scu["commit_en"] = 1
        scu["commit_is_store"] = 1
        scu["commit_addr"] = 0
        scu["commit_data"] = 0x99
        scu.evaluate()
        assert scu["dmem_wen"] == 1
        assert scu["dmem_waddr"] == 0
        assert scu["dmem_wdata"] == 0x99

    def test_deasserting_commit_clears_the_write_ports(self):
        scu = StoreCommitUnit()
        scu["commit_en"] = 1
        scu["commit_is_store"] = 1
        scu["commit_addr"] = 0x40
        scu["commit_data"] = 0x1234
        scu.evaluate()
        scu["commit_en"] = 0
        scu.evaluate()
        assert (scu["dmem_wen"], scu["dmem_waddr"], scu["dmem_wdata"]) == (0, 0, 0)

    def test_rob_store_commit_pair_drives_the_write_ports_end_to_end(self):
        rob = ReorderBuffer()
        scu = StoreCommitUnit()

        _rob_dispatch(rob, rd=0, is_store=1, reg_write=0, store_data_reg=7)
        _cycle(rob)
        _rob_dispatch_off(rob)
        _rob_complete(rob, tag=0, value=0x2C)   # the store address
        _cycle(rob)
        _rob_complete_off(rob)

        rob.evaluate()
        scu["commit_en"] = rob["commit_en"]
        scu["commit_is_store"] = rob["commit_is_store"]
        scu["commit_addr"] = rob["commit_value"]
        scu["commit_data"] = 0xCAFE             # regfile rs3 read at commit
        scu.evaluate()

        assert rob["commit_store_data_reg"] == 7
        assert scu["dmem_wen"] == 1
        assert scu["dmem_waddr"] == 0x2C
        assert scu["dmem_wdata"] == 0xCAFE


# ═══════════════════════════════════════════════════════════════════
# ForwardingUnit (5-stage pipeline)
# ═══════════════════════════════════════════════════════════════════

class TestForwardingUnit:

    def _unit(self, **kw):
        fu = ForwardingUnit(**kw)
        fu["id_ex_rs1"] = 1
        fu["id_ex_rs2"] = 2
        fu["rs1_data_in"] = 0xAA
        fu["rs2_data_in"] = 0xBB
        return fu

    def test_no_matching_producer_uses_the_register_file_values(self):
        fu = self._unit()
        fu.evaluate()
        assert (fu["forward_a"], fu["forward_b"]) == (0, 0)
        assert (fu["rs1_data_out"], fu["rs2_data_out"]) == (0xAA, 0xBB)
        assert fu.get_state() == {"forward_a": "REG", "forward_b": "REG"}

    def test_ex_mem_match_forwards_the_alu_result(self):
        fu = self._unit()
        fu["ex_mem_rd"] = 1
        fu["ex_mem_reg_write"] = 1
        fu["ex_mem_alu_result"] = 0x11
        fu.evaluate()
        assert fu["forward_a"] == 1
        assert fu["rs1_data_out"] == 0x11
        assert fu["forward_b"] == 0, "rs2 does not match, so it stays on the regfile"
        assert fu.get_state()["forward_a"] == "EX/MEM"

    def test_mem_wb_match_forwards_the_writeback_data(self):
        fu = self._unit()
        fu["mem_wb_rd"] = 2
        fu["mem_wb_reg_write"] = 1
        fu["mem_wb_data"] = 0x22
        fu.evaluate()
        assert fu["forward_b"] == 2
        assert fu["rs2_data_out"] == 0x22
        assert fu.get_state()["forward_b"] == "MEM/WB"

    def test_ex_mem_beats_mem_wb_when_both_write_the_same_register(self):
        """The nearer (younger) producer holds the correct value."""
        fu = self._unit()
        fu["ex_mem_rd"] = 1
        fu["ex_mem_reg_write"] = 1
        fu["ex_mem_alu_result"] = 0x11
        fu["mem_wb_rd"] = 1
        fu["mem_wb_reg_write"] = 1
        fu["mem_wb_data"] = 0x22
        fu.evaluate()
        assert fu["forward_a"] == 1
        assert fu["rs1_data_out"] == 0x11

        fu["ex_mem_reg_write"] = 0      # only the older producer remains
        fu.evaluate()
        assert fu["forward_a"] == 2
        assert fu["rs1_data_out"] == 0x22

    def test_reg_write_low_never_forwards(self):
        fu = self._unit()
        fu["ex_mem_rd"] = 1
        fu["ex_mem_reg_write"] = 0
        fu["ex_mem_alu_result"] = 0x11
        fu["mem_wb_rd"] = 1
        fu["mem_wb_reg_write"] = 0
        fu["mem_wb_data"] = 0x22
        fu.evaluate()
        assert fu["forward_a"] == 0
        assert fu["rs1_data_out"] == 0xAA

    def test_both_operands_forward_independently(self):
        fu = self._unit()
        fu["ex_mem_rd"] = 1
        fu["ex_mem_reg_write"] = 1
        fu["ex_mem_alu_result"] = 0x11
        fu["mem_wb_rd"] = 2
        fu["mem_wb_reg_write"] = 1
        fu["mem_wb_data"] = 0x22
        fu.evaluate()
        assert (fu["forward_a"], fu["forward_b"]) == (1, 2)
        assert (fu["rs1_data_out"], fu["rs2_data_out"]) == (0x11, 0x22)

    @pytest.mark.parametrize("zero_kwargs,zero_idx,real_idx,label", [
        ({}, 0, 31, "riscv: x0 hardwired, x31 real"),
        ({"zero_reg_index": 31}, 31, 0, "arm: XZR=31 hardwired, x0 real"),
    ])
    def test_the_hardwired_zero_register_is_never_a_forwarding_source(
            self, zero_kwargs, zero_idx, real_idx, label):
        fu = ForwardingUnit(**zero_kwargs)
        fu["id_ex_rs1"] = zero_idx
        fu["rs1_data_in"] = 0
        fu["ex_mem_rd"] = zero_idx
        fu["ex_mem_reg_write"] = 1
        fu["ex_mem_alu_result"] = 0x99
        fu.evaluate()
        assert fu["forward_a"] == 0, label
        assert fu["rs1_data_out"] == 0

        # The OTHER index is a real GPR on this ISA and must forward normally.
        fu["id_ex_rs1"] = real_idx
        fu["ex_mem_rd"] = real_idx
        fu.evaluate()
        assert fu["forward_a"] == 1, label
        assert fu["rs1_data_out"] == 0x99

    def test_x86_has_no_zero_register_so_index_zero_forwards(self):
        fu = ForwardingUnit(zero_reg_index=None)
        fu["id_ex_rs1"] = 0                 # EAX
        fu["rs1_data_in"] = 0
        fu["ex_mem_rd"] = 0
        fu["ex_mem_reg_write"] = 1
        fu["ex_mem_alu_result"] = 0x99
        fu.evaluate()
        assert fu["forward_a"] == 1
        assert fu["rs1_data_out"] == 0x99


# ═══════════════════════════════════════════════════════════════════
# HazardDetector (5-stage pipeline)
# ═══════════════════════════════════════════════════════════════════

class TestHazardDetector:

    def _det(self):
        det = HazardDetector()
        det["id_ex_mem_read"] = 1
        det["id_ex_rd"] = 3
        det["if_id_rs1"] = 9
        det["if_id_rs2"] = 9
        return det

    def test_idle_detector_does_not_stall(self):
        det = HazardDetector()
        det.evaluate()
        assert det["stall"] == 0
        assert det.get_state() == {"stall": "OK"}

    def test_load_destination_matching_rs1_stalls(self):
        det = self._det()
        det["if_id_rs1"] = 3
        det.evaluate()
        assert det["stall"] == 1
        state = det.get_state()
        assert state["stall"] == "STALL"
        assert state["hazard_type"] == "Load-Use"
        assert state["stall_rd"] == 3
        assert state["conflicting_regs"] == [3]

    def test_load_destination_matching_rs2_stalls(self):
        det = self._det()
        det["if_id_rs2"] = 3
        det.evaluate()
        assert det["stall"] == 1
        assert det.get_state()["conflicting_regs"] == [3]

    def test_a_match_on_both_sources_is_reported_once(self):
        det = self._det()
        det["if_id_rs1"] = 3
        det["if_id_rs2"] = 3
        det.evaluate()
        assert det["stall"] == 1
        assert det.get_state()["conflicting_regs"] == [3], \
            "rs2 is only listed when it differs from rs1"

    def test_no_stall_when_the_consumer_does_not_read_the_load_destination(self):
        det = self._det()
        det["if_id_rs1"] = 1
        det["if_id_rs2"] = 2
        det.evaluate()
        assert det["stall"] == 0

    def test_no_stall_when_the_ex_stage_instruction_is_not_a_load(self):
        """A non-load producer is handled by forwarding, not a stall."""
        det = self._det()
        det["id_ex_mem_read"] = 0
        det["if_id_rs1"] = 3
        det.evaluate()
        assert det["stall"] == 0

    def test_a_load_into_register_index_zero_never_stalls(self):
        det = self._det()
        det["id_ex_rd"] = 0
        det["if_id_rs1"] = 0
        det["if_id_rs2"] = 0
        det.evaluate()
        assert det["stall"] == 0

    def test_hazard_detector_does_not_gate_rs2_on_reads_rs2(self):
        """PINS CURRENT (conservative) BEHAVIOUR. Unlike CrossLaneHazardDetector,
        HazardDetector has no `reads_rs2` gate, so an I-type consumer whose rs2
        FIELD merely holds matching immediate bits also stalls. That costs a
        cycle but is never incorrect — pinned so adding the gate is deliberate."""
        det = HazardDetector()
        det["id_ex_mem_read"] = 1
        det["id_ex_rd"] = 5
        det["if_id_rs1"] = 9
        det["if_id_rs2"] = 5     # e.g. RISC-V ADDI x?, x9, 5 -> imm bits land here
        det.evaluate()
        assert det["stall"] == 1


class TestHazardDetectorZeroRegisterGap:
    """REGRESSION PINS for a FIXED bug (2026-08-08).

    ``hazard_detector.py`` used to hardcode ``if rd != 0`` with the comment
    "Don't stall on x0". Every other zero-register-aware component in the repo
    (ForwardingUnit, RegisterAliasTable, CrossLaneHazardDetector,
    WideForwardingUnit, WideRegisterFile) takes a ``zero_reg_index`` /
    ``has_zero_reg`` parameter precisely because index 0 is NOT the zero
    register on every ISA:

        RISC-V  x0  == 0   -> hardcoded 0 happened to be correct
        ARM     XZR == 31  -> X0 is a REAL general-purpose register
        x86     none       -> EAX is register index 0 and is REAL

    So on the ARM and x86 pipeline presets a load into X0 / EAX skipped the
    load-use interlock. The consumer then forwarded from EX/MEM, which for a
    load holds the computed ADDRESS, and read the address instead of the data
    (x86: EDX == 16, the address, instead of 77).

    FIXED by giving HazardDetector the same ``zero_reg_index: int | None = 0``
    parameter and passing 31 from the ARM preset / None from x86
    (sim/isa/arm/presets/pipeline.py, sim/isa/x86/presets/pipeline.py) —
    mirroring the ForwardingUnit construction one line below each.
    """

    X86_LOAD_USE = (
        "MOV ECX, 16\n"
        "MOV EBX, 77\n"
        "MOV [ECX], EBX\n"
        "MOV EAX, [ECX]\n"
        "ADD EDX, EAX\n" + "NOP\n" * 12
    )
    ARM_LOAD_USE = (
        "MOVZ X5, #16\n"
        "MOVZ X6, #77\n"
        "STR X6, [X5]\n"
        "LDR X0, [X5]\n"
        "ADD X7, X0, XZR\n" + "NOP\n" * 12
    )

    def test_riscv_load_use_into_x1_is_correctly_interlocked(self):
        """Control case: on RISC-V index 0 genuinely IS the zero register."""
        asm = ("ADDI x5, x0, 16\n"
               "ADDI x6, x0, 77\n"
               "SW x6, 0(x5)\n"
               "LW x1, 0(x5)\n"
               "ADD x7, x1, x0\n" + "NOP\n" * 12)
        oracle = simulate("riscv", "single_cycle", asm=asm, cycles=80).reg(7)
        assert simulate("riscv", "pipeline", asm=asm, cycles=80).reg(7) == oracle == 77

    def test_x86_load_use_into_eax_stalls(self):
        """x86 has no zero register, so a load into EAX must interlock."""
        oracle = simulate("x86", "single_cycle", asm=self.X86_LOAD_USE, cycles=80).reg("EDX")
        assert oracle == 77
        assert simulate("x86", "pipeline", asm=self.X86_LOAD_USE, cycles=80).reg("EDX") == oracle

    def test_arm_load_use_into_x0_stalls(self):
        """ARM's zero register is XZR (31), so X0 must interlock."""
        oracle = simulate("arm", "single_cycle", asm=self.ARM_LOAD_USE, cycles=80).reg(7)
        assert oracle == 77
        assert simulate("arm", "pipeline", asm=self.ARM_LOAD_USE, cycles=80).reg(7) == oracle

    def test_zero_register_still_skips_the_interlock_per_isa(self):
        """The parameter must actually take effect, not just default to 0.

        A load into each ISA's REAL zero register needs no interlock, because
        the consumer reads 0 regardless. Asserted at the unit level so this
        cannot silently regress into "always stall".
        """
        for zero_idx, rd in ((0, 0), (31, 31), (None, None)):
            det = HazardDetector(zero_reg_index=zero_idx)
            det["id_ex_mem_read"] = 1
            det["if_id_rs1"] = rd if rd is not None else 0
            if rd is None:
                # x86: index 0 is real, so it MUST stall.
                det["id_ex_rd"] = 0
                det.evaluate()
                assert det["stall"] == 1
            else:
                det["id_ex_rd"] = rd
                det.evaluate()
                assert det["stall"] == 0, f"zero_reg_index={zero_idx} should not stall"


# ═══════════════════════════════════════════════════════════════════
# Narrow (1-wide) pipeline registers
# ═══════════════════════════════════════════════════════════════════

class TestIF_ID:

    def test_reset_state_is_a_nop_bubble(self):
        reg = IF_ID()
        reg.evaluate()
        assert reg["instr_out"] == 0x00000013
        assert reg["valid"] == 0
        assert reg.get_state()["valid"] == "BUBBLE"

    def test_custom_nop_encoding_is_used_for_bubbles(self):
        reg = IF_ID(nop_encoding=0xD503201F)     # ARM NOP
        reg.evaluate()
        assert reg["instr_out"] == 0xD503201F

    def test_latching_carries_pc_pc4_instruction_and_prediction(self):
        reg = IF_ID()
        reg["pc_in"] = 0x10
        reg["pc4_in"] = 0x14
        reg["instr_in"] = 0x11223344
        reg["predicted_taken_in"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["pc_out"] == 0x10
        assert reg["pc4_out"] == 0x14
        assert reg["instr_out"] == 0x11223344
        assert reg["valid"] == 1
        assert reg["predicted_taken_out"] == 1
        assert reg.get_state()["valid"] == "VALID"

    def test_the_instruction_path_is_64_bits_wide_for_x86(self):
        """A 32-bit port silently truncated 5-6 byte x86 instructions."""
        reg = IF_ID()
        reg["instr_in"] = 0x1122334455
        reg.rising_edge()
        reg.evaluate()
        assert reg["instr_out"] == 0x1122334455
        assert reg.ports_spec["instr_in"].width == 64
        assert reg.ports_spec["instr_out"].width == 64

    def test_stall_holds_the_latched_instruction(self):
        reg = IF_ID()
        reg["instr_in"] = 0xAAAA
        reg.rising_edge()
        reg["stall"] = 1
        reg["instr_in"] = 0xBBBB
        reg.rising_edge()
        reg.evaluate()
        assert reg["instr_out"] == 0xAAAA
        assert reg["valid"] == 1

    def test_predict_flush_inserts_a_bubble(self):
        reg = IF_ID()
        reg["instr_in"] = 0xAAAA
        reg.rising_edge()
        reg["predict_flush"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["instr_out"] == 0x00000013
        assert reg["valid"] == 0
        assert reg["predicted_taken_out"] == 0

    def test_flush_outranks_both_stall_and_predict_flush(self):
        reg = IF_ID()
        reg["instr_in"] = 0xAAAA
        reg.rising_edge()
        reg["flush"] = 1
        reg["stall"] = 1
        reg["predict_flush"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["valid"] == 0
        assert reg["instr_out"] == 0x00000013
        assert reg["pc_out"] == 0


class TestID_EX:

    def test_hold_is_not_the_same_as_stall(self):
        """The subtle distinction that broke a first attempt at cache stalling.

        `stall` on ID/EX INSERTS A BUBBLE — that is how the load-use interlock
        works: the stalled IF/ID instruction must not enter EX, while the
        instruction already in EX proceeds. So driving `stall` from a
        downstream (MEM) stall DESTROYS the instruction in EX, because nothing
        downstream is consuming it either.

        `hold` (added 2026-08-08) freezes the stage in place instead, which is
        what a MEM-stage cache miss needs. Both ports exist and mean different
        things; conflating them silently loses instructions.
        """
        assert "stall" in ID_EX.ports_spec
        assert "hold" in ID_EX.ports_spec

        # stall -> bubble: the latched instruction is DESTROYED.
        reg = ID_EX()
        reg["rd_in"] = 7
        reg["valid_in"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["rd_out"] == 7
        reg["rd_in"] = 9
        reg["stall"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["rd_out"] == 0 and reg["valid"] == 0, "stall must bubble"

        # hold -> freeze: the latched instruction SURVIVES.
        reg2 = ID_EX()
        reg2["rd_in"] = 7
        reg2["valid_in"] = 1
        reg2.rising_edge()
        reg2.evaluate()
        reg2["rd_in"] = 9
        reg2["hold"] = 1
        reg2.rising_edge()
        reg2.evaluate()
        assert reg2["rd_out"] == 7 and reg2["valid"] == 1, "hold must preserve"

    def test_flush_outranks_hold(self):
        reg = ID_EX()
        reg["rd_in"] = 7
        reg["valid_in"] = 1
        reg.rising_edge()
        reg["hold"] = 1
        reg["flush"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["rd_out"] == 0 and reg["valid"] == 0

    def test_hold_is_inert_when_unwired(self):
        """Every existing preset leaves `hold` undriven, so it must read 0."""
        reg = ID_EX()
        reg["rd_in"] = 3
        reg["valid_in"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["rd_out"] == 3 and reg["valid"] == 1

    def test_reset_state_is_a_bubble(self):
        reg = ID_EX()
        reg.evaluate()
        assert reg["valid"] == 0
        assert reg["rd_out"] == 0

    def test_all_declared_fields_round_trip_through_the_latch(self):
        reg = ID_EX()
        for field in ID_EX._FIELDS:
            reg[f"{field}_in"] = 1
        reg["valid_in"] = 1
        reg.rising_edge()
        reg.evaluate()
        for field in ID_EX._FIELDS:
            assert reg[f"{field}_out"] == 1, f"{field} did not propagate"
        assert reg["valid"] == 1

    def test_latched_control_and_data_are_readable(self):
        reg = ID_EX()
        reg["rd_in"] = 5
        reg["alu_op_in"] = 3
        reg["pc_in"] = 0x20
        reg["mem_read_in"] = 1
        reg["write_flags_in"] = 1
        reg["valid_in"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert (reg["rd_out"], reg["alu_op_out"], reg["mem_read_out"]) == (5, 3, 1)
        assert reg["write_flags_out"] == 1
        assert reg.get_state() == {"pc": "0x00000020", "rd": 5, "alu_op": 3, "valid": "VALID"}

    def test_stall_inserts_a_bubble_rather_than_holding(self):
        """The EX instruction already advanced via EX/MEM, so the stalled ID
        instruction must NOT enter EX — ID/EX becomes a bubble."""
        reg = ID_EX()
        reg["rd_in"] = 5
        reg["valid_in"] = 1
        reg.rising_edge()
        reg["stall"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["valid"] == 0
        assert reg["rd_out"] == 0

    def test_flush_inserts_a_bubble(self):
        reg = ID_EX()
        reg["rd_in"] = 5
        reg["valid_in"] = 1
        reg.rising_edge()
        reg["flush"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["valid"] == 0

    def test_upstream_invalid_clears_valid_but_still_latches_the_fields(self):
        reg = ID_EX()
        reg["rd_in"] = 5
        reg["valid_in"] = 0
        reg.rising_edge()
        reg.evaluate()
        assert reg["valid"] == 0
        assert reg["rd_out"] == 5, "data still flows; only the valid bit gates it"
        assert reg.get_state()["valid"] == "BUBBLE"


class TestEX_MEM:

    def test_all_declared_fields_round_trip_through_the_latch(self):
        reg = EX_MEM()
        for field in EX_MEM._FIELDS:
            reg[f"{field}_in"] = 1
        reg["valid_in"] = 1
        reg.rising_edge()
        reg.evaluate()
        for field in EX_MEM._FIELDS:
            assert reg[f"{field}_out"] == 1, f"{field} did not propagate"

    def test_condition_flags_travel_with_the_branch_through_the_stage(self):
        """PINS THE FIX: ARM/x86 flags are captured at EX and carried in the
        pipeline register so a YOUNGER flag-writer cannot clobber the branch's
        own condition."""
        reg = EX_MEM()
        reg["flags_zero_in"] = 1
        reg["flags_result_in"] = 0x30
        reg["branch_in"] = 1
        reg["valid_in"] = 1
        reg.rising_edge()

        # A younger flag-writer now drives the INPUT ports...
        reg["flags_zero_in"] = 0
        reg["flags_result_in"] = 0xFF
        reg.evaluate()
        # ...but the latched outputs still belong to the branch.
        assert reg["flags_zero_out"] == 1
        assert reg["flags_result_out"] == 0x30

    def test_flush_clears_the_stage(self):
        reg = EX_MEM()
        reg["alu_result_in"] = 0x77
        reg["valid_in"] = 1
        reg.rising_edge()
        reg["flush"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["valid"] == 0
        assert reg["alu_result_out"] == 0

    def test_ex_mem_stall_holds_and_is_inert_when_unwired(self):
        """EX/MEM gained a `stall` (hold) port for MEM-stage stalls.

        Added 2026-08-08 for the timing-accurate cache: a cache miss must hold
        the instruction in MEM. The load-use bubble is still inserted upstream
        at ID/EX — this is a different mechanism. Unwired the port reads 0, so
        every existing preset behaves exactly as before, which is the property
        that actually matters and is what this test pins.
        """
        assert "stall" in EX_MEM.ports_spec
        assert "flush" in EX_MEM.ports_spec

        # Unwired (stall == 0): latches normally.
        reg = EX_MEM()
        reg["rd_in"] = 7
        reg["valid_in"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["rd_out"] == 7

        # stall == 1: holds the previous value instead of latching.
        reg["rd_in"] = 9
        reg["stall"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["rd_out"] == 7, "stall must hold, not latch"

        # flush outranks stall.
        reg["flush"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["rd_out"] == 0 and reg["valid"] == 0

    def test_get_state_summarises_the_stage(self):
        reg = EX_MEM()
        reg["pc_in"] = 0x30
        reg["rd_in"] = 4
        reg["alu_result_in"] = 0x77
        reg["valid_in"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg.get_state() == {"pc": "0x00000030", "rd": 4,
                                   "alu_result": 0x77, "valid": "VALID"}


class TestMEM_WB:

    def test_all_declared_fields_round_trip_through_the_latch(self):
        reg = MEM_WB()
        for field in MEM_WB._FIELDS:
            reg[f"{field}_in"] = 1
        reg["valid_in"] = 1
        reg.rising_edge()
        reg.evaluate()
        for field in MEM_WB._FIELDS:
            assert reg[f"{field}_out"] == 1, f"{field} did not propagate"
        assert reg["valid"] == 1

    def test_valid_propagates_from_upstream(self):
        reg = MEM_WB()
        reg["rd_in"] = 6
        reg["valid_in"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["valid"] == 1
        reg["valid_in"] = 0
        reg.rising_edge()
        reg.evaluate()
        assert reg["valid"] == 0

    def test_mem_wb_stall_bubbles_and_is_inert_when_unwired(self):
        """MEM/WB gained a `stall` (bubble) port; it still has no `flush`.

        Added 2026-08-08 for the timing-accurate cache: while the MEM stage is
        servicing a miss, NOTHING completes, so MEM/WB must take a bubble
        rather than latch a value memory has not produced yet. Nothing after
        writeback can be squashed, so there is still no `flush`. Unwired the
        port reads 0 and behaviour is unchanged.
        """
        assert "flush" not in MEM_WB.ports_spec
        assert "stall" in MEM_WB.ports_spec

        # Unwired: latches normally.
        reg = MEM_WB()
        reg["rd_in"] = 5
        reg["reg_write_in"] = 1
        reg["valid_in"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["rd_out"] == 5 and reg["reg_write_out"] == 1

        # stall == 1: bubble, so no register write is handed to WB.
        reg["rd_in"] = 6
        reg["stall"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg["reg_write_out"] == 0, "a MEM stall must not complete an instr"
        assert reg["valid"] == 0

    def test_get_state_summarises_the_stage(self):
        reg = MEM_WB()
        reg["pc_in"] = 0x50
        reg["rd_in"] = 6
        reg["wb_sel_in"] = 1
        reg["valid_in"] = 1
        reg.rising_edge()
        reg.evaluate()
        assert reg.get_state() == {"pc": "0x00000050", "rd": 6,
                                   "wb_sel": 1, "valid": "VALID"}
