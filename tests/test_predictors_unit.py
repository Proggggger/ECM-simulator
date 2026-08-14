"""
UNIT tests for the branch predictors (`sim/components/branch/predictors/`) and
the branch resolution unit (`sim/components/branch/unit.py`).

WHY THIS FILE EXISTS
--------------------
`tests/test_btb.py` covers the BTB shim and the IF/ID prediction-stage toggle,
and every other predictor is exercised only through a full pipeline run where
"the loop finished with x1 == 0" is true regardless of whether the predictor
predicted anything.  A predictor that always returned 0 would pass most of
that suite (it would just be slower).

Here each predictor is driven DIRECTLY — construct it, write input ports via
``pred._ports[...]``, call ``evaluate()`` / ``rising_edge()`` by hand, and
assert on output ports, internal tables, and ``get_state()`` — so the 2-bit
saturating state machine, the GHR shift/XOR indexing, and the BTB's
tag/eviction protocol are each independently falsifiable.

CLOCKING CONTRACT (mirrors ``CPU.tick()`` in sim/component/wire.py)
-------------------------------------------------------------------
``evaluate()`` is the combinational read (drives ``prediction`` /
``predict_target`` from the CURRENT ``pc``); ``rising_edge()`` is the clocked
training write (consumes ``update_en`` / ``update_pc`` / ``actual`` /
``update_target``).  Prediction and training are on INDEPENDENT port sets, so
a test may train and then predict without an intervening full tick — which is
exactly how the real pipeline uses them (IF predicts, WB trains).

Everything here is instant: no CPU is built and no simulation is run.

Run from repo root:  python3 -m pytest tests/test_predictors_unit.py -q
"""
import importlib
import inspect
import os
import pkgutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sim.components.branch.predictors as predictors_pkg
from sim.component.base import ComponentBase
from sim.components.branch.predictors.always_taken import AlwaysTaken
from sim.components.branch.predictors.base import BranchPredictorBase
from sim.components.branch.predictors.bimodal import BimodalPredictor
from sim.components.branch.predictors.btb import BTBPredictor
from sim.components.branch.predictors.gshare import GSharePredictor
from sim.components.branch.predictors.never_taken import NeverTaken
from sim.components.branch.predictors.no_predict import NoPrediction
from sim.components.branch.unit import BranchResolutionUnit
from sim.core.signals import BranchCond
from sim.harness import PREDICTOR_CLASSES, PREDICTORS

# 2-bit saturating counter states, by name.
STRONG_NT, WEAK_NT, WEAK_T, STRONG_T = 0, 1, 2, 3


# ── driving helpers ──────────────────────────────────────────────────────────

def predict(pred, pc, *, is_branch=1, is_jal=0, imm=0):
    """Combinational read: set the fetch-side ports and evaluate()."""
    pred._ports["pc"] = pc
    pred._ports["is_branch"] = is_branch
    pred._ports["is_jal"] = is_jal
    pred._ports["imm"] = imm & 0xFFFF_FFFF
    pred.evaluate()
    return pred["prediction"]


def train(pred, pc, taken, *, target=0, enable=1):
    """Clocked write: set the feedback ports and rising_edge()."""
    pred._ports["update_en"] = enable
    pred._ports["update_pc"] = pc
    pred._ports["actual"] = 1 if taken else 0
    pred._ports["update_target"] = target & 0xFFFF_FFFF
    pred.rising_edge()


ALL_PREDICTOR_CLASSES = [
    AlwaysTaken, NeverTaken, NoPrediction, BimodalPredictor,
    GSharePredictor, BTBPredictor,
]


# ════════════════════════════════════════════════════════════════════════════
# Shared port contract — every predictor is interchangeable
# ════════════════════════════════════════════════════════════════════════════

class TestPredictorPortContract:
    """All predictors expose EXACTLY the base port set (that is the swap point)."""

    EXPECTED_PORTS = {
        "pc", "is_branch", "is_jal", "imm",
        "prediction", "predict_target",
        "update_en", "update_pc", "actual", "update_target",
    }

    @pytest.mark.parametrize("cls", ALL_PREDICTOR_CLASSES,
                             ids=[c.__name__ for c in ALL_PREDICTOR_CLASSES])
    def test_port_set_is_the_base_contract(self, cls):
        assert set(cls.ports_spec) == self.EXPECTED_PORTS
        assert cls.ports_spec is BranchPredictorBase.ports_spec

    @pytest.mark.parametrize("cls", ALL_PREDICTOR_CLASSES,
                             ids=[c.__name__ for c in ALL_PREDICTOR_CLASSES])
    def test_port_widths_and_directions(self, cls):
        spec = cls.ports_spec
        assert spec["prediction"].width == 1
        assert spec["prediction"].direction == "out"
        assert spec["predict_target"].width == 32
        assert spec["predict_target"].direction == "out"
        for name in ("pc", "imm", "update_pc", "update_target"):
            assert spec[name].width == 32
            assert spec[name].direction == "in"
        for name in ("is_branch", "is_jal", "update_en", "actual"):
            assert spec[name].width == 1
            assert spec[name].direction == "in"

    @pytest.mark.parametrize("cls", ALL_PREDICTOR_CLASSES,
                             ids=[c.__name__ for c in ALL_PREDICTOR_CLASSES])
    def test_is_a_component_with_a_control_ui_category(self, cls):
        pred = cls()
        assert isinstance(pred, ComponentBase)
        assert isinstance(pred, BranchPredictorBase)
        assert cls.ui_category == "control"

    @pytest.mark.parametrize("cls", ALL_PREDICTOR_CLASSES,
                             ids=[c.__name__ for c in ALL_PREDICTOR_CLASSES])
    def test_prediction_port_is_masked_to_one_bit(self, cls):
        pred = cls()
        pred["prediction"] = 0xFF
        assert pred["prediction"] == 1

    @pytest.mark.parametrize("cls", ALL_PREDICTOR_CLASSES,
                             ids=[c.__name__ for c in ALL_PREDICTOR_CLASSES])
    def test_get_state_is_a_serialisable_dict(self, cls):
        pred = cls()
        predict(pred, 0x40)
        state = pred.get_state()
        assert isinstance(state, dict)
        for key, value in state.items():
            assert isinstance(key, str)
            assert isinstance(value, (str, int, bool)), (key, value)

    @pytest.mark.parametrize("cls", ALL_PREDICTOR_CLASSES,
                             ids=[c.__name__ for c in ALL_PREDICTOR_CLASSES])
    def test_evaluate_is_idempotent(self, cls):
        """Re-evaluating without a clock edge must not change the answer."""
        pred = cls()
        first = predict(pred, 0x40, imm=8)
        pred.evaluate()
        assert pred["prediction"] == first

    @pytest.mark.parametrize("cls", ALL_PREDICTOR_CLASSES,
                             ids=[c.__name__ for c in ALL_PREDICTOR_CLASSES])
    def test_training_is_a_no_op_when_update_en_is_low(self, cls):
        pred = cls()
        before = predict(pred, 0x100, imm=8)
        for _ in range(8):
            train(pred, 0x100, taken=not before, target=0x200, enable=0)
        assert predict(pred, 0x100, imm=8) == before


# ════════════════════════════════════════════════════════════════════════════
# Base class — the abstract template + the BTB target-cache layer
# ════════════════════════════════════════════════════════════════════════════

class _BareSubclass(BranchPredictorBase):
    """A predictor that overrides NOTHING — exercises the base's defaults."""
    name = "_bare_for_tests"


class TestBranchPredictorBaseDefaults:

    def test_default_predict_is_not_taken_with_no_target(self):
        # base.py:74-75 — the un-overridden _predict() template hook.
        pred = _BareSubclass()
        assert predict(pred, 0x100, is_branch=1, is_jal=1, imm=8) == 0
        assert pred["predict_target"] == 0

    def test_default_update_is_a_no_op(self):
        pred = _BareSubclass()
        train(pred, 0x100, taken=True, target=0x200)
        assert predict(pred, 0x100) == 0

    def test_default_predictor_state_is_empty(self):
        assert _BareSubclass()._get_predictor_state() == {}
        assert _BareSubclass().get_state() == {}

    def test_default_prediction_stage_is_id_with_no_btb(self):
        pred = _BareSubclass()
        assert pred.prediction_stage == "id"
        assert pred._btb is None
        assert pred._btb_size == 0

    def test_if_stage_allocates_a_btb_of_the_requested_size(self):
        pred = _BareSubclass(prediction_stage="if", btb_size=16)
        assert pred._btb_size == 16
        assert pred._btb == [None] * 16


class TestComputeTarget:
    """`_compute_target()` = pc + imm when taken, 0 when not."""

    def test_target_is_pc_plus_imm_when_taken(self):
        pred = AlwaysTaken()
        predict(pred, 0x100, is_branch=1, imm=0x20)
        assert pred["predict_target"] == 0x120

    def test_target_is_zero_when_not_taken(self):
        pred = NeverTaken()
        predict(pred, 0x100, is_branch=1, imm=0x20)
        assert pred["predict_target"] == 0

    def test_negative_offsets_arrive_as_two_complement_and_go_backwards(self):
        pred = AlwaysTaken()
        predict(pred, 0x100, is_branch=1, imm=-0x20 & 0xFFFF_FFFF)
        assert pred["predict_target"] == 0xE0

    def test_target_wraps_at_32_bits(self):
        pred = AlwaysTaken()
        predict(pred, 0xFFFF_FFFC, is_branch=1, imm=8)
        assert pred["predict_target"] == 0x4


# ════════════════════════════════════════════════════════════════════════════
# Static predictors
# ════════════════════════════════════════════════════════════════════════════

class TestNoPrediction:
    """Always not-taken, no target, untrainable. The safest default."""

    @pytest.mark.parametrize("is_branch,is_jal", [(0, 0), (1, 0), (0, 1), (1, 1)])
    def test_never_predicts_taken_for_any_input(self, is_branch, is_jal):
        pred = NoPrediction()
        assert predict(pred, 0x40, is_branch=is_branch, is_jal=is_jal, imm=8) == 0
        assert pred["predict_target"] == 0

    def test_training_never_changes_the_answer(self):
        pred = NoPrediction()
        for _ in range(16):
            train(pred, 0x40, taken=True, target=0x80)
        assert predict(pred, 0x40, is_branch=1, imm=8) == 0
        assert pred["predict_target"] == 0

    def test_state_is_empty_at_id_stage(self):
        # NoPrediction does not override _get_predictor_state().
        assert NoPrediction().get_state() == {}

    def test_if_stage_btb_hit_cannot_force_a_taken_prediction(self):
        # The base wrapper zeroes prediction whenever _predict() says 0, so a
        # strong BTB entry still yields not-taken.
        pred = NoPrediction(prediction_stage="if", btb_size=8)
        train(pred, 0x100, taken=True, target=0x200)
        assert predict(pred, 0x100) == 0
        assert pred["predict_target"] == 0


class TestNeverTaken:
    """Not-taken for conditional branches, TAKEN for unconditional jumps."""

    def test_branches_are_not_taken(self):
        pred = NeverTaken()
        assert predict(pred, 0x40, is_branch=1, imm=8) == 0
        assert pred["predict_target"] == 0

    def test_jal_is_always_taken(self):
        pred = NeverTaken()
        assert predict(pred, 0x40, is_branch=0, is_jal=1, imm=8) == 1
        assert pred["predict_target"] == 0x48

    def test_non_control_instructions_are_not_taken(self):
        pred = NeverTaken()
        assert predict(pred, 0x40, is_branch=0, is_jal=0, imm=8) == 0

    def test_a_branch_that_is_also_flagged_jal_follows_jal(self):
        pred = NeverTaken()
        assert predict(pred, 0x40, is_branch=1, is_jal=1, imm=8) == 1

    @pytest.mark.parametrize("taken", [True, False])
    def test_training_never_changes_the_answer(self, taken):
        pred = NeverTaken()
        for _ in range(16):
            train(pred, 0x40, taken=taken, target=0x80)
        assert predict(pred, 0x40, is_branch=1, imm=8) == 0

    def test_state_label(self):
        assert NeverTaken().get_state() == {"prediction": "NT (branches)"}


class TestAlwaysTaken:
    """Taken for anything control-flow, not-taken for everything else."""

    @pytest.mark.parametrize("is_branch,is_jal,expected", [
        (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 0, 0),
    ])
    def test_constant_behaviour(self, is_branch, is_jal, expected):
        pred = AlwaysTaken()
        assert predict(pred, 0x40, is_branch=is_branch, is_jal=is_jal, imm=8) == expected

    def test_target_is_computed_for_taken_predictions(self):
        pred = AlwaysTaken()
        predict(pred, 0x40, is_branch=1, imm=0x10)
        assert pred["predict_target"] == 0x50

    @pytest.mark.parametrize("taken", [True, False])
    def test_training_never_changes_the_answer(self, taken):
        pred = AlwaysTaken()
        for _ in range(16):
            train(pred, 0x40, taken=taken, target=0x80)
        assert predict(pred, 0x40, is_branch=1, imm=8) == 1

    def test_state_label(self):
        assert AlwaysTaken().get_state() == {"prediction": "T (always)"}

    def test_if_stage_still_needs_a_btb_hit(self):
        # Even AlwaysTaken predicts not-taken at IF until the BTB learns the
        # PC — the base wrapper zeroes is_branch on a miss.
        pred = AlwaysTaken(prediction_stage="if", btb_size=8)
        assert predict(pred, 0x100, imm=8) == 0
        train(pred, 0x100, taken=True, target=0x200)
        assert predict(pred, 0x100, imm=8) == 1
        assert pred["predict_target"] == 0x200


# ════════════════════════════════════════════════════════════════════════════
# Bimodal — the 2-bit saturating counter state machine
# ════════════════════════════════════════════════════════════════════════════

class TestBimodalIndexing:

    def test_index_drops_the_two_low_pc_bits(self):
        pred = BimodalPredictor(table_size=256)
        assert pred._index(0x00) == 0
        assert pred._index(0x04) == 1
        assert pred._index(0x08) == 2
        # Sub-word bits are ignored (instructions are word-aligned).
        assert pred._index(0x04) == pred._index(0x05) == pred._index(0x07)

    def test_index_wraps_modulo_table_size(self):
        pred = BimodalPredictor(table_size=4)
        assert pred._index(0x00) == pred._index(0x10) == 0
        assert pred._index(0x04) == pred._index(0x14) == 1

    def test_index_is_in_range_for_any_32_bit_pc(self):
        pred = BimodalPredictor(table_size=8)
        for pc in (0, 4, 0x1000, 0x7FFF_FFFC, 0xFFFF_FFFC, 0xFFFF_FFFF):
            assert 0 <= pred._index(pc) < 8

    def test_table_is_sized_and_initialised_weakly_taken(self):
        pred = BimodalPredictor(table_size=32)
        assert pred._table == [WEAK_T] * 32


class TestBimodalStateMachine:
    """Walk the full 2-bit saturating counter, both directions."""

    PC = 0x100

    @pytest.fixture()
    def pred(self):
        return BimodalPredictor(table_size=16)

    def _counter(self, pred):
        return pred._table[pred._index(self.PC)]

    def test_initial_state_is_weakly_taken(self, pred):
        assert self._counter(pred) == WEAK_T
        assert predict(pred, self.PC) == 1

    def test_walk_down_from_weakly_taken_to_strongly_not_taken(self, pred):
        assert self._counter(pred) == WEAK_T
        train(pred, self.PC, taken=False)
        assert self._counter(pred) == WEAK_NT
        assert predict(pred, self.PC) == 0
        train(pred, self.PC, taken=False)
        assert self._counter(pred) == STRONG_NT
        assert predict(pred, self.PC) == 0

    def test_walk_up_from_weakly_taken_to_strongly_taken(self, pred):
        train(pred, self.PC, taken=True)
        assert self._counter(pred) == STRONG_T
        assert predict(pred, self.PC) == 1

    def test_full_round_trip_through_every_state(self, pred):
        # 2 -> 1 -> 0 -> 1 -> 2 -> 3, with the prediction flipping at 2.
        expected = [
            (False, WEAK_NT, 0),
            (False, STRONG_NT, 0),
            (True, WEAK_NT, 0),
            (True, WEAK_T, 1),
            (True, STRONG_T, 1),
        ]
        for actual, counter, prediction in expected:
            train(pred, self.PC, taken=actual)
            assert self._counter(pred) == counter
            assert predict(pred, self.PC) == prediction

    def test_saturates_at_strongly_taken(self, pred):
        for _ in range(10):
            train(pred, self.PC, taken=True)
        assert self._counter(pred) == STRONG_T
        assert predict(pred, self.PC) == 1

    def test_saturates_at_strongly_not_taken(self, pred):
        for _ in range(10):
            train(pred, self.PC, taken=False)
        assert self._counter(pred) == STRONG_NT
        assert predict(pred, self.PC) == 0

    def test_one_mispredict_does_not_flip_a_strongly_held_taken_state(self, pred):
        """The whole point of 2 bits: hysteresis on a single anomaly."""
        for _ in range(4):
            train(pred, self.PC, taken=True)
        assert self._counter(pred) == STRONG_T
        train(pred, self.PC, taken=False)            # one anomaly
        assert self._counter(pred) == WEAK_T
        assert predict(pred, self.PC) == 1           # STILL predicts taken

    def test_one_mispredict_does_not_flip_a_strongly_held_not_taken_state(self, pred):
        for _ in range(4):
            train(pred, self.PC, taken=False)
        assert self._counter(pred) == STRONG_NT
        train(pred, self.PC, taken=True)
        assert self._counter(pred) == WEAK_NT
        assert predict(pred, self.PC) == 0           # STILL predicts not-taken

    def test_two_consecutive_mispredicts_do_flip_the_prediction(self, pred):
        for _ in range(4):
            train(pred, self.PC, taken=True)
        train(pred, self.PC, taken=False)
        train(pred, self.PC, taken=False)
        assert predict(pred, self.PC) == 0


class TestBimodalPerPcIndependence:

    def test_distinct_pcs_train_independent_counters(self):
        pred = BimodalPredictor(table_size=256)
        for _ in range(4):
            train(pred, 0x100, taken=True)
            train(pred, 0x104, taken=False)
        assert predict(pred, 0x100) == 1
        assert predict(pred, 0x104) == 0

    def test_pcs_that_share_an_index_alias(self):
        """No tags in the bimodal table — collisions are destructive."""
        pred = BimodalPredictor(table_size=4)
        assert pred._index(0x00) == pred._index(0x10)
        for _ in range(4):
            train(pred, 0x00, taken=True)
        assert predict(pred, 0x10) == 1               # inherits 0x00's history
        for _ in range(4):
            train(pred, 0x10, taken=False)
        assert predict(pred, 0x00) == 0               # 0x00's history destroyed


class TestBimodalPredictionGating:

    def test_jal_is_predicted_taken_regardless_of_the_counter(self):
        pred = BimodalPredictor(table_size=4)
        pred._table = [STRONG_NT] * 4
        assert predict(pred, 0x00, is_branch=0, is_jal=1, imm=8) == 1
        assert pred["predict_target"] == 8

    def test_non_control_instructions_are_never_taken(self):
        pred = BimodalPredictor(table_size=4)
        pred._table = [STRONG_T] * 4
        assert predict(pred, 0x00, is_branch=0, is_jal=0, imm=8) == 0
        assert pred["predict_target"] == 0

    def test_the_counter_is_only_consulted_for_branches(self):
        pred = BimodalPredictor(table_size=4)
        assert predict(pred, 0x00, is_branch=1, imm=8) == 1
        assert predict(pred, 0x00, is_branch=0, imm=8) == 0


class TestBimodalState:

    def test_state_reports_the_indexed_counter_and_direction(self):
        pred = BimodalPredictor(table_size=16)
        predict(pred, 0x100)
        assert pred.get_state() == {
            "prediction": "T", "counter": "2/3", "index": pred._index(0x100),
        }

    def test_state_follows_training(self):
        pred = BimodalPredictor(table_size=16)
        train(pred, 0x100, taken=False)
        train(pred, 0x100, taken=False)
        predict(pred, 0x100)
        state = pred.get_state()
        assert state["prediction"] == "NT"
        assert state["counter"] == "0/3"


# ════════════════════════════════════════════════════════════════════════════
# GShare — global history XOR PC
# ════════════════════════════════════════════════════════════════════════════

class TestGShareConstruction:

    @pytest.mark.parametrize("bits", [1, 2, 3, 8, 10])
    def test_table_size_is_two_to_the_history_bits(self, bits):
        pred = GSharePredictor(history_bits=bits)
        assert len(pred._table) == 1 << bits
        assert pred._table == [WEAK_T] * (1 << bits)

    def test_ghr_starts_cleared(self):
        assert GSharePredictor(history_bits=4)._ghr == 0

    def test_default_history_is_eight_bits(self):
        pred = GSharePredictor()
        assert pred._history_bits == 8
        assert len(pred._table) == 256


class TestGShareIndexing:

    def test_index_is_pc_xor_ghr_masked(self):
        pred = GSharePredictor(history_bits=4)
        pred._ghr = 0b0101
        assert pred._index(0x00) == (0 ^ 0b0101) & 0xF
        assert pred._index(0x0C) == ((0x0C >> 2) ^ 0b0101) & 0xF

    def test_index_with_a_zero_ghr_matches_a_bimodal_index(self):
        gshare = GSharePredictor(history_bits=4)
        bimodal = BimodalPredictor(table_size=16)
        for pc in (0x00, 0x04, 0x10, 0x40):
            assert gshare._index(pc) == bimodal._index(pc)

    def test_index_is_in_range_for_any_pc_and_ghr(self):
        pred = GSharePredictor(history_bits=3)
        for ghr in range(8):
            pred._ghr = ghr
            for pc in (0, 4, 0x1000, 0xFFFF_FFFF):
                assert 0 <= pred._index(pc) < 8

    def test_history_length_boundary_masks_the_ghr(self):
        """Only `history_bits` of history survive the shift."""
        pred = GSharePredictor(history_bits=3)
        for _ in range(10):
            train(pred, 0x00, taken=True)
        assert pred._ghr == 0b111                 # saturated, not 0b1111111111
        for _ in range(10):
            train(pred, 0x00, taken=False)
        assert pred._ghr == 0


class TestGShareHistoryRegister:

    def test_ghr_shifts_in_the_actual_outcome(self):
        pred = GSharePredictor(history_bits=4)
        train(pred, 0x00, taken=True)
        assert pred._ghr == 0b0001
        train(pred, 0x00, taken=True)
        assert pred._ghr == 0b0011
        train(pred, 0x00, taken=False)
        assert pred._ghr == 0b0110
        train(pred, 0x00, taken=True)
        assert pred._ghr == 0b1101

    def test_ghr_only_advances_when_update_en_is_high(self):
        pred = GSharePredictor(history_bits=4)
        train(pred, 0x00, taken=True, enable=0)
        assert pred._ghr == 0
        train(pred, 0x00, taken=True, enable=1)
        assert pred._ghr == 1

    def test_ghr_is_global_across_pcs(self):
        pred = GSharePredictor(history_bits=4)
        train(pred, 0x100, taken=True)
        train(pred, 0x200, taken=True)
        assert pred._ghr == 0b0011           # one shared register, not per-PC

    def test_the_same_pc_indexes_different_entries_as_history_changes(self):
        pred = GSharePredictor(history_bits=4)
        before = pred._index(0x00)
        train(pred, 0x00, taken=True)
        assert pred._index(0x00) != before


class TestGShareLearning:

    def test_learns_a_taken_branch_under_a_stable_history(self):
        pred = GSharePredictor(history_bits=4)
        pred._table = [STRONG_NT] * 16
        # With ghr == 0 the entry for pc 0x40 is index (0x10 ^ 0) & 0xF.
        idx = pred._index(0x40)
        assert predict(pred, 0x40) == 0
        pred._table[idx] = STRONG_T
        assert predict(pred, 0x40) == 1

    def test_training_updates_the_entry_the_update_pc_selects(self):
        pred = GSharePredictor(history_bits=4)
        idx = pred._index(0x40)
        train(pred, 0x40, taken=False)
        assert pred._table[idx] == WEAK_NT

    def test_counter_saturates_in_both_directions(self):
        pred = GSharePredictor(history_bits=1)   # 2 entries: heavy aliasing
        for _ in range(10):
            train(pred, 0x00, taken=True)
        assert max(pred._table) == STRONG_T
        assert all(c <= STRONG_T for c in pred._table)
        for _ in range(20):
            train(pred, 0x00, taken=False)
        assert min(pred._table) == STRONG_NT
        assert all(c >= STRONG_NT for c in pred._table)

    def test_two_pcs_that_map_to_the_same_index_alias(self):
        """No tags — a shared index means shared (destructive) history."""
        pred = GSharePredictor(history_bits=2)
        assert pred._index(0x00) == pred._index(0x10)     # (pc>>2) & 3
        pred._table = [STRONG_NT] * 4
        pred._table[pred._index(0x00)] = STRONG_T
        assert predict(pred, 0x00) == 1
        assert predict(pred, 0x10) == 1                   # aliased hit

    def test_history_separates_two_dynamic_instances_of_one_branch(self):
        """The point of gshare: same PC, different history -> different entry."""
        pred = GSharePredictor(history_bits=2)
        pc = 0x00
        idx_h0 = pred._index(pc)
        pred._ghr = 0b01
        idx_h1 = pred._index(pc)
        assert idx_h0 != idx_h1

    def test_jal_bypasses_the_table(self):
        pred = GSharePredictor(history_bits=2)
        pred._table = [STRONG_NT] * 4
        assert predict(pred, 0x00, is_branch=0, is_jal=1, imm=16) == 1
        assert pred["predict_target"] == 16

    def test_non_control_instructions_are_not_taken(self):
        pred = GSharePredictor(history_bits=2)
        pred._table = [STRONG_T] * 4
        assert predict(pred, 0x00, is_branch=0, is_jal=0, imm=16) == 0
        assert pred["predict_target"] == 0


class TestGShareState:

    def test_state_reports_counter_ghr_and_index(self):
        pred = GSharePredictor(history_bits=3)
        train(pred, 0x00, taken=True)
        predict(pred, 0x00)
        state = pred.get_state()
        assert state["ghr"] == "0b001"
        assert state["counter"].endswith("/3")
        assert state["prediction"] in ("T", "NT")
        assert state["index"] == pred._index(0x00)

    def test_ghr_is_rendered_at_the_configured_width(self):
        pred = GSharePredictor(history_bits=8)
        predict(pred, 0)
        assert pred.get_state()["ghr"] == "0b00000000"


# ════════════════════════════════════════════════════════════════════════════
# BTB — target cache in the base class, direction from the bimodal counters
# ════════════════════════════════════════════════════════════════════════════

class TestBTBConstruction:

    def test_defaults_to_if_stage_with_a_btb_allocated(self):
        pred = BTBPredictor()
        assert pred.prediction_stage == "if"
        assert pred._btb is not None
        assert len(pred._btb) == pred._btb_size

    def test_is_a_bimodal_subclass(self):
        assert issubclass(BTBPredictor, BimodalPredictor)
        assert BTBPredictor().name == "btb"

    def test_can_be_forced_back_to_id_stage(self):
        pred = BTBPredictor(prediction_stage="id")
        assert pred.prediction_stage == "id"
        assert pred._btb is None

    def test_table_size_and_btb_size_are_independent(self):
        pred = BTBPredictor(table_size=8, btb_size=4)
        assert len(pred._table) == 8
        assert len(pred._btb) == 4


class TestBTBHitAndMiss:

    def test_a_cold_btb_misses_and_predicts_not_taken(self):
        pred = BTBPredictor(btb_size=16)
        assert predict(pred, 0x100, imm=8) == 0
        assert pred["predict_target"] == 0
        assert pred.get_state()["btb_hit"] is False
        assert pred.get_state()["prediction"] == "NT (miss)"

    def test_a_taken_outcome_allocates_the_entry_weakly_taken(self):
        pred = BTBPredictor(btb_size=16)
        train(pred, 0x100, taken=True, target=0x80)
        entry = pred._btb[pred._btb_index(0x100)]
        assert entry == {"pc": 0x100, "target": 0x80, "counter": 2}

    def test_a_hit_predicts_taken_with_the_cached_target(self):
        pred = BTBPredictor(btb_size=16)
        train(pred, 0x100, taken=True, target=0x80)
        assert predict(pred, 0x100, imm=0xDEAD) == 1
        # The CACHED target wins over pc+imm (imm is unwired at IF stage).
        assert pred["predict_target"] == 0x80

    def test_the_cached_target_is_updated_on_each_taken_outcome(self):
        pred = BTBPredictor(btb_size=16)
        train(pred, 0x100, taken=True, target=0x80)
        train(pred, 0x100, taken=True, target=0x200)
        assert predict(pred, 0x100) == 1
        assert pred["predict_target"] == 0x200

    def test_a_not_taken_outcome_never_allocates_an_entry(self):
        pred = BTBPredictor(btb_size=16)
        train(pred, 0x100, taken=False, target=0x80)
        assert pred._btb[pred._btb_index(0x100)] is None
        assert predict(pred, 0x100) == 0

    def test_state_exposes_the_cached_target_and_counter_on_a_hit(self):
        pred = BTBPredictor(btb_size=16)
        train(pred, 0x100, taken=True, target=0x80)
        predict(pred, 0x100)
        state = pred.get_state()
        assert state["btb_hit"] is True
        assert state["btb_target"] == "0x00000080"
        assert state["btb_counter"] == "2/3"
        assert state["prediction"] == "T"


class TestBTBCounter:
    """The BTB entry carries its OWN 2-bit counter, separate from the table."""

    def test_counter_saturates_at_three(self):
        pred = BTBPredictor(btb_size=16)
        for _ in range(6):
            train(pred, 0x100, taken=True, target=0x80)
        assert pred._btb[pred._btb_index(0x100)]["counter"] == 3

    def test_a_not_taken_outcome_decrements_an_existing_entry(self):
        pred = BTBPredictor(btb_size=16)
        train(pred, 0x100, taken=True, target=0x80)          # counter 2
        train(pred, 0x100, taken=False, target=0x80)         # counter 1
        entry = pred._btb[pred._btb_index(0x100)]
        assert entry["counter"] == 1
        assert entry["target"] == 0x80                        # target preserved
        assert predict(pred, 0x100) == 0                      # weak -> no steer

    def test_counter_saturates_at_zero_and_the_entry_survives(self):
        pred = BTBPredictor(btb_size=16)
        train(pred, 0x100, taken=True, target=0x80)
        for _ in range(6):
            train(pred, 0x100, taken=False)
        entry = pred._btb[pred._btb_index(0x100)]
        assert entry["counter"] == 0
        assert entry["target"] == 0x80
        assert predict(pred, 0x100) == 0

    def test_a_weak_entry_can_be_promoted_back_to_a_hit(self):
        pred = BTBPredictor(btb_size=16)
        train(pred, 0x100, taken=True, target=0x80)
        train(pred, 0x100, taken=False)
        assert predict(pred, 0x100) == 0
        train(pred, 0x100, taken=True, target=0x80)
        assert predict(pred, 0x100) == 1

    def test_a_weak_entry_still_reports_a_hit_in_state(self):
        pred = BTBPredictor(btb_size=16)
        train(pred, 0x100, taken=True, target=0x80)
        train(pred, 0x100, taken=False)
        predict(pred, 0x100)
        state = pred.get_state()
        assert state["btb_hit"] is True
        assert state["btb_counter"] == "1/3"
        assert state["prediction"] == "NT"


class TestBTBIndexingAndEviction:

    def test_index_drops_the_two_low_pc_bits_and_wraps(self):
        pred = BTBPredictor(btb_size=4)
        assert pred._btb_index(0x00) == 0
        assert pred._btb_index(0x04) == 1
        assert pred._btb_index(0x10) == 0        # collides with 0x00

    def test_a_tag_mismatch_is_a_miss_not_a_wrong_target(self):
        pred = BTBPredictor(btb_size=4)
        train(pred, 0x00, taken=True, target=0x80)
        assert predict(pred, 0x10) == 0          # same index, different tag
        assert pred["predict_target"] == 0

    def test_a_colliding_pc_evicts_the_previous_entry(self):
        pred = BTBPredictor(btb_size=4)
        train(pred, 0x00, taken=True, target=0x80)
        train(pred, 0x10, taken=True, target=0x90)
        entry = pred._btb[0]
        assert entry["pc"] == 0x10 and entry["target"] == 0x90
        assert predict(pred, 0x10) == 1
        assert pred["predict_target"] == 0x90
        assert predict(pred, 0x00) == 0          # evicted -> miss
        assert pred.get_state()["btb_hit"] is False

    def test_an_evicting_write_resets_the_counter_to_weakly_taken(self):
        pred = BTBPredictor(btb_size=4)
        for _ in range(4):
            train(pred, 0x00, taken=True, target=0x80)       # counter 3
        train(pred, 0x10, taken=True, target=0x90)
        assert pred._btb[0]["counter"] == 2                   # fresh allocation

    def test_a_not_taken_outcome_on_a_colliding_pc_does_not_touch_the_entry(self):
        pred = BTBPredictor(btb_size=4)
        train(pred, 0x00, taken=True, target=0x80)
        train(pred, 0x10, taken=False, target=0x90)
        entry = pred._btb[0]
        assert entry["pc"] == 0x00 and entry["counter"] == 2

    def test_distinct_indices_coexist(self):
        pred = BTBPredictor(btb_size=8)
        train(pred, 0x00, taken=True, target=0x80)
        train(pred, 0x04, taken=True, target=0x90)
        assert predict(pred, 0x00) == 1 and pred["predict_target"] == 0x80
        assert predict(pred, 0x04) == 1 and pred["predict_target"] == 0x90


class TestBTBDirectionPredictorInteraction:
    """A hit steers fetch only if the DIRECTION predictor also says taken."""

    def test_a_strong_btb_hit_is_vetoed_by_a_not_taken_direction_counter(self):
        # Distinct BTB indices (btb_size default 256) but ALIASED bimodal
        # indices (table_size 4): 0x100 and 0x110 share bimodal entry 0.
        pred = BTBPredictor(table_size=4, btb_size=256)
        assert pred._index(0x100) == pred._index(0x110)
        assert pred._btb_index(0x100) != pred._btb_index(0x110)
        for _ in range(3):
            train(pred, 0x100, taken=True, target=0x200)
        assert predict(pred, 0x100) == 1
        # Now drive the SHARED bimodal counter down via the aliasing PC.
        for _ in range(4):
            train(pred, 0x110, taken=False)
        assert pred._table[pred._index(0x100)] == STRONG_NT
        assert pred._btb[pred._btb_index(0x100)]["counter"] == 3   # BTB still hot
        assert predict(pred, 0x100) == 0                            # direction vetoes
        assert pred["predict_target"] == 0

    def test_the_direction_predictor_also_trains_on_every_update(self):
        pred = BTBPredictor(table_size=16, btb_size=16)
        idx = pred._index(0x100)
        train(pred, 0x100, taken=True, target=0x200)
        assert pred._table[idx] == STRONG_T

    def test_gshare_at_if_stage_uses_the_shared_btb_layer(self):
        pred = GSharePredictor(history_bits=4, prediction_stage="if")
        train(pred, 0x100, taken=True, target=0x200)
        assert predict(pred, 0x100) == 1
        assert pred["predict_target"] == 0x200

    def test_id_stage_bimodal_has_no_btb_and_uses_pc_plus_imm(self):
        pred = BimodalPredictor(table_size=16)          # id stage
        assert pred._btb is None
        assert predict(pred, 0x100, imm=0x20) == 1
        assert pred["predict_target"] == 0x120


class TestBTBSideEffectsOnInputPorts:
    """The IF-stage wrapper WRITES is_branch — a documented internal effect."""

    def test_a_hit_forces_is_branch_high_before_delegating(self):
        pred = BTBPredictor(btb_size=8)
        train(pred, 0x100, taken=True, target=0x200)
        pred._ports["is_branch"] = 0                # decoder ports unwired at IF
        predict(pred, 0x100, is_branch=0)
        assert pred["is_branch"] == 1               # injected by the wrapper

    def test_a_miss_forces_is_branch_low(self):
        pred = BTBPredictor(btb_size=8)
        predict(pred, 0x100, is_branch=1)
        assert pred["is_branch"] == 0


# ════════════════════════════════════════════════════════════════════════════
# Predictor discovery registry (`sim/harness.py` -> also used by api/app.py)
# ════════════════════════════════════════════════════════════════════════════

class TestPredictorRegistry:

    EXPECTED_NAMES = {
        "always_taken", "never_taken", "no_prediction",
        "bimodal", "gshare", "btb",
    }

    def test_registry_contains_exactly_the_shipped_predictors(self):
        assert set(PREDICTOR_CLASSES) == self.EXPECTED_NAMES

    def test_predictors_tuple_is_the_sorted_registry_keys(self):
        assert PREDICTORS == tuple(sorted(PREDICTOR_CLASSES))

    @pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
    def test_every_registered_class_satisfies_the_base_interface(self, name):
        cls = PREDICTOR_CLASSES[name]
        assert issubclass(cls, BranchPredictorBase)
        assert cls is not BranchPredictorBase
        assert cls.name == name
        assert set(cls.ports_spec) == TestPredictorPortContract.EXPECTED_PORTS

    @pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
    def test_every_registered_class_is_constructible_with_no_arguments(self, name):
        pred = PREDICTOR_CLASSES[name]()
        assert pred.prediction_stage in ("id", "if")
        assert isinstance(pred.get_state(), dict)

    @pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
    def test_every_registered_class_accepts_a_prediction_stage(self, name):
        for stage in ("id", "if"):
            pred = PREDICTOR_CLASSES[name](prediction_stage=stage)
            assert pred.prediction_stage == stage
            assert (pred._btb is not None) == (stage == "if")

    @pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
    def test_every_registered_class_predicts_and_trains_without_error(self, name):
        pred = PREDICTOR_CLASSES[name]()
        assert predict(pred, 0x100, is_branch=1, imm=8) in (0, 1)
        train(pred, 0x100, taken=True, target=0x200)
        assert predict(pred, 0x100, is_branch=1, imm=8) in (0, 1)

    def test_names_are_unique(self):
        names = [cls.name for cls in PREDICTOR_CLASSES.values()]
        assert len(names) == len(set(names))

    def test_every_class_has_a_distinct_non_empty_ui_label(self):
        labels = [cls.ui_label for cls in PREDICTOR_CLASSES.values()]
        assert all(isinstance(v, str) and v.strip() for v in labels)
        assert len(labels) == len(set(labels))

    def test_only_btb_defaults_to_if_stage(self):
        if_stage = {n for n, c in PREDICTOR_CLASSES.items()
                    if c().prediction_stage == "if"}
        assert if_stage == {"btb"}

    def test_discovery_finds_every_module_in_the_package(self):
        """Re-run the harness's scan and require the same answer.

        Guards the "a new predictor file is picked up automatically" contract
        that /predictors and the UI dropdown rely on.
        """
        rediscovered = {}
        for info in pkgutil.iter_modules(predictors_pkg.__path__):
            if info.name in ("base", "__init__"):
                continue
            mod = importlib.import_module(
                f"sim.components.branch.predictors.{info.name}")
            for attr in vars(mod).values():
                if (inspect.isclass(attr) and issubclass(attr, BranchPredictorBase)
                        and attr is not BranchPredictorBase
                        and hasattr(attr, "name")):
                    rediscovered[attr.name] = attr
        assert rediscovered == PREDICTOR_CLASSES

    def test_the_base_module_is_excluded_from_discovery(self):
        assert "" not in PREDICTOR_CLASSES
        assert BranchPredictorBase not in PREDICTOR_CLASSES.values()


# ════════════════════════════════════════════════════════════════════════════
# BranchResolutionUnit — the consumer of `prediction`
# ════════════════════════════════════════════════════════════════════════════

def resolve(compare_mode="slt", serialized=False, **ports):
    unit = BranchResolutionUnit(compare_mode=compare_mode, serialized=serialized)
    for name, value in ports.items():
        unit[name] = value
    unit.evaluate()
    return unit


class TestBranchResolutionConditions:

    def test_beq_taken_when_zero(self):
        unit = resolve(pc=0x10, pc4=0x14, imm=8, branch=1,
                       branch_cond=BranchCond.EQ, alu_zero=1)
        assert unit["branch_taken"] == 1
        assert unit["next_pc"] == 0x18

    def test_beq_not_taken_when_nonzero(self):
        unit = resolve(pc=0x10, pc4=0x14, imm=8, branch=1,
                       branch_cond=BranchCond.EQ, alu_zero=0)
        assert unit["branch_taken"] == 0
        assert unit["next_pc"] == 0x14

    def test_bne_is_the_inverse_of_beq(self):
        for zero in (0, 1):
            eq = resolve(pc=0, pc4=4, imm=8, branch=1,
                         branch_cond=BranchCond.EQ, alu_zero=zero)
            ne = resolve(pc=0, pc4=4, imm=8, branch=1,
                         branch_cond=BranchCond.NEQ, alu_zero=zero)
            assert eq["branch_taken"] != ne["branch_taken"]

    def test_never_condition_is_never_taken(self):
        unit = resolve(pc=0, pc4=4, imm=8, branch=1,
                       branch_cond=BranchCond.NEVER, alu_zero=1)
        assert unit["branch_taken"] == 0
        assert unit["next_pc"] == 4

    @pytest.mark.parametrize("cond,slt_result,expected", [
        (BranchCond.LT, 1, 1), (BranchCond.LT, 0, 0),
        (BranchCond.GE, 1, 0), (BranchCond.GE, 0, 1),
    ])
    def test_slt_mode_reads_bit_zero_of_the_alu_result(self, cond, slt_result, expected):
        unit = resolve("slt", pc=0, pc4=4, imm=8, branch=1,
                       branch_cond=cond, alu_result=slt_result)
        assert unit["branch_taken"] == expected

    @pytest.mark.parametrize("cond,alu_result,zero,expected", [
        (BranchCond.LT, 0xFFFF_FFFF, 0, 1),    # a - b < 0
        (BranchCond.LT, 5, 0, 0),
        (BranchCond.GE, 5, 0, 1),
        (BranchCond.GE, 0xFFFF_FFFF, 0, 0),
        (BranchCond.LE, 0, 1, 1),              # equal
        (BranchCond.LE, 0xFFFF_FFFF, 0, 1),    # less
        (BranchCond.LE, 5, 0, 0),
        (BranchCond.GT, 5, 0, 1),
        (BranchCond.GT, 0, 1, 0),              # equal is not greater
        (BranchCond.GT, 0xFFFF_FFFF, 0, 0),
    ])
    def test_sub_mode_reads_the_sign_of_the_alu_result(self, cond, alu_result,
                                                       zero, expected):
        unit = resolve("sub", pc=0, pc4=4, imm=8, branch=1, branch_cond=cond,
                       alu_result=alu_result, alu_zero=zero)
        assert unit["branch_taken"] == expected

    def test_compare_mode_is_validated(self):
        with pytest.raises(AssertionError):
            BranchResolutionUnit(compare_mode="bogus")

    @pytest.mark.parametrize("mode", ["slt", "sub"])
    def test_ordering_falls_through_to_not_taken_for_a_non_ordering_code(self, mode):
        # unit.py:67 — the defensive tail of _ordering(). evaluate() only calls
        # it for LT/GE/LE/GT, so this is only reachable directly.
        unit = BranchResolutionUnit(compare_mode=mode)
        assert unit._ordering(BranchCond.NEVER) == 0
        assert unit._ordering(BranchCond.EQ) == 0
        assert unit._ordering(BranchCond.NEQ) == 0


class TestBranchResolutionNextPc:

    def test_jal_target_is_pc_plus_imm(self):
        unit = resolve(pc=0x20, pc4=0x24, imm=0x10, jal=1)
        assert unit["next_pc"] == 0x30
        assert unit["branch_taken"] == 1

    def test_jalr_target_is_rs1_plus_imm_with_bit_zero_cleared(self):
        unit = resolve(pc=0x20, pc4=0x24, imm=3, jalr=1, rs1_data=0x100)
        assert unit["next_pc"] == 0x102          # 0x103 & ~1

    def test_jal_outranks_a_branch(self):
        unit = resolve(pc=0, pc4=4, imm=8, jal=1, branch=1,
                       branch_cond=BranchCond.EQ, alu_zero=0)
        assert unit["next_pc"] == 8

    def test_a_non_control_instruction_falls_through_to_pc4(self):
        unit = resolve(pc=0x40, pc4=0x44, imm=0x100)
        assert unit["next_pc"] == 0x44
        assert unit["branch_taken"] == 0
        assert unit["is_control"] == 0

    def test_next_pc_wraps_at_32_bits(self):
        unit = resolve(pc=0xFFFF_FFFC, pc4=0, imm=8, jal=1)
        assert unit["next_pc"] == 4


class TestBranchResolutionMispredict:
    """`mispredict` is the predictor's report card: taken != predicted."""

    @pytest.mark.parametrize("predicted,zero,expected", [
        (0, 1, 1),      # predicted not-taken, actually taken   -> mispredict
        (1, 1, 0),      # predicted taken, actually taken       -> correct
        (1, 0, 1),      # predicted taken, actually not-taken   -> mispredict
        (0, 0, 0),      # predicted not-taken, not-taken        -> correct
    ])
    def test_mispredict_compares_actual_against_prediction(self, predicted, zero,
                                                           expected):
        unit = resolve(pc=0, pc4=4, imm=8, branch=1, branch_cond=BranchCond.EQ,
                       alu_zero=zero, predicted_taken=predicted)
        assert unit["mispredict"] == expected
        assert unit["is_control"] == 1

    def test_an_unpredicted_jal_counts_as_a_mispredict(self):
        unit = resolve(pc=0, pc4=4, imm=8, jal=1, predicted_taken=0)
        assert unit["mispredict"] == 1

    def test_a_correctly_predicted_jal_is_not_a_mispredict(self):
        unit = resolve(pc=0, pc4=4, imm=8, jal=1, predicted_taken=1)
        assert unit["mispredict"] == 0

    def test_non_control_instructions_never_mispredict(self):
        # unit.py:117 — even a stale predicted_taken cannot flush the pipe.
        unit = resolve(pc=0, pc4=4, predicted_taken=1)
        assert unit["mispredict"] == 0
        assert unit["is_control"] == 0

    @pytest.mark.parametrize("kind", ["branch", "jal", "jalr"])
    def test_is_control_is_set_for_every_control_instruction(self, kind):
        unit = resolve(pc=0, pc4=4, imm=8, **{kind: 1})
        assert unit["is_control"] == 1


class TestBranchResolutionSerialization:
    """OoO mode holds a conditional branch until the ROB drains."""

    def test_a_non_serialized_unit_never_stalls(self):
        unit = resolve(pc=0, pc4=4, imm=8, branch=1,
                       branch_cond=BranchCond.EQ, alu_zero=1, rob_empty=0)
        assert unit["stall"] == 0
        assert unit["branch_taken"] == 1

    def test_a_serialized_branch_stalls_while_the_rob_is_busy(self):
        unit = resolve(pc=0, pc4=4, imm=8, branch=1, branch_cond=BranchCond.EQ,
                       alu_zero=1, rob_empty=0, serialized=True)
        assert unit["stall"] == 1
        assert unit["branch_taken"] == 0        # held, not resolved
        assert unit["next_pc"] == 4             # fall through while stalled
        assert unit["mispredict"] == 0          # no flush while stalled
        assert unit["is_control"] == 1

    def test_a_serialized_branch_resolves_once_the_rob_drains(self):
        unit = resolve(pc=0, pc4=4, imm=8, branch=1, branch_cond=BranchCond.EQ,
                       alu_zero=1, rob_empty=1, serialized=True)
        assert unit["stall"] == 0
        assert unit["branch_taken"] == 1
        assert unit["next_pc"] == 8

    def test_jalr_is_not_serialized(self):
        # Documented in unit.py:74-78: JALR dispatches to the ROB itself, so
        # stalling on it would deadlock.
        unit = resolve(pc=0, pc4=4, imm=0, jalr=1, rs1_data=0x40,
                       rob_empty=0, serialized=True)
        assert unit["stall"] == 0
        assert unit["next_pc"] == 0x40


class TestBranchResolutionState:

    def test_state_reports_next_pc_taken_and_mispredict(self):
        unit = resolve(pc=0x10, pc4=0x14, imm=8, branch=1,
                       branch_cond=BranchCond.EQ, alu_zero=1)
        assert unit.get_state() == {
            "next_pc": "0x00000018", "branch_taken": 1, "mispredict": 1,
        }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
