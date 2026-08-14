"""
Hazard detection unit — detects load-use (RAW) hazards and generates stall signals.
"""
from sim.component.base import ComponentBase, Port


class HazardDetector(ComponentBase):
    """
    Detects load-use hazards: if the ID/EX stage has a load (mem_read=1)
    and the instruction in IF/ID depends on the load destination.
    """
    name = "hazard_detector"
    ui_label = "Hazard Detector"
    ui_category = "control"

    def __init__(self, zero_reg_index: int | None = 0, **kw):
        super().__init__(**kw)
        # The register that always reads 0 and therefore never needs a
        # load-use interlock (RISC-V x0 = 0, ARM XZR = 31). x86 has no zero
        # register, so pass None — every register, including EAX (index 0),
        # must stall. Mirrors ForwardingUnit's parameterisation; hardcoding 0
        # here silently skipped the interlock for ARM X0 / x86 EAX, and the
        # consumer then forwarded the load's ADDRESS instead of its data.
        self._zero = -1 if zero_reg_index is None else zero_reg_index

    ports_spec = {
        # From ID/EX register
        "id_ex_mem_read": Port(1, "in",  "ID/EX stage is a load"),
        "id_ex_rd":       Port(5, "in",  "ID/EX destination register"),
        # From decoder (current instruction in ID stage)
        "if_id_rs1":      Port(5, "in",  "IF/ID source register 1"),
        "if_id_rs2":      Port(5, "in",  "IF/ID source register 2"),
        # Outputs
        "stall":          Port(1, "out", "Stall the pipeline (IF and IF/ID hold)"),
    }

    def evaluate(self):
        stall = 0
        if self["id_ex_mem_read"]:
            rd = self["id_ex_rd"]
            if rd != self._zero:  # the zero register never needs an interlock
                if rd == self["if_id_rs1"] or rd == self["if_id_rs2"]:
                    stall = 1
        self["stall"] = stall

    def get_state(self):
        result = {
            "stall": "STALL" if self["stall"] else "OK",
        }
        if self["stall"]:
            rd = self["id_ex_rd"]
            rs1 = self["if_id_rs1"]
            rs2 = self["if_id_rs2"]
            result["hazard_type"] = "Load-Use"
            result["stall_rd"] = rd
            conflicting = []
            if rd == rs1:
                conflicting.append(rs1)
            if rd == rs2 and rs2 != rs1:
                conflicting.append(rs2)
            result["conflicting_regs"] = conflicting
        return result
