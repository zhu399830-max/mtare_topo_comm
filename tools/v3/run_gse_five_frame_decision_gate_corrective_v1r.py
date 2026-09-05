#!/usr/bin/env python3
"""V1R runner correcting only the frozen SciPy patch-version contract."""

from __future__ import annotations

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_evidence_integrity import verify_failed_component_run_seal
import run_gse_five_frame_decision_gate_corrective_v1 as implementation


implementation.RUN_ID = "gate3_20260828_gse_five_frame_decision_gate_corrective_v1r_seed0"
implementation.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_FIVE_FRAME_DECISION_GATE_CORRECTIVE_V1R"
_base_sources = implementation._sources
_failed = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_five_frame_decision_gate_corrective_v1_seed0"


def _sources():
    return {
        **_base_sources(),
        "failed_environment_checker_v1": verify_failed_component_run_seal(
            PROJECT_ROOT, _failed, "FAIL_GSE_FIVE_FRAME_DECISION_GATE_C09_V1"
        ),
    }


implementation._sources = _sources


if __name__ == "__main__":
    raise SystemExit(implementation.main())
