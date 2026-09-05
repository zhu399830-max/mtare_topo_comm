#!/usr/bin/env python3
"""Explicitly authorized V1R3 wrapper for the unchanged ERCSS evaluator."""

from __future__ import annotations

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json
import run_gse_registered_skeleton_feasibility_v1r as base


def main() -> int:
    previous_root = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_gse_registered_skeleton_feasibility_v1r2_seed0"
    previous = load_json(previous_root / "metrics/summary.json")
    if (
        "frozen tool drift: tests/v3/unit/test_gse_registered_structural_skeleton.py" not in str(previous.get("error", ""))
        or previous.get("registered_skeleton_feasibility")
        or previous.get("returncode") is not None
    ):
        raise RuntimeError("V1R2 was not the expected pre-execution stale-test-hash failure")
    base.RUN_ID = "gate3_20260830_gse_registered_skeleton_feasibility_v1r3_seed0"
    base.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_REGISTERED_SKELETON_FEASIBILITY_V1R3"
    return base.main()


if __name__ == "__main__": raise SystemExit(main())
