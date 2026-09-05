#!/usr/bin/env python3
"""Configure the generic immutable topology runner for C09 node-gated V4."""

from __future__ import annotations

from _bootstrap import PROJECT_ROOT
import run_gse_offline_topology_validation_v3 as runner


RUN_ID = "gate4_20260826_gse_offline_topology_validation_v4_seed0"
PASS_STATUS = "PASS_GSE_OFFLINE_TOPOLOGY_VALIDATION_V4"
FAIL_STATUS = "FAIL_GSE_OFFLINE_TOPOLOGY_VALIDATION_V4"


def main() -> int:
    runner.RUN_ID = RUN_ID
    runner.PASS_STATUS = PASS_STATUS
    runner.FAIL_STATUS = FAIL_STATUS
    runner.EVALUATOR = PROJECT_ROOT / "tools/v3/evaluate_gse_offline_topology_v4.py"
    runner.NODE_CALIBRATION = PROJECT_ROOT / "results/gate4_topology/gate4_20260826_gse_node_matchability_ensemble_calibration_v1_seed0"
    runner.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_OFFLINE_TOPOLOGY_VALIDATION_V4"
    runner.RUN_SCHEMA = "gse_offline_topology_run_v4"
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
