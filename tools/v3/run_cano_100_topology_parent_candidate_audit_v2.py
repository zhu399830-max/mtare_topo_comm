#!/usr/bin/env python3
"""Run and seal the approved Cano V2 bounded-resampling topology audit once."""

from __future__ import annotations

from pathlib import Path

from _bootstrap import PROJECT_ROOT
import run_cano_100_topology_parent_candidate_audit_v1 as runner


runner.RUN_ID = "gate0_20260811_cano_100_topology_parent_candidate_audit_v2_bounded_resampling_seed0"
runner.RUNNER_PATH = Path(__file__).resolve()
runner.EXECUTOR_PATH = PROJECT_ROOT / "tools/v3/execute_cano_100_topology_parent_candidate_audit_v2.py"
runner.LOG_NAME = "01_cano_100_topology_parent_candidate_audit_v2.log"
runner.METHOD_ID = "v2_bounded_parameter_resampling_exact_cycle_rank"
runner.EXTRA_TOOL_PATHS = {
    "runner_base": PROJECT_ROOT / "tools/v3/run_cano_100_topology_parent_candidate_audit_v1.py",
    "executor_base": PROJECT_ROOT / "tools/v3/execute_cano_100_topology_parent_candidate_audit_v1.py",
}


if __name__ == "__main__":
    raise SystemExit(runner.main())
