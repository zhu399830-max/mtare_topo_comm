#!/usr/bin/env python3
"""Run once and seal the approved Gate-1 V2R export."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import run_cano_phase2_supervised_range_dataset_v2 as base

from mtare_topo.governance import load_json, write_json

RUN_ID = "gate1_20260812_cano_phase2_supervised_range_dataset_v2r_seed0"
EXECUTOR = base.PROJECT_ROOT / "tools/v3/execute_cano_phase2_supervised_range_dataset_v2r.py"


def main() -> int:
    base.RUN_ID = RUN_ID
    base.EXECUTOR = EXECUTOR
    result = base.main()
    run_dir = Path(sys.argv[sys.argv.index("--run-dir") + 1]).resolve()
    runner_summary = load_json(run_dir / "metrics/runner_summary.json")
    run_state = load_json(run_dir / "RUN_STATE.json")
    passed = result == 0
    overall = (
        "PASS_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V2R"
        if passed else "FAIL_CANO_PHASE2_SUPERVISED_RANGE_DATASET_V2R"
    )
    runner_summary["overall_status"] = overall
    runner_summary["selector_version"] = "V2R"
    runner_summary["tunnel_quota_basis"] = "eligible_candidate_capacity_after_objective_audit"
    write_json(run_dir / "metrics/runner_summary.json", runner_summary)
    summary_path = run_dir / "metrics/summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        summary["overall_status"] = overall
        write_json(summary_path, summary)
    run_state["overall_status"] = overall
    run_state["note"] = "Formal eligibility-first V2R dataset; zero training or benchmark reads."
    write_json(run_dir / "RUN_STATE.json", run_state)
    sealed = base._seal_manifest(run_dir)
    print(json.dumps({"overall_status": overall, "final_v2r_sealed_files": sealed}, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
