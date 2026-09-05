#!/home/zeng-workstation/anaconda3/bin/python
"""Retry the exact V5 matrix after binding the inherited case executor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import run_aee_composite_v9_combined_correction_stochastic_v1 as base
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate6_20260823_aee_composite_v9_combined_correction_stochastic_v1r2_seed20260820"
STATUS_PASS = "PASS_AEE_COMPOSITE_V9_COMBINED_CORRECTION_STOCHASTIC_V1R2"
STATUS_FAIL = "FAIL_AEE_COMPOSITE_V9_COMBINED_CORRECTION_STOCHASTIC_V1R2"
FAILED_STATUS = "FAIL_AEE_COMPOSITE_V9_COMBINED_CORRECTION_STOCHASTIC_V1R"
FAILED_REASON = "module 'run_mtare_single_robot_stochastic_v2' has no attribute 'run_case'"


def verify_failed_source(spec: dict, project_root: Path = base.base.PROJECT_ROOT) -> dict:
    root = project_root.resolve()
    source = (root / spec["failed_stochastic_run"]).resolve()
    source.relative_to(root)
    state = load_json(source / "RUN_STATE.json")
    summary = load_json(source / "metrics/summary.json")
    seal_path = source / "artifacts/evidence_sha256.txt"
    if (
        state.get("state") != "FAILED"
        or state.get("overall_status") != FAILED_STATUS
        or summary.get("overall_status") != FAILED_STATUS
        or summary.get("completed_case_count") != 0
        or summary.get("failure_reason") != FAILED_REASON
    ):
        raise RuntimeError("V1R2 source is not the exact zero-case executor-binding failure")
    if base.base.matrix_base.sha256(seal_path) != spec["failed_stochastic_seal_sha256"]:
        raise RuntimeError("V1R2 failed-source seal identity drift")
    entries = 0
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if base.base.matrix_base.sha256(root / relative) != digest:
            raise RuntimeError(f"V1R2 failed-source evidence drift: {relative}")
        entries += 1
    return {"schema_version": "aee_combined_stochastic_failed_source_v1r2", "run": spec["failed_stochastic_run"], "status": FAILED_STATUS, "seal_sha256": spec["failed_stochastic_seal_sha256"], "sealed_files": entries, "completed_cases": 0, "partial_reuse": False}


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    known, _ = parser.parse_known_args(sys.argv[1:])
    spec = load_json(known.spec.resolve())
    write_json(known.run_dir.resolve() / "config/failed_stochastic_source.json", verify_failed_source(spec))
    base.RUN_ID = RUN_ID
    base.STATUS_PASS = STATUS_PASS
    base.STATUS_FAIL = STATUS_FAIL
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
