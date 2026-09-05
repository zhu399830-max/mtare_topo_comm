#!/home/zeng-workstation/anaconda3/bin/python
"""Run the six-case V4 readiness after the matched re-anchor probe passes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from mtare_topo.deployment.m1d_checkpoint import COMPOSITE_V9_MODE
from mtare_topo.governance import load_json, write_json
import run_aee_adapted_readiness_v1 as base
from run_aee_composite_v9_reanchor_probe_v1 import audit_reanchor_mechanism
from run_mtare_single_robot_stochastic_v1 import sha256


RUN_ID = "gate6_20260823_aee_composite_v9_reanchor_readiness_v1_seed20260822"
STATUS_PASS = "PASS_AEE_COMPOSITE_V9_REANCHOR_READINESS_V1"
STATUS_FAIL = "FAIL_AEE_COMPOSITE_V9_REANCHOR_READINESS_V1"
DEPLOYMENT_STATUS_PASS = "PASS_AEE_COMPOSITE_V9_ROS_DEPLOYMENT_EXPORT"
BASE_CASE_COMMAND = base.case_command
BASE_AUDIT_CASE = base.audit_case
ACTIVE_SPEC: dict[str, Any] | None = None


def validate_probe_source(spec: dict[str, Any], *, project_root: Path = base.PROJECT_ROOT) -> dict[str, Any]:
    root = project_root.resolve()
    source = (root / spec["reanchor_probe_run"]).resolve()
    source.relative_to(root)
    state_path = source / "RUN_STATE.json"
    summary_path = source / "metrics/summary.json"
    case_id = spec["reanchor_probe_case_id"]
    case_summary_path = source / f"artifacts/cases/{case_id}/summary.json"
    seal_path = source / "artifacts/evidence_sha256.txt"
    state, summary, case_summary = (
        load_json(state_path), load_json(summary_path), load_json(case_summary_path)
    )
    if state.get("state") != "COMPLETED" or state.get("overall_status") != spec["reanchor_probe_status"]:
        raise RuntimeError("V4 readiness probe source is not the exact completed PASS")
    if summary.get("overall_status") != spec["reanchor_probe_status"] or summary.get("completed_cases") != 1:
        raise RuntimeError("V4 readiness probe summary is not the exact one-case PASS")
    mechanism = summary.get("mechanism_audit", {})
    if (
        int(mechanism.get("verified_reanchor_count", 0)) < 1
        or mechanism.get("post_reanchor_route_growth") is not True
        or mechanism.get("uses_evaluator_gt") is not False
    ):
        raise RuntimeError("V4 readiness probe lacks causal re-anchor and post-transition progress")
    if case_summary.get("status") != "PASS_SINGLE_ROBOT_CASE_V2":
        raise RuntimeError("V4 readiness probe case is not finalized V2 PASS")
    if sha256(seal_path) != spec["reanchor_probe_seal_sha256"]:
        raise RuntimeError("V4 readiness probe seal identity drift")
    entries = {
        path: digest
        for digest, path in (
            line.split("  ", 1) for line in seal_path.read_text(encoding="utf-8").splitlines()
        )
    }
    bound = {}
    for path in (state_path, summary_path, case_summary_path):
        relative = path.relative_to(root).as_posix()
        digest = sha256(path)
        if entries.get(relative) != digest:
            raise RuntimeError(f"V4 readiness probe bound evidence drift: {relative}")
        bound[relative] = digest
    return {
        "schema_version": "aee_composite_v9_reanchor_probe_source_v1",
        "run": spec["reanchor_probe_run"],
        "status": spec["reanchor_probe_status"],
        "seal_sha256": spec["reanchor_probe_seal_sha256"],
        "case_id": case_id,
        "verified_reanchor_count": mechanism["verified_reanchor_count"],
        "post_reanchor_route_growth": True,
        "uses_evaluator_gt": False,
        "verified_bound_files": bound,
    }


def case_command(
    run_dir: Path,
    case: dict[str, Any],
    checkpoint: dict[str, Any],
    topic_contract: str = "configs/v3/gate6/closed_loop_recording_topics_aee_v2.json",
) -> tuple[list[str], str]:
    if ACTIVE_SPEC is None:
        raise RuntimeError("V4 readiness active spec is unavailable")
    probe_path = run_dir / "config/reanchor_probe_source.json"
    if not probe_path.exists():
        write_json(probe_path, validate_probe_source(ACTIVE_SPEC))
    command, old_name = BASE_CASE_COMMAND(run_dir, case, checkpoint, topic_contract)
    new_name = f"aee-v4-readiness-{case['case_id']}"
    command = [new_name if item == old_name else item for item in command]
    command[-1] = command[-1].replace(
        "/workspace/tools/v3/run_mtare_single_robot_case_v1.py",
        "/workspace/tools/v3/run_mtare_single_robot_case_v2.py",
    )
    if "/workspace/tools/v3/run_mtare_single_robot_case_v2.py" not in command[-1]:
        raise RuntimeError("V4 readiness failed to select the V2 case wrapper")
    return command, new_name


def audit_case(case_dir: Path, expected_case: dict[str, Any]) -> dict[str, Any]:
    qualification = BASE_AUDIT_CASE(case_dir, expected_case)
    qualification["reanchor_mechanism"] = audit_reanchor_mechanism(
        case_dir, require_reanchor=False
    )
    return qualification


def main() -> int:
    global ACTIVE_SPEC
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--spec", required=True, type=Path)
    known, _ = parser.parse_known_args(sys.argv[1:])
    ACTIVE_SPEC = load_json(known.spec.resolve())
    original_case_command = base.case_command
    original_audit_case = base.audit_case
    base.RUN_ID = RUN_ID
    base.STATUS_PASS = STATUS_PASS
    base.STATUS_FAIL = STATUS_FAIL
    base.DEPLOYMENT_STATUS_PASS = DEPLOYMENT_STATUS_PASS
    base.EXPECTED_CHECKPOINT_MODE = COMPOSITE_V9_MODE
    base.case_command = case_command
    base.audit_case = audit_case
    try:
        return base.main()
    finally:
        base.case_command = original_case_command
        base.audit_case = original_audit_case


if __name__ == "__main__":
    raise SystemExit(main())
