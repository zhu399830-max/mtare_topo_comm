#!/usr/bin/env python3
"""Run and seal one approved selector-v2 full-spline coverage audit."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_cano_anchor_selector_zero_raycast_audit import (
    E1_PYTHON,
    SOURCE_RUN,
    _environment,
    _environment_identity,
    _project_file,
    _seal,
    _sha256,
)


RUN_ID = "gate0_20260811_cano_anchor_selector_coverage_audit_v2_seed0"


def _run_executor(run_dir: Path, environment: dict[str, str]) -> tuple[int, float]:
    argv = [
        str(E1_PYTHON),
        str(PROJECT_ROOT / "tools/v3/execute_cano_anchor_selector_coverage_audit_v2.py"),
        "--run-dir",
        str(run_dir),
    ]
    started = time.monotonic()
    completed = subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
        check=False,
    )
    duration = time.monotonic() - started
    (run_dir / "logs/01_selector_v2_coverage_audit.log").write_text(
        "argv="
        + json.dumps(argv)
        + "\n"
        + f"finished_at_utc={datetime.now(timezone.utc).isoformat()}\n"
        + completed.stdout
        + f"\nduration_seconds={duration:.6f}\nexit_code={completed.returncode}\n",
        encoding="utf-8",
    )
    print(completed.stdout, end="", flush=True)
    return completed.returncode, duration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or not run_dir.is_dir():
        raise RuntimeError(f"runner accepts only {RUN_ID}")
    if load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("run is not in one-time CREATED_NOT_EXECUTED state")
    if (
        spec.get("operation") != "infrastructure"
        or spec.get("gate") != 0
        or spec.get("seed") != 0
        or spec.get("user_authorization", {}).get("status") != "APPROVED"
    ):
        raise RuntimeError("spec is not the approved Gate-0 selector-v2 audit")
    expected_scope = {
        "sealed_graph_spline_parents_reused": 4,
        "new_graph_only_topology_constructions": 1,
        "new_meshes": 0,
        "canonical_anchors": 250,
        "raycast_observations": 0,
        "rays": 0,
        "formal_dataset_worlds": 0,
        "formal_dataset_samples": 0,
        "training_samples": 0,
        "models": 0,
        "trajectories": 0,
        "topology_runtime_changes": 0,
        "mtare_changes": 0,
    }
    if spec.get("data_scope") != expected_scope:
        raise RuntimeError("approved selector-v2 scope drifted")

    tool_paths = {
        "runner": Path(__file__).resolve(),
        "executor": PROJECT_ROOT / "tools/v3/execute_cano_anchor_selector_coverage_audit_v2.py",
        "v1_runner_helper": PROJECT_ROOT / "tools/v3/run_cano_anchor_selector_zero_raycast_audit.py",
        "v1_executor_helper": PROJECT_ROOT / "tools/v3/execute_cano_anchor_selector_zero_raycast_audit.py",
        "contract": PROJECT_ROOT / "src/mtare_topo/data/cano_contract_pilot.py",
        "support": PROJECT_ROOT / "tools/v3/cano_five_topology_support.py",
        "unit_test": PROJECT_ROOT / "tests/v3/unit/test_cano_contract_pilot.py",
        "external_test": PROJECT_ROOT / "tests/v3/external/test_cano_adapter_contract.py",
        "proposal": _project_file(spec.get("config_path"), "config_path"),
    }
    observed_hashes = {name: _sha256(path) for name, path in tool_paths.items()}
    for name, frozen in spec.get("frozen_tools", {}).items():
        if observed_hashes.get(name) != frozen.get("sha256"):
            raise RuntimeError(
                f"frozen hash mismatch for {name}: {observed_hashes.get(name)} != {frozen.get('sha256')}"
            )
    if load_json(SOURCE_RUN / "RUN_STATE.json").get("state") != "FAILED":
        raise RuntimeError("source v2b run is not sealed FAILED evidence")
    for relative, expected_hash in spec.get("frozen_source_files", {}).items():
        if _sha256(_project_file(relative, "frozen_source_files")) != expected_hash:
            raise RuntimeError(f"frozen source identity mismatch: {relative}")
    environment = _environment()
    identity = _environment_identity(environment)
    write_json(run_dir / "config/tool_hashes.json", observed_hashes)
    write_json(run_dir / "config/executor_environment_identity.json", identity)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "Approved selector-v2 graph/spline-only coverage audit; zero mesh and rays.",
        },
    )

    exit_code, duration = _run_executor(run_dir, environment)
    summary_path = run_dir / "metrics/summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else None
    parent_metrics = sorted((run_dir / "metrics").glob("P*.json"))
    anchors = sorted((run_dir / "artifacts/anchors").glob("P*.json"))
    previews = sorted((run_dir / "previews").glob("*.png"))
    p05_files = sorted((run_dir / "artifacts/p05_graph_only").glob("*.json"))
    result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    passed = bool(
        exit_code == 0
        and duration <= 300
        and summary is not None
        and summary.get("overall_status") == "PASS_CANO_ANCHOR_SELECTOR_COVERAGE_AUDIT_V2"
        and summary.get("scope") == expected_scope
        and summary.get("all_parent_audits_passed") is True
        and summary.get("all_arc_length_fill_quotas_match") is True
        and summary.get("all_replays_identical") is True
        and summary.get("minimum_pairwise_distance_across_parents_m", 0.0) >= 5.0
        and summary.get("maximum_full_spline_coverage_radius_across_parents_m", 99.0) <= 7.5
        and len(parent_metrics) == 5
        and len(anchors) == 5
        and len(previews) == 6
        and len(p05_files) == 2
        and result_bytes <= 100 * 1024**2
    )
    overall = (
        "PASS_CANO_ANCHOR_SELECTOR_COVERAGE_AUDIT_V2"
        if passed
        else "FAIL_CANO_ANCHOR_SELECTOR_COVERAGE_AUDIT_V2"
    )
    runner_summary = {
        "schema_version": "cano_anchor_selector_coverage_runner_summary_v2",
        "run_id": RUN_ID,
        "overall_status": overall,
        "executor_exit_code": exit_code,
        "duration_seconds": duration,
        "counts": {
            "parent_metrics": len(parent_metrics),
            "anchor_artifacts": len(anchors),
            "previews": len(previews),
            "p05_graph_spline_files": len(p05_files),
        },
        "result_bytes_before_seal": result_bytes,
        "disk_limit_bytes": 100 * 1024**2,
        "free_disk_bytes_after": shutil.disk_usage(PROJECT_ROOT).free,
        "claim_boundary": "Selector-v2 sampling evidence only; no mesh, rays, formal dataset, training, topology runtime or M-TARE changes.",
    }
    write_json(run_dir / "metrics/runner_summary.json", runner_summary)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if passed else "FAILED",
            "overall_status": overall,
            "note": runner_summary["claim_boundary"],
        },
    )
    sealed_files = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "sealed_files": sealed_files}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
