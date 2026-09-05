#!/usr/bin/env python3
"""Run and seal one approved read-only Cano topology recipe reclassification."""

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
from run_cano_five_topology_cpu_contract_pilot import (
    E1_PYTHON,
    _approved_project_file,
    _seal_manifest,
    _sha256,
)


RUN_ID = "gate0_20260811_cano_100_topology_parent_recipe_reclassification_v2r_seed0"
SOURCE_RUN = PROJECT_ROOT / (
    "results/gate0_baseline/"
    "gate0_20260811_cano_100_topology_parent_candidate_audit_v2_bounded_resampling_seed0"
)
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_cano_100_topology_parent_recipe_reclassification_v2r.py"
EXPECTED_SCOPE = {
    "sealed_candidates_read": 120,
    "selected_parent_references": 100,
    "new_topology_constructions": 0,
    "new_replays": 0,
    "new_graphs": 0,
    "new_splines": 0,
    "new_previews": 0,
    "meshes": 0,
    "lidar_observations": 0,
    "teacher_labels": 0,
    "formal_dataset_samples": 0,
    "training_samples": 0,
    "models": 0,
    "trajectories": 0,
    "mtare_changes": 0,
}


def _verify_source_seal() -> dict:
    manifest = SOURCE_RUN / "artifacts/evidence_sha256.txt"
    lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    mismatches = []
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = PROJECT_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            mismatches.append(relative)
    return {
        "manifest": str(manifest.relative_to(PROJECT_ROOT)),
        "listed_file_count": len(lines),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "manifest_sha256": _sha256(manifest),
    }


def _run_executor(run_dir: Path) -> tuple[int, float]:
    argv = [str(E1_PYTHON), str(EXECUTOR), "--run-dir", str(run_dir)]
    started = time.monotonic()
    completed = subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    duration = time.monotonic() - started
    (run_dir / "logs/01_recipe_reclassification_v2r.log").write_text(
        "argv=" + json.dumps(argv) + "\n"
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
        or spec.get("data_scope") != EXPECTED_SCOPE
    ):
        raise RuntimeError("spec approval or read-only scope is incompatible")
    proposal_path = _approved_project_file(spec.get("config_path"), "config_path")
    card_path = _approved_project_file(spec.get("topology_card"), "topology_card")
    proposal = load_json(proposal_path)
    card = load_json(card_path)
    if (
        proposal.get("status") != "APPROVED_FOR_ONE_EXECUTION"
        or proposal.get("approval", {}).get("status") != "APPROVED"
        or card.get("status") != "APPROVED_FOR_ONE_EXECUTION"
        or card.get("approval", {}).get("status") != "APPROVED"
    ):
        raise RuntimeError("approved V2R proposal/card is missing")

    tool_paths = {
        "runner": Path(__file__).resolve(),
        "executor": EXECUTOR,
        "recipe_helper": PROJECT_ROOT / "src/mtare_topo/data/cano_topology_recipe_reclassification.py",
        "unit_test": PROJECT_ROOT / "tests/v3/unit/test_cano_topology_recipe_reclassification.py",
        "runner_helpers": PROJECT_ROOT / "tools/v3/run_cano_five_topology_cpu_contract_pilot.py",
        "governance": PROJECT_ROOT / "src/mtare_topo/governance.py",
        "proposal": proposal_path,
        "data_card": card_path,
    }
    observed_hashes = {name: _sha256(path) for name, path in tool_paths.items()}
    for name, frozen in spec.get("frozen_tools", {}).items():
        if observed_hashes.get(name) != frozen.get("sha256"):
            raise RuntimeError(f"frozen tool mismatch: {name}")
    if load_json(SOURCE_RUN / "RUN_STATE.json").get("state") != "FAILED":
        raise RuntimeError("sealed V2 source state is not FAILED")
    source_summary = load_json(SOURCE_RUN / "metrics/summary.json")
    if (
        source_summary.get("overall_status") != "FAIL_CANO_100_TOPOLOGY_PARENT_CANDIDATE_AUDIT"
        or source_summary.get("scope", {}).get("candidate_topology_constructions") != 120
        or source_summary.get("scope", {}).get("same_seed_topology_replays_attempted") != 120
    ):
        raise RuntimeError("sealed V2 source summary identity is incompatible")
    source_seal = _verify_source_seal()
    if source_seal["listed_file_count"] != 391 or source_seal["mismatch_count"] != 0:
        raise RuntimeError(f"sealed V2 evidence verification failed: {source_seal}")
    frozen_source = spec.get("frozen_source_files", {})
    for relative, expected in frozen_source.items():
        if _sha256(_approved_project_file(relative, "frozen_source_files")) != expected:
            raise RuntimeError(f"frozen source identity mismatch: {relative}")

    write_json(run_dir / "config/tool_hashes.json", observed_hashes)
    write_json(run_dir / "config/source_seal_verification.json", source_seal)
    shutil.copy2(card_path, run_dir / "config/topology_card.json")
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "Approved read-only V2R; zero topology generation, mesh, LiDAR and training.",
        },
    )
    exit_code, duration = _run_executor(run_dir)
    summary = load_json(run_dir / "metrics/summary.json") if (run_dir / "metrics/summary.json").is_file() else None
    artifact_json = sorted((run_dir / "artifacts").glob("*.json"))
    executor_metric_json = sorted((run_dir / "metrics").glob("*.json"))
    previews = sorted((run_dir / "previews").glob("*.png"))
    result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    passed = bool(
        exit_code == 0
        and duration <= 60
        and summary is not None
        and summary.get("overall_status") == "PASS_CANO_100_TOPOLOGY_PARENT_RECIPE_RECLASSIFICATION_V2R"
        and summary.get("scope") == EXPECTED_SCOPE
        and summary.get("batch_audit", {}).get("passed") is True
        and len(artifact_json) == 3
        and len(executor_metric_json) == 2
        and len(previews) == 0
        and result_bytes <= 10 * 1024**2
    )
    overall = (
        "PASS_CANO_100_TOPOLOGY_PARENT_RECIPE_RECLASSIFICATION_V2R"
        if passed
        else "FAIL_CANO_100_TOPOLOGY_PARENT_RECIPE_RECLASSIFICATION_V2R"
    )
    runner_summary = {
        "schema_version": "cano_topology_recipe_reclassification_runner_summary_v2r",
        "run_id": RUN_ID,
        "overall_status": overall,
        "executor_exit_code": exit_code,
        "duration_seconds": duration,
        "counts": {
            "artifact_manifests": len(artifact_json),
            "executor_metric_summaries": len(executor_metric_json),
            "new_previews": len(previews),
        },
        "source_seal": source_seal,
        "result_bytes_before_seal": result_bytes,
        "disk_limit_bytes": 10 * 1024**2,
        "claim_boundary": "Topology recipe metadata and split only; zero topology generation, mesh, LiDAR, labels, training, models, trajectories and M-TARE changes.",
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
    sealed_files = _seal_manifest(run_dir)
    print(json.dumps({"overall_status": overall, "sealed_files": sealed_files}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
