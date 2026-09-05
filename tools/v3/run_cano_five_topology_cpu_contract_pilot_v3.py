#!/usr/bin/env python3
"""Run and seal one approved five-topology selector-v2 CPU LiDAR pilot."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_cano_five_topology_cpu_contract_pilot import (
    E1_PYTHON,
    PREREQUISITE_DIR,
    _approved_project_file,
    _e1_identity,
    _executor_environment,
    _inspect_outputs,
    _seal_manifest,
    _sha256,
    _subt_proc_gen_import_identity,
    _upstream_identity,
)


RUN_ID = "gate0_20260811_cano_five_topology_cpu_contract_pilot_v3_selector_coverage_seed0"
SELECTOR_AUDIT_DIR = PROJECT_ROOT / (
    "results/gate0_baseline/"
    "gate0_20260811_cano_anchor_selector_coverage_audit_v2_seed0"
)


EXPECTED_SCOPE = {
    "topology_parents": 5,
    "primary_topology_constructions": 5,
    "topology_replay_constructions": 5,
    "native_perception_meshes": 5,
    "canonical_anchors": 250,
    "diagnostic_observations": 750,
    "rays_per_observation": 11520,
    "independent_scene_passes": 2,
    "total_primary_rays": 17280000,
    "formal_dataset_worlds": 0,
    "formal_dataset_samples": 0,
    "labels_for_training": 0,
    "training_samples": 0,
    "models": 0,
    "ai_labels": 0,
    "trajectories": 0,
    "isaac_runs": 0,
    "gazebo_runs": 0,
    "topology_runtime_changes": 0,
    "mtare_changes": 0,
}


def _run_executor(run_dir: Path) -> tuple[int, float]:
    argv = [
        str(E1_PYTHON),
        str(PROJECT_ROOT / "tools/v3/execute_cano_five_topology_cpu_contract_pilot_v3.py"),
        "--run-dir",
        str(run_dir),
    ]
    started = time.monotonic()
    output = ""
    exit_code = 124
    try:
        completed = subprocess.run(
            argv,
            cwd=PROJECT_ROOT,
            env=_executor_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=7200,
            check=False,
        )
        output = completed.stdout
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        captured = exc.stdout or ""
        output = captured.decode() if isinstance(captured, bytes) else captured
        output += "\nTIMEOUT: frozen 7200 second limit reached.\n"
    duration = time.monotonic() - started
    (run_dir / "logs/01_five_topology_cpu_contract_v3.log").write_text(
        "argv="
        + json.dumps(argv, ensure_ascii=False)
        + "\n"
        + f"finished_at_utc={datetime.now(timezone.utc).isoformat()}\n"
        + output
        + f"\nduration_seconds={duration:.6f}\nexit_code={exit_code}\n",
        encoding="utf-8",
    )
    print(output, end="", flush=True)
    return exit_code, duration


def _completed_prerequisite(path: Path, expected_status: str) -> dict:
    summary = load_json(path / "metrics/summary.json")
    state = load_json(path / "RUN_STATE.json")
    if summary.get("overall_status") != expected_status or state.get("state") != "COMPLETED":
        raise RuntimeError(f"prerequisite is not a sealed completed PASS: {path}")
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "overall_status": expected_status,
        "summary_sha256": _sha256(path / "metrics/summary.json"),
        "run_state_sha256": _sha256(path / "RUN_STATE.json"),
    }


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
    authorization = spec.get("user_authorization", {})
    if (
        spec.get("operation") != "sensor_contract_pilot"
        or spec.get("gate") != 0
        or spec.get("seed") != 0
        or authorization.get("status") != "APPROVED"
    ):
        raise RuntimeError("spec is not the approved Gate-0 selector-v2 LiDAR pilot")
    if spec.get("data_scope") != EXPECTED_SCOPE:
        raise RuntimeError(f"approved pilot scope drifted: {spec.get('data_scope')}")

    tool_paths = {
        "runner": Path(__file__).resolve(),
        "executor": PROJECT_ROOT / "tools/v3/execute_cano_five_topology_cpu_contract_pilot_v3.py",
        "legacy_runner_helper": PROJECT_ROOT / "tools/v3/run_cano_five_topology_cpu_contract_pilot.py",
        "legacy_executor": PROJECT_ROOT / "tools/v3/execute_cano_five_topology_cpu_contract_pilot.py",
        "support": PROJECT_ROOT / "tools/v3/cano_five_topology_support.py",
        "contract": PROJECT_ROOT / "src/mtare_topo/data/cano_contract_pilot.py",
        "sensor": PROJECT_ROOT / "src/mtare_topo/data/cano_sensor_smoke.py",
        "baseline": PROJECT_ROOT / "src/mtare_topo/semantics/range_exit_baseline.py",
        "governance": PROJECT_ROOT / "src/mtare_topo/governance.py",
        "unit_test": PROJECT_ROOT / "tests/v3/unit/test_cano_contract_pilot.py",
        "external_test": PROJECT_ROOT / "tests/v3/external/test_cano_adapter_contract.py",
        "proposal": _approved_project_file(spec.get("config_path"), "config_path"),
        "data_card": _approved_project_file(spec.get("data_card"), "data_card"),
    }
    observed_hashes = {name: _sha256(path) for name, path in tool_paths.items()}
    for name, frozen in spec.get("frozen_tools", {}).items():
        if observed_hashes.get(name) != frozen.get("sha256"):
            raise RuntimeError(
                f"frozen hash mismatch for {name}: "
                f"{observed_hashes.get(name)} != {frozen.get('sha256')}"
            )

    lidar_prerequisite = _completed_prerequisite(
        PREREQUISITE_DIR, "PASS_CANO_CPU_RAYCAST_LIDAR_CONTRACT"
    )
    selector_prerequisite = _completed_prerequisite(
        SELECTOR_AUDIT_DIR, "PASS_CANO_ANCHOR_SELECTOR_COVERAGE_AUDIT_V2"
    )
    frozen_selector = spec.get("selector_v2_prerequisite", {})
    if (
        selector_prerequisite["summary_sha256"]
        != frozen_selector.get("summary_sha256")
        or selector_prerequisite["run_state_sha256"]
        != frozen_selector.get("run_state_sha256")
        or selector_prerequisite["path"] != frozen_selector.get("path")
    ):
        raise RuntimeError("selector-v2 prerequisite identity drifted from approved spec")
    selector_summary = load_json(SELECTOR_AUDIT_DIR / "metrics/summary.json")
    if (
        selector_summary.get("all_parent_audits_passed") is not True
        or selector_summary.get("all_arc_length_fill_quotas_match") is not True
        or selector_summary.get("minimum_pairwise_distance_across_parents_m", 0.0) < 5.0
        or selector_summary.get("maximum_full_spline_coverage_radius_across_parents_m", 99.0) > 7.5
    ):
        raise RuntimeError("selector-v2 prerequisite metrics are not a full PASS")
    upstream = _upstream_identity()
    if upstream != spec.get("upstream_identity"):
        raise RuntimeError(f"upstream identity mismatch: {upstream}")
    environment = _e1_identity()
    if environment["pip_freeze_evidence_sha256"] != spec.get("pip_freeze_evidence_sha256"):
        raise RuntimeError("frozen E1 pip-freeze evidence identity mismatch")
    import_identity = _subt_proc_gen_import_identity(_executor_environment())

    write_json(run_dir / "config/tool_hashes.json", observed_hashes)
    write_json(run_dir / "config/upstream_identity.json", upstream)
    write_json(run_dir / "config/e1_environment.json", environment)
    write_json(run_dir / "config/subt_proc_gen_import_identity.json", import_identity)
    write_json(
        run_dir / "config/prerequisite_identities.json",
        {"one_world_lidar": lidar_prerequisite, "selector_v2": selector_prerequisite},
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "Approved five-topology selector-v2 diagnostic LiDAR contract; zero formal data and training.",
        },
    )

    exit_code, duration = _run_executor(run_dir)
    inspection = _inspect_outputs(run_dir)
    summary_path = run_dir / "metrics/summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else None
    counts = inspection["counts"]
    result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    expected_executor_scope = {
        key: value
        for key, value in EXPECTED_SCOPE.items()
        if key not in {"primary_topology_constructions", "topology_replay_constructions"}
    }
    expected_executor_scope["online_graph_metrics"] = (
        "NOT_APPLICABLE_STATIC_ANCHORS_HAVE_NO_CAUSAL_MOVEMENT_EDGES"
    )
    selector = summary.get("selector_v2_audit", {}) if summary else {}
    counts_pass = counts == {
        "world_bundles": 5,
        "diagnostic_shards": 5,
        "parent_metric_files": 5,
        "world_maps": 5,
        "contact_pages": 30,
        "manifest_observations": 750,
    }
    passed = bool(
        exit_code == 0
        and duration <= 7200
        and summary is not None
        and summary.get("schema_version") == "cano_five_topology_cpu_contract_pilot_summary_v3"
        and summary.get("overall_status") == "PASS_CANO_FIVE_TOPOLOGY_CPU_CONTRACT_PILOT"
        and summary.get("scope") == expected_executor_scope
        and summary.get("source_unchanged") is True
        and selector.get("parents_passed") == 5
        and selector.get("all_arc_length_fill_quotas_match") is True
        and selector.get("minimum_pairwise_distance_across_parents_m", 0.0) >= 5.0
        and selector.get("maximum_full_spline_coverage_radius_across_parents_m", 99.0) <= 7.5
        and not inspection["missing_required_evidence"]
        and inspection["shards_pass"]
        and counts_pass
        and result_bytes <= 2 * 1024**3
    )
    overall = (
        "PASS_CANO_FIVE_TOPOLOGY_CPU_CONTRACT_PILOT"
        if passed
        else "FAIL_CANO_FIVE_TOPOLOGY_CPU_CONTRACT_PILOT"
    )
    runner_summary = {
        "schema_version": "cano_five_topology_cpu_contract_runner_summary_v3",
        "run_id": RUN_ID,
        "overall_status": overall,
        "executor_exit_code": exit_code,
        "duration_seconds": duration,
        "inspection": inspection,
        "selector_v2_audit": selector,
        "counts_pass": counts_pass,
        "result_bytes_before_seal": result_bytes,
        "disk_limit_bytes": 2 * 1024**3,
        "host": {
            "platform": platform.platform(),
            "free_disk_bytes_after": shutil.disk_usage(PROJECT_ROOT).free,
        },
        "claim_boundary": (
            "Diagnostic static-view evidence only; no formal dataset, learned model, "
            "causal trajectory graph, navigation result or M-TARE comparison."
        ),
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
