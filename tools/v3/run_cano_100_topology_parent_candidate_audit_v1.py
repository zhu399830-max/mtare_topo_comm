#!/usr/bin/env python3
"""Run and seal the approved Cano topology-only 120-candidate audit once."""

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
from mtare_topo.data.cano_topology_parent_audit import canonical_json_hash
from run_cano_five_topology_cpu_contract_pilot import (
    E1_PYTHON,
    _approved_project_file,
    _e1_identity,
    _executor_environment,
    _seal_manifest,
    _sha256,
    _subt_proc_gen_import_identity,
    _upstream_identity,
)


RUN_ID = "gate0_20260811_cano_100_topology_parent_candidate_audit_v1_seed0"
RUNNER_PATH = Path(__file__).resolve()
EXECUTOR_PATH = PROJECT_ROOT / "tools/v3/execute_cano_100_topology_parent_candidate_audit_v1.py"
LOG_NAME = "01_cano_100_topology_parent_candidate_audit.log"
METHOD_ID = "v1_single_parameter_draw"
EXTRA_TOOL_PATHS: dict[str, Path] = {}
EXPECTED_SCOPE = {
    "candidate_topology_constructions": 120,
    "same_seed_topology_replays_maximum": 120,
    "retained_topology_parents": 100,
    "train_parents": 80,
    "validation_parents": 10,
    "development_test_parents": 10,
    "meshes": 0,
    "anchors": 0,
    "lidar_observations": 0,
    "rays": 0,
    "teacher_labels": 0,
    "formal_dataset_samples": 0,
    "training_samples": 0,
    "models": 0,
    "trajectories": 0,
    "mtare_changes": 0,
}


def _run_executor(run_dir: Path) -> tuple[int, float]:
    argv = [
        str(E1_PYTHON),
        str(EXECUTOR_PATH),
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
    (run_dir / "logs" / LOG_NAME).write_text(
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
        spec.get("operation") != "infrastructure"
        or spec.get("gate") != 0
        or spec.get("seed") != 0
        or authorization.get("status") != "APPROVED"
    ):
        raise RuntimeError("spec is not the approved Gate-0 topology candidate audit")
    if spec.get("data_scope") != EXPECTED_SCOPE:
        raise RuntimeError(f"approved topology scope drifted: {spec.get('data_scope')}")

    proposal_path = _approved_project_file(spec.get("config_path"), "config_path")
    data_card_path = _approved_project_file(spec.get("topology_card"), "topology_card")
    proposal = load_json(proposal_path)
    data_card = load_json(data_card_path)
    if (
        proposal.get("status") != "APPROVED_FOR_ONE_EXECUTION"
        or proposal.get("approval", {}).get("status") != "APPROVED"
        or data_card.get("status") != "APPROVED_FOR_ONE_EXECUTION"
        or data_card.get("approval", {}).get("status") != "APPROVED"
        or data_card.get("approval", {}).get("authorized_operations") != ["infrastructure"]
        or data_card.get("approval", {}).get("authorized_gates") != [0]
    ):
        raise RuntimeError("proposal/card approval is absent or scope-incompatible")
    strata = proposal.get("frozen_candidate_scope", {}).get("strata", [])
    from execute_cano_100_topology_parent_candidate_audit_v1 import _candidate_registry

    candidates = [
        item
        for stratum_index, stratum in enumerate(strata, start=1)
        for item in _candidate_registry(proposal, stratum, stratum_index)
    ]
    if (
        len(strata) != 10
        or len(candidates) != 120
        or len({item["candidate_id"] for item in candidates}) != 120
        or len({item["topology_seed"] for item in candidates}) != 120
        or len({item["reserved_geometry_seed"] for item in candidates}) != 120
    ):
        raise RuntimeError("approved candidate registry is not exactly 10x12 with unique identities")
    registry_payload = {"candidates": candidates}
    registry_sha256 = canonical_json_hash(registry_payload)
    if registry_sha256 != spec.get("candidate_registry_sha256"):
        raise RuntimeError(
            f"frozen candidate registry hash mismatch: {registry_sha256}"
        )

    tool_paths = {
        "runner": RUNNER_PATH,
        "executor": EXECUTOR_PATH,
        "audit_helpers": PROJECT_ROOT / "src/mtare_topo/data/cano_topology_parent_audit.py",
        "cano_export_adapter": PROJECT_ROOT / "tools/v3/generate_cano_audited_bundle.py",
        "runner_helpers": PROJECT_ROOT / "tools/v3/run_cano_five_topology_cpu_contract_pilot.py",
        "governance": PROJECT_ROOT / "src/mtare_topo/governance.py",
        "unit_test": PROJECT_ROOT / "tests/v3/unit/test_cano_topology_parent_audit.py",
        "external_test": PROJECT_ROOT / "tests/v3/external/test_cano_topology_parent_generation.py",
        "proposal": proposal_path,
        "data_card": data_card_path,
    }
    tool_paths.update(EXTRA_TOOL_PATHS)
    observed_hashes = {name: _sha256(path) for name, path in tool_paths.items()}
    for name, frozen in spec.get("frozen_tools", {}).items():
        if observed_hashes.get(name) != frozen.get("sha256"):
            raise RuntimeError(
                f"frozen hash mismatch for {name}: {observed_hashes.get(name)} != {frozen.get('sha256')}"
            )
    upstream = _upstream_identity()
    if upstream != spec.get("upstream_identity"):
        raise RuntimeError(f"upstream identity mismatch: {upstream}")
    environment = _e1_identity()
    if environment["pip_freeze_evidence_sha256"] != spec.get("pip_freeze_evidence_sha256"):
        raise RuntimeError("frozen E1 pip-freeze identity mismatch")
    import_identity = _subt_proc_gen_import_identity(_executor_environment())

    write_json(run_dir / "config/tool_hashes.json", observed_hashes)
    write_json(run_dir / "config/frozen_candidate_registry.json", registry_payload)
    write_json(
        run_dir / "config/frozen_candidate_registry_identity.json",
        {"candidate_count": len(candidates), "canonical_sha256": registry_sha256},
    )
    write_json(run_dir / "config/upstream_identity.json", upstream)
    write_json(run_dir / "config/e1_environment.json", environment)
    write_json(run_dir / "config/subt_proc_gen_import_identity.json", import_identity)
    shutil.copy2(data_card_path, run_dir / "config/topology_card.json")
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "Approved topology-only 120-candidate audit; zero mesh, LiDAR and training.",
        },
    )

    exit_code, duration = _run_executor(run_dir)
    summary_path = run_dir / "metrics/summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else None
    candidate_metrics = [
        path for path in (run_dir / "metrics").glob("S*_C*.json") if path.is_file()
    ]
    candidate_dirs = [path for path in (run_dir / "artifacts/candidates").glob("S*_C*") if path.is_dir()]
    maps = list((run_dir / "previews/train_only_maps").glob("*.png"))
    manifests = [
        run_dir / "artifacts/candidate_manifest.json",
        run_dir / "artifacts/accepted_parent_manifest.json",
        run_dir / "artifacts/split_manifest.json",
        run_dir / "previews/provenance.json",
    ]
    result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    summary_scope = summary.get("scope", {}) if summary else {}
    passed = bool(
        exit_code == 0
        and duration <= 7200
        and summary is not None
        and summary.get("overall_status") == "PASS_CANO_100_TOPOLOGY_PARENT_CANDIDATE_AUDIT"
        and summary.get("method_id") == METHOD_ID
        and summary_scope.get("candidate_topology_constructions") == 120
        and summary_scope.get("retained_topology_parents") == 100
        and summary_scope.get("train_parents") == 80
        and summary_scope.get("validation_parents") == 10
        and summary_scope.get("development_test_parents") == 10
        and all(summary_scope.get(key) == 0 for key in ("meshes", "anchors", "lidar_observations", "rays", "teacher_labels", "formal_dataset_samples", "training_samples", "models", "trajectories", "mtare_changes"))
        and summary.get("diversity_audit", {}).get("passed") is True
        and summary.get("split_audit", {}).get("parent_and_identity_sets_pairwise_disjoint") is True
        and summary.get("source_unchanged") is True
        and len(candidate_metrics) == 120
        and len(candidate_dirs) >= 100
        and len(maps) == 10
        and all(path.is_file() for path in manifests)
        and result_bytes <= 2 * 1024**3
    )
    overall = (
        "PASS_CANO_100_TOPOLOGY_PARENT_CANDIDATE_AUDIT"
        if passed
        else "FAIL_CANO_100_TOPOLOGY_PARENT_CANDIDATE_AUDIT"
    )
    runner_summary = {
        "schema_version": "cano_100_topology_parent_candidate_runner_summary_v1",
        "method_id": METHOD_ID,
        "run_id": RUN_ID,
        "overall_status": overall,
        "executor_exit_code": exit_code,
        "duration_seconds": duration,
        "counts": {
            "candidate_metric_files": len(candidate_metrics),
            "valid_candidate_artifact_directories": len(candidate_dirs),
            "train_only_maps": len(maps),
        },
        "result_bytes_before_seal": result_bytes,
        "disk_limit_bytes": 2 * 1024**3,
        "free_disk_bytes_after": shutil.disk_usage(PROJECT_ROOT).free,
        "host": {"platform": platform.platform()},
        "claim_boundary": "Topology graph/spline source and split audit only; no mesh, LiDAR, labels, formal dataset, training, online graph or M-TARE change.",
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
