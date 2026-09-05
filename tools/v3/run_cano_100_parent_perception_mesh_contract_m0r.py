#!/usr/bin/env python3
"""Run and seal the approved Cano perception-mesh M0R geometric replay once."""

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
from run_cano_100_parent_perception_mesh_contract_m0 import (
    EXPECTED_SCOPE,
    SOURCE_V2,
    SOURCE_V2R,
    _verify_seal,
)
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


RUN_ID = "gate0_20260811_cano_100_parent_perception_mesh_contract_m0r_geometric_replay_seed0"
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_cano_100_parent_perception_mesh_contract_m0r.py"
FAILED_M0 = PROJECT_ROOT / (
    "results/gate0_baseline/"
    "gate0_20260811_cano_100_parent_perception_mesh_contract_m0_seed0"
)


def _run_executor(run_dir: Path) -> tuple[int, float]:
    argv = [str(E1_PYTHON), str(EXECUTOR), "--run-dir", str(run_dir)]
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
            timeout=3600,
            check=False,
        )
        output = completed.stdout
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        captured = exc.stdout or ""
        output = captured.decode() if isinstance(captured, bytes) else captured
        output += "\nTIMEOUT: frozen 3600 second M0R limit reached.\n"
    duration = time.monotonic() - started
    (run_dir / "logs/01_cano_100_parent_perception_mesh_contract_m0r.log").write_text(
        "argv=" + json.dumps(argv, ensure_ascii=False) + "\n"
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
    if (
        spec.get("operation") != "infrastructure"
        or spec.get("gate") != 0
        or spec.get("seed") != 0
        or spec.get("user_authorization", {}).get("status") != "APPROVED"
        or spec.get("data_scope") != EXPECTED_SCOPE
    ):
        raise RuntimeError("spec approval or unchanged M0R scope is incompatible")

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
        raise RuntimeError("approved M0R proposal/card is missing")

    tool_paths = {
        "runner": Path(__file__).resolve(),
        "executor": EXECUTOR,
        "m0_executor": PROJECT_ROOT / "tools/v3/execute_cano_100_parent_perception_mesh_contract_m0.py",
        "m0_runner_helpers": PROJECT_ROOT / "tools/v3/run_cano_100_parent_perception_mesh_contract_m0.py",
        "mesh_contract": PROJECT_ROOT / "src/mtare_topo/data/cano_perception_mesh_contract.py",
        "topology_audit": PROJECT_ROOT / "src/mtare_topo/data/cano_topology_parent_audit.py",
        "v2_executor": PROJECT_ROOT / "tools/v3/execute_cano_100_topology_parent_candidate_audit_v2.py",
        "v1_executor": PROJECT_ROOT / "tools/v3/execute_cano_100_topology_parent_candidate_audit_v1.py",
        "cano_adapter": PROJECT_ROOT / "tools/v3/generate_cano_audited_bundle.py",
        "runner_helpers": PROJECT_ROOT / "tools/v3/run_cano_five_topology_cpu_contract_pilot.py",
        "governance": PROJECT_ROOT / "src/mtare_topo/governance.py",
        "unit_test": PROJECT_ROOT / "tests/v3/unit/test_cano_perception_mesh_contract.py",
        "external_test": PROJECT_ROOT / "tests/v3/external/test_cano_perception_mesh_reconstruction.py",
        "proposal": proposal_path,
        "data_card": card_path,
    }
    observed_hashes = {name: _sha256(path) for name, path in tool_paths.items()}
    if set(spec.get("frozen_tools", {})) != set(tool_paths):
        raise RuntimeError("frozen M0R tool set is not exact")
    for name, frozen in spec["frozen_tools"].items():
        if observed_hashes[name] != frozen.get("sha256"):
            raise RuntimeError(f"frozen tool mismatch: {name}")

    failed_state = load_json(FAILED_M0 / "RUN_STATE.json")
    if (
        failed_state.get("state") != "FAILED"
        or failed_state.get("overall_status")
        != "FAIL_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0"
    ):
        raise RuntimeError("M0 predecessor is not the sealed exact-replay FAIL")
    m0_seal = _verify_seal(FAILED_M0, 33)
    v2r_seal = _verify_seal(SOURCE_V2R, 17)
    v2_seal = _verify_seal(SOURCE_V2, 391)
    for relative, expected in spec.get("frozen_source_files", {}).items():
        path = _approved_project_file(relative, "frozen_source_files")
        if _sha256(path) != expected:
            raise RuntimeError(f"frozen source mismatch: {relative}")

    upstream = _upstream_identity()
    if upstream != spec.get("upstream_identity"):
        raise RuntimeError("upstream identity mismatch")
    environment = _e1_identity()
    if environment["pip_freeze_evidence_sha256"] != spec.get("pip_freeze_evidence_sha256"):
        raise RuntimeError("E1 environment identity mismatch")
    import_identity = _subt_proc_gen_import_identity(_executor_environment())
    if shutil.disk_usage(PROJECT_ROOT).free < 2 * 1024**3:
        raise RuntimeError("less than 2 GiB free disk before M0R")

    write_json(run_dir / "config/tool_hashes.json", observed_hashes)
    write_json(run_dir / "config/upstream_identity.json", upstream)
    write_json(run_dir / "config/e1_environment.json", environment)
    write_json(run_dir / "config/subt_proc_gen_import_identity.json", import_identity)
    write_json(run_dir / "config/m0_source_seal.json", m0_seal)
    write_json(run_dir / "config/v2r_source_seal.json", v2r_seal)
    write_json(run_dir / "config/v2_source_seal.json", v2_seal)
    shutil.copy2(card_path, run_dir / "config/topology_card.json")
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "Approved M0R single correction: noise-bounded geometric replay; zero data/training.",
        },
    )
    exit_code, duration = _run_executor(run_dir)

    summary_path = run_dir / "metrics/summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else None
    mesh_paths = sorted(run_dir.glob("artifacts/meshes/S*_C01/*/mesh.obj"))
    primary_meshes = [path for path in mesh_paths if path.parent.name == "primary"]
    replay_meshes = [path for path in mesh_paths if path.parent.name == "replay"]
    parent_metrics = sorted(run_dir.glob("metrics/S*_C01.json"))
    maps = sorted(run_dir.glob("previews/train_complete_maps/*.png"))
    result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    passed = bool(
        exit_code == 0
        and duration <= 3600
        and summary is not None
        and summary.get("overall_status") == "PASS_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0R"
        and summary.get("scope") == EXPECTED_SCOPE
        and summary.get("batch_audit", {}).get("passed") is True
        and len(primary_meshes) == 10
        and len(replay_meshes) == 10
        and len(parent_metrics) == 10
        and len(maps) == 10
        and (run_dir / "artifacts/mesh_manifest.json").is_file()
        and (run_dir / "previews/provenance.json").is_file()
        and result_bytes <= int(1.5 * 1024**3)
    )
    overall = (
        "PASS_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0R"
        if passed
        else "FAIL_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0R"
    )
    runner_summary = {
        "schema_version": "cano_100_parent_perception_mesh_contract_runner_summary_m0r",
        "run_id": RUN_ID,
        "overall_status": overall,
        "executor_exit_code": exit_code,
        "duration_seconds": duration,
        "counts": {
            "primary_meshes": len(primary_meshes),
            "replay_meshes": len(replay_meshes),
            "parent_metrics": len(parent_metrics),
            "train_complete_previews": len(maps),
        },
        "source_seals": {"m0": m0_seal, "v2r": v2r_seal, "v2": v2_seal},
        "result_bytes_before_seal": result_bytes,
        "disk_limit_bytes": int(1.5 * 1024**3),
        "free_disk_bytes_after": shutil.disk_usage(PROJECT_ROOT).free,
        "host": {"platform": platform.platform()},
        "claim_boundary": "Train-only native perception-mesh geometric replay; zero LiDAR, labels, formal data, training, models, trajectories and M-TARE changes.",
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
