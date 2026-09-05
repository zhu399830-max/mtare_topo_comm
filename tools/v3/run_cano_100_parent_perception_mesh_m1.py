#!/usr/bin/env python3
"""Run and seal the approved full-100 immutable perception-asset batch once."""

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
from execute_cano_100_parent_perception_mesh_m1 import EXPECTED_SCOPE
from mtare_topo.governance import load_json, write_json
from run_cano_100_parent_perception_mesh_contract_m0 import SOURCE_V2, SOURCE_V2R, _verify_seal
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


RUN_ID = "gate0_20260811_cano_100_parent_perception_mesh_m1_immutable_assets_seed0"
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_cano_100_parent_perception_mesh_m1.py"
SOURCE_M0F = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_contract_m0f_immutable_assets_seed0"
TIME_LIMIT_SECONDS = 10800
DISK_LIMIT_BYTES = 4 * 1024**3


def _run(run_dir: Path) -> tuple[int, float]:
    argv = [str(E1_PYTHON), str(EXECUTOR), "--run-dir", str(run_dir)]
    started = time.monotonic()
    output = ""
    code = 124
    try:
        done = subprocess.run(
            argv,
            cwd=PROJECT_ROOT,
            env=_executor_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=TIME_LIMIT_SECONDS,
            check=False,
        )
        output, code = done.stdout, done.returncode
    except subprocess.TimeoutExpired as exc:
        captured = exc.stdout or ""
        output = captured.decode() if isinstance(captured, bytes) else captured
        output += f"\nTIMEOUT: frozen {TIME_LIMIT_SECONDS} second M1 limit reached.\n"
    duration = time.monotonic() - started
    (run_dir / "logs/01_cano_100_parent_perception_mesh_m1.log").write_text(
        "argv=" + json.dumps(argv, ensure_ascii=False) + "\n"
        + f"finished_at_utc={datetime.now(timezone.utc).isoformat()}\n"
        + output
        + f"\nduration_seconds={duration:.6f}\nexit_code={code}\n",
        encoding="utf-8",
    )
    print(output, end="", flush=True)
    return code, duration


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
        raise RuntimeError("run is not in one-time state")
    if (
        spec.get("operation") != "infrastructure"
        or spec.get("gate") != 0
        or spec.get("seed") != 0
        or spec.get("user_authorization", {}).get("status") != "APPROVED"
        or spec.get("data_scope") != EXPECTED_SCOPE
    ):
        raise RuntimeError("approval or M1 scope mismatch")

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
        raise RuntimeError("approved M1 proposal/card missing")

    paths = {
        "runner": Path(__file__).resolve(),
        "executor": EXECUTOR,
        "m0_executor_helpers": PROJECT_ROOT / "tools/v3/execute_cano_100_parent_perception_mesh_contract_m0.py",
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
    observed = {name: _sha256(path) for name, path in paths.items()}
    if set(spec.get("frozen_tools", {})) != set(paths):
        raise RuntimeError("frozen M1 tool set mismatch")
    for name, frozen in spec["frozen_tools"].items():
        if observed[name] != frozen.get("sha256"):
            raise RuntimeError(f"frozen tool mismatch: {name}")

    m0f_state = load_json(SOURCE_M0F / "RUN_STATE.json")
    if m0f_state.get("state") != "COMPLETED" or m0f_state.get("overall_status") != "PASS_CANO_100_PARENT_PERCEPTION_MESH_CONTRACT_M0F":
        raise RuntimeError("M0F route evidence invalid")
    m0f_seal = _verify_seal(SOURCE_M0F, 110)
    v2r_seal = _verify_seal(SOURCE_V2R, 17)
    v2_seal = _verify_seal(SOURCE_V2, 391)
    for seal in (m0f_seal, v2r_seal, v2_seal):
        if seal.get("mismatch_count") != 0:
            raise RuntimeError(f"source seal mismatch: {seal}")
    for relative, expected in spec.get("frozen_source_files", {}).items():
        if _sha256(_approved_project_file(relative, "frozen_source_files")) != expected:
            raise RuntimeError(f"frozen source mismatch: {relative}")

    upstream = _upstream_identity()
    environment = _e1_identity()
    imports = _subt_proc_gen_import_identity(_executor_environment())
    if upstream != spec.get("upstream_identity") or environment["pip_freeze_evidence_sha256"] != spec.get("pip_freeze_evidence_sha256"):
        raise RuntimeError("environment/upstream mismatch")
    if shutil.disk_usage(PROJECT_ROOT).free < 6 * 1024**3:
        raise RuntimeError("less than 6 GiB free before M1")

    for name, value in (
        ("tool_hashes", observed),
        ("upstream_identity", upstream),
        ("e1_environment", environment),
        ("subt_proc_gen_import_identity", imports),
        ("m0f_source_seal", m0f_seal),
        ("v2r_source_seal", v2r_seal),
        ("v2_source_seal", v2_seal),
    ):
        write_json(run_dir / f"config/{name}.json", value)
    shutil.copy2(card_path, run_dir / "config/topology_card.json")
    write_json(
        run_dir / "RUN_STATE.json",
        {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING", "note": "Approved M1 100 immutable primary assets; 80 train previews; zero replay/data/training."},
    )

    code, duration = _run(run_dir)
    summary_path = run_dir / "metrics/summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else None
    primary = sorted(run_dir.glob("artifacts/meshes/S*_C*/primary/mesh.obj"))
    replay = sorted(run_dir.glob("artifacts/meshes/S*_C*/replay/mesh.obj"))
    metrics = sorted(run_dir.glob("metrics/S*_C*.json"))
    maps = sorted(run_dir.glob("previews/train_complete_maps/*.png"))
    size = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    passed = bool(
        code == 0
        and duration <= TIME_LIMIT_SECONDS
        and summary
        and summary.get("overall_status") == "PASS_CANO_100_PARENT_PERCEPTION_MESH_M1"
        and summary.get("scope") == EXPECTED_SCOPE
        and summary.get("batch_audit", {}).get("passed") is True
        and len(primary) == len(metrics) == 100
        and len(maps) == 80
        and len(replay) == 0
        and (run_dir / "artifacts/mesh_manifest.json").is_file()
        and (run_dir / "previews/provenance.json").is_file()
        and size <= DISK_LIMIT_BYTES
    )
    overall = "PASS_CANO_100_PARENT_PERCEPTION_MESH_M1" if passed else "FAIL_CANO_100_PARENT_PERCEPTION_MESH_M1"
    boundary = "100 immutable perception assets only; zero replay, LiDAR, labels, formal data, training, models, trajectories and M-TARE changes."
    runner_summary = {
        "schema_version": "cano_100_parent_perception_mesh_runner_summary_m1",
        "run_id": RUN_ID,
        "overall_status": overall,
        "executor_exit_code": code,
        "duration_seconds": duration,
        "counts": {"primary_meshes": len(primary), "replay_meshes": len(replay), "parent_metrics": len(metrics), "train_complete_previews": len(maps)},
        "source_seals": {"m0f": m0f_seal, "v2r": v2r_seal, "v2": v2_seal},
        "result_bytes_before_seal": size,
        "disk_limit_bytes": DISK_LIMIT_BYTES,
        "host": {"platform": platform.platform()},
        "claim_boundary": boundary,
    }
    write_json(run_dir / "metrics/runner_summary.json", runner_summary)
    write_json(
        run_dir / "RUN_STATE.json",
        {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if passed else "FAILED", "overall_status": overall, "note": boundary},
    )
    sealed = _seal_manifest(run_dir)
    print(json.dumps({"overall_status": overall, "sealed_files": sealed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
