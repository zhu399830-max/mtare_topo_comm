#!/usr/bin/env python3
"""Run and seal the path-corrected 20-parent perception-mesh batch once."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from execute_aee_corrective_perception_mesh_v1 import EXPECTED_SCOPE, SOURCE_RUN
from mtare_topo.governance import load_json, write_json
from run_aee_corrective_topology_candidate_audit_v1 import (
    E1,
    _approved_project_file,
    _environment,
    _sha256,
    _verify_environment_and_sources,
)
from run_cano_five_topology_cpu_contract_pilot import _seal_manifest


RUN_ID = "gate2_20260821_aee_corrective_perception_mesh_v1r_seed20260821"
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_aee_corrective_perception_mesh_v1.py"
TIME_LIMIT_SECONDS = 7200
DISK_LIMIT_BYTES = 1024**3


def _verify_source_seal() -> dict:
    manifest = SOURCE_RUN / "artifacts/evidence_sha256.txt"
    lines = [line for line in manifest.read_text().splitlines() if line.strip()]
    mismatch = []
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = PROJECT_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            mismatch.append(relative)
    return {
        "entries": len(lines),
        "mismatch_count": len(mismatch),
        "mismatches": mismatch,
        "manifest_sha256": _sha256(manifest),
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
        raise RuntimeError("run is not in one-time state")
    if (
        spec.get("operation") != "infrastructure"
        or spec.get("gate") != 2
        or spec.get("seed") != 20260821
        or spec.get("user_authorization", {}).get("status") != "APPROVED"
        or spec.get("data_scope") != EXPECTED_SCOPE
    ):
        raise RuntimeError("approval or corrective mesh scope mismatch")

    proposal_path = _approved_project_file(spec.get("config_path"), "config_path")
    card_path = _approved_project_file(spec.get("data_card"), "data_card")
    proposal = load_json(proposal_path)
    card = load_json(card_path)
    if (
        proposal.get("status") != "APPROVED_FOR_ONE_EXECUTION"
        or proposal.get("approval", {}).get("status") != "APPROVED"
        or card.get("status") != "APPROVED_FOR_ONE_EXECUTION"
        or card.get("approval", {}).get("status") != "APPROVED"
    ):
        raise RuntimeError("approved proposal/data card missing")

    paths = {
        "runner": Path(__file__).resolve(),
        "executor": EXECUTOR,
        "m0_helpers": PROJECT_ROOT / "tools/v3/execute_cano_100_parent_perception_mesh_contract_m0.py",
        "m1r_helpers": PROJECT_ROOT / "tools/v3/execute_cano_100_parent_perception_mesh_m1r.py",
        "mesh_contract": PROJECT_ROOT / "src/mtare_topo/data/cano_perception_mesh_contract.py",
        "topology_executor": PROJECT_ROOT / "tools/v3/execute_aee_corrective_topology_candidate_audit_v1.py",
        "proposal": proposal_path,
        "data_card": card_path,
        "environment": _approved_project_file(
            spec.get("environment", {}).get("path"), "environment.path"
        ),
    }
    observed = {name: _sha256(path) for name, path in paths.items()}
    if set(observed) != set(spec.get("frozen_tools", {})):
        raise RuntimeError("frozen tool set mismatch")
    for name, digest in observed.items():
        if spec["frozen_tools"][name].get("sha256") != digest:
            raise RuntimeError(f"frozen tool mismatch: {name}")

    source_state = load_json(SOURCE_RUN / "RUN_STATE.json")
    source_seal = _verify_source_seal()
    if (
        source_state.get("overall_status")
        != "PASS_AEE_CORRECTIVE_TOPOLOGY_CANDIDATE_AUDIT_V1"
        or source_seal["entries"] != 376
        or source_seal["mismatch_count"] != 0
    ):
        raise RuntimeError("sealed topology source invalid")
    environment = _verify_environment_and_sources(spec)
    write_json(run_dir / "config/tool_hashes.json", observed)
    write_json(run_dir / "config/source_seal_audit.json", source_seal)
    write_json(run_dir / "config/environment_and_source_audit.json", environment)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "Twenty corrective perception meshes only; frozen checkout environment propagated.",
        },
    )

    argv = [str(E1), str(EXECUTOR), "--run-dir", str(run_dir)]
    started = time.monotonic()
    code = 124
    output = ""
    try:
        done = subprocess.run(
            argv,
            cwd=PROJECT_ROOT,
            env=_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=TIME_LIMIT_SECONDS,
            check=False,
        )
        code, output = done.returncode, done.stdout
    except subprocess.TimeoutExpired as exc:
        output = (
            exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
        ) + "\nTIMEOUT\n"
    duration = time.monotonic() - started
    (run_dir / "logs/01_corrective_mesh.log").write_text(
        "argv="
        + json.dumps(argv)
        + "\nfinished_at_utc="
        + datetime.now(timezone.utc).isoformat()
        + "\n"
        + output
        + f"\nduration_seconds={duration:.6f}\nexit_code={code}\n",
        encoding="utf-8",
    )
    print(output, end="", flush=True)
    summary = (
        load_json(run_dir / "metrics/summary.json")
        if (run_dir / "metrics/summary.json").is_file()
        else None
    )
    meshes = list(run_dir.glob("artifacts/meshes/S*_C*/primary/mesh.obj"))
    sanitation = list(run_dir.glob("artifacts/meshes/S*_C*/primary/sanitation.json"))
    metrics = list(run_dir.glob("metrics/S*_C*.json"))
    previews = list(run_dir.glob("previews/train_complete_maps/*.png"))
    size = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    passed = bool(
        code == 0
        and summary
        and summary.get("overall_status") == "PASS_AEE_CORRECTIVE_PERCEPTION_MESH_V1"
        and summary.get("scope") == EXPECTED_SCOPE
        and summary.get("checks")
        and all(summary["checks"].values())
        and len(meshes) == len(sanitation) == len(metrics) == 20
        and len(previews) == 10
        and duration <= TIME_LIMIT_SECONDS
        and size <= DISK_LIMIT_BYTES
    )
    overall = (
        "PASS_AEE_CORRECTIVE_PERCEPTION_MESH_V1R"
        if passed
        else "FAIL_AEE_CORRECTIVE_PERCEPTION_MESH_V1R"
    )
    boundary = "Twenty immutable perception meshes only; zero LiDAR, teacher, formal data, training, C09/C10 or planner change."
    write_json(
        run_dir / "metrics/runner_summary.json",
        {
            "schema_version": "aee_corrective_perception_mesh_runner_v1r",
            "run_id": RUN_ID,
            "overall_status": overall,
            "executor_exit_code": code,
            "duration_seconds": duration,
            "counts": {
                "meshes": len(meshes),
                "sanitation": len(sanitation),
                "metrics": len(metrics),
                "train_previews": len(previews),
            },
            "result_bytes_before_seal": size,
            "disk_limit_bytes": DISK_LIMIT_BYTES,
            "host": {"platform": platform.platform()},
            "claim_boundary": boundary,
            "corrective_change": "Propagate the already-qualified frozen checkout environment to the executor child process.",
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if passed else "FAILED",
            "overall_status": overall,
            "note": boundary,
        },
    )
    sealed = _seal_manifest(run_dir)
    print(json.dumps({"overall_status": overall, "sealed_files": sealed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
