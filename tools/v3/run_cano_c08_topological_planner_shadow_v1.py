#!/usr/bin/env python3
"""Run and seal one approved Gate-5 C08 planner shadow replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate5_20260820_cano_c08_topological_planner_shadow_v1_seed0"
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_cano_c08_topological_planner_shadow_v1.py"
PYTHON = Path("/tmp/mtare_gate4_torch290_cu129_zarr2187/bin/python")
SOURCE_RUN = PROJECT_ROOT / "results/gate4_topology/gate4_20260817_cano_c08_route_conditioned_causal_topology_replay_v2r_seed0"
EXPECTED_FREEZE_SHA256 = "fccb0fe667ca7164294619bef3f9f6c5be81deb42b1c5c9554e7b562bd9b23ae"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_seal(path: Path) -> int:
    entries = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if "_C09" in relative or "_C10" in relative:
            raise RuntimeError(f"forbidden validation/test source in C08 seal: {relative}")
        if sha(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"source seal mismatch: {relative}")
        entries += 1
    return entries


def seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    destination.write_text(
        "".join(f"{sha(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files),
        encoding="utf-8",
    )
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("run identity/state mismatch")
    if spec.get("gate") != 5 or spec.get("operation") != "shadow":
        raise RuntimeError("Gate-5 shadow scope mismatch")
    if spec.get("user_authorization", {}).get("status") != "APPROVED":
        raise RuntimeError("formal execution is not approved")
    for relative, expected in spec["frozen_inputs"].items():
        if sha(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"frozen input drift: {relative}")
    source_entries = verify_source_seal(SOURCE_RUN / "artifacts/evidence_sha256.txt")
    observed_tools = {}
    for name, item in spec["frozen_tools"].items():
        observed_tools[name] = sha(PROJECT_ROOT / item["path"])
        if observed_tools[name] != item["sha256"]:
            raise RuntimeError(f"frozen tool drift: {name}")
    write_json(run_dir / "config/input_integrity.json", {"source_seal_entries": source_entries})
    write_json(run_dir / "config/tool_hashes.json", observed_tools)

    freeze = subprocess.check_output([str(PYTHON), "-m", "pip", "freeze"], text=True)
    normalized = "\n".join(sorted(line for line in freeze.splitlines() if line.strip())) + "\n"
    freeze_hash = hashlib.sha256(normalized.encode()).hexdigest()
    if freeze_hash != EXPECTED_FREEZE_SHA256:
        raise RuntimeError(f"shadow environment drift: {freeze_hash}")
    check = subprocess.run([str(PYTHON), "-m", "pip", "check"], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if check.returncode != 0:
        raise RuntimeError(f"shadow environment pip check failed: {check.stdout}")
    (run_dir / "config/shadow_pip_freeze.txt").write_text(normalized, encoding="utf-8")
    identity = json.loads(subprocess.check_output([
        str(PYTHON), "-c",
        "import json,sys,numpy,zarr;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'zarr':zarr.__version__}))",
    ], text=True))
    identity.update({"pip_freeze_sha256": freeze_hash, "pip_check": check.stdout.strip()})
    write_json(run_dir / "config/executor_environment_identity.json", identity)

    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"
    })
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools/v3")))
    environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    argv = [str(PYTHON), str(EXECUTOR), "--run-dir", str(run_dir)]
    started = time.monotonic()
    completed = subprocess.run(
        argv, cwd=PROJECT_ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600, check=False,
    )
    duration = time.monotonic() - started
    (run_dir / "logs/01_shadow_replay.log").write_text(
        f"argv={json.dumps(argv)}\nfinished_at_utc={datetime.now(timezone.utc).isoformat()}\n"
        f"{completed.stdout}\nduration_seconds={duration:.6f}\nexit_code={completed.returncode}\n",
        encoding="utf-8",
    )
    print(completed.stdout, end="", flush=True)
    result = load_json(run_dir / "metrics/summary.json") if (run_dir / "metrics/summary.json").is_file() else {}
    passed = bool(
        completed.returncode == 0
        and result.get("overall_status") == "PASS_CANO_C08_TOPOLOGICAL_PLANNER_SHADOW_V1"
        and result.get("unique_source_frames") == 4773
        and result.get("planner_cycles") == 23865
        and result.get("all_waypoints_finite_and_bounded") is True
        and result.get("all_routes_use_verified_edges") is True
        and result.get("all_decision_replays_deterministic") is True
        and result.get("c09_worlds_read") == result.get("c10_worlds_read") == 0
        and result.get("mtare_closed_loop_runs") == 0
    )
    overall = "PASS_CANO_C08_TOPOLOGICAL_PLANNER_SHADOW_V1" if passed else "FAIL_CANO_C08_TOPOLOGICAL_PLANNER_SHADOW_V1"
    write_json(run_dir / "metrics/runner_summary.json", {
        "schema_version": "cano_c08_topological_planner_shadow_runner_v1",
        "overall_status": overall,
        "executor_exit_code": completed.returncode,
        "duration_seconds": duration,
        "unique_source_frames": result.get("unique_source_frames", 0),
        "planner_cycles": result.get("planner_cycles", 0),
        "source_seal_entries": source_entries,
        "result_bytes_before_seal": sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file()),
        "free_disk_bytes_after": shutil.disk_usage(PROJECT_ROOT).free,
        "claim_boundary": "C08 offline shadow interface only; no robot control, exploration benefit, C09/C10 or M-TARE result.",
    })
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if passed else "FAILED", "overall_status": overall,
    })
    count = seal(run_dir)
    print(json.dumps({"overall_status": overall, "sealed_files": count}, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
