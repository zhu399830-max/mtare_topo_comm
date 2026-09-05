#!/usr/bin/env python3
"""Run and seal the approved S10 local lateral pose audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate4_20260813_cano_s10_local_lateral_pose_feasibility_v1_seed0"
E1 = Path("/tmp/mtare_cano_e1_zarr2187/bin/python")
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_cano_s10_local_lateral_pose_feasibility_v1.py"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    destination.write_text("".join(f"{sha(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files), encoding="utf-8")
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
    if spec.get("gate") != 4 or spec.get("operation") != "topology_replay" or spec.get("local_lateral_pose_audit_only") is not True:
        raise RuntimeError("scope mismatch")
    card = load_json(PROJECT_ROOT / spec["data_card"])
    if card.get("approval", {}).get("status") != "APPROVED" or "topology_replay" not in card["approval"].get("authorized_operations", []):
        raise RuntimeError("data card is not approved")
    for relative, expected in spec["frozen_inputs"].items():
        if sha(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"frozen input drift: {relative}")
    observed = {}
    for name, item in spec["frozen_tools"].items():
        observed[name] = sha(PROJECT_ROOT / item["path"])
        if observed[name] != item["sha256"]:
            raise RuntimeError(f"frozen tool drift: {name}")
    write_json(run_dir / "config/tool_hashes.json", observed)
    identity = json.loads(subprocess.check_output([
        str(E1), "-c",
        "import json,sys,numpy,open3d,matplotlib;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'open3d':open3d.__version__,'matplotlib':matplotlib.__version__}))",
    ], text=True))
    write_json(run_dir / "config/executor_environment_identity.json", identity)
    write_json(run_dir / "RUN_STATE.json", {"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING"})
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools/v3")))
    environment.update({"OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"})
    argv = [str(E1), str(EXECUTOR), "--run-dir", str(run_dir)]
    started = time.monotonic()
    completed = subprocess.run(argv, cwd=PROJECT_ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300, check=False)
    duration = time.monotonic() - started
    (run_dir / "logs/01_local_lateral_pose_feasibility.log").write_text(
        f"argv={json.dumps(argv)}\nfinished_at_utc={datetime.now(timezone.utc).isoformat()}\n{completed.stdout}\nduration_seconds={duration:.6f}\nexit_code={completed.returncode}\n",
        encoding="utf-8",
    )
    print(completed.stdout, end="", flush=True)
    summary = load_json(run_dir / "metrics/summary.json") if (run_dir / "metrics/summary.json").is_file() else {}
    passed = bool(
        completed.returncode == 0
        and summary.get("overall_status") == "PASS_CANO_S10_LOCAL_LATERAL_POSE_FEASIBILITY_AUDIT_V1"
        and summary.get("route_failure_poses") == 9
        and summary.get("pose_xy_candidates") == 3969
        and summary.get("horizontal_rays") == 2857680
        and summary.get("vertical_rays") == 11907
        and summary.get("inference_frames") == summary.get("graph_updates") == summary.get("training_samples_consumed") == 0
        and summary.get("c09_worlds_read") == summary.get("c10_worlds_read") == summary.get("mtare_worlds_read") == 0
    )
    overall = "PASS_CANO_S10_LOCAL_LATERAL_POSE_FEASIBILITY_AUDIT_V1" if passed else "FAIL_CANO_S10_LOCAL_LATERAL_POSE_FEASIBILITY_AUDIT_V1"
    write_json(run_dir / "metrics/runner_summary.json", {
        "schema_version":"cano_s10_local_lateral_pose_feasibility_runner_v1", "overall_status":overall,
        "executor_exit_code":completed.returncode, "duration_seconds":duration,
        "route_failure_poses":summary.get("route_failure_poses", 0), "pose_xy_candidates":summary.get("pose_xy_candidates", 0),
        "unresolved_route_poses":summary.get("unresolved_route_poses"),
        "claim_boundary":"Local diagnostic only; not a complete trajectory or dynamic-navigation qualification.",
    })
    write_json(run_dir / "RUN_STATE.json", {"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if passed else "FAILED","overall_status":overall})
    count = seal(run_dir)
    print(json.dumps({"overall_status":overall,"sealed_files":count}, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
