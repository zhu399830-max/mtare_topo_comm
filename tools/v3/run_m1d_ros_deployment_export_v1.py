#!/usr/bin/env python3
"""Export, cross-load, compare, and seal three frozen M1D checkpoints."""

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

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.deployment.m1d_checkpoint import export_m1d_ros_checkpoint
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate5_20260820_m1d_ros_deployment_export_v1_seed0"
PYTHON = Path("/tmp/mtare_gate4_torch290_cu129_zarr2187/bin/python")
ROS_IMAGE = "mtare-semantic-runtime:local"
VERIFY_TOOL = PROJECT_ROOT / "tools/v3/verify_m1d_ros_checkpoint_v1.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(run_dir: Path) -> int:
    target = run_dir / "artifacts/evidence_sha256.txt"
    paths = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != target)
    target.write_text("".join(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in paths), encoding="utf-8")
    return len(paths)


def verify_runtime(python: str, checkpoint: Path, output: Path, *, docker_run_dir: Path | None = None) -> dict:
    if docker_run_dir is None:
        command = [python, str(VERIFY_TOOL), "--checkpoint", str(checkpoint), "--output", str(output)]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, text=True, capture_output=True, check=False)
    else:
        relative_checkpoint = checkpoint.relative_to(docker_run_dir)
        relative_output = output.relative_to(docker_run_dir)
        command = [
            "docker", "run", "--rm", "--network", "none",
            "-v", "/etc/localtime:/etc/localtime:ro",
            "-v", f"{PROJECT_ROOT}:/workspace:ro", "-v", f"{docker_run_dir}:/run:rw",
            "-e", "PYTHONPATH=/workspace/src", "--entrypoint", "/bin/bash", ROS_IMAGE, "-lc",
            "python3 /workspace/tools/v3/verify_m1d_ros_checkpoint_v1.py "
            f"--checkpoint /run/{relative_checkpoint} --output /run/{relative_output}",
        ]
        completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"runtime checkpoint verification failed: {completed.stdout}\n{completed.stderr}")
    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if len(lines) != 1:
        raise RuntimeError(f"runtime verification emitted unexpected output: {completed.stdout}")
    return json.loads(lines[0])


def compare_outputs(first: Path, second: Path) -> dict:
    with np.load(first, allow_pickle=False) as left, np.load(second, allow_pickle=False) as right:
        if sorted(left.files) != sorted(right.files):
            raise RuntimeError("source/ROS forward output keys differ")
        maximum = {}
        decisions = {}
        for name in sorted(left.files):
            if left[name].shape != right[name].shape or not np.all(np.isfinite(left[name])) or not np.all(np.isfinite(right[name])):
                raise RuntimeError(f"invalid cross-runtime output: {name}")
            maximum[name] = float(np.max(np.abs(left[name].astype(np.float64) - right[name].astype(np.float64))))
        decisions["direction_sign_equal"] = bool(np.array_equal(left["direction_logits"] >= 0.0, right["direction_logits"] >= 0.0))
        decisions["count_argmax_equal"] = bool(np.array_equal(np.argmax(left["count_logits"], axis=1), np.argmax(right["count_logits"], axis=1)))
        decisions["role_argmax_equal"] = bool(np.array_equal(np.argmax(left["role_logits"], axis=1), np.argmax(right["role_logits"], axis=1)))
    return {"maximum_absolute_error": maximum, "decisions": decisions, "pass": all(value <= 1e-6 for value in maximum.values()) and all(decisions.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec, run_dir = load_json(args.spec.resolve()), args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("run identity/state mismatch")
    if spec.get("gate") != 5 or spec.get("operation") != "infrastructure" or spec.get("user_authorization", {}).get("status") != "APPROVED":
        raise RuntimeError("approved Gate-5 infrastructure scope required")
    for relative, expected in spec["frozen_inputs"].items():
        if sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"frozen checkpoint drift: {relative}")
    for name, item in spec["frozen_tools"].items():
        if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen tool drift: {name}")
    image_id = subprocess.check_output(["docker", "image", "inspect", ROS_IMAGE, "--format", "{{.Id}}"], text=True).strip()
    if image_id != spec["ros_image_id"]:
        raise RuntimeError(f"ROS image drift: {image_id}")
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
    started = time.monotonic(); records = []
    checkpoint_dir = run_dir / "artifacts/checkpoints"; parity_dir = run_dir / "artifacts/parity"
    checkpoint_dir.mkdir(parents=True, exist_ok=True); parity_dir.mkdir(parents=True, exist_ok=True)
    for item in spec["checkpoints"]:
        seed = int(item["seed"]); source = PROJECT_ROOT / item["path"]
        deployment = checkpoint_dir / f"m1d_seed{seed}_ros.pt"
        export_audit = export_m1d_ros_checkpoint(source=source, destination=deployment, expected_source_sha256=item["sha256"], expected_seed=seed)
        source_npz = parity_dir / f"m1d_seed{seed}_torch29.npz"; ros_npz = parity_dir / f"m1d_seed{seed}_torch20.npz"
        source_report = verify_runtime(str(PYTHON), deployment, source_npz)
        ros_report = verify_runtime("python3", deployment, ros_npz, docker_run_dir=run_dir)
        if source_report["tensor_sha256"] != ros_report["tensor_sha256"]:
            raise RuntimeError(f"seed {seed}: tensor hashes differ across runtimes")
        comparison = compare_outputs(source_npz, ros_npz)
        if not comparison["pass"]:
            raise RuntimeError(f"seed {seed}: forward parity failed: {comparison}")
        record = {"seed": seed, "export": export_audit, "torch29": source_report, "torch20_ros": ros_report, "cross_runtime": comparison}
        write_json(parity_dir / f"m1d_seed{seed}_audit.json", record); records.append(record)
    duration = time.monotonic() - started
    summary = {
        "schema_version": "m1d_ros_deployment_export_summary_v1",
        "overall_status": "PASS_M1D_ROS_DEPLOYMENT_EXPORT_V1",
        "checkpoint_count": len(records), "seeds": [item["seed"] for item in records],
        "all_tensor_hashes_equal": True, "all_cross_runtime_forward_pass": True,
        "training_steps": 0, "optimizer_steps": 0, "duration_seconds": duration,
        "ros_image": ROS_IMAGE, "ros_image_id": image_id,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    (run_dir / "logs/01_export.log").write_text(
        f"finished_at_utc={datetime.now(timezone.utc).isoformat()}\nduration_seconds={duration:.6f}\nstatus={summary['overall_status']}\n",
        encoding="utf-8",
    )
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED", "overall_status": summary["overall_status"]})
    count = seal(run_dir)
    print(json.dumps({**summary, "sealed_files": count}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
