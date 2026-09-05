#!/home/zeng-workstation/anaconda3/bin/python
"""Losslessly export three qualified AEE-adapted checkpoints to ROS Torch 2.0."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.deployment.m1d_checkpoint import export_m1d_ros_checkpoint
from mtare_topo.deployment.m1d_checkpoint import COMPOSITE_V9_MODE, COMPOSITE_V9_RUNTIME_CONTRACT
from mtare_topo.governance import load_json, write_json
from run_m1d_ros_deployment_export_v1 import (
    PYTHON,
    ROS_IMAGE,
    compare_outputs,
    seal,
    sha256,
    verify_runtime,
)


RUN_ID = "gate5_20260820_aee_adapted_ros_deployment_export_v1_seed20260820"
SOURCE_MODE = "M1D_AEE_HEAD_ADAPTED_V1"
STATUS_PASS = "PASS_AEE_ADAPTED_ROS_DEPLOYMENT_EXPORT_V1"
STATUS_FAIL = "FAIL_AEE_ADAPTED_ROS_DEPLOYMENT_EXPORT_V1"
EXPECTED_GATE = 5


def validate_adaptation_source(source_run: Path, expected_seal_sha256: str) -> list[dict[str, Any]]:
    state = load_json(source_run / "RUN_STATE.json")
    summary = load_json(source_run / "metrics/summary.json")
    if state.get("state") != "COMPLETED" or state.get("overall_status") != "PASS_AEE_HEAD_ADAPTATION_V1":
        raise RuntimeError("AEE adaptation source is not a completed PASS")
    if summary.get("overall_status") != "PASS_AEE_HEAD_ADAPTATION_V1" or summary.get("completed_seeds") != 3:
        raise RuntimeError("AEE adaptation aggregate identity/count drift")
    if summary.get("c10_frames_read") != 0 or summary.get("later_sealed_world_frames_read") != 0:
        raise RuntimeError("AEE adaptation source read forbidden sealed data")
    seal_path = source_run / "artifacts/evidence_sha256.txt"
    if sha256(seal_path) != expected_seal_sha256:
        raise RuntimeError("AEE adaptation source seal identity drift")
    prefix = source_run.relative_to(PROJECT_ROOT).as_posix() + "/"
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if not relative.startswith(prefix) or sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"AEE adaptation source seal mismatch: {relative}")
    records = []
    for seed in (0, 1, 2):
        checkpoint = source_run / f"artifacts/models/m1d_seed{seed}/best.pt"
        seed_summary = load_json(source_run / f"artifacts/models/m1d_seed{seed}/summary.json")
        if seed_summary.get("status") != "PASS_AEE_HEAD_ADAPTATION_SEED_V1" or seed_summary.get("seed") != seed:
            raise RuntimeError(f"AEE adapted checkpoint qualification drift: seed {seed}")
        if seed_summary.get("best_checkpoint_sha256") != sha256(checkpoint):
            raise RuntimeError(f"AEE adapted checkpoint hash drift: seed {seed}")
        records.append({"seed": seed, "path": checkpoint, "sha256": sha256(checkpoint)})
    return records


def failure(run_dir: Path, started: float, completed: int, message: str) -> None:
    write_json(run_dir / "metrics/summary.json", {"schema_version": "aee_adapted_ros_deployment_export_summary_v1", "overall_status": STATUS_FAIL, "failure_reason": message, "completed_checkpoints": completed, "planned_checkpoints": 3, "training_steps": 0, "optimizer_steps": 0, "duration_seconds": time.monotonic() - started})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "FAILED", "overall_status": STATUS_FAIL})
    seal(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec, run_dir = load_json(args.spec.resolve()), args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("AEE ROS deployment run identity/state mismatch")
    started = time.monotonic()
    records = []
    try:
        if spec.get("gate") != EXPECTED_GATE or spec.get("operation") != "infrastructure" or spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("approved Gate-5 adapted deployment scope required")
        for name, item in spec["frozen_tools"].items():
            if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"frozen adapted-deployment tool drift: {name}")
        source_run = (PROJECT_ROOT / spec["source_adaptation_run"]).resolve()
        source_run.relative_to(PROJECT_ROOT)
        checkpoints = validate_adaptation_source(source_run, spec["source_adaptation_seal_sha256"])
        import subprocess

        image_id = subprocess.check_output(["docker", "image", "inspect", ROS_IMAGE, "--format", "{{.Id}}"], text=True).strip()
        if image_id != spec["ros_image_id"]:
            raise RuntimeError("adapted ROS deployment image drift")
        write_json(run_dir / "config/input_integrity.json", {"source_run": str(source_run.relative_to(PROJECT_ROOT)), "source_seal_sha256": spec["source_adaptation_seal_sha256"], "checkpoints": [{**item, "path": str(item["path"].relative_to(PROJECT_ROOT))} for item in checkpoints], "ros_image_id": image_id})
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        checkpoint_dir = run_dir / "artifacts/checkpoints"
        parity_dir = run_dir / "artifacts/parity"
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        parity_dir.mkdir(parents=True, exist_ok=False)
        for item in checkpoints:
            seed = item["seed"]
            deployment = checkpoint_dir / f"m1d_aee_seed{seed}_ros.pt"
            exported = export_m1d_ros_checkpoint(source=item["path"], destination=deployment, expected_source_sha256=item["sha256"], expected_seed=seed, expected_mode=SOURCE_MODE)
            source_npz = parity_dir / f"m1d_aee_seed{seed}_torch29.npz"
            ros_npz = parity_dir / f"m1d_aee_seed{seed}_torch20.npz"
            source_report = verify_runtime(str(PYTHON), deployment, source_npz)
            ros_report = verify_runtime("python3", deployment, ros_npz, docker_run_dir=run_dir)
            if source_report.get("mode") != SOURCE_MODE or ros_report.get("mode") != SOURCE_MODE:
                raise RuntimeError(f"adapted deployment mode lost across runtimes: seed {seed}")
            expected_contract = COMPOSITE_V9_RUNTIME_CONTRACT if SOURCE_MODE == COMPOSITE_V9_MODE else None
            if source_report.get("runtime_contract") != expected_contract or ros_report.get("runtime_contract") != expected_contract:
                raise RuntimeError(f"adapted deployment runtime contract lost across runtimes: seed {seed}")
            if source_report["tensor_sha256"] != ros_report["tensor_sha256"]:
                raise RuntimeError(f"adapted tensor hashes differ across runtimes: seed {seed}")
            comparison = compare_outputs(source_npz, ros_npz)
            if not comparison["pass"]:
                raise RuntimeError(f"adapted cross-runtime parity failed: seed {seed}")
            record = {"seed": seed, "mode": SOURCE_MODE, "export": exported, "torch29": source_report, "torch20_ros": ros_report, "cross_runtime": comparison, "deployment_checkpoint": str(deployment.relative_to(run_dir)), "deployment_checkpoint_sha256": sha256(deployment)}
            write_json(parity_dir / f"m1d_aee_seed{seed}_audit.json", record)
            records.append(record)
        summary = {"schema_version": "aee_adapted_ros_deployment_export_summary_v1", "overall_status": STATUS_PASS, "checkpoint_count": 3, "seeds": [0, 1, 2], "mode": SOURCE_MODE, "all_tensor_hashes_equal": True, "all_cross_runtime_forward_pass": True, "training_steps": 0, "optimizer_steps": 0, "ros_image": ROS_IMAGE, "ros_image_id": image_id, "duration_seconds": time.monotonic() - started, "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED", "overall_status": STATUS_PASS})
        sealed = seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure(run_dir, started, len(records), str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
