#!/home/zeng-workstation/anaconda3/bin/python
"""Losslessly export the three sealed V9 composite checkpoints to ROS Torch 2.0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import run_aee_adapted_ros_deployment_export_v1 as base
from _bootstrap import PROJECT_ROOT
from mtare_topo.deployment.m1d_checkpoint import COMPOSITE_V9_MODE
from mtare_topo.governance import load_json


RUN_ID = "gate6_20260822_aee_composite_v9_ros_deployment_export_seed20260822"
STATUS_PASS = "PASS_AEE_COMPOSITE_V9_ROS_DEPLOYMENT_EXPORT"
STATUS_FAIL = "FAIL_AEE_COMPOSITE_V9_ROS_DEPLOYMENT_EXPORT"
SOURCE_STATUS = "PASS_AEE_CORRECTIVE_COMPOSITE_V9"
SEED_STATUS = "COMPLETED_AEE_CORRECTIVE_COMPOSITE_SEED_V9"


def validate_v9_source(source_run: Path, expected_seal_sha256: str) -> list[dict[str, Any]]:
    state = load_json(source_run / "RUN_STATE.json")
    summary = load_json(source_run / "metrics/summary.json")
    if state.get("state") != "COMPLETED" or state.get("overall_status") != SOURCE_STATUS:
        raise RuntimeError("V9 composite source is not a completed PASS")
    if summary.get("overall_status") != SOURCE_STATUS or summary.get("completed_seeds") != 3:
        raise RuntimeError("V9 composite aggregate identity/count drift")
    if any(summary.get(key) != 0 for key in ("c09_frames_read", "c10_frames_read", "formal_benchmark_frames_read")):
        raise RuntimeError("V9 composite source read forbidden sealed data")
    seal_path = source_run / "artifacts/evidence_sha256.txt"
    if base.sha256(seal_path) != expected_seal_sha256:
        raise RuntimeError("V9 composite source seal identity drift")
    prefix = source_run.relative_to(PROJECT_ROOT).as_posix() + "/"
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if not relative.startswith(prefix) or base.sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"V9 composite source seal mismatch: {relative}")
    records = []
    for seed in (0, 1, 2):
        checkpoint = source_run / f"artifacts/models/m1d_seed{seed}/best.pt"
        seed_summary = load_json(source_run / f"artifacts/models/m1d_seed{seed}/summary.json")
        if seed_summary.get("status") != SEED_STATUS or seed_summary.get("seed") != seed:
            raise RuntimeError(f"V9 checkpoint qualification drift: seed {seed}")
        if seed_summary.get("best_checkpoint_sha256") != base.sha256(checkpoint):
            raise RuntimeError(f"V9 checkpoint hash drift: seed {seed}")
        records.append({"seed": seed, "path": checkpoint, "sha256": base.sha256(checkpoint)})
    return records


def main() -> int:
    base.RUN_ID = RUN_ID
    base.EXPECTED_GATE = 6
    base.SOURCE_MODE = COMPOSITE_V9_MODE
    base.STATUS_PASS = STATUS_PASS
    base.STATUS_FAIL = STATUS_FAIL
    base.validate_adaptation_source = validate_v9_source
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
