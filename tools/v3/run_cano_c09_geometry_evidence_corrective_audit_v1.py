#!/usr/bin/env python3
"""Verify, execute once and seal the C09 V1R2 corrective evidence audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_cano_five_topology_cpu_contract_pilot import _seal_manifest

RUN_ID = "gate4_20260820_cano_c09_geometry_evidence_corrective_audit_v1_seed0"
SOURCE_RUN = PROJECT_ROOT / "results/gate4_topology/gate4_20260819_cano_c09_route_conditioned_geometry_qualification_v1r2_seed0"
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_cano_c09_geometry_evidence_corrective_audit_v1.py"
PASS_STATUS = "PASS_CANO_C09_GEOMETRY_EVIDENCE_CORRECTIVE_AUDIT_V1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_seal() -> dict:
    seal = SOURCE_RUN / "artifacts/evidence_sha256.txt"
    lines = seal.read_text(encoding="utf-8").splitlines()
    mismatches = []
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = PROJECT_ROOT / relative
        if not path.is_file() or sha(path) != expected:
            mismatches.append(relative)
    return {"entries": len(lines), "mismatches": mismatches, "seal_sha256": sha(seal), "passed": len(lines) == 506 and not mismatches}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run = args.run_dir.resolve()
    if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("one-time run state mismatch")
    if spec.get("gate") != 4 or spec.get("operation") != "topology_replay" or spec.get("corrective_evidence_audit_only") is not True:
        raise RuntimeError("corrective audit scope mismatch")
    card = load_json(PROJECT_ROOT / spec["data_card"])
    if card.get("approval", {}).get("status") != "APPROVED" or card["approval"].get("authorized_operations") != ["topology_replay"]:
        raise RuntimeError("data card approval mismatch")
    for relative, expected in spec["frozen_inputs"].items():
        if sha(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"frozen input drift: {relative}")
    for name, record in spec["frozen_tools"].items():
        if sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"frozen tool drift: {name}")
    before = verify_source_seal()
    if not before["passed"]:
        raise RuntimeError("source V1R2 seal precheck failed")
    write_json(run / "config/source_seal_before.json", before)
    write_json(run / "RUN_STATE.json", {"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING"})
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools/v3")))
    started = time.monotonic()
    done = subprocess.run(
        ["/home/zeng-workstation/anaconda3/bin/python", str(EXECUTOR), "--run-dir", str(run)],
        cwd=PROJECT_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=600, check=False,
    )
    duration = time.monotonic() - started
    (run / "logs/01_corrective_evidence_audit.log").write_text(done.stdout + f"\nduration_seconds={duration:.6f}\nexit_code={done.returncode}\n", encoding="utf-8")
    print(done.stdout, end="")
    after = verify_source_seal()
    summary = load_json(run / "metrics/summary.json") if (run / "metrics/summary.json").is_file() else {}
    size = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    passed = bool(
        done.returncode == 0 and duration <= 600 and before == after and after["passed"]
        and summary.get("overall_status") == PASS_STATUS and summary.get("visual_patch_count") == 426
        and summary.get("physical_window_layers") == 142 and summary.get("frames") == 15833
        and summary.get("failed_frames") == 0 and summary.get("outside_window_pose_exact") is True
        and summary.get("inference_frames") == summary.get("graph_updates") == summary.get("training_samples_consumed") == 0
        and size <= 100 * 1024**2
    )
    overall = PASS_STATUS if passed else "FAIL_CANO_C09_GEOMETRY_EVIDENCE_CORRECTIVE_AUDIT_V1"
    write_json(run / "metrics/runner_summary.json", {
        "schema_version":"cano_c09_geometry_evidence_corrective_audit_runner_v1",
        "overall_status":overall,"executor_exit_code":done.returncode,"duration_seconds":duration,
        "source_seal_before":before,"source_seal_after":after,"source_seal_unchanged":before == after,
        "result_bytes_before_seal":size,"disk_limit_bytes":100 * 1024**2,
        "claim_boundary":"Read-only correction of the V1R2 patch-count evidence contract; no geometry generation, inference or graph replay.",
    })
    write_json(run / "RUN_STATE.json", {"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if passed else "FAILED","overall_status":overall})
    sealed = _seal_manifest(run)
    print(json.dumps({"overall_status":overall,"sealed_files":sealed}, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
