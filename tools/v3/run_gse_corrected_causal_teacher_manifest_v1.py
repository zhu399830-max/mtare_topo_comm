#!/usr/bin/env python3
"""Run once and seal the corrected C01-C08 causal Teacher manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate2_20260826_gse_corrected_causal_teacher_manifest_v1_seed0"
PASS_STATUS = "PASS_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1"
FAIL_STATUS = "FAIL_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1"
PYTHON = Path("/home/zeng-workstation/anaconda3/bin/python")
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_gse_corrected_causal_teacher_manifest_v1.py"
OLD_TEACHER_RUN = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0"
PROOF_RUN = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_causal_change_point_proof_v1_seed0"
TIME_LIMIT_SECONDS = 300
DISK_LIMIT_BYTES = 512 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_exact_seal(run_dir: Path, expected_status: str) -> dict:
    state = load_json(run_dir / "RUN_STATE.json")
    if state.get("state") != "COMPLETED" or state.get("overall_status") != expected_status:
        raise RuntimeError(f"sealed source is not expected PASS: {run_dir.name}")
    seal = run_dir / "artifacts/evidence_sha256.txt"
    sealed: set[Path] = set()
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, raw = line.split("  ", 1)
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"source seal mismatch: {path}")
        sealed.add(path.resolve())
    actual = {
        path.resolve()
        for path in run_dir.rglob("*")
        if path.is_file() and path != seal
    }
    if sealed != actual:
        raise RuntimeError(f"source seal coverage mismatch: {run_dir.name}")
    return {
        "run": str(run_dir.relative_to(PROJECT_ROOT)),
        "status": expected_status,
        "seal_sha256": _sha256(seal),
        "sealed_file_count": len(sealed),
    }


def _verify_sources() -> dict:
    old = _verify_exact_seal(OLD_TEACHER_RUN, "PASS_GSE_TEACHER_MANIFEST_V1")
    proof = _verify_exact_seal(PROOF_RUN, "PASS_GSE_CAUSAL_CHANGE_POINT_PROOF_V1")
    return {
        "old_teacher": old,
        "causal_change_point_proof": proof,
        "c09_records_consumed": 0,
        "c10_records_consumed": 0,
        "mtare_records_consumed": 0,
    }


def _seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    destination.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files),
        encoding="utf-8",
    )
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID:
        raise RuntimeError("unexpected corrected Teacher run identity")
    if load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("corrected Teacher manifest may execute only once")
    if spec.get("gate") != 2 or spec.get("operation") != "teacher_generation" or spec.get("seed") != 0:
        raise RuntimeError("corrected Teacher run scope mismatch")
    if spec.get("user_authorization", {}).get("status") != "APPROVED":
        raise RuntimeError("user decision A is not bound to the run")
    card = load_json(PROJECT_ROOT / spec["data_card"])
    if card.get("approval", {}).get("status") != "APPROVED":
        raise RuntimeError("corrected Teacher Data Card is not approved")
    for name, record in spec["frozen_tools"].items():
        if _sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"frozen tool mismatch: {name}")
    for raw_path, expected in spec["frozen_inputs"].items():
        if _sha256(PROJECT_ROOT / raw_path) != expected:
            raise RuntimeError(f"frozen input mismatch: {raw_path}")
    if not PYTHON.is_file():
        raise RuntimeError("frozen Python environment is missing")
    environment = json.loads(
        subprocess.check_output(
            [
                str(PYTHON),
                "-c",
                "import json,numpy,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__},sort_keys=True))",
            ],
            text=True,
        )
    )
    if environment != {"python": "3.13.5", "numpy": "2.1.3"}:
        raise RuntimeError(f"corrected Teacher environment drift: {environment}")
    if shutil.disk_usage(PROJECT_ROOT).free < 2 * 1024**3:
        raise RuntimeError("less than 2 GiB free")
    source_before = _verify_sources()
    write_json(
        run_dir / "config/environment.json",
        {
            "executable": str(PYTHON),
            "versions": environment,
            "cpu_only": True,
            "source_verification_before": source_before,
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "Immutable C01-C08 corrected causal Teacher manifest; zero C09/C10/M-TARE/model/training.",
        },
    )
    command = [
        str(PYTHON),
        str(EXECUTOR),
        "--run-dir",
        str(run_dir),
        "--old-teacher-run",
        str(OLD_TEACHER_RUN),
        "--proof-run",
        str(PROOF_RUN),
    ]
    (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    started = time.monotonic()
    code = 124
    output = ""
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=TIME_LIMIT_SECONDS,
            check=False,
        )
        code = completed.returncode
        output = completed.stdout
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or "") + "\nTIMEOUT\n"
    duration = time.monotonic() - started
    (run_dir / "logs/01_corrected_teacher_manifest.log").write_text(
        output + f"\nduration_seconds={duration:.6f}\nexit_code={code}\n",
        encoding="utf-8",
    )
    print(output, end="")
    source_after = _verify_sources()
    summary_path = run_dir / "metrics/summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    required = tuple(
        run_dir / f"artifacts/{name}.jsonl"
        for name in (
            "teacher_observations",
            "traversal_manifest",
            "association_pairs",
            "structural_identities",
            "label_application_audit",
            "world_manifest_summary",
        )
    )
    result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    passed = bool(
        code == 0
        and duration <= TIME_LIMIT_SECONDS
        and summary.get("overall_status") == PASS_STATUS
        and summary.get("world_count") == 80
        and summary.get("observation_count") == 188126
        and summary.get("applied_change_point_label_count") == 1031
        and summary.get("identity_counts", {}).get("geometry_transition") == 76
        and summary.get("c09_worlds_consumed") == 0
        and summary.get("strict_test_worlds_read") == 0
        and summary.get("mtare_worlds_read") == 0
        and summary.get("optimizer_steps") == 0
        and all(path.is_file() for path in required)
        and source_before == source_after
        and result_bytes <= DISK_LIMIT_BYTES
    )
    overall = PASS_STATUS if passed else FAIL_STATUS
    write_json(
        run_dir / "metrics/runner_summary.json",
        {
            "overall_status": overall,
            "executor_exit_code": code,
            "duration_seconds": duration,
            "source_verification_before": source_before,
            "source_verification_after": source_after,
            "source_unchanged": source_before == source_after,
            "result_bytes_before_seal": result_bytes,
            "disk_limit_bytes": DISK_LIMIT_BYTES,
            "c09_worlds_consumed": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
            "model_inference_frames": 0,
            "training_samples_consumed": 0,
            "optimizer_steps": 0,
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if passed else "FAILED",
            "overall_status": overall,
        },
    )
    sealed_files = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "sealed_files": sealed_files}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
