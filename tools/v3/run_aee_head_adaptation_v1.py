#!/home/zeng-workstation/anaconda3/bin/python
"""Execute, aggregate and seal the approved three-seed AEE head adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.aee_domain_adaptation import sha256
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate2_20260820_aee_head_adaptation_v1_seed20260820"
PYTHON = Path("/tmp/mtare_phase3_torch290_zarr2187/bin/python")
TRAINER = PROJECT_ROOT / "tools/v3/train_aee_head_adaptation_v1.py"
STATUS_PASS = "PASS_AEE_HEAD_ADAPTATION_V1"
STATUS_FAIL = "FAIL_AEE_HEAD_ADAPTATION_V1"
SUMMARY_SCHEMA = "aee_head_adaptation_summary_v1"
PROGRESS_SCHEMA = "aee_head_adaptation_progress_v1"
EXTRA_SUMMARY: dict[str, Any] = {}
TIME_LIMIT_SECONDS = 4 * 3600
DISK_LIMIT_BYTES = 8 * 1024**3
EXPECTED_ENVIRONMENT = {
    "torch": "2.9.0+cu129",
    "cuda": "12.9",
    "gpu": "NVIDIA GeForce RTX 5090 D",
    "zarr": "2.18.7",
    "numcodecs": "0.15.1",
    "numpy": "2.1.3",
    "sklearn": "1.6.1",
    "cuda_available": True,
}


def sorted_pip_freeze_sha256(text: str) -> str:
    """Hash pip-freeze output with the same deterministic sort as the freeze."""

    normalized = "\n".join(sorted(text.splitlines())) + "\n"
    return hashlib.sha256(normalized.encode()).hexdigest()


def validate_pip_check(returncode: int, stdout: str, stderr: str) -> str:
    """Require the exact healthy pip dependency result and return its evidence."""

    evidence = stdout.strip()
    if returncode != 0 or evidence != "No broken requirements found.":
        raise RuntimeError(
            "AEE training pip check failed: " + (stdout + stderr).strip()
        )
    return evidence


def preserve_venv_executable(raw_path: str) -> Path:
    """Make an executable path absolute without dereferencing its venv symlink."""

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        raise RuntimeError("AEE training executable must be an absolute venv path")
    absolute = Path(os.path.abspath(candidate))
    if not absolute.is_file():
        raise RuntimeError(f"AEE training executable missing: {absolute}")
    return absolute


def seal(run_dir: Path) -> int:
    target = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != target)
    target.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files),
        encoding="utf-8",
    )
    return len(files)


def verify_seal(run_dir: Path, expected_sha256: str) -> int:
    seal_path = run_dir / "artifacts/evidence_sha256.txt"
    if sha256(seal_path) != expected_sha256:
        raise RuntimeError(f"source seal identity drift: {run_dir.name}")
    prefix = run_dir.relative_to(PROJECT_ROOT).as_posix() + "/"
    entries = 0
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if not relative.startswith(prefix):
            raise RuntimeError(f"source seal path escapes run: {relative}")
        if sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"source seal mismatch: {relative}")
        entries += 1
    if entries < 1:
        raise RuntimeError("source seal is empty")
    return entries


def probe_environment(python: Path = PYTHON) -> dict[str, Any]:
    code = (
        "import json,numpy,zarr,numcodecs,sklearn,torch;"
        "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
        "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
        "'zarr':zarr.__version__,'numcodecs':numcodecs.__version__,"
        "'numpy':numpy.__version__,'sklearn':sklearn.__version__,"
        "'cuda_available':torch.cuda.is_available()}))"
    )
    completed = subprocess.run(
        [str(python), "-c", code], text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"AEE training environment probe failed: {completed.stderr}")
    observed = json.loads(completed.stdout)
    if observed != EXPECTED_ENVIRONMENT:
        raise RuntimeError(f"AEE training environment identity drift: {observed}")
    return observed


def child_argv(
    spec: dict[str, Any], run_dir: Path, checkpoint: dict[str, Any], python: Path = PYTHON
) -> list[str]:
    seed = int(checkpoint["seed"])
    if seed not in (0, 1, 2):
        raise RuntimeError("AEE checkpoint seed must be 0, 1 or 2")
    return [
        str(python),
        str(TRAINER),
        "--cano-dataset-run",
        str((PROJECT_ROOT / spec["cano_dataset_run"]).resolve()),
        "--aee-sensor-run",
        str((PROJECT_ROOT / spec["source_sensor_run"]).resolve()),
        "--aee-teacher-run",
        str((PROJECT_ROOT / spec["source_teacher_run"]).resolve()),
        "--source-checkpoint",
        str((PROJECT_ROOT / checkpoint["path"]).resolve()),
        "--source-checkpoint-sha256",
        checkpoint["sha256"],
        "--output-dir",
        str(run_dir / f"artifacts/models/m1d_seed{seed}"),
        "--seed",
        str(seed),
        "--epochs",
        "10",
        "--batch-size",
        "128",
        "--learning-rate",
        "0.0001",
        "--weight-decay",
        "0.0001",
        "--workers",
        "0",
    ]


def validate_seed_summary(summary: dict[str, Any], seed: int) -> None:
    if summary.get("status") != "PASS_AEE_HEAD_ADAPTATION_SEED_V1":
        raise RuntimeError(f"AEE head adaptation seed {seed} failed qualification")
    if summary.get("seed") != seed or summary.get("epochs_completed") != 10:
        raise RuntimeError(f"AEE head adaptation seed identity/epoch drift: {seed}")
    if summary.get("train_samples_per_epoch") != {"cano": 3000, "aee": 3000}:
        raise RuntimeError(f"AEE balanced-domain sample drift: {seed}")
    if summary.get("aee_validation_frames") != 3000 or summary.get("cano_validation_frames") != 12500:
        raise RuntimeError(f"AEE/Cano validation count drift: {seed}")
    if summary.get("strict_test_frames_read") != 0 or summary.get("c10_frames_read") != 0:
        raise RuntimeError(f"strict-test data read by seed {seed}")
    if summary.get("cano_c09_validation_frames_read") != 12500:
        raise RuntimeError(f"sealed Cano retention read count drift: {seed}")
    if summary.get("later_sealed_world_frames_read") != 0:
        raise RuntimeError(f"later sealed-world data read by seed {seed}")
    gates = summary.get("gate")
    if not isinstance(gates, dict) or set(gates) != {
        "aee_direction_vs_b0",
        "aee_empty_rate",
        "aee_count_macro_f1",
        "aee_role_macro_f1",
        "cano_direction_retention",
        "frozen_tensor_identity",
        "fixed_probe_z_role_identity",
        "semantic_heads_changed",
    } or not all(gates.values()):
        raise RuntimeError(f"AEE representation gate failed: seed {seed}")


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if [item.get("seed") for item in summaries] != [0, 1, 2]:
        raise RuntimeError("AEE adaptation must contain seeds 0/1/2 exactly once in order")
    for item in summaries:
        validate_seed_summary(item, int(item["seed"]))
    return {
        "aee_direction_f1_by_seed": [item["adapted_aee_validation"]["direction"]["f1"] for item in summaries],
        "aee_direction_f1_median": float(np.median([item["adapted_aee_validation"]["direction"]["f1"] for item in summaries])),
        "aee_empty_rate_by_seed": [item["adapted_aee_validation"]["direction"]["empty_rate"] for item in summaries],
        "aee_count_macro_f1_by_seed": [item["adapted_aee_validation"]["count"]["macro_f1_present"] for item in summaries],
        "aee_role_macro_f1_by_seed": [item["adapted_aee_validation"]["role"]["macro_f1_present"] for item in summaries],
        "cano_direction_f1_change_by_seed": [
            item["adapted_cano_validation"]["direction"]["f1"]
            - item["source_cano_validation"]["direction"]["f1"]
            for item in summaries
        ],
        "all_seed_gates_passed": True,
    }


def fail(run_dir: Path, started: float, completed: int, message: str) -> None:
    write_json(
        run_dir / "metrics/summary.json",
        {
            "schema_version": SUMMARY_SCHEMA,
            "overall_status": STATUS_FAIL,
            "failure_reason": message,
            "completed_seeds": completed,
            "planned_seeds": 3,
            "duration_seconds": time.monotonic() - started,
            "c10_frames_read": 0,
            "later_sealed_world_frames_read": 0,
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "FAILED", "overall_status": STATUS_FAIL},
    )
    seal(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("AEE adaptation run identity/state mismatch")
    started = time.monotonic()
    summaries: list[dict[str, Any]] = []
    try:
        authorization = spec.get("user_authorization", {})
        if spec.get("gate") != 2 or spec.get("operation") != "training" or authorization.get("status") != "APPROVED":
            raise RuntimeError("approved Gate-2 AEE head-adaptation scope required")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        expected_card_status = spec.get(
            "expected_data_card_status",
            "APPROVED_FOR_DATA_EXPORT_TEACHER_AND_HEAD_ADAPTATION",
        )
        if card.get("status") != expected_card_status:
            raise RuntimeError("AEE adaptation Data Card status drift")
        approval = card.get("approval", {})
        if approval.get("authorized_gates") != [2] or not {"training", "checkpoint_selection"}.issubset(approval.get("authorized_operations", [])):
            raise RuntimeError("AEE adaptation Data Card authorization drift")
        for name, item in spec["frozen_tools"].items():
            if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"frozen AEE adaptation tool drift: {name}")
        source_runs = {
            "sensor": (PROJECT_ROOT / spec["source_sensor_run"]).resolve(),
            "teacher": (PROJECT_ROOT / spec["source_teacher_run"]).resolve(),
            "cano": (PROJECT_ROOT / spec["cano_dataset_run"]).resolve(),
        }
        source_evidence = {}
        for name, source in source_runs.items():
            source.relative_to(PROJECT_ROOT)
            expected = spec["source_seals"][name]
            source_evidence[name] = {"run": str(source.relative_to(PROJECT_ROOT)), "seal_entries": verify_seal(source, expected), "seal_sha256": expected}
        sensor_summary = load_json(source_runs["sensor"] / "metrics/summary.json")
        teacher_summary = load_json(source_runs["teacher"] / "metrics/summary.json")
        if sensor_summary.get("overall_status") != "PASS_AEE_DOMAIN_SENSOR_EXPORT_V1" or sensor_summary.get("effective_frames") != 6000:
            raise RuntimeError("AEE sensor source is not exact PASS/6000")
        expected_teacher_status = spec.get(
            "expected_teacher_status", "PASS_AEE_OBJECTIVE_TEACHER_EXPORT_V1"
        )
        if teacher_summary.get("overall_status") != expected_teacher_status or teacher_summary.get("samples") != 6000:
            raise RuntimeError("AEE teacher source is not exact PASS/6000")
        checkpoints = spec.get("source_checkpoints")
        if not isinstance(checkpoints, list) or [item.get("seed") for item in checkpoints] != [0, 1, 2]:
            raise RuntimeError("source checkpoints must be seeds 0/1/2 in order")
        for checkpoint in checkpoints:
            if sha256(PROJECT_ROOT / checkpoint["path"]) != checkpoint["sha256"]:
                raise RuntimeError(f"source checkpoint identity drift: seed {checkpoint['seed']}")
        python = preserve_venv_executable(
            str(spec.get("python_executable", str(PYTHON)))
        )
        environment = probe_environment(python)
        freeze_path = PROJECT_ROOT / spec["sidecar_freeze"]["path"]
        if sha256(freeze_path) != spec["sidecar_freeze"]["sha256"]:
            raise RuntimeError("AEE adaptation sidecar freeze drift")
        pip_freeze_path = PROJECT_ROOT / spec["sidecar_freeze"]["pip_freeze_path"]
        expected_pip_freeze_sha256 = spec["sidecar_freeze"]["pip_freeze_sha256"]
        if sha256(pip_freeze_path) != expected_pip_freeze_sha256:
            raise RuntimeError("AEE adaptation frozen pip-freeze evidence drift")
        completed = subprocess.run(
            [str(python), "-m", "pip", "freeze"],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"AEE training pip-freeze probe failed: {completed.stderr}")
        live_pip_freeze_sha256 = sorted_pip_freeze_sha256(completed.stdout)
        if live_pip_freeze_sha256 != expected_pip_freeze_sha256:
            raise RuntimeError("AEE adaptation live pip-freeze identity drift")
        pip_check = subprocess.run(
            [str(python), "-m", "pip", "check"],
            text=True,
            capture_output=True,
            check=False,
        )
        pip_check_evidence = validate_pip_check(
            pip_check.returncode, pip_check.stdout, pip_check.stderr
        )
        write_json(
            run_dir / "config/input_integrity.json",
            {
                "sources": source_evidence,
                "checkpoints": checkpoints,
                "environment": environment,
                "python_executable": str(python),
                "live_pip_freeze_sha256": live_pip_freeze_sha256,
                "pip_check": pip_check_evidence,
                "host": platform.platform(),
            },
        )
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        (run_dir / "artifacts/models").mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        for checkpoint in checkpoints:
            seed = int(checkpoint["seed"])
            remaining = max(1, TIME_LIMIT_SECONDS - int(time.monotonic() - started))
            completed = subprocess.run(child_argv(spec, run_dir, checkpoint, python), cwd=PROJECT_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=remaining, check=False)
            (run_dir / f"logs/m1d_seed{seed}.log").write_text(completed.stdout + f"\nexit_code={completed.returncode}\n", encoding="utf-8")
            if completed.returncode != 0 or "does not have a deterministic implementation" in completed.stdout:
                raise RuntimeError(f"AEE head adaptation child failed: seed {seed}, exit {completed.returncode}")
            summary = load_json(run_dir / f"artifacts/models/m1d_seed{seed}/summary.json")
            validate_seed_summary(summary, seed)
            summaries.append(summary)
            write_json(run_dir / "metrics/progress.json", {"schema_version": PROGRESS_SCHEMA, "completed_seeds": len(summaries), "planned_seeds": 3, "last_seed": seed, "duration_seconds": time.monotonic() - started})
        aggregate_metrics = aggregate(summaries)
        result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if result_bytes > DISK_LIMIT_BYTES or time.monotonic() - started > TIME_LIMIT_SECONDS:
            raise RuntimeError("AEE adaptation resource limit exceeded")
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "overall_status": STATUS_PASS,
            "completed_seeds": 3,
            "seeds": [0, 1, 2],
            "aggregate": aggregate_metrics,
            "per_seed": summaries,
            "result_bytes_before_seal": result_bytes,
            "disk_limit_bytes": DISK_LIMIT_BYTES,
            "duration_seconds": time.monotonic() - started,
            "time_limit_seconds": TIME_LIMIT_SECONDS,
            "train_samples_per_seed_epoch": {"cano": 3000, "aee": 3000},
            "aee_validation_frames_per_seed": 3000,
            "cano_c09_retention_frames_per_seed": 12500,
            "c10_frames_read": 0,
            "later_sealed_world_frames_read": 0,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            **EXTRA_SUMMARY,
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED", "overall_status": STATUS_PASS})
        sealed = seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        fail(run_dir, started, len(summaries), str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
