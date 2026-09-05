#!/usr/bin/env python3
"""Execute one approved original Cano native-export smoke with frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import selectors
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


EXPECTED_REMOTE = "https://github.com/LorenzoCanoAn/procedural-subt-gen"
EXPECTED_COMMIT = "b6c77621187404b4dfab1249c7a1b40f63ad9ab3"
EXPECTED_ENTRYPOINT_SHA256 = (
    "7c765d160b21198d611c297b160664aaffc3c12c9d18c36fc9851a9d182113a4"
)
EXPECTED_FREEZE_SHA256 = (
    "e917255cba6815804085d1b873b0bb4ad3e84764aaf7e43140944fb9d257e975"
)
EXPECTED_ENVIRONMENT = Path("/tmp/mtare_cano_compat_e1_py312_np126_sp112")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _run_logged(
    argv: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_s: int | None = None,
) -> tuple[int, float]:
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("argv=" + json.dumps(argv, ensure_ascii=False) + "\n")
        log.write(f"cwd={cwd}\n")
        log.write(f"started_at_utc={datetime.now(timezone.utc).isoformat()}\n")
        log.flush()
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        return_code: int | None = None
        while return_code is None:
            if timeout_s is not None and time.monotonic() - started > timeout_s:
                process.kill()
                process.wait()
                log.write(f"TIMEOUT_AFTER_SECONDS={timeout_s}\n")
                return_code = 124
                break
            for key, _ in selector.select(timeout=0.5):
                line = key.fileobj.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
            polled = process.poll()
            if polled is not None:
                remainder = process.stdout.read()
                if remainder:
                    sys.stdout.write(remainder)
                    sys.stdout.flush()
                    log.write(remainder)
                return_code = polled
        selector.close()
        duration = time.monotonic() - started
        log.write(f"finished_at_utc={datetime.now(timezone.utc).isoformat()}\n")
        log.write(f"duration_seconds={duration:.6f}\n")
        log.write(f"exit_code={return_code}\n")
        return return_code, duration


def _source_snapshot(source: Path, entrypoint: Path) -> dict[str, Any]:
    status = _git(source, "status", "--porcelain=v1")
    return {
        "remote": _git(source, "remote", "get-url", "origin"),
        "commit": _git(source, "rev-parse", "HEAD"),
        "tracked_clean": not bool(status),
        "status_porcelain": status,
        "entrypoint_sha256": _sha256(entrypoint),
    }


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _write_manifest(run_dir: Path) -> int:
    manifest = run_dir / "artifacts" / "evidence_sha256.txt"
    files = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path != manifest
    )
    lines = [f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}" for path in files]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _validate_fixed_scope(spec: dict[str, Any], run_dir: Path) -> None:
    if spec.get("user_authorization", {}).get("status") != "APPROVED":
        raise RuntimeError("run authorization is not approved")
    if spec.get("source", {}).get("remote") != EXPECTED_REMOTE:
        raise RuntimeError("spec source remote mismatch")
    if spec.get("source", {}).get("commit") != EXPECTED_COMMIT:
        raise RuntimeError("spec source commit mismatch")
    if spec.get("source", {}).get("entrypoint_sha256") != EXPECTED_ENTRYPOINT_SHA256:
        raise RuntimeError("spec entrypoint hash mismatch")
    scope = spec.get("data_scope", {})
    expected_scope = {
        "native_export_attempts": 1,
        "temporary_smoke_environments": 1,
        "requested_grown_tunnels": 3,
        "requested_connector_tunnels": 1,
        "formal_dataset_worlds": 0,
        "lidar_samples": 0,
        "labels": 0,
        "training_samples": 0,
    }
    for key, expected in expected_scope.items():
        if scope.get(key) != expected:
            raise RuntimeError(f"fixed data_scope mismatch for {key}")
    command = spec.get("command")
    if not isinstance(command, list):
        raise RuntimeError("spec command is not a list")
    expected_target = run_dir / "artifacts" / "native_export"
    if str(expected_target) not in command:
        raise RuntimeError("command does not target this immutable run directory")
    if expected_target.exists():
        raise RuntimeError("native export target already exists")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    spec_path = args.spec if args.spec.is_absolute() else PROJECT_ROOT / args.spec
    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    spec = load_json(spec_path)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    _validate_fixed_scope(spec, run_dir)

    source = PROJECT_ROOT / spec["source"]["checkout"]
    entrypoint = source / spec["source"]["entrypoint"]
    python = EXPECTED_ENVIRONMENT / "bin" / "python"
    evidence_freeze = (
        PROJECT_ROOT
        / "results/gate0_baseline/gate0_20260810_cano_dependency_compatibility_matrix_v1_seed0/config/pip_freeze_E1.txt"
    )
    if not python.is_file():
        raise RuntimeError(f"frozen E1 Python is missing: {python}")
    if _sha256(evidence_freeze) != EXPECTED_FREEZE_SHA256:
        raise RuntimeError("stored E1 freeze evidence hash mismatch")

    pre_source = _source_snapshot(source, entrypoint)
    if pre_source != {
        "remote": EXPECTED_REMOTE,
        "commit": EXPECTED_COMMIT,
        "tracked_clean": True,
        "status_porcelain": "",
        "entrypoint_sha256": EXPECTED_ENTRYPOINT_SHA256,
    }:
        raise RuntimeError(f"third-party source precheck failed: {pre_source}")

    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": run_dir.name,
            "state": "RUNNING",
            "note": "One approved original Cano native-export attempt is running.",
        },
    )
    print("[1/4] Verifying frozen E1 environment", flush=True)
    freeze = subprocess.check_output(
        [str(python), "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT
    )
    runtime_freeze = run_dir / "config" / "runtime_pip_freeze_E1.txt"
    runtime_freeze.write_text(freeze, encoding="utf-8")
    freeze_matches = _sha256(runtime_freeze) == EXPECTED_FREEZE_SHA256
    pip_check_rc, pip_check_seconds = _run_logged(
        [str(python), "-m", "pip", "check"],
        cwd=PROJECT_ROOT,
        log_path=run_dir / "logs" / "01_e1_pip_check.log",
        timeout_s=120,
    )
    if not freeze_matches or pip_check_rc != 0:
        raise RuntimeError("frozen E1 environment drifted or pip check failed")
    versions = subprocess.check_output(
        [
            str(python),
            "-c",
            "import numpy,scipy,open3d,pyvista,sys;"
            "print(sys.version);"
            "print('numpy='+numpy.__version__);"
            "print('scipy='+scipy.__version__);"
            "print('open3d='+open3d.__version__);"
            "print('pyvista='+pyvista.__version__)",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    (run_dir / "config" / "runtime_versions.txt").write_text(versions, encoding="utf-8")
    write_json(run_dir / "artifacts" / "source_before.json", pre_source)

    print("[2/4] Running the exact original generate_environments.py command", flush=True)
    export_rc, export_seconds = _run_logged(
        list(spec["command"]),
        cwd=source,
        log_path=run_dir / "logs" / "02_original_native_export.log",
        timeout_s=1860,
    )
    post_source = _source_snapshot(source, entrypoint)
    write_json(run_dir / "artifacts" / "source_after.json", post_source)
    source_unchanged = post_source == pre_source

    world_dir = run_dir / "artifacts" / "native_export" / "env_001"
    expected_files = ["mesh.obj", "axis.txt", "fta_dist.txt", "model.sdf"]
    file_presence = {name: (world_dir / name).is_file() for name in expected_files}
    export_dir = run_dir / "artifacts" / "native_export"
    environment_directories = (
        sorted(path.name for path in export_dir.iterdir() if path.is_dir())
        if export_dir.is_dir()
        else []
    )

    validator_rc: int | None = None
    validator_seconds: float | None = None
    validation: dict[str, Any] | None = None
    if export_rc == 0 and source_unchanged and all(file_presence.values()):
        print("[3/4] Auditing native mesh, axis labels, FTA, and SDF", flush=True)
        validation_path = run_dir / "metrics" / "native_export_validation.json"
        validator_argv = [
            str(python),
            str(PROJECT_ROOT / "tools/v3/validate_cano_native_export.py"),
            "--world-dir",
            str(world_dir),
            "--result",
            str(validation_path),
            "--preview",
            str(run_dir / "previews" / "native_mesh_axis_diagnostic.png"),
            "--run-id",
            run_dir.name,
        ]
        validator_rc, validator_seconds = _run_logged(
            validator_argv,
            cwd=PROJECT_ROOT,
            log_path=run_dir / "logs" / "03_native_export_validation.log",
            timeout_s=900,
        )
        if validation_path.is_file():
            validation = load_json(validation_path)

    if export_rc != 0:
        overall = "FAIL_ORIGINAL_NATIVE_EXPORT"
    elif not source_unchanged:
        overall = "FAIL_THIRD_PARTY_SOURCE_CHANGED"
    elif environment_directories != ["env_001"] or not all(file_presence.values()):
        overall = "FAIL_NATIVE_OUTPUT_CONTRACT"
    elif validator_rc != 0 or validation is None:
        overall = "FAIL_NATIVE_VALIDATION"
    else:
        overall = str(validation["overall_status"])

    print("[4/4] Writing immutable summary and evidence hashes", flush=True)
    output_hashes = {
        name: _sha256(world_dir / name)
        for name in expected_files
        if (world_dir / name).is_file()
    }
    summary = {
        "schema_version": "cano_native_export_smoke_summary_v1",
        "run_id": run_dir.name,
        "overall_status": overall,
        "execution": {
            "original_export_exit_code": export_rc,
            "original_export_duration_seconds": export_seconds,
            "validator_exit_code": validator_rc,
            "validator_duration_seconds": validator_seconds,
            "requested_environments": 1,
            "requested_grown_tunnels": 3,
            "requested_connector_tunnels": 1,
            "environment_directories": environment_directories,
        },
        "environment": {
            "id": "E1",
            "prefix": str(EXPECTED_ENVIRONMENT),
            "freeze_matches_compatibility_evidence": freeze_matches,
            "pip_check_exit_code": pip_check_rc,
            "pip_check_duration_seconds": pip_check_seconds,
        },
        "source": {
            "before": pre_source,
            "after": post_source,
            "unchanged": source_unchanged,
        },
        "outputs": {
            "files_present": file_presence,
            "sha256": output_hashes,
            "native_export_bytes": _directory_size(export_dir) if export_dir.exists() else 0,
        },
        "validation": validation,
        "counts": {
            "temporary_native_export_worlds": int(environment_directories == ["env_001"]),
            "formal_dataset_worlds": 0,
            "lidar_samples": 0,
            "labels": 0,
            "training_samples": 0,
        },
        "reproducibility": {
            "run_spec_seed": 0,
            "original_entrypoint_seed_argument": False,
            "numpy_rng_seeded_by_original_entrypoint": False,
            "deterministic_replay_claim_allowed": False,
        },
        "host": {
            "platform": platform.platform(),
            "orchestrator_python": sys.version,
            "free_disk_bytes_after": shutil.disk_usage(PROJECT_ROOT).free,
        },
        "claim_boundary": (
            "This one-world smoke can establish native file emission and mesh readability "
            "only. It cannot establish topology success, per-tunnel ground-truth validity, "
            "deterministic replay, graph-mesh consistency, navigation feasibility, dataset "
            "quality, Isaac/LiDAR validity, labels, or learning readiness."
        ),
    }
    write_json(run_dir / "metrics" / "summary.json", summary)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": run_dir.name,
            "state": overall,
            "note": summary["claim_boundary"],
        },
    )
    manifest_entries = _write_manifest(run_dir)
    print(
        json.dumps(
            {
                "overall_status": overall,
                "manifest_entries": manifest_entries,
                "summary": str(run_dir / "metrics" / "summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if overall.startswith("PARTIAL_NATIVE_MESH_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
