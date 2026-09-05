#!/usr/bin/env python3
"""Run one approved fixed-seed Cano read-only adapter smoke and seal evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
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


SOURCE = PROJECT_ROOT / "external/procedural-subt-gen"
EXPECTED_COMMIT = "b6c77621187404b4dfab1249c7a1b40f63ad9ab3"
EXPECTED_REMOTE = "https://github.com/LorenzoCanoAn/procedural-subt-gen"
EXPECTED_ENTRYPOINT_SHA256 = "7c765d160b21198d611c297b160664aaffc3c12c9d18c36fc9851a9d182113a4"
EXPECTED_FREEZE_SHA256 = "e917255cba6815804085d1b873b0bb4ad3e84764aaf7e43140944fb9d257e975"
E1 = Path("/tmp/mtare_cano_compat_e1_py312_np126_sp112")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _source_snapshot() -> dict[str, Any]:
    return {
        "remote": _git("remote", "get-url", "origin"),
        "commit": _git("rev-parse", "HEAD"),
        "tracked_clean": not bool(_git("status", "--porcelain=v1")),
        "entrypoint_sha256": _sha256(SOURCE / "scripts/generate_environments.py"),
    }


def _run_logged(
    argv: list[str], cwd: Path, log_path: Path, timeout_s: int
) -> tuple[int, float]:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("argv=" + json.dumps(argv, ensure_ascii=False) + "\n")
        log.write(f"cwd={cwd}\nstarted_at_utc={datetime.now(timezone.utc).isoformat()}\n")
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
            if time.monotonic() - started > timeout_s:
                process.kill()
                process.wait()
                log.write(f"TIMEOUT_AFTER_SECONDS={timeout_s}\n")
                return_code = 124
                break
            for key, _ in selector.select(timeout=0.5):
                line = key.fileobj.readline()
                if line:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
            polled = process.poll()
            if polled is not None:
                remainder = process.stdout.read()
                if remainder:
                    print(remainder, end="", flush=True)
                    log.write(remainder)
                return_code = polled
        selector.close()
        duration = time.monotonic() - started
        log.write(
            f"finished_at_utc={datetime.now(timezone.utc).isoformat()}\n"
            f"duration_seconds={duration:.6f}\nexit_code={return_code}\n"
        )
        return return_code, duration


def _manifest(run_dir: Path) -> int:
    manifest = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != manifest)
    manifest.write_text(
        "\n".join(
            f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}" for path in files
        )
        + "\n",
        encoding="utf-8",
    )
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec_path = args.spec if args.spec.is_absolute() else PROJECT_ROOT / args.spec
    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    spec = load_json(spec_path)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if spec.get("user_authorization", {}).get("status") != "APPROVED":
        raise RuntimeError("adapter smoke is not approved")
    scope = spec.get("data_scope", {})
    expected_scope = {
        "adapter_generation_attempts": 1,
        "temporary_smoke_worlds": 1,
        "requested_grown_tunnels": 3,
        "requested_connector_tunnels": 1,
        "formal_dataset_worlds": 0,
        "lidar_samples": 0,
        "labels": 0,
        "training_samples": 0,
    }
    for key, value in expected_scope.items():
        if scope.get(key) != value:
            raise RuntimeError(f"scope mismatch: {key}")
    bundle = run_dir / "artifacts/world_000"
    if str(bundle) not in spec.get("command", []):
        raise RuntimeError("frozen command does not target this run")
    if bundle.exists():
        raise RuntimeError("bundle already exists")

    expected_source = {
        "remote": EXPECTED_REMOTE,
        "commit": EXPECTED_COMMIT,
        "tracked_clean": True,
        "entrypoint_sha256": EXPECTED_ENTRYPOINT_SHA256,
    }
    source_before = _source_snapshot()
    if source_before != expected_source:
        raise RuntimeError(f"source precheck failed: {source_before}")
    python = E1 / "bin/python"
    if not python.is_file():
        raise RuntimeError("E1 Python missing")
    stored_freeze = PROJECT_ROOT / (
        "results/gate0_baseline/"
        "gate0_20260810_cano_dependency_compatibility_matrix_v1_seed0/"
        "config/pip_freeze_E1.txt"
    )
    if _sha256(stored_freeze) != EXPECTED_FREEZE_SHA256:
        raise RuntimeError("stored E1 freeze mismatch")

    tool_paths = {
        "runner": Path(__file__).resolve(),
        "adapter": PROJECT_ROOT / "tools/v3/generate_cano_audited_bundle.py",
        "validator": PROJECT_ROOT / "tools/v3/validate_cano_audited_bundle.py",
        "spec": spec_path,
    }
    write_json(
        run_dir / "config/tool_hashes.json",
        {
            name: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": _sha256(path)}
            for name, path in tool_paths.items()
        },
    )
    write_json(run_dir / "artifacts/source_before.json", source_before)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": run_dir.name,
            "state": "RUNNING",
            "note": "Exactly one approved fixed-seed read-only Cano adapter candidate is running.",
        },
    )

    print("[1/4] Verify frozen E1", flush=True)
    runtime_freeze = subprocess.check_output(
        [str(python), "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT
    )
    freeze_path = run_dir / "config/runtime_pip_freeze_E1.txt"
    freeze_path.write_text(runtime_freeze, encoding="utf-8")
    freeze_match = _sha256(freeze_path) == EXPECTED_FREEZE_SHA256
    pip_rc, pip_seconds = _run_logged(
        [str(python), "-m", "pip", "check"],
        PROJECT_ROOT,
        run_dir / "logs/01_e1_pip_check.log",
        120,
    )
    if not freeze_match or pip_rc != 0:
        raise RuntimeError("E1 environment drift")

    print("[2/4] Generate one audited fixed-seed candidate", flush=True)
    generation_rc, generation_seconds = _run_logged(
        list(spec["command"]),
        PROJECT_ROOT,
        run_dir / "logs/02_adapter_generation.log",
        1860,
    )
    source_after = _source_snapshot()
    write_json(run_dir / "artifacts/source_after.json", source_after)
    source_unchanged = source_after == source_before

    validation: dict[str, Any] | None = None
    validator_rc: int | None = None
    validator_seconds: float | None = None
    result_path = run_dir / "metrics/adapter_validation.json"
    if generation_rc == 0 and source_unchanged:
        print("[3/4] Apply strict graph, GT, and mesh gates", flush=True)
        validator_command = [
            str(python),
            str(PROJECT_ROOT / "tools/v3/validate_cano_audited_bundle.py"),
            "--bundle",
            str(bundle),
            "--result",
            str(result_path),
            "--preview",
            str(run_dir / "previews/adapter_mesh_splines_full.png"),
            "--run-id",
            run_dir.name,
        ]
        validator_rc, validator_seconds = _run_logged(
            validator_command,
            PROJECT_ROOT,
            run_dir / "logs/03_strict_validation.log",
            900,
        )
        if result_path.is_file():
            validation = load_json(result_path)

    if not source_unchanged:
        overall = "FAIL_ADAPTER_SOURCE_CHANGED"
    elif generation_rc != 0:
        audit_path = bundle / "generation_audit.json"
        audit = load_json(audit_path) if audit_path.is_file() else {}
        overall = "FAIL_ADAPTER_GENERATION_RETURN_" + str(
            audit.get("overall_status", f"EXIT_{generation_rc}")
        )
    elif validation is None:
        overall = "FAIL_ADAPTER_VALIDATOR_NO_RESULT"
    else:
        overall = str(validation["overall_status"])

    print("[4/4] Seal evidence", flush=True)
    summary = {
        "schema_version": "cano_audited_adapter_smoke_summary_v1",
        "run_id": run_dir.name,
        "overall_status": overall,
        "execution": {
            "generation_exit_code": generation_rc,
            "generation_duration_seconds": generation_seconds,
            "validator_exit_code": validator_rc,
            "validator_duration_seconds": validator_seconds,
            "single_attempt_enforced": True,
        },
        "environment": {
            "id": "E1",
            "prefix": str(E1),
            "freeze_match": freeze_match,
            "pip_check_exit_code": pip_rc,
            "pip_check_duration_seconds": pip_seconds,
        },
        "source": {"before": source_before, "after": source_after, "unchanged": source_unchanged},
        "validation": validation,
        "counts": {
            "temporary_smoke_worlds": int(bundle.is_dir()),
            "formal_dataset_worlds": 0,
            "lidar_samples": 0,
            "labels": 0,
            "training_samples": 0,
        },
        "host": {
            "platform": platform.platform(),
            "orchestrator_python": sys.version,
            "free_disk_bytes_after": shutil.disk_usage(PROJECT_ROOT).free,
        },
        "claim_boundary": (
            "Even PASS establishes only one adapter candidate and would require a separate "
            "fixed-seed replay approval. This run never creates a formal dataset world or "
            "authorizes batch, Isaac/LiDAR, labels, training, or M-TARE changes."
        ),
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": run_dir.name,
            "state": overall,
            "note": summary["claim_boundary"],
        },
    )
    (run_dir / "previews/README.md").write_text(
        "# Preview provenance\n\n"
        f"- Run: `{run_dir.name}`\n"
        "- Split: `NONE_ADAPTER_SMOKE_ONLY`; formal dataset worlds: 0.\n"
        "- World: fixed-seed adapter candidate `world_000`; no trajectory/LiDAR sample.\n"
        "- Preview: actual complete OBJ vertices with every exported per-tunnel spline in X-Y/X-Z/Y-Z.\n"
        "- Units: upstream source units expected metres; scale not independently certified.\n"
        "- Supports: extent, tunnel identity, spline alignment, and failure inspection.\n"
        "- Does not support: Isaac rendering, navigation, LiDAR, labels, data sufficiency, or learning.\n",
        encoding="utf-8",
    )
    entries = _manifest(run_dir)
    print(json.dumps({"overall_status": overall, "manifest_entries": entries}, indent=2))
    return 0 if overall == "PASS_ADAPTER_SINGLE_WORLD_SMOKE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
