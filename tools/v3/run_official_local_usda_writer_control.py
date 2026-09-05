#!/usr/bin/env python3
"""Run and seal the approved exact-local-official-USDA Writer control v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import selectors
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


IMAGE = "nvcr.io/nvidia/isaac-sim:6.0.1"
EXPECTED_IMAGE_DIGEST = "sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9"
OFFICIAL_REFERENCE = "/isaac-sim/standalone_examples/api/isaacsim.sensors.experimental.rtx/inspect_lidar_gmo.py"
EXPECTED_OFFICIAL_REFERENCE_SHA256 = "fb6ab0cb1599d3a84bdc47f92e0db8cee7481dca240880cb20b440c23da89d46"
ASSET_PATH = PROJECT_ROOT / "results/gate0_baseline/gate0_20260810_isaac_official_lidar_asset_freeze_v1_seed0/artifacts/official_asset/Example_Rotary.usda"
EXPECTED_ASSET_SHA256 = "0812faf5c310f40316d5a11ab0c6786e19e18ac12cea10edc2e7ee44fc56c8c6"
EXPECTED_ASSET_BYTES = 15137
RUN_ID = "gate0_20260810_isaac_official_local_usda_writer_control_v2_seed0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_logged(argv: list[str], log_path: Path, timeout_s: int) -> tuple[int, float]:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("argv=" + json.dumps(argv, ensure_ascii=False) + "\n")
        log.write(f"cwd={PROJECT_ROOT}\nstarted_at_utc={datetime.now(timezone.utc).isoformat()}\n")
        log.flush()
        process = subprocess.Popen(
            argv,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        return_code = None
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
    return int(return_code), duration


def _image_identity() -> dict[str, Any]:
    raw = subprocess.check_output(
        ["docker", "image", "inspect", IMAGE], text=True, stderr=subprocess.STDOUT
    )
    data = json.loads(raw)[0]
    repo_digests = data.get("RepoDigests", [])
    if not any(item.endswith(EXPECTED_IMAGE_DIGEST) for item in repo_digests):
        raise RuntimeError(f"Isaac image digest mismatch: {repo_digests}")
    return {"image": IMAGE, "id": data.get("Id"), "repo_digests": repo_digests}


def _official_reference_hash() -> str:
    raw = subprocess.check_output(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "sha256sum", IMAGE, OFFICIAL_REFERENCE],
        text=True,
        stderr=subprocess.STDOUT,
    )
    observed = raw.split()[0]
    if observed != EXPECTED_OFFICIAL_REFERENCE_SHA256:
        raise RuntimeError(f"official reference hash mismatch: {observed}")
    return observed


def _seal_manifest(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    destination.write_text(
        "\n".join(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}" for path in files) + "\n",
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
    if run_dir.name != RUN_ID or not run_dir.is_dir():
        raise RuntimeError(f"runner accepts only created run directory {RUN_ID}")
    if spec.get("operation") != "sensor_smoke" or spec.get("seed") != 0:
        raise RuntimeError("spec is not the approved local official Writer runtime control")
    scope = spec.get("data_scope", {})
    if (
        scope.get("official_control_worlds") != 1
        or scope.get("cano_worlds") != 0
        or scope.get("exact_official_local_sensors") != 1
        or scope.get("custom_project_sensors") != 0
        or scope.get("formal_dataset_samples") != 0
    ):
        raise RuntimeError("local official control scope drifted")

    observed_asset_hash = _sha256(ASSET_PATH)
    observed_asset_bytes = ASSET_PATH.stat().st_size
    if observed_asset_hash != EXPECTED_ASSET_SHA256 or observed_asset_bytes != EXPECTED_ASSET_BYTES:
        raise RuntimeError(
            f"frozen official asset mismatch: sha256={observed_asset_hash}, bytes={observed_asset_bytes}"
        )
    image_identity = _image_identity()
    official_reference_hash = _official_reference_hash()
    tool_paths = {
        "runner": Path(__file__).resolve(),
        "isaac_probe": PROJECT_ROOT / "tools/v3/isaac/probe_official_local_usda_writer_runtime.py",
        "unit_test": PROJECT_ROOT / "tests/v3/unit/test_official_local_usda_writer_control.py",
    }
    observed_hashes = {name: _sha256(path) for name, path in tool_paths.items()}
    for name, frozen in spec.get("frozen_tools", {}).items():
        if name not in observed_hashes or observed_hashes[name] != frozen.get("sha256"):
            raise RuntimeError(f"frozen hash mismatch for {name}: {observed_hashes.get(name)}")

    write_json(run_dir / "config/tool_hashes.json", observed_hashes)
    write_json(run_dir / "config/isaac_image.json", image_identity)
    write_json(
        run_dir / "config/official_reference.json",
        {"path_in_image": OFFICIAL_REFERENCE, "sha256": official_reference_hash},
    )
    write_json(
        run_dir / "config/frozen_official_asset.json",
        {
            "path": str(ASSET_PATH.relative_to(PROJECT_ROOT)),
            "sha256": observed_asset_hash,
            "bytes": observed_asset_bytes,
            "dependency_audit": "PASS_SELF_CONTAINED_USDA",
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "Exact frozen official local USDA Writer health control; zero project data/training/model counts.",
        },
    )

    container_run = f"/workspace/{run_dir.relative_to(PROJECT_ROOT)}"
    container_asset = f"/workspace/{ASSET_PATH.relative_to(PROJECT_ROOT)}"
    command = [
        "docker",
        "run",
        "--rm",
        "--user",
        "0:0",
        "--gpus",
        "all",
        "--network",
        "none",
        "--shm-size",
        "2g",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "PRIVACY_CONSENT=Y",
        "-v",
        f"{PROJECT_ROOT}:/workspace:rw",
        "--entrypoint",
        "/isaac-sim/python.sh",
        IMAGE,
        "/workspace/tools/v3/isaac/probe_official_local_usda_writer_runtime.py",
        "--official-usda",
        container_asset,
        "--expected-usda-sha256",
        EXPECTED_ASSET_SHA256,
        "--control-output-dir",
        container_run + "/artifacts/runtime",
        "--control-metrics-dir",
        container_run + "/metrics",
    ]
    capture_rc, duration = _run_logged(
        command, run_dir / "logs/01_official_local_usda_writer_control.log", 600
    )
    subprocess.run(
        [
            "docker", "run", "--rm", "--network", "none", "--user", "0:0",
            "-v", f"{run_dir}:/target:rw", "--entrypoint", "chown", IMAGE,
            "-R", f"{os.getuid()}:{os.getgid()}", "/target",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    runtime_path = run_dir / "metrics/runtime_control.json"
    failure_path = run_dir / "metrics/runtime_control_failure.json"
    runtime = load_json(runtime_path) if runtime_path.is_file() else None
    failure = load_json(failure_path) if failure_path.is_file() else None
    log_text = (run_dir / "logs/01_official_local_usda_writer_control.log").read_text(
        encoding="utf-8", errors="replace"
    )
    drain_timeout = "Timed out while waiting for pending Replicator writer schedules to drain" in log_text
    if runtime is None:
        overall = "FAIL_OFFICIAL_LOCAL_USDA_WRITER_CONTROL_NO_SUMMARY"
        classification = "LOCAL_USDA_CONTROL_IMPLEMENTATION_OR_RUNTIME_EXCEPTION"
    elif runtime.get("overall_status") == "PASS_OFFICIAL_LOCAL_USDA_WRITER_RUNTIME_HEALTH":
        overall = "PASS_OFFICIAL_LOCAL_USDA_WRITER_RUNTIME_CONTROL"
        classification = "CUSTOM_SENSOR_OR_SCENE_COUPLING_REMAINS"
    else:
        overall = "FAIL_OFFICIAL_LOCAL_USDA_WRITER_RUNTIME_CONTROL"
        classification = runtime.get("classification", "UNKNOWN_RUNTIME_FAILURE")

    required = [
        runtime_path,
        run_dir / "artifacts/runtime/official_local_usda_control_stage.usda",
    ]
    missing = [str(path.relative_to(run_dir)) for path in required if not path.is_file()]
    if missing:
        overall = "FAIL_OFFICIAL_LOCAL_USDA_WRITER_CONTROL_MISSING_EVIDENCE"

    evidence_state = runtime or failure or {}
    execution_state = evidence_state.get("execution_state", {})
    summary = {
        "schema_version": "official_local_usda_writer_runtime_control_summary_v2",
        "run_id": RUN_ID,
        "overall_status": overall,
        "classification": classification,
        "isaac_exit_code": capture_rc,
        "duration_seconds": duration,
        "writer_drain_timeout_in_log": drain_timeout,
        "missing_required_evidence": missing,
        "runtime_metrics": runtime,
        "runtime_failure": failure,
        "execution_state": execution_state,
        "counts": {
            "official_control_worlds": 1,
            "cano_worlds": 0,
            "exact_official_local_sensor_creation_success": int(
                execution_state.get("sensor_creation_success", 0)
            ),
            "writer_attach_success": int(execution_state.get("writer_attach_success", 0)),
            "render_frames_updated": int(execution_state.get("render_frames_updated", 0)),
            "custom_project_sensors": 0,
            "saved_point_samples": 0,
            "formal_dataset_samples": 0,
            "labels": 0,
            "training_samples": 0,
            "models": 0,
        },
        "frozen_official_asset": {
            "sha256": observed_asset_hash,
            "bytes": observed_asset_bytes,
        },
        "host": {
            "platform": platform.platform(),
            "free_disk_bytes_after": shutil.disk_usage(PROJECT_ROOT).free,
        },
        "claim_boundary": (
            "This result classifies only Writer runtime behavior for the exact frozen official "
            "local USDA in the official control scene. The official sensor cannot replace the "
            "project sensor or enter any dataset, training, topology, or M-TARE pipeline."
        ),
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if overall.startswith("PASS") else "FAILED",
            "overall_status": overall,
            "classification": classification,
            "note": summary["claim_boundary"],
        },
    )
    sealed_files = _seal_manifest(run_dir)
    print(
        json.dumps(
            {"overall_status": overall, "classification": classification, "sealed_files": sealed_files},
            indent=2,
        )
    )
    return 0 if overall.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
