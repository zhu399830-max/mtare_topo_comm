#!/usr/bin/env python3
"""Run and seal the approved 24-pose Cano/Isaac LiDAR diagnostic smoke."""

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


E1_PYTHON = Path("/tmp/mtare_cano_compat_e1_py312_np126_sp112/bin/python")
IMAGE = "nvcr.io/nvidia/isaac-sim:6.0.1"
EXPECTED_IMAGE_DIGEST = "sha256:783444"
WORLD_RELATIVE = Path(
    "results/gate0_baseline/"
    "gate0_20260810_cano_readonly_audited_adapter_smoke_v1_seed0/"
    "artifacts/world_000"
)
SENSOR_RELATIVE = Path("configs/v3/gate0/sensors/mtare_vlp16_720_50m_v1.json")
CONFIG_CONTAINER_PATH = (
    "/isaac-sim/extscache/omni.sensors.nv.common-3.0.0+f9bf0dda.lx64.r.cp312/"
    "data/lidar/MTARE_VLP16_720_50M_V1.json"
)


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


def _seal_manifest(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    destination.write_text(
        "\n".join(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}" for path in files)
        + "\n",
        encoding="utf-8",
    )
    return len(files)


def _image_identity() -> dict[str, Any]:
    raw = subprocess.check_output(
        ["docker", "image", "inspect", IMAGE], text=True, stderr=subprocess.STDOUT
    )
    data = json.loads(raw)[0]
    repo_digests = data.get("RepoDigests", [])
    identity = {"image": IMAGE, "id": data.get("Id"), "repo_digests": repo_digests}
    if not any(EXPECTED_IMAGE_DIGEST in item for item in repo_digests):
        raise RuntimeError(f"Isaac image digest mismatch: {identity}")
    return identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    run_dir = args.run_dir.resolve()
    spec = load_json(spec_path)
    if not run_dir.is_dir() or run_dir.name != "gate0_20260810_cano_native_mesh_lidar_label_smoke_v1_seed0":
        raise RuntimeError("runner accepts only the frozen Gate-0 sensor-smoke directory")
    if spec.get("operation") != "sensor_smoke" or spec.get("seed") != 0:
        raise RuntimeError("run spec is not the approved seed-0 sensor smoke")
    if spec.get("data_scope", {}).get("diagnostic_lidar_poses") != 24:
        raise RuntimeError("run spec must freeze exactly 24 diagnostic poses")
    if not E1_PYTHON.is_file():
        raise RuntimeError("frozen E1 Python is missing")

    world_dir = PROJECT_ROOT / WORLD_RELATIVE
    sensor_config = PROJECT_ROOT / SENSOR_RELATIVE
    image_identity = _image_identity()
    tool_paths = {
        "runner": Path(__file__).resolve(),
        "prepare": PROJECT_ROOT / "tools/v3/prepare_cano_lidar_sensor_smoke.py",
        "isaac_capture": PROJECT_ROOT / "tools/v3/isaac/capture_cano_lidar_sensor_smoke.py",
        "finalize": PROJECT_ROOT / "tools/v3/finalize_cano_lidar_sensor_smoke.py",
        "geometry_module": PROJECT_ROOT / "src/mtare_topo/data/cano_sensor_smoke.py",
        "unit_test": PROJECT_ROOT / "tests/v3/unit/test_cano_sensor_smoke.py",
        "sensor_config": sensor_config,
        "spec": spec_path,
    }
    observed_tool_hashes = {name: _sha256(path) for name, path in tool_paths.items()}
    for name, frozen in spec.get("frozen_tools", {}).items():
        if name not in observed_tool_hashes:
            raise RuntimeError(f"frozen tool is not checked by runner: {name}")
        if observed_tool_hashes[name] != frozen.get("sha256"):
            raise RuntimeError(
                f"frozen tool hash mismatch for {name}: {observed_tool_hashes[name]}"
            )
    write_json(
        run_dir / "config/tool_hashes.json",
        {
            name: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": observed_tool_hashes[name]}
            for name, path in tool_paths.items()
        },
    )
    write_json(run_dir / "config/isaac_image.json", image_identity)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": run_dir.name,
            "state": "RUNNING",
            "note": "Approved one-world 24-pose sensor diagnostic; not a formal dataset.",
        },
    )

    print("[1/3] Prepare frozen poses, spline labels, USD, and CPU reference", flush=True)
    prepare_command = [
        str(E1_PYTHON),
        "tools/v3/prepare_cano_lidar_sensor_smoke.py",
        "--world-dir",
        str(world_dir),
        "--run-dir",
        str(run_dir),
        "--sensor-config",
        str(sensor_config),
    ]
    prepare_rc, prepare_seconds = _run_logged(
        prepare_command, run_dir / "logs/01_prepare_cpu_reference.log", 600
    )

    capture_rc = None
    capture_seconds = None
    if prepare_rc == 0:
        print("[2/3] Capture exactly 24 stationary scans in Isaac Sim 6.0.1 RTX", flush=True)
        capture_command = [
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
            "-v",
            f"{sensor_config}:{CONFIG_CONTAINER_PATH}:ro",
            "--entrypoint",
            "/isaac-sim/python.sh",
            IMAGE,
            "/workspace/tools/v3/isaac/capture_cano_lidar_sensor_smoke.py",
            "--capture-stage-path",
            f"/workspace/{run_dir.relative_to(PROJECT_ROOT)}/artifacts/isaac_stage.usda",
            "--capture-manifest-path",
            f"/workspace/{run_dir.relative_to(PROJECT_ROOT)}/artifacts/pose_manifest.json",
            "--capture-output-dir",
            f"/workspace/{run_dir.relative_to(PROJECT_ROOT)}/artifacts/rtx_capture",
            "--capture-metrics-path",
            f"/workspace/{run_dir.relative_to(PROJECT_ROOT)}/metrics/isaac_capture.json",
            "--capture-config-name",
            "MTARE_VLP16_720_50M_V1",
        ]
        capture_rc, capture_seconds = _run_logged(
            capture_command, run_dir / "logs/02_isaac_rtx_capture.log", 1800
        )
        ownership_command = [
            "docker",
            "run",
            "--rm",
            "--user",
            "0:0",
            "-v",
            f"{run_dir}:/target:rw",
            "--entrypoint",
            "chown",
            IMAGE,
            "-R",
            f"{os.getuid()}:{os.getgid()}",
            "/target",
        ]
        subprocess.run(ownership_command, check=True, stdout=subprocess.DEVNULL)
    finalize_rc = None
    finalize_seconds = None
    if capture_rc == 0:
        print("[3/3] Compare RTX/CPU and render all 24 samples", flush=True)
        finalize_command = [
            str(E1_PYTHON),
            "tools/v3/finalize_cano_lidar_sensor_smoke.py",
            "--run-dir",
            str(run_dir),
            "--world-dir",
            str(world_dir),
        ]
        finalize_rc, finalize_seconds = _run_logged(
            finalize_command, run_dir / "logs/03_finalize_compare_visualize.log", 900
        )

    if prepare_rc != 0:
        overall = "FAIL_CPU_OR_POSE_PREPARATION"
    elif capture_rc != 0:
        overall = "FAIL_ISAAC_RTX_CAPTURE"
    elif finalize_rc != 0:
        overall = "FAIL_RTX_CPU_ACCEPTANCE"
    else:
        overall = "PASS_SENSOR_SMOKE"
    summary = {
        "schema_version": "cano_native_mesh_lidar_label_smoke_summary_v1",
        "run_id": run_dir.name,
        "overall_status": overall,
        "durations_seconds": {
            "prepare": prepare_seconds,
            "isaac_capture": capture_seconds,
            "finalize": finalize_seconds,
        },
        "exit_codes": {
            "prepare": prepare_rc,
            "isaac_capture": capture_rc,
            "finalize": finalize_rc,
        },
        "counts": {
            "source_worlds": 1,
            "planned_poses": 24,
            "formal_dataset_worlds": 0,
            "training_samples": 0,
            "models_trained": 0,
        },
        "host": {
            "platform": platform.platform(),
            "free_disk_bytes_after": shutil.disk_usage(PROJECT_ROOT).free,
        },
        "claim_boundary": (
            "PASS proves only that one audited Cano native mesh can yield stable Isaac RTX "
            "LiDAR and deterministic 5 m spline-direction labels at 24 frozen static poses. "
            "It is not a dataset, representation result, trained model, topology planner, or benchmark."
        ),
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": run_dir.name,
            "state": "COMPLETED" if overall == "PASS_SENSOR_SMOKE" else "FAILED",
            "overall_status": overall,
            "note": summary["claim_boundary"],
        },
    )
    count = _seal_manifest(run_dir)
    print(json.dumps({"overall_status": overall, "sealed_files": count}, indent=2))
    return 0 if overall == "PASS_SENSOR_SMOKE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
