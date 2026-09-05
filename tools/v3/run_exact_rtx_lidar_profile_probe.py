#!/usr/bin/env python3
"""Run and seal the approved one-sensor exact-profile Isaac RTX probe."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


IMAGE = "nvcr.io/nvidia/isaac-sim:6.0.1"
EXPECTED_IMAGE_DIGEST = "sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9"
RUN_ID = "gate0_20260810_cano_exact_profile_writer_probe_v2_seed0"


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


def _seal_manifest(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    destination.write_text(
        "\n".join(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}" for path in files)
        + "\n",
        encoding="utf-8",
    )
    return len(files)


def _render_range_image(scan_path: Path, output_path: Path) -> dict[str, Any]:
    with np.load(scan_path) as data:
        ranges = np.asarray(data["range_m"], dtype=np.float32)
        valid = np.asarray(data["valid_mask"], dtype=bool)
    if ranges.shape != (16, 720) or valid.shape != (16, 720):
        raise RuntimeError(f"unexpected raster shapes: ranges={ranges.shape}, valid={valid.shape}")
    valid_ranges = ranges[valid]
    if valid_ranges.size == 0 or not np.all(np.isfinite(valid_ranges)):
        raise RuntimeError("complete scan contains no finite valid ranges")
    display = np.where(valid, ranges, np.nan)
    figure, axis = plt.subplots(figsize=(13.0, 4.8), constrained_layout=True)
    image = axis.imshow(
        display,
        origin="lower",
        aspect="auto",
        extent=(0.0, 360.0, -16.0, 16.0),
        interpolation="nearest",
        cmap="viridis",
        vmin=0.3,
        vmax=50.0,
    )
    axis.set_xlabel("Azimuth in sensor frame (deg)")
    axis.set_ylabel("Elevation (deg)")
    axis.set_title(
        "Gate 0 exact-profile creation probe — first complete RTX scan\n"
        "1 analytic box / 1 sensor / 1 pose / 0 formal dataset samples"
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Range (m); invalid cells are blank")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return {
        "raster_shape": list(ranges.shape),
        "valid_count": int(valid.sum()),
        "valid_ratio": float(valid.mean()),
        "valid_range_min_m": float(np.min(valid_ranges)),
        "valid_range_median_m": float(np.median(valid_ranges)),
        "valid_range_max_m": float(np.max(valid_ranges)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    run_dir = args.run_dir.resolve()
    spec = load_json(spec_path)
    if run_dir.name != RUN_ID or not run_dir.is_dir():
        raise RuntimeError(f"runner accepts only created run directory {RUN_ID}")
    if spec.get("operation") != "sensor_smoke" or spec.get("seed") != 0:
        raise RuntimeError("spec is not the approved exact-profile probe")
    scope = spec.get("data_scope", {})
    if scope.get("cano_worlds") != 0 or scope.get("diagnostic_complete_scans") != 1:
        raise RuntimeError("probe scope drifted from 0 Cano worlds and 1 diagnostic scan")

    source_json = PROJECT_ROOT / spec["source_profile"]
    sensor_usda = PROJECT_ROOT / spec["sensor_usda"]
    image_identity = _image_identity()
    tool_paths = {
        "runner": Path(__file__).resolve(),
        "isaac_probe": PROJECT_ROOT / "tools/v3/isaac/probe_exact_rtx_lidar_profile.py",
        "unit_test": PROJECT_ROOT / "tests/v3/unit/test_exact_rtx_profile_probe.py",
        "source_profile": source_json,
        "sensor_usda": sensor_usda,
    }
    observed_hashes = {name: _sha256(path) for name, path in tool_paths.items()}
    for name, frozen in spec.get("frozen_tools", {}).items():
        if name not in observed_hashes:
            raise RuntimeError(f"unverified frozen tool: {name}")
        if observed_hashes[name] != frozen.get("sha256"):
            raise RuntimeError(f"frozen hash mismatch for {name}: {observed_hashes[name]}")

    write_json(run_dir / "config/tool_hashes.json", observed_hashes)
    write_json(run_dir / "config/isaac_image.json", image_identity)
    shutil.copy2(source_json, run_dir / "config/source_profile.json")
    shutil.copy2(sensor_usda, run_dir / "artifacts/sensor_asset.usda")
    (run_dir / "artifacts/sensor_asset.sha256").write_text(
        observed_hashes["sensor_usda"] + "\n", encoding="utf-8"
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "One exact-profile sensor creation probe; zero Cano/formal data/training/model counts.",
        },
    )

    container_run = f"/workspace/{run_dir.relative_to(PROJECT_ROOT)}"
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
        "/workspace/tools/v3/isaac/probe_exact_rtx_lidar_profile.py",
        "--probe-sensor-usda",
        "/workspace/" + str(sensor_usda.relative_to(PROJECT_ROOT)),
        "--probe-source-json",
        "/workspace/" + str(source_json.relative_to(PROJECT_ROOT)),
        "--probe-output-dir",
        container_run + "/artifacts/runtime",
        "--probe-metrics-dir",
        container_run + "/metrics",
    ]
    capture_rc, capture_seconds = _run_logged(
        command, run_dir / "logs/01_isaac_exact_profile_probe.log", 600
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

    visualization_metrics = None
    visualization_error = None
    scan_path = run_dir / "artifacts/runtime/complete_scan.npz"
    capture_summary_path = run_dir / "metrics/capture_summary.json"
    if scan_path.is_file() and capture_summary_path.is_file():
        try:
            visualization_metrics = _render_range_image(
                scan_path,
                run_dir / "previews/range_image.png",
            )
            write_json(
                run_dir / "previews/README.json",
                {
                    "schema_version": "exact_rtx_profile_preview_v1",
                    "run_id": RUN_ID,
                    "question": "Did the exact local-USDA profile produce a non-empty complete RTX scan?",
                    "provenance": "First complete GMO scan in a deterministic analytic closed box.",
                    "supports": "Sensor creation, raster shape, finite non-empty return, and gross range sanity only.",
                    "does_not_support": "Cano mesh quality, dataset quality, labels, learning, topology, planning, or M-TARE performance.",
                    "metrics": visualization_metrics,
                },
            )
        except Exception as error:
            visualization_error = f"{type(error).__name__}: {error}"

    required_runtime_files = [
        run_dir / "metrics/attribute_readback.json",
        run_dir / "metrics/sensor_checker.json",
        run_dir / "metrics/gmo_summary.json",
        run_dir / "metrics/capture_summary.json",
        run_dir / "artifacts/runtime/complete_scan.npz",
        run_dir / "artifacts/runtime/probe_stage.usda",
        run_dir / "previews/range_image.png",
    ]
    missing = [str(path.relative_to(run_dir)) for path in required_runtime_files if not path.is_file()]
    complete_scan_evidence = (
        scan_path.is_file()
        and capture_summary_path.is_file()
        and load_json(capture_summary_path).get("overall_status")
        == "PASS_EXACT_PROFILE_COMPLETE_SCAN"
    )
    overall = (
        "PASS_EXACT_PROFILE_CREATION_PROBE"
        if complete_scan_evidence and not missing and visualization_error is None
        else "FAIL_EXACT_PROFILE_CREATION_PROBE"
    )
    summary = {
        "schema_version": "exact_rtx_profile_probe_summary_v1",
        "run_id": RUN_ID,
        "overall_status": overall,
        "isaac_exit_code": capture_rc,
        "duration_seconds": capture_seconds,
        "missing_required_evidence": missing,
        "visualization_error": visualization_error,
        "visualization_metrics": visualization_metrics,
        "counts": {
            "analytic_test_worlds": 1,
            "cano_worlds": 0,
            "sensors": 1,
            "poses": 1,
            "complete_diagnostic_scans": 1 if complete_scan_evidence else 0,
            "formal_dataset_worlds": 0,
            "formal_dataset_samples": 0,
            "training_samples": 0,
            "models_trained": 0,
        },
        "host": {
            "platform": platform.platform(),
            "free_disk_bytes_after": shutil.disk_usage(PROJECT_ROOT).free,
        },
        "claim_boundary": (
            "PASS proves only that the exact frozen custom profile can be loaded through the "
            "Isaac Sim 6.0.1 local-USDA path, validated, read back, and used for one complete "
            "RTX scan in an analytic box. It does not test Cano, create a dataset, train a "
            "model, build topology, or authorize the 24-pose rerun."
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
            "note": summary["claim_boundary"],
        },
    )
    sealed_files = _seal_manifest(run_dir)
    print(json.dumps({"overall_status": overall, "sealed_files": sealed_files}, indent=2))
    return 0 if overall.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
