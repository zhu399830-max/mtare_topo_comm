#!/usr/bin/env python3
"""Acquire, audit, and seal one approved official Isaac LiDAR USDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


IMAGE = "nvcr.io/nvidia/isaac-sim:6.0.1"
EXPECTED_IMAGE_DIGEST = "sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9"
SOURCE_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/"
    "Isaac/Sensors/NVIDIA/Example_Rotary.usda"
)
SOURCE_HOST = "omniverse-content-production.s3-us-west-2.amazonaws.com"
RUN_ID = "gate0_20260810_isaac_official_lidar_asset_freeze_v1_seed0"
MAX_BYTES = 10 * 1024 * 1024
USD_EXT = "/isaac-sim/extscache/omni.usd.libs-1.0.3+f9bf0dda.lx64.r.cp312"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_identity() -> dict[str, Any]:
    raw = subprocess.check_output(
        ["docker", "image", "inspect", IMAGE], text=True, stderr=subprocess.STDOUT
    )
    value = json.loads(raw)[0]
    repo_digests = value.get("RepoDigests", [])
    if not any(item.endswith(EXPECTED_IMAGE_DIGEST) for item in repo_digests):
        raise RuntimeError(f"Isaac image digest mismatch: {repo_digests}")
    return {"image": IMAGE, "id": value.get("Id"), "repo_digests": repo_digests}


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
    if spec.get("operation") != "audit" or spec.get("seed") != 0:
        raise RuntimeError("spec is not the approved official asset freeze audit")
    source = spec.get("source", {})
    if source.get("url") != SOURCE_URL:
        raise RuntimeError("source URL drifted from the approved exact registry asset")
    parsed = urlparse(SOURCE_URL)
    if parsed.scheme != "https" or parsed.hostname != SOURCE_HOST or parsed.query or parsed.fragment:
        raise RuntimeError("source URL violates the frozen HTTPS host/path contract")
    scope = spec.get("data_scope", {})
    if (
        scope.get("official_usda_requested") != 1
        or scope.get("maximum_followed_usd_dependencies") != 0
        or scope.get("gpu_runs") != 0
        or scope.get("formal_dataset_samples") != 0
    ):
        raise RuntimeError("asset-freeze scope drifted")

    image_identity = _image_identity()
    tool_paths = {
        "runner": Path(__file__).resolve(),
        "usd_auditor": PROJECT_ROOT / "tools/v3/isaac/audit_frozen_usda.py",
        "unit_test": PROJECT_ROOT / "tests/v3/unit/test_official_lidar_asset_freeze.py",
    }
    observed_hashes = {name: _sha256(path) for name, path in tool_paths.items()}
    for name, frozen in spec.get("frozen_tools", {}).items():
        if name not in observed_hashes or observed_hashes[name] != frozen.get("sha256"):
            raise RuntimeError(f"frozen hash mismatch for {name}: {observed_hashes.get(name)}")

    write_json(run_dir / "config/tool_hashes.json", observed_hashes)
    write_json(run_dir / "config/isaac_image.json", image_identity)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "One-file official USDA provenance acquisition; no Isaac/GPU/Writer/data operation.",
        },
    )

    asset_dir = run_dir / "artifacts/official_asset"
    asset_dir.mkdir(parents=True, exist_ok=False)
    asset_path = asset_dir / "Example_Rotary.usda"
    headers_path = run_dir / "config/http_response_headers.txt"
    curl_stderr = run_dir / "logs/01_https_acquisition.stderr.log"
    curl_command = [
        "curl",
        "--fail-with-body",
        "--silent",
        "--show-error",
        "--proto",
        "=https",
        "--connect-timeout",
        "20",
        "--max-time",
        "120",
        "--max-redirs",
        "0",
        "--max-filesize",
        str(MAX_BYTES),
        "--dump-header",
        str(headers_path),
        "--output",
        str(asset_path),
        "--write-out",
        "%{json}",
        SOURCE_URL,
    ]
    started = time.monotonic()
    with curl_stderr.open("w", encoding="utf-8") as stderr:
        acquired = subprocess.run(
            curl_command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            timeout=150,
            check=False,
        )
    acquisition_duration = time.monotonic() - started
    curl_metrics: dict[str, Any]
    try:
        curl_metrics = json.loads(acquired.stdout) if acquired.stdout else {}
    except json.JSONDecodeError:
        curl_metrics = {"raw_write_out": acquired.stdout}
    write_json(run_dir / "metrics/curl_writeout.json", curl_metrics)
    write_json(
        run_dir / "config/acquisition_command.json",
        {"argv": curl_command, "request_count": 1, "redirect_following": False},
    )

    http_code = int(curl_metrics.get("http_code", 0) or 0)
    effective_url = str(curl_metrics.get("url_effective", ""))
    redirect_value = curl_metrics.get("redirect_url")
    redirect_url = "" if redirect_value in (None, "") else str(redirect_value)
    bytes_downloaded = asset_path.stat().st_size if asset_path.is_file() else 0
    acquisition_ok = (
        acquired.returncode == 0
        and http_code == 200
        and effective_url == SOURCE_URL
        and not redirect_url
        and 0 < bytes_downloaded <= MAX_BYTES
    )

    dependency_report = None
    audit_exit_code = None
    audit_log = run_dir / "logs/02_openusd_dependency_audit.log"
    if acquisition_ok:
        audit_command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{PROJECT_ROOT}:/workspace:ro",
            "--entrypoint",
            "/bin/bash",
            IMAGE,
            "-lc",
            (
                f"PYTHONPATH={USD_EXT} LD_LIBRARY_PATH={USD_EXT}/bin:/isaac-sim/kit "
                "/isaac-sim/kit/python/bin/python3 /workspace/tools/v3/isaac/audit_frozen_usda.py "
                f"--asset /workspace/{asset_path.relative_to(PROJECT_ROOT)} "
                "--output -"
            ),
        ]
        audited = subprocess.run(
            audit_command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
        )
        audit_exit_code = audited.returncode
        audit_log.write_text(
            "stdout:\n" + audited.stdout + "\nstderr:\n" + audited.stderr,
            encoding="utf-8",
        )
        try:
            dependency_report = json.loads(audited.stdout) if audited.stdout else None
        except json.JSONDecodeError:
            dependency_report = None
        if isinstance(dependency_report, dict):
            write_json(run_dir / "metrics/usda_dependency_audit.json", dependency_report)

    if not acquisition_ok:
        overall = "FAIL_OFFICIAL_USDA_ACQUISITION"
    elif dependency_report is None:
        overall = "FAIL_OFFICIAL_USDA_PARSE_OR_EVIDENCE"
    elif dependency_report.get("overall_status") != "PASS_SELF_CONTAINED_USDA":
        overall = "FAIL_OFFICIAL_USDA_EXTERNAL_DEPENDENCIES"
    elif not dependency_report.get("is_usda_text"):
        overall = "FAIL_OFFICIAL_ASSET_NOT_USDA_TEXT"
    else:
        overall = "PASS_OFFICIAL_USDA_FROZEN_SELF_CONTAINED"

    summary = {
        "schema_version": "official_lidar_asset_freeze_summary_v1",
        "run_id": RUN_ID,
        "overall_status": overall,
        "source_url": SOURCE_URL,
        "source_host": SOURCE_HOST,
        "request_count": 1,
        "redirect_following": False,
        "curl_exit_code": acquired.returncode,
        "http_code": http_code,
        "effective_url": effective_url,
        "redirect_url": redirect_url,
        "downloaded_bytes": bytes_downloaded,
        "asset_sha256": _sha256(asset_path) if asset_path.is_file() else None,
        "acquisition_duration_seconds": acquisition_duration,
        "openusd_audit_exit_code": audit_exit_code,
        "dependency_audit": dependency_report,
        "counts": {
            "official_usda_requested": 1,
            "official_usda_downloaded": int(acquisition_ok),
            "followed_usd_dependencies": 0,
            "gpu_runs": 0,
            "isaac_simulation_runs": 0,
            "writer_runs": 0,
            "official_control_worlds": 0,
            "cano_worlds": 0,
            "project_sensors": 0,
            "saved_point_samples": 0,
            "formal_dataset_samples": 0,
            "labels": 0,
            "training_samples": 0,
            "models": 0,
        },
        "host": {
            "platform": platform.platform(),
            "free_disk_bytes_after": shutil.disk_usage(PROJECT_ROOT).free,
        },
        "claim_boundary": (
            "PASS freezes only one official diagnostic USDA and its dependency status. It does "
            "not authorize or establish sensor creation, Writer health, RTX output, project data, "
            "training, topology, or M-TARE behavior."
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
