#!/usr/bin/env python3
"""Run and seal the approved one-world Cano CPU raycast contract exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate0_20260810_cano_cpu_raycast_lidar_contract_v1_seed0"
E1_PYTHON = Path("/tmp/mtare_cano_compat_e1_py312_np126_sp112/bin/python")
WORLD_DIR = PROJECT_ROOT / "results/gate0_baseline/gate0_20260810_cano_readonly_audited_adapter_smoke_v1_seed0/artifacts/world_000"
REFERENCE_DIR = PROJECT_ROOT / "results/gate0_baseline/gate0_20260810_cano_native_mesh_lidar_label_smoke_v1_seed0/artifacts/cpu_reference"
SENSOR_CONFIG = PROJECT_ROOT / "configs/v3/gate0/sensors/mtare_vlp16_720_50m_v1.json"
FREEZE_EVIDENCE = PROJECT_ROOT / "results/gate0_baseline/gate0_20260810_cano_dependency_compatibility_matrix_v1_seed0/config/pip_freeze_E1.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_manifest_sha256(directory: Path) -> tuple[str, list[dict[str, Any]]]:
    records = [
        {"name": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(directory.glob("*.npz"))
    ]
    canonical = "".join(
        f"{item['sha256']}  {item['name']}  {item['bytes']}\n" for item in records
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), records


def _e1_identity() -> dict[str, Any]:
    if not E1_PYTHON.is_file():
        raise RuntimeError(f"frozen E1 Python is absent: {E1_PYTHON}")
    program = (
        "import json,matplotlib,numpy,open3d,scipy,sys;"
        "print(json.dumps({'python':sys.version.split()[0],'executable':sys.executable,"
        "'numpy':numpy.__version__,'scipy':scipy.__version__,"
        "'open3d':open3d.__version__,'matplotlib':matplotlib.__version__},sort_keys=True))"
    )
    identity = json.loads(subprocess.check_output([str(E1_PYTHON), "-c", program], text=True))
    expected = {
        "python": "3.12.3",
        "numpy": "1.26.4",
        "scipy": "1.12.0",
        "open3d": "0.19.0",
        "matplotlib": "3.11.1",
    }
    observed = {key: identity[key] for key in expected}
    if observed != expected:
        raise RuntimeError(f"frozen E1 version mismatch: observed={observed}, expected={expected}")
    pip_check = subprocess.run(
        [str(E1_PYTHON), "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    identity["pip_check_exit_code"] = pip_check.returncode
    identity["pip_check_output"] = pip_check.stdout.strip()
    if pip_check.returncode != 0:
        raise RuntimeError(f"frozen E1 pip check failed: {pip_check.stdout}")
    identity["pip_freeze_evidence"] = str(FREEZE_EVIDENCE.relative_to(PROJECT_ROOT))
    identity["pip_freeze_evidence_sha256"] = _sha256(FREEZE_EVIDENCE)
    return identity


def _seal_manifest(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    destination.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files),
        encoding="utf-8",
    )
    return len(files)


def _run_executor(run_dir: Path) -> tuple[int, float]:
    argv = [
        str(E1_PYTHON),
        str(PROJECT_ROOT / "tools/v3/execute_cano_cpu_raycast_lidar_contract.py"),
        "--world-dir",
        str(WORLD_DIR),
        "--reference-dir",
        str(REFERENCE_DIR),
        "--run-dir",
        str(run_dir),
        "--sensor-config",
        str(SENSOR_CONFIG),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    started = time.monotonic()
    completed = subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
    )
    duration = time.monotonic() - started
    log_path = run_dir / "logs/01_cpu_raycast_contract.log"
    log_path.write_text(
        "argv=" + json.dumps(argv, ensure_ascii=False) + "\n"
        + f"started_at_utc={datetime.now(timezone.utc).isoformat()}\n"
        + completed.stdout
        + f"\nduration_seconds={duration:.6f}\nexit_code={completed.returncode}\n",
        encoding="utf-8",
    )
    print(completed.stdout, end="", flush=True)
    return completed.returncode, duration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    spec = load_json(spec_path)
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or not run_dir.is_dir():
        raise RuntimeError(f"runner accepts only the created run directory {RUN_ID}")
    authorization = spec.get("user_authorization", {})
    if (
        spec.get("operation") != "sensor_smoke"
        or spec.get("gate") != 0
        or spec.get("seed") != 0
        or authorization.get("status") != "APPROVED"
    ):
        raise RuntimeError("spec is not the approved Gate-0 CPU raycast sensor smoke")
    expected_scope = {
        "source_worlds": 1,
        "new_worlds_generated": 0,
        "poses": 24,
        "rays_per_pose": 11520,
        "primary_rays_per_pass": 276480,
        "independent_scene_passes": 2,
        "formal_dataset_worlds": 0,
        "formal_dataset_samples": 0,
        "labels_for_training": 0,
        "training_samples": 0,
        "models": 0,
        "isaac_runs": 0,
        "gazebo_runs": 0,
        "topology_changes": 0,
        "mtare_changes": 0,
    }
    if spec.get("data_scope") != expected_scope:
        raise RuntimeError(f"approved CPU contract scope drifted: {spec.get('data_scope')}")

    paths = {
        "runner": Path(__file__).resolve(),
        "executor": PROJECT_ROOT / "tools/v3/execute_cano_cpu_raycast_lidar_contract.py",
        "geometry": PROJECT_ROOT / "src/mtare_topo/data/cano_sensor_smoke.py",
        "unit_test": PROJECT_ROOT / "tests/v3/unit/test_cano_cpu_raycast_lidar_contract.py",
        "sensor_config": SENSOR_CONFIG,
        "proposal": PROJECT_ROOT / "configs/v3/gate0/cano_cpu_raycast_lidar_contract_v1.proposal.json",
        "data_card": PROJECT_ROOT / "configs/v3/gate0/data_cards/cano_cpu_raycast_lidar_contract_v1.json",
    }
    observed_tool_hashes = {name: _sha256(path) for name, path in paths.items()}
    for name, frozen in spec.get("frozen_tools", {}).items():
        if observed_tool_hashes.get(name) != frozen.get("sha256"):
            raise RuntimeError(
                f"frozen hash mismatch for {name}: {observed_tool_hashes.get(name)} != {frozen.get('sha256')}"
            )

    observed_world_hashes = {
        name: _sha256(WORLD_DIR / name)
        for name in ("mesh.obj", "graph.json", "splines.json", "fta_dist.txt", "metadata.json")
    }
    if observed_world_hashes != spec.get("frozen_world_hashes"):
        raise RuntimeError("frozen source-world identity mismatch")
    reference_manifest_hash, reference_records = _directory_manifest_sha256(REFERENCE_DIR)
    if len(reference_records) != 24:
        raise RuntimeError(f"expected exactly 24 prior references, found {len(reference_records)}")
    if reference_manifest_hash != spec.get("prior_reference_manifest_sha256"):
        raise RuntimeError("prior-reference manifest identity mismatch")
    environment_identity = _e1_identity()
    if environment_identity["pip_freeze_evidence_sha256"] != spec.get("pip_freeze_evidence_sha256"):
        raise RuntimeError("frozen E1 pip-freeze evidence identity mismatch")

    write_json(run_dir / "config/tool_hashes.json", observed_tool_hashes)
    write_json(run_dir / "config/source_world_hashes.json", observed_world_hashes)
    write_json(
        run_dir / "config/prior_reference_manifest.json",
        {"manifest_sha256": reference_manifest_hash, "records": reference_records},
    )
    write_json(run_dir / "config/e1_environment.json", environment_identity)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "Approved one-world CPU diagnostic contract; no formal data or training.",
        },
    )

    exit_code, duration = _run_executor(run_dir)
    contract_path = run_dir / "metrics/contract.json"
    contract = load_json(contract_path) if contract_path.is_file() else None
    required = [
        run_dir / "metrics/analytic_control.json",
        run_dir / "metrics/per_pose.json",
        contract_path,
        run_dir / "artifacts/pose_manifest.json",
        run_dir / "previews/full_world_pose_map.png",
        run_dir / "previews/full_world_pose_map.provenance.json",
        run_dir / "previews/all_24_range_label_contact_sheet.png",
        run_dir / "previews/all_24_range_label_contact_sheet.provenance.json",
    ]
    missing = [str(path.relative_to(run_dir)) for path in required if not path.is_file()]
    scan_count = len(list((run_dir / "artifacts/scans").glob("*.npz"))) if (run_dir / "artifacts/scans").is_dir() else 0
    result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    passed = bool(
        exit_code == 0
        and contract is not None
        and contract.get("overall_status") == "PASS"
        and not missing
        and scan_count == 24
        and result_bytes <= int(0.05 * 1024**3)
    )
    overall = "PASS_CANO_CPU_RAYCAST_LIDAR_CONTRACT" if passed else "FAIL_CANO_CPU_RAYCAST_LIDAR_CONTRACT"
    summary = {
        "schema_version": "cano_cpu_raycast_lidar_contract_summary_v1",
        "run_id": RUN_ID,
        "overall_status": overall,
        "executor_exit_code": exit_code,
        "duration_seconds": duration,
        "missing_required_evidence": missing,
        "result_bytes_before_seal": result_bytes,
        "disk_limit_bytes": int(0.05 * 1024**3),
        "counts": {
            "source_worlds": 1,
            "new_worlds_generated": 0,
            "diagnostic_poses": 24,
            "diagnostic_scan_npz": scan_count,
            "primary_rays_per_pose": 11520,
            "primary_rays_per_pass": 276480,
            "independent_scene_passes": 2,
            "formal_dataset_worlds": 0,
            "formal_dataset_samples": 0,
            "labels_for_training": 0,
            "training_samples": 0,
            "models": 0,
            "isaac_runs": 0,
            "gazebo_runs": 0,
            "topology_changes": 0,
            "mtare_changes": 0,
        },
        "contract_metrics": contract,
        "host": {
            "platform": platform.platform(),
            "free_disk_bytes_after": shutil.disk_usage(PROJECT_ROOT).free,
        },
        "next_step_boundary": (
            "A pass authorizes only preparation of a separately approved five-topology CPU-raycast "
            "contract pilot. It does not authorize formal data generation, training, topology or M-TARE work."
        ),
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if passed else "FAILED",
            "overall_status": overall,
            "note": summary["next_step_boundary"],
        },
    )
    sealed_files = _seal_manifest(run_dir)
    print(json.dumps({"overall_status": overall, "sealed_files": sealed_files}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
