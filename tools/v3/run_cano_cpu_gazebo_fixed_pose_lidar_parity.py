#!/usr/bin/env python3
"""Run and seal the approved CPU--Gazebo parity contract exactly once."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from execute_cano_cpu_gazebo_fixed_pose_lidar_parity import EXPECTED_SCOPE, IMAGE, IMAGE_ID, PLUGIN_PATH, PLUGIN_SHA256, SOURCE_RUN
from mtare_topo.governance import load_json, write_json
from run_cano_five_topology_cpu_contract_pilot import E1_PYTHON, _executor_environment, _seal_manifest, _sha256


RUN_ID = "gate0_20260812_cano_cpu_gazebo_fixed_pose_lidar_parity_v1_seed0"
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_cano_cpu_gazebo_fixed_pose_lidar_parity.py"
TIME_LIMIT_SECONDS = 3600
DISK_LIMIT_BYTES = 1024**3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or not run_dir.is_dir() or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("runner accepts only the frozen one-time parity run")
    if spec.get("operation") != "sensor_contract_pilot" or spec.get("gate") != 0 or spec.get("seed") != 0 or spec.get("data_scope") != EXPECTED_SCOPE or spec.get("user_authorization", {}).get("status") != "APPROVED":
        raise RuntimeError("parity approval/scope mismatch")
    proposal = PROJECT_ROOT / spec["config_path"]
    card = PROJECT_ROOT / spec["data_card"]
    if load_json(proposal).get("approval", {}).get("status") != "APPROVED" or load_json(card).get("approval", {}).get("status") != "APPROVED":
        raise RuntimeError("approved proposal/data card missing")

    paths = {
        "runner": Path(__file__).resolve(), "executor": EXECUTOR,
        "parity_module": PROJECT_ROOT / "src/mtare_topo/data/cano_gazebo_parity.py",
        "sensor_module": PROJECT_ROOT / "src/mtare_topo/data/cano_sensor_smoke.py",
        "collector": PROJECT_ROOT / "tools/v3/gazebo/collect_fixed_lidar_scans.py",
        "session": PROJECT_ROOT / "tools/v3/gazebo/run_fixed_lidar_session.sh",
        "unit_test": PROJECT_ROOT / "tests/v3/unit/test_cano_gazebo_parity.py",
        "external_test": PROJECT_ROOT / "tests/v3/external/run_cano_gazebo_analytic_control.py",
        "proposal": proposal, "data_card": card,
    }
    observed = {name: _sha256(path) for name, path in paths.items()}
    if set(observed) != set(spec.get("frozen_tools", {})) or any(observed[name] != spec["frozen_tools"][name]["sha256"] for name in observed):
        raise RuntimeError("frozen parity tool mismatch")
    source_state = load_json(SOURCE_RUN / "RUN_STATE.json")
    if source_state.get("state") != "COMPLETED" or source_state.get("overall_status") != "PASS_CANO_100_PARENT_PERCEPTION_MESH_M1R":
        raise RuntimeError("M1R source run is not PASS")
    for relative, expected in spec.get("frozen_source_files", {}).items():
        if _sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"frozen source mismatch: {relative}")
    image_id = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
    plugin_sha = subprocess.check_output(["docker", "run", "--rm", "--network", "none", "-v", "/etc/localtime:/etc/localtime:ro", IMAGE, "sha256sum", PLUGIN_PATH], text=True).split()[0]
    if image_id != IMAGE_ID or plugin_sha != PLUGIN_SHA256:
        raise RuntimeError("container/plugin identity mismatch")
    if shutil.disk_usage(PROJECT_ROOT).free < 2 * 1024**3:
        raise RuntimeError("less than 2 GiB free")
    environment_check = subprocess.run(
        [str(E1_PYTHON), "-c", "import json,sys,numpy,scipy,open3d,matplotlib; print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'scipy':scipy.__version__,'open3d':open3d.__version__,'matplotlib':matplotlib.__version__}))"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    expected_environment = {"python": "3.12.3", "numpy": "1.26.4", "scipy": "1.12.0", "open3d": "0.19.0", "matplotlib": "3.11.1"}
    observed_environment = json.loads(environment_check.stdout) if environment_check.returncode == 0 else {}
    pip_check = subprocess.run([str(E1_PYTHON), "-m", "pip", "check"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if observed_environment != expected_environment or pip_check.returncode != 0:
        raise RuntimeError(f"E1 environment mismatch: {observed_environment}; pip_check={pip_check.stdout}")
    write_json(run_dir / "config/tool_hashes.json", observed)
    write_json(run_dir / "config/container_identity.json", {"image_id": image_id, "plugin_sha256": plugin_sha})
    write_json(run_dir / "config/e1_environment.json", {"observed": observed_environment, "expected": expected_environment, "pip_check_exit_code": pip_check.returncode, "pip_check_output": pip_check.stdout.strip()})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING", "note": "Approved diagnostic-only CPU--Gazebo parity."})

    argv = [str(E1_PYTHON), str(EXECUTOR), "--run-dir", str(run_dir)]
    started = time.monotonic()
    code = 124
    output = ""
    try:
        done = subprocess.run(argv, cwd=PROJECT_ROOT, env=_executor_environment(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=TIME_LIMIT_SECONDS, check=False)
        code, output = done.returncode, done.stdout
    except subprocess.TimeoutExpired as exc:
        captured = exc.stdout or ""
        output = captured.decode() if isinstance(captured, bytes) else captured
        output += f"\nTIMEOUT after {TIME_LIMIT_SECONDS}s\n"
    duration = time.monotonic() - started
    (run_dir / "logs/01_parity.log").write_text("argv=" + json.dumps(argv) + f"\nfinished_at_utc={datetime.now(timezone.utc).isoformat()}\n" + output + f"\nduration_seconds={duration:.6f}\nexit_code={code}\n", encoding="utf-8")
    print(output, end="")

    summary_path = run_dir / "metrics/summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else None
    pose_metrics = list(run_dir.glob("metrics/S*_C01__*.json"))
    pages = list((run_dir / "previews/parity_pages").glob("*.png")) if (run_dir / "previews/parity_pages").is_dir() else []
    size = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    passed = bool(code == 0 and duration <= TIME_LIMIT_SECONDS and size <= DISK_LIMIT_BYTES and summary and summary.get("overall_status") == "PASS_CANO_CPU_GAZEBO_FIXED_POSE_LIDAR_PARITY_V1" and summary.get("scope") == EXPECTED_SCOPE and len(pose_metrics) == 24 and len(pages) == 6)
    overall = "PASS_CANO_CPU_GAZEBO_FIXED_POSE_LIDAR_PARITY_V1" if passed else "FAIL_CANO_CPU_GAZEBO_FIXED_POSE_LIDAR_PARITY_V1"
    boundary = "Diagnostic sensor parity only; zero formal dataset, labels, training, models, trajectories, online graph and M-TARE changes."
    write_json(run_dir / "metrics/runner_summary.json", {"schema_version": "cano_cpu_gazebo_parity_runner_summary_v1", "run_id": RUN_ID, "overall_status": overall, "executor_exit_code": code, "duration_seconds": duration, "pose_metric_count": len(pose_metrics), "complete_page_count": len(pages), "result_bytes_before_seal": size, "host": platform.platform(), "claim_boundary": boundary})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if passed else "FAILED", "overall_status": overall, "note": boundary})
    sealed = _seal_manifest(run_dir)
    print(json.dumps({"overall_status": overall, "sealed_files": sealed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
