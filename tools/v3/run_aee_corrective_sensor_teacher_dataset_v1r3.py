#!/home/zeng-workstation/anaconda3/bin/python
"""Monotonic-pairing replacement for the sealed V1R2 startup-order failure."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import run_aee_corrective_sensor_teacher_dataset_v1 as implementation
from mtare_topo.evaluation.process_tree_resource import run_monitored_process
from mtare_topo.governance import write_json


implementation.RUN_ID = "gate2_20260822_aee_corrective_sensor_teacher_dataset_v1r3_seed20260822"
implementation.STATUS_PASS = "PASS_AEE_CORRECTIVE_SENSOR_TEACHER_DATASET_V1R3"
implementation.STATUS_FAIL = "FAIL_AEE_CORRECTIVE_SENSOR_TEACHER_DATASET_V1R3"
implementation.CANO_EXECUTOR = "tools/v3/execute_aee_corrective_cano_dataset_v1r2.py"
implementation.CANO_STATUS_PASS = "PASS_AEE_CORRECTIVE_CANO_DATASET_V1R2"
implementation.CANO_EXPECTED_METRICS = {
    "candidate_clusters": 6_058,
    "candidate_frames_qualified": 30_290,
    "eligible_clusters": 6_043,
    "eligible_frames": 30_215,
    "ineligible_clusters": 15,
    "ineligible_frames": 21,
    "selected_clusters": 2_000,
    "frames": 10_000,
}
implementation.SENSOR_REQUIRED_PAIRING_METHOD = "strict_monotonic_nearest_timestamp_v1"

RSS_LIMIT_BYTES = 4 * 1024**3
RSS_SAMPLE_INTERVAL_SECONDS = 0.25
_base_collector_command = implementation.collector_command
_base_teacher_command = implementation.teacher_command


def _add_docker_memory_contract(command: list[str]) -> list[str]:
    if command[:2] != ["docker", "run"]:
        raise RuntimeError("expected docker run command for memory contract")
    return [*command[:2], "--memory", str(RSS_LIMIT_BYTES), "--memory-swap", str(RSS_LIMIT_BYTES), *command[2:]]


def collector_command(run_dir, trajectory):
    command, name = _base_collector_command(run_dir, trajectory)
    old = "/workspace/tools/v3/collect_aee_corrective_trajectory_v1r.py"
    new = "/workspace/tools/v3/collect_aee_corrective_trajectory_v1r3.py"
    replacements = sum(part.count(old) for part in command)
    if replacements != 1:
        raise RuntimeError("V1R3 collector command replacement drift")
    command = [part.replace(old, new) for part in command]
    return _add_docker_memory_contract(command), name


def teacher_command(run_dir, trajectory, complete_map):
    command, name = _base_teacher_command(run_dir, trajectory, complete_map)
    return _add_docker_memory_contract(command), name


def run_logged(command: list[str], name: str, log_path: Path, timeout_sec: float) -> None:
    resource_root = log_path.parents[1] / "metrics/resources"
    trace_root = log_path.parents[1] / "metrics/resource_traces"
    resource_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)
    resource_path = resource_root / f"{log_path.stem}.json"
    trace_path = trace_root / f"{log_path.stem}.jsonl"
    if log_path.exists() or resource_path.exists() or trace_path.exists():
        raise RuntimeError(f"resource/log output already exists for {name}")
    exit_code, _, resource = run_monitored_process(
        command,
        cwd=implementation.PROJECT_ROOT,
        environment=os.environ.copy(),
        log_path=log_path,
        trace_path=trace_path,
        time_limit_seconds=timeout_sec,
        rss_limit_bytes=RSS_LIMIT_BYTES,
        sample_interval_seconds=RSS_SAMPLE_INTERVAL_SECONDS,
    )
    resource.update(
        {
            "case_name": name,
            "docker_memory_limit_bytes": RSS_LIMIT_BYTES if command[:2] == ["docker", "run"] else None,
            "docker_memory_swap_limit_bytes": RSS_LIMIT_BYTES if command[:2] == ["docker", "run"] else None,
        }
    )
    write_json(resource_path, resource)
    if exit_code != 0 or not resource["within_rss_limit"] or resource["stop_reason"] is not None:
        if command[:2] == ["docker", "run"]:
            subprocess.run(["docker", "stop", "--time", "30", name], check=False, capture_output=True)
        raise RuntimeError(f"case failed resource/exit contract: {name}: {resource}")


implementation.collector_command = collector_command
implementation.teacher_command = teacher_command
implementation.run_logged = run_logged


if __name__ == "__main__":
    raise SystemExit(implementation.main())
