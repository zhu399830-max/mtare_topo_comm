#!/usr/bin/env python3
"""Run and seal the 2-GiB-RSS-qualified 20-parent perception-mesh batch once."""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from _bootstrap import PROJECT_ROOT
from execute_aee_corrective_perception_mesh_v1 import EXPECTED_SCOPE, SOURCE_RUN
from mtare_topo.governance import load_json, write_json
from run_aee_corrective_topology_candidate_audit_v1 import (
    E1,
    _approved_project_file,
    _environment,
    _sha256,
    _verify_environment_and_sources,
)
from run_cano_five_topology_cpu_contract_pilot import _seal_manifest


RUN_ID = "gate2_20260822_aee_corrective_perception_mesh_v1r2_seed20260821"
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_aee_corrective_perception_mesh_v1.py"
TIME_LIMIT_SECONDS = 7200
DISK_LIMIT_BYTES = 1024**3
RSS_LIMIT_BYTES = 2 * 1024**3
RSS_SAMPLE_INTERVAL_SECONDS = 0.25
STOP_GRACE_SECONDS = 30.0


def _verify_source_seal() -> dict:
    manifest = SOURCE_RUN / "artifacts/evidence_sha256.txt"
    lines = [line for line in manifest.read_text().splitlines() if line.strip()]
    mismatch = []
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = PROJECT_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            mismatch.append(relative)
    return {
        "entries": len(lines),
        "mismatch_count": len(mismatch),
        "mismatches": mismatch,
        "manifest_sha256": _sha256(manifest),
    }


def _process_tree_rss_bytes(root_pid: int) -> tuple[int, list[int]]:
    """Return the sum of RSS over a live root process and all live descendants."""
    try:
        root = psutil.Process(root_pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0, []
    rss = 0
    pids = []
    for process in processes:
        try:
            rss += int(process.memory_info().rss)
            pids.append(int(process.pid))
        except psutil.Error:
            continue
    return rss, sorted(set(pids))


def _signal_process_group(process: subprocess.Popen, sig: signal.Signals) -> None:
    """Signal the isolated executor process group, failing back to its root."""
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            pass


def _run_monitored(
    argv: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    trace_path: Path,
    time_limit_seconds: float,
    rss_limit_bytes: int,
    sample_interval_seconds: float,
) -> tuple[int, float, dict]:
    """Run one child while enforcing time and process-tree RSS contracts."""
    started = time.monotonic()
    peak_rss = 0
    peak_pids: list[int] = []
    samples = 0
    stop_reason = None
    stop_sent_at = None
    forced_kill = False
    with log_path.open("w", encoding="utf-8") as log_stream, trace_path.open(
        "w", encoding="utf-8"
    ) as trace_stream:
        log_stream.write("argv=" + json.dumps(argv) + "\n")
        log_stream.write(
            "started_at_utc=" + datetime.now(timezone.utc).isoformat() + "\n"
        )
        log_stream.flush()
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=environment,
            text=True,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            rss, pids = _process_tree_rss_bytes(os.getpid())
            samples += 1
            if rss > peak_rss:
                peak_rss, peak_pids = rss, pids
            trace_stream.write(
                json.dumps(
                    {
                        "elapsed_seconds": elapsed,
                        "process_tree_rss_bytes": rss,
                        "process_count": len(pids),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            trace_stream.flush()
            if stop_reason is None and rss > rss_limit_bytes:
                stop_reason = "RSS_LIMIT_EXCEEDED"
                stop_sent_at = time.monotonic()
                _signal_process_group(process, signal.SIGINT)
            elif stop_reason is None and elapsed > time_limit_seconds:
                stop_reason = "TIME_LIMIT_EXCEEDED"
                stop_sent_at = time.monotonic()
                _signal_process_group(process, signal.SIGINT)
            elif (
                stop_reason is not None
                and stop_sent_at is not None
                and time.monotonic() - stop_sent_at > STOP_GRACE_SECONDS
            ):
                _signal_process_group(process, signal.SIGKILL)
                forced_kill = True
            time.sleep(sample_interval_seconds)
        exit_code = int(process.wait())
        duration = time.monotonic() - started
        final_rss, final_pids = _process_tree_rss_bytes(os.getpid())
        samples += 1
        if final_rss > peak_rss:
            peak_rss, peak_pids = final_rss, final_pids
        trace_stream.write(
            json.dumps(
                {
                    "elapsed_seconds": duration,
                    "process_tree_rss_bytes": final_rss,
                    "process_count": len(final_pids),
                    "final": True,
                },
                sort_keys=True,
            )
            + "\n"
        )
        resource = {
            "schema_version": "process_tree_rss_contract_v1",
            "definition": "Sum of psutil RSS for the runner process and all live recursive descendants.",
            "rss_limit_bytes": int(rss_limit_bytes),
            "sample_interval_seconds": float(sample_interval_seconds),
            "sample_count": int(samples),
            "peak_process_tree_rss_bytes": int(peak_rss),
            "peak_process_ids": peak_pids,
            "within_rss_limit": bool(peak_rss <= rss_limit_bytes),
            "time_limit_seconds": float(time_limit_seconds),
            "within_time_limit": bool(duration <= time_limit_seconds),
            "stop_reason": stop_reason,
            "forced_kill": forced_kill,
            "executor_exit_code": exit_code,
            "duration_seconds": duration,
        }
        log_stream.write(
            "\nfinished_at_utc=" + datetime.now(timezone.utc).isoformat() + "\n"
        )
        log_stream.write("resource_summary=" + json.dumps(resource, sort_keys=True) + "\n")
        log_stream.flush()
    return exit_code, duration, resource


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or not run_dir.is_dir():
        raise RuntimeError(f"runner accepts only {RUN_ID}")
    if load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("run is not in one-time state")
    resource_contract = spec.get("resource_contract", {})
    if (
        spec.get("operation") != "infrastructure"
        or spec.get("gate") != 2
        or spec.get("seed") != 20260821
        or spec.get("user_authorization", {}).get("status") != "APPROVED"
        or spec.get("data_scope") != EXPECTED_SCOPE
        or resource_contract.get("process_tree_rss_limit_bytes") != RSS_LIMIT_BYTES
        or resource_contract.get("rss_sample_interval_seconds")
        != RSS_SAMPLE_INTERVAL_SECONDS
    ):
        raise RuntimeError("approval, corrective mesh scope or resource contract mismatch")

    proposal_path = _approved_project_file(spec.get("config_path"), "config_path")
    card_path = _approved_project_file(spec.get("data_card"), "data_card")
    proposal = load_json(proposal_path)
    card = load_json(card_path)
    if (
        proposal.get("status") != "APPROVED_FOR_ONE_EXECUTION"
        or proposal.get("approval", {}).get("status") != "APPROVED"
        or card.get("status") != "APPROVED_FOR_ONE_EXECUTION"
        or card.get("approval", {}).get("status") != "APPROVED"
    ):
        raise RuntimeError("approved proposal/data card missing")

    paths = {
        "runner": Path(__file__).resolve(),
        "executor": EXECUTOR,
        "m0_helpers": PROJECT_ROOT / "tools/v3/execute_cano_100_parent_perception_mesh_contract_m0.py",
        "m1r_helpers": PROJECT_ROOT / "tools/v3/execute_cano_100_parent_perception_mesh_m1r.py",
        "mesh_contract": PROJECT_ROOT / "src/mtare_topo/data/cano_perception_mesh_contract.py",
        "topology_executor": PROJECT_ROOT / "tools/v3/execute_aee_corrective_topology_candidate_audit_v1.py",
        "proposal": proposal_path,
        "data_card": card_path,
        "environment": _approved_project_file(
            spec.get("environment", {}).get("path"), "environment.path"
        ),
    }
    observed = {name: _sha256(path) for name, path in paths.items()}
    if set(observed) != set(spec.get("frozen_tools", {})):
        raise RuntimeError("frozen tool set mismatch")
    for name, digest in observed.items():
        if spec["frozen_tools"][name].get("sha256") != digest:
            raise RuntimeError(f"frozen tool mismatch: {name}")

    source_state = load_json(SOURCE_RUN / "RUN_STATE.json")
    source_seal = _verify_source_seal()
    if (
        source_state.get("overall_status")
        != "PASS_AEE_CORRECTIVE_TOPOLOGY_CANDIDATE_AUDIT_V1"
        or source_seal["entries"] != 376
        or source_seal["mismatch_count"] != 0
    ):
        raise RuntimeError("sealed topology source invalid")
    environment_audit = _verify_environment_and_sources(spec)
    write_json(run_dir / "config/tool_hashes.json", observed)
    write_json(run_dir / "config/source_seal_audit.json", source_seal)
    write_json(
        run_dir / "config/environment_and_source_audit.json", environment_audit
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "RUNNING",
            "note": "Twenty corrective perception meshes under a measured 2-GiB process-tree RSS hard limit.",
        },
    )

    argv = [str(E1), str(EXECUTOR), "--run-dir", str(run_dir)]
    code, duration, resource = _run_monitored(
        argv,
        cwd=PROJECT_ROOT,
        environment=_environment(),
        log_path=run_dir / "logs/01_corrective_mesh.log",
        trace_path=run_dir / "metrics/process_tree_rss_trace.jsonl",
        time_limit_seconds=TIME_LIMIT_SECONDS,
        rss_limit_bytes=RSS_LIMIT_BYTES,
        sample_interval_seconds=RSS_SAMPLE_INTERVAL_SECONDS,
    )
    write_json(run_dir / "metrics/resource_summary.json", resource)
    summary = (
        load_json(run_dir / "metrics/summary.json")
        if (run_dir / "metrics/summary.json").is_file()
        else None
    )
    meshes = list(run_dir.glob("artifacts/meshes/S*_C*/primary/mesh.obj"))
    sanitation = list(run_dir.glob("artifacts/meshes/S*_C*/primary/sanitation.json"))
    metrics = list(run_dir.glob("metrics/S*_C*.json"))
    previews = list(run_dir.glob("previews/train_complete_maps/*.png"))
    size = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    passed = bool(
        code == 0
        and summary
        and summary.get("overall_status") == "PASS_AEE_CORRECTIVE_PERCEPTION_MESH_V1"
        and summary.get("scope") == EXPECTED_SCOPE
        and summary.get("checks")
        and all(summary["checks"].values())
        and len(meshes) == len(sanitation) == len(metrics) == 20
        and len(previews) == 10
        and resource["within_rss_limit"]
        and resource["within_time_limit"]
        and resource["stop_reason"] is None
        and size <= DISK_LIMIT_BYTES
    )
    overall = (
        "PASS_AEE_CORRECTIVE_PERCEPTION_MESH_V1R2"
        if passed
        else "FAIL_AEE_CORRECTIVE_PERCEPTION_MESH_V1R2"
    )
    boundary = "Twenty immutable perception meshes only; zero LiDAR, teacher, formal data, training, C09/C10 or planner change."
    write_json(
        run_dir / "metrics/runner_summary.json",
        {
            "schema_version": "aee_corrective_perception_mesh_runner_v1r2",
            "run_id": RUN_ID,
            "overall_status": overall,
            "executor_exit_code": code,
            "duration_seconds": duration,
            "counts": {
                "meshes": len(meshes),
                "sanitation": len(sanitation),
                "metrics": len(metrics),
                "train_previews": len(previews),
            },
            "resource_contract": resource,
            "result_bytes_before_seal": size,
            "disk_limit_bytes": DISK_LIMIT_BYTES,
            "host": {"platform": platform.platform()},
            "claim_boundary": boundary,
            "corrective_change": "Measure runner-plus-executor process-tree RSS and enforce the approved 2-GiB hard limit.",
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": RUN_ID,
            "state": "COMPLETED" if passed else "FAILED",
            "overall_status": overall,
            "note": boundary,
        },
    )
    sealed = _seal_manifest(run_dir)
    print(
        json.dumps(
            {
                "overall_status": overall,
                "sealed_files": sealed,
                "peak_process_tree_rss_bytes": resource[
                    "peak_process_tree_rss_bytes"
                ],
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
