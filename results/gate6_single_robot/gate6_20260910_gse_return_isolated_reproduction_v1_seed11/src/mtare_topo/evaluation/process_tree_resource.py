"""Fail-closed process-tree resource monitoring for formal subprocesses."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def process_tree_rss_bytes(root_pid: int) -> tuple[int, list[int]]:
    """Return summed RSS and live PIDs for a root plus recursive descendants."""
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
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            pass


def run_monitored_process(
    argv: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    trace_path: Path,
    time_limit_seconds: float,
    rss_limit_bytes: int,
    sample_interval_seconds: float,
    stop_grace_seconds: float = 30.0,
) -> tuple[int, float, dict]:
    """Run a fresh process group and enforce elapsed-time and tree-RSS limits."""
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
            rss, pids = process_tree_rss_bytes(os.getpid())
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
                and time.monotonic() - stop_sent_at > stop_grace_seconds
            ):
                _signal_process_group(process, signal.SIGKILL)
                forced_kill = True
            time.sleep(sample_interval_seconds)
        exit_code = int(process.wait())
        duration = time.monotonic() - started
        final_rss, final_pids = process_tree_rss_bytes(os.getpid())
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
            "definition": "Sum of psutil RSS for the batch runner and its live recursive descendants during one fresh child process.",
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
