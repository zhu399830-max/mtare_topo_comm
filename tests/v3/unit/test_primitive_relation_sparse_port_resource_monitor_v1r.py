from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from run_primitive_relation_sparse_port_three_seed_training_v1r import (
    _parse_proc_status,
    _read_process_memory,
    _run_training_monitored,
)


def test_proc_status_memory_is_typed_and_converted_from_kib() -> None:
    parsed = _parse_proc_status(
        "Name:\tpython\nVmHWM:\t2048 kB\nVmRSS:\t1024 kB\n"
    )
    assert parsed == {
        "current_rss_bytes": 1024 * 1024,
        "peak_rss_bytes": 2048 * 1024,
    }


def test_proc_status_rejects_missing_peak_memory() -> None:
    with pytest.raises(ValueError, match="lacks VmRSS or VmHWM"):
        _parse_proc_status("VmRSS:\t1024 kB\n")


def test_live_process_memory_reader_observes_current_process() -> None:
    value = _read_process_memory(os.getpid())
    assert value is not None
    assert value["peak_rss_bytes"] >= value["current_rss_bytes"] > 0


def test_monitor_terminates_process_that_exceeds_host_limit(tmp_path: Path) -> None:
    result = _run_training_monitored(
        [
            sys.executable,
            "-c",
            "import time; value=bytearray(1024*1024); time.sleep(5)",
        ],
        log=tmp_path / "over_limit.log",
        environment=os.environ.copy(),
        timeout_seconds=10,
        poll_seconds=0.01,
        maximum_host_rss_bytes=1,
    )
    assert result["killed_for_host_limit"] is True
    assert result["returncode"] != 0
