#!/usr/bin/env python3
"""V4 complete kTestID-contract wrapper for the frozen planner-seed trial V3."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import mtare_planner_seed_trial_v3 as trial_v3


FULL_COMMS_SINGLE_ROBOT_TEST_ID = "0001"
_rewrite_v3 = trial_v3.rewrite_planner_command


def rewrite_planner_command_v4(command: str) -> str:
    rewritten = _rewrite_v3(command)
    if trial_v3.PLANNER_MARKER not in command:
        return rewritten
    if not rewritten.endswith(" test_id:=0"):
        raise RuntimeError("V3 intermediate kTestID mapping contract drift")
    return rewritten[:-1] + FULL_COMMS_SINGLE_ROBOT_TEST_ID


def start_v4(command: str, log_path: Path) -> tuple[subprocess.Popen[bytes], Any]:
    return trial_v3._start_v2(rewrite_planner_command_v4(command), log_path)


if __name__ == "__main__":
    trial_v3.trial_v2.trial_v1.start = start_v4
    raise SystemExit(trial_v3.trial_v2.trial_v1.main())
