#!/usr/bin/env python3
"""V3 parameter-mapping wrapper for the frozen planner-seed trial V2."""

from __future__ import annotations

import re
from pathlib import Path
import subprocess
from typing import Any

import mtare_planner_seed_trial_v2 as trial_v2


PLANNER_MARKER = "roslaunch /workspace/configs/v3/gate5/roslaunch/explore_seeded.launch "
TEST_ID_PATTERN = re.compile(r" test_id:=\d+$")


def rewrite_planner_command(command: str) -> str:
    """Keep experiment seeds separate from M-TARE's communication-mode ID."""

    if PLANNER_MARKER not in command:
        return command
    if command.count(" planner_seed:=") != 1:
        raise RuntimeError("planner command must contain exactly one planner_seed")
    rewritten, count = TEST_ID_PATTERN.subn(" test_id:=0", command)
    if count != 1:
        raise RuntimeError("planner command must end in exactly one numeric test_id")
    return rewritten


_start_v2 = trial_v2.trial_v1.start


def start_v3(command: str, log_path: Path) -> tuple[subprocess.Popen[bytes], Any]:
    return _start_v2(rewrite_planner_command(command), log_path)


trial_v2.trial_v1.start = start_v3


if __name__ == "__main__":
    raise SystemExit(trial_v2.trial_v1.main())
