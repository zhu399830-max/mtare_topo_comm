from __future__ import annotations

import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[3] / "tools" / "v3"
sys.path.insert(0, str(TOOLS))
import mtare_planner_seed_trial_v4 as trial_v4
import run_mtare_planner_seed_qualification_v4 as qualification_v4


def test_v4_uses_complete_full_comms_single_robot_test_id() -> None:
    command = (
        "roslaunch /workspace/configs/v3/gate5/roslaunch/explore_seeded.launch "
        "scenario:=tunnel planner_seed:=23 robot_num:=1 test_id:=23"
    )
    rewritten = trial_v4.rewrite_planner_command_v4(command)
    assert rewritten.endswith("test_id:=0001")
    test_id = rewritten.rsplit("test_id:=", 1)[1]
    assert test_id[0] == "0"
    assert int(test_id[2:4]) == 1
    assert "planner_seed:=23" in rewritten


def test_v4_preserves_nonplanner_command() -> None:
    command = "roslaunch system_seeded.launch gazebo_seed:=23"
    assert trial_v4.rewrite_planner_command_v4(command) == command


def test_v4_has_version_correct_status() -> None:
    assert qualification_v4.RUN_ID.endswith("qualification_v4_seed11")
    assert qualification_v4.STATUS_PASS.endswith("_V4")
    assert qualification_v4.STATUS_FAIL.endswith("_V4")
