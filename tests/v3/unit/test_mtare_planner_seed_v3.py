from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[3] / "tools" / "v3"
sys.path.insert(0, str(TOOLS))
import mtare_planner_seed_trial_v3 as trial_v3
import run_mtare_planner_seed_qualification_v3 as qualification_v3


def test_v3_catkin_environment_extends_aee_with_tare() -> None:
    environment = trial_v3.trial_v2.trial_v1.SOURCE_ENV
    assert "autonomous_exploration_development_environment/devel/setup.bash" in environment
    assert "tare_system/devel/setup.bash --extend" in environment


def test_v3_maps_test_id_to_full_comms_without_changing_seed() -> None:
    command = (
        "roslaunch /workspace/configs/v3/gate5/roslaunch/explore_seeded.launch "
        "scenario:=tunnel planner_seed:=23 rviz:=false test_id:=23"
    )
    rewritten = trial_v3.rewrite_planner_command(command)
    assert "planner_seed:=23" in rewritten
    assert rewritten.endswith("test_id:=0")


def test_v3_does_not_rewrite_nonplanner_commands() -> None:
    command = "roslaunch system_seeded.launch gazebo_seed:=23 test_id:=23"
    assert trial_v3.rewrite_planner_command(command) == command


@pytest.mark.parametrize(
    "command",
    [
        trial_v3.PLANNER_MARKER + "scenario:=tunnel test_id:=11",
        trial_v3.PLANNER_MARKER + "scenario:=tunnel planner_seed:=11",
        trial_v3.PLANNER_MARKER + "planner_seed:=11 planner_seed:=23 test_id:=11",
    ],
)
def test_v3_rejects_ambiguous_planner_mapping(command: str) -> None:
    with pytest.raises(RuntimeError):
        trial_v3.rewrite_planner_command(command)


def test_v3_has_version_correct_evidence_labels() -> None:
    assert qualification_v3.RUN_ID.endswith("qualification_v3_seed11")
    assert qualification_v3.STATUS_PASS.endswith("_V3")
    assert qualification_v3.STATUS_FAIL.endswith("_V3")
