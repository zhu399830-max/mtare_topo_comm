#!/usr/bin/env python3
"""V4 kTestID-contract wrapper for the frozen planner-seed qualification V3."""

from __future__ import annotations

import subprocess
from pathlib import Path

from _bootstrap import PROJECT_ROOT
import run_mtare_planner_seed_qualification_v3 as qualification_v3


RUN_ID = "gate5_20260820_mtare_planner_seed_qualification_v4_seed11"
STATUS_PASS = "PASS_MTARE_PLANNER_SEED_QUALIFICATION_V4"
STATUS_FAIL = "FAIL_MTARE_PLANNER_SEED_QUALIFICATION_V4"


def docker_trial(run_dir: Path, trial_id: str, seed: int, log_path: Path) -> None:
    command = [
        "docker", "run", "--rm", "--name", f"mtare-seed-qualification-v4-{trial_id}",
        "-v", f"{PROJECT_ROOT}:/workspace:ro",
        "-v", f"{run_dir / 'artifacts'}:/evidence:rw",
        "--entrypoint", "/bin/bash", qualification_v3.IMAGE, "-lc",
        "source /opt/ros/noetic/setup.bash && "
        "source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash && "
        "source /home/docker-user/mtare/tare_system/devel/setup.bash --extend && "
        "export PYTHONPATH=/workspace/src:$PYTHONPATH && "
        "python3 /workspace/tools/v3/mtare_planner_seed_trial_v4.py "
        f"--trial-dir /evidence/trials/{trial_id} --seed {seed} "
        "--duration-sec 60 --audit-frames 100 --audit-waypoints 20",
    ]
    with log_path.open("xb") as stream:
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode != 0:
        raise RuntimeError(f"trial {trial_id} failed with exit code {completed.returncode}")


def main() -> int:
    qualification_v3.RUN_ID = RUN_ID
    qualification_v3.STATUS_PASS = STATUS_PASS
    qualification_v3.STATUS_FAIL = STATUS_FAIL
    qualification_v3.docker_trial = docker_trial
    return qualification_v3.main()


if __name__ == "__main__":
    raise SystemExit(main())
