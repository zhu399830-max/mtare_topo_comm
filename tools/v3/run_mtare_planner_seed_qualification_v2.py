#!/usr/bin/env python3
"""V2 environment-only wrapper for the frozen planner-seed qualification V1."""

from __future__ import annotations

import subprocess
from pathlib import Path

from _bootstrap import PROJECT_ROOT
import run_mtare_planner_seed_qualification_v1 as qualification_v1


RUN_ID = "gate5_20260820_mtare_planner_seed_qualification_v2_seed11"


def docker_trial(run_dir: Path, trial_id: str, seed: int, log_path: Path) -> None:
    command = [
        "docker", "run", "--rm", "--name", f"mtare-seed-qualification-v2-{trial_id}",
        "-v", f"{PROJECT_ROOT}:/workspace:ro",
        "-v", f"{run_dir / 'artifacts'}:/evidence:rw",
        "--entrypoint", "/bin/bash", qualification_v1.IMAGE, "-lc",
        "source /opt/ros/noetic/setup.bash && "
        "source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash && "
        "source /home/docker-user/mtare/tare_system/devel/setup.bash --extend && "
        "export PYTHONPATH=/workspace/src:$PYTHONPATH && "
        "python3 /workspace/tools/v3/mtare_planner_seed_trial_v2.py "
        f"--trial-dir /evidence/trials/{trial_id} --seed {seed} "
        "--duration-sec 60 --audit-frames 100 --audit-waypoints 20",
    ]
    with log_path.open("xb") as stream:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"trial {trial_id} failed with exit code {completed.returncode}")


qualification_v1.RUN_ID = RUN_ID
qualification_v1.docker_trial = docker_trial


if __name__ == "__main__":
    raise SystemExit(qualification_v1.main())
