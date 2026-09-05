#!/usr/bin/env python3
"""V2 environment-only wrapper for the frozen planner-seed trial V1."""

from __future__ import annotations

import mtare_planner_seed_trial_v1 as trial_v1


NEEDLE = "source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash\n"
INSERTION = NEEDLE + "source /home/docker-user/mtare/tare_system/devel/setup.bash --extend\n"
if trial_v1.SOURCE_ENV.count(NEEDLE) != 1:
    raise RuntimeError("frozen V1 source-environment contract drift")
trial_v1.SOURCE_ENV = trial_v1.SOURCE_ENV.replace(NEEDLE, INSERTION)


if __name__ == "__main__":
    raise SystemExit(trial_v1.main())
