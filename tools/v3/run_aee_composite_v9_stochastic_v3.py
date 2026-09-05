#!/home/zeng-workstation/anaconda3/bin/python
"""Run the frozen 90-case matrix with the readiness-qualified V9 runtime."""

from __future__ import annotations

import run_mtare_single_robot_stochastic_v2 as base
from mtare_topo.deployment.m1d_checkpoint import COMPOSITE_V9_MODE


RUN_ID = "gate6_20260822_aee_composite_v9_stochastic_v3_seed20260820"
STATUS_PASS = "PASS_AEE_COMPOSITE_V9_STOCHASTIC_V3"
STATUS_FAIL = "FAIL_AEE_COMPOSITE_V9_STOCHASTIC_V3"


def main() -> int:
    base.RUN_ID = RUN_ID
    base.STATUS_PASS = STATUS_PASS
    base.STATUS_FAIL = STATUS_FAIL
    base.EXPECTED_READINESS_STATUS = "PASS_AEE_COMPOSITE_V9_READINESS_V1R3"
    base.EXPECTED_CHECKPOINT_MODE = COMPOSITE_V9_MODE
    base.TOPIC_CONTRACT = "configs/v3/gate6/closed_loop_recording_topics_aee_v2.json"
    base.CONTAINER_PREFIX = "mtare-v9-v3"
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
