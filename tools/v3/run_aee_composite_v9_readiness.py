#!/home/zeng-workstation/anaconda3/bin/python
"""Run and seal six AEE closed-loop readiness cases with the V9 composite runtime."""

from __future__ import annotations

import run_aee_adapted_readiness_v1 as base
from mtare_topo.deployment.m1d_checkpoint import COMPOSITE_V9_MODE


RUN_ID = "gate6_20260822_aee_composite_v9_readiness_v1r3_seed20260822"
STATUS_PASS = "PASS_AEE_COMPOSITE_V9_READINESS_V1R3"
STATUS_FAIL = "FAIL_AEE_COMPOSITE_V9_READINESS_V1R3"
DEPLOYMENT_STATUS_PASS = "PASS_AEE_COMPOSITE_V9_ROS_DEPLOYMENT_EXPORT"


def main() -> int:
    base.RUN_ID = RUN_ID
    base.STATUS_PASS = STATUS_PASS
    base.STATUS_FAIL = STATUS_FAIL
    base.DEPLOYMENT_STATUS_PASS = DEPLOYMENT_STATUS_PASS
    base.EXPECTED_CHECKPOINT_MODE = COMPOSITE_V9_MODE
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
