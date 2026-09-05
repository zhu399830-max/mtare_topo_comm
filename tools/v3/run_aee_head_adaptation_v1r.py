#!/home/zeng-workstation/anaconda3/bin/python
"""Execute the immutable V1R AEE three-seed head-only adaptation."""

from __future__ import annotations

import run_aee_head_adaptation_v1 as base


base.RUN_ID = "gate2_20260821_aee_head_adaptation_v1r_seed20260820"
base.STATUS_PASS = "PASS_AEE_HEAD_ADAPTATION_V1R"
base.STATUS_FAIL = "FAIL_AEE_HEAD_ADAPTATION_V1R"


if __name__ == "__main__":
    raise SystemExit(base.main())
