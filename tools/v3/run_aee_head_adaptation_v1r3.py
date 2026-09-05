#!/home/zeng-workstation/anaconda3/bin/python
"""Execute the V1R3 venv-path-semantics replacement for AEE adaptation."""

from __future__ import annotations

import run_aee_head_adaptation_v1 as base


base.RUN_ID = "gate2_20260821_aee_head_adaptation_v1r3_seed20260820"
base.STATUS_PASS = "PASS_AEE_HEAD_ADAPTATION_V1R3"
base.STATUS_FAIL = "FAIL_AEE_HEAD_ADAPTATION_V1R3"


if __name__ == "__main__":
    raise SystemExit(base.main())
