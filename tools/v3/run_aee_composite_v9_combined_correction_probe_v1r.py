#!/home/zeng-workstation/anaconda3/bin/python
"""Run the V5 probe selected by the sealed V1R composed mechanism audit."""

from __future__ import annotations

import run_aee_composite_v9_combined_correction_probe_v1 as base


RUN_ID = "gate6_20260823_aee_composite_v9_combined_correction_probe_v1r_seed20260823"
STATUS_PASS = "PASS_AEE_COMPOSITE_V9_COMBINED_CORRECTION_PROBE_V1R"
STATUS_FAIL = "FAIL_AEE_COMPOSITE_V9_COMBINED_CORRECTION_PROBE_V1R"
AUDIT_STATUS_PASS = "PASS_COMPOSED_FRONTIER_ATTEMPT_AUDIT_V1R"


def main() -> int:
    base.RUN_ID = RUN_ID
    base.STATUS_PASS = STATUS_PASS
    base.STATUS_FAIL = STATUS_FAIL
    base.AUDIT_STATUS_PASS = AUDIT_STATUS_PASS
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
